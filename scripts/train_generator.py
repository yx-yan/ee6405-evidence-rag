"""Fine-tune the shared answer generator (BART) on data/generator/*.jsonl. Owner: D.

    python scripts/train_generator.py --out models/generator/bart-base           # full run (GPU)
    python scripts/train_generator.py --out /tmp/smoke --limit 64 --max-steps 3  # CPU smoke test
    python scripts/train_generator.py --out models/generator/bart-base --resume  # continue after a time limit

Inputs are formatted with evrag.generator.format_input, so training and inference agree. Logs a
line every --log-every steps (loss, lr, speed, ETA) to stdout and <out>/log.jsonl; evaluates
validation loss and ROUGE-L (closed and context examples separately) every --eval-every steps
and at the end of each epoch; keeps the best checkpoint by validation loss in <out>/best and the
latest in <out>/last (with optimizer state for --resume). At the end it decodes a test sample with
the shared decoding settings and writes <out>/test_eval.json and <out>/test_samples.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evrag.config import DATA_DIR, load_config  # noqa: E402
from evrag.generator import decoding, format_input, rouge_l  # noqa: E402


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def encode(rows, tokenizer, max_source, max_target):
    sources = [format_input(r["question"], r["contexts"]) for r in rows]
    enc = tokenizer(sources, max_length=max_source, truncation=True)
    labels = tokenizer(text_target=[r["answer"] for r in rows], max_length=max_target, truncation=True)
    return [{"input_ids": i, "labels": l} for i, l in zip(enc["input_ids"], labels["input_ids"])]


def batches(features, batch_size, rng, shuffle=True):
    """Length-grouped batches: sort within pools of 50 batches to cut padding, then shuffle."""
    order = list(range(len(features)))
    if shuffle:
        rng.shuffle(order)
    pool = batch_size * 50
    out = []
    for start in range(0, len(order), pool):
        block = sorted(order[start:start + pool], key=lambda i: len(features[i]["input_ids"]))
        out += [block[k:k + batch_size] for k in range(0, len(block), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return out


def collate(features, idx, pad_id, device):
    ids = [features[i]["input_ids"] for i in idx]
    labels = [features[i]["labels"] for i in idx]
    width, lwidth = max(map(len, ids)), max(map(len, labels))
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(ids), width), dtype=torch.long)
    label_ids = torch.full((len(ids), lwidth), -100, dtype=torch.long)
    for k, (x, y) in enumerate(zip(ids, labels)):
        input_ids[k, :len(x)] = torch.tensor(x)
        attention[k, :len(x)] = 1
        label_ids[k, :len(y)] = torch.tensor(y)
    return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device), "labels": label_ids.to(device)}


def autocast(device, bf16):
    if device == "cuda" and bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def val_loss(model, features, batch_size, pad_id, device, bf16):
    model.eval()
    total, n = 0.0, 0
    for idx in batches(features, batch_size, random.Random(0), shuffle=False):
        batch = collate(features, idx, pad_id, device)
        with autocast(device, bf16):
            loss = model(**batch).loss
        total += loss.item() * len(idx)
        n += len(idx)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def decode(model, tokenizer, rows, device, max_source, gen, batch_size=16):
    model.eval()
    preds = []
    for k in range(0, len(rows), batch_size):
        chunk = rows[k:k + batch_size]
        enc = tokenizer([format_input(r["question"], r["contexts"]) for r in chunk], max_length=max_source,
                        truncation=True, padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, **gen)
        preds += tokenizer.batch_decode(out, skip_special_tokens=True)
    model.train()
    return [p.strip() for p in preds]


def rouge_by_mode(rows, preds):
    scores = {}
    for mode in ("closed", "context"):
        pairs = [(p, r["answer"]) for p, r in zip(preds, rows) if r["mode"] == mode]
        if pairs:
            scores[f"rougeL_{mode}"] = round(sum(rouge_l(p, a) for p, a in pairs) / len(pairs), 4)
            scores[f"n_{mode}"] = len(pairs)
    return scores


def balanced_sample(rows, n, seed):
    rng = random.Random(seed)
    closed = [r for r in rows if r["mode"] == "closed"]
    context = [r for r in rows if r["mode"] == "context"]
    return rng.sample(closed, min(n // 2, len(closed))) + rng.sample(context, min(n - n // 2, len(context)))


def save(model, tokenizer, path: Path, state: dict | None = None, optimizer=None, scheduler=None):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if state is not None:
        (path / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path / "optimizer.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA_DIR / "generator")
    p.add_argument("--model", default="facebook/bart-base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-source", type=int, default=1024)
    p.add_argument("--max-target", type=int, default=256)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast on GPUs that support it")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-loss-samples", type=int, default=1000)
    p.add_argument("--eval-gen-samples", type=int, default=200)
    p.add_argument("--test-samples", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="use only the first N train/val/test rows (smoke test)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    seed = load_config()["seed"] if args.seed is None else args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    def log(record):
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    start_from = args.out / "last" if args.resume and (args.out / "last" / "trainer_state.json").exists() else None
    tokenizer = AutoTokenizer.from_pretrained(start_from or args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(start_from or args.model).to(device)
    model.train()

    train_rows = read_jsonl(args.data / "train.jsonl", args.limit)
    val_rows = read_jsonl(args.data / "val.jsonl", args.limit)
    test_rows = read_jsonl(args.data / "test.jsonl", args.limit)
    t0 = time.time()
    train = encode(train_rows, tokenizer, args.max_source, args.max_target)
    val_loss_rows = random.Random(seed).sample(val_rows, min(args.eval_loss_samples, len(val_rows)))
    val_feats = encode(val_loss_rows, tokenizer, args.max_source, args.max_target)
    val_gen_rows = balanced_sample(val_rows, args.eval_gen_samples, seed)
    log({"event": "start", "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
         "bf16": bf16, "model": str(start_from or args.model), "train": len(train), "val": len(val_rows),
         "test": len(test_rows), "tokenize_s": round(time.time() - t0, 1), "args": {k: str(v) for k, v in vars(args).items()}})

    steps_per_epoch = math.ceil(math.ceil(len(train) / args.batch_size) / args.grad_accum)
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    no_decay = ("bias", "layer_norm.weight", "layernorm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n.lower() for k in no_decay)],
               "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n.lower() for k in no_decay)],
               "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    warmup = max(1, int(total_steps * args.warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))
    state = {"step": 0, "epoch": 0, "best_val_loss": None}
    if start_from:
        state = json.loads((start_from / "trainer_state.json").read_text(encoding="utf-8"))
        saved = torch.load(start_from / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        log({"event": "resume", **state})

    last_eval = {"step": None}

    def evaluate(tag):
        last_eval["step"] = state["step"]
        loss = val_loss(model, val_feats, args.batch_size, tokenizer.pad_token_id, device, bf16)
        preds = decode(model, tokenizer, val_gen_rows, device, args.max_source,
                       {"num_beams": 1, "max_new_tokens": 128})
        record = {"event": "eval", "tag": tag, "step": state["step"], "val_loss": round(loss, 4),
                  **rouge_by_mode(val_gen_rows, preds)}
        if state["best_val_loss"] is None or loss < state["best_val_loss"]:
            state["best_val_loss"] = loss
            save(model, tokenizer, args.out / "best", dict(state))
            record["saved_best"] = True
        log(record)

    rng = random.Random(seed + state["epoch"])
    t_start, seen, start_step = time.time(), 0, state["step"]
    done = state["step"] >= total_steps
    while not done:
        plan = batches(train, args.batch_size, rng)
        # On resume, skip the micro-batches already trained in this epoch (order is re-drawn).
        skip = (state["step"] - state["epoch"] * steps_per_epoch) * args.grad_accum if start_from else 0
        for micro, idx in enumerate(plan[skip:], skip + 1):
            batch = collate(train, idx, tokenizer.pad_token_id, device)
            with autocast(device, bf16):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            seen += len(idx)
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                elapsed = time.time() - t_start
                rate = (state["step"] - start_step) / max(elapsed, 1e-9)
                log({"event": "train", "step": state["step"], "of": total_steps,
                     "epoch": round(state["epoch"] + micro / len(plan), 3), "loss": round(loss.item() * args.grad_accum, 4),
                     "lr": round(scheduler.get_last_lr()[0], 8), "examples_per_s": round(seen / max(elapsed, 1e-9), 1),
                     "eta_min": round((total_steps - state["step"]) / rate / 60, 1) if rate else None})
            if args.eval_every and state["step"] % args.eval_every == 0:
                evaluate(f"step{state['step']}")
                save(model, tokenizer, args.out / "last", dict(state), optimizer, scheduler)
            if state["step"] >= total_steps:
                done = True
                break
        else:
            state["epoch"] += 1
            evaluate(f"epoch{state['epoch']}")
            save(model, tokenizer, args.out / "last", dict(state), optimizer, scheduler)
            start_from = None
    if last_eval["step"] != state["step"]:
        evaluate("final")
    save(model, tokenizer, args.out / "last", dict(state), optimizer, scheduler)

    # Final test decode with the shared decoding settings, from the best checkpoint.
    best = AutoModelForSeq2SeqLM.from_pretrained(args.out / "best").to(device).eval()
    gen = decoding()
    test_sample = balanced_sample(test_rows, args.test_samples, seed + 1)
    preds = decode(best, tokenizer, test_sample, device, args.max_source, gen, batch_size=8)
    scores = {"split": "test", "n": len(test_sample), "decoding": gen, **rouge_by_mode(test_sample, preds)}
    (args.out / "test_eval.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    with (args.out / "test_samples.jsonl").open("w", encoding="utf-8") as f:
        for r, pred in zip(test_sample, preds):
            f.write(json.dumps({"id": r["id"], "mode": r["mode"], "question": r["question"], "answer": r["answer"],
                                "prediction": pred, "rougeL": round(rouge_l(pred, r["answer"]), 4)},
                               ensure_ascii=False) + "\n")
    log({"event": "done", **scores, "best_val_loss": state["best_val_loss"], "minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
