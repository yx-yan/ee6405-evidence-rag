"""Trial of baseline and evidence on the MedQuAD A/B/C draft, leaving data/processed untouched. Owner: D.

    python scripts/d_run_medquad_trial.py --split dev --per-scenario 20 --methods baseline evidence
    python scripts/d_run_medquad_trial.py --split dev --per-scenario 60 --retrieval-only   # no LLM calls

Reads data/medquad/{corpus.jsonl,dataset.csv} (built by `python -m evrag.data.medquad`), injects a
retriever over that corpus into each method, and writes
results/runs/<method>_medquad_<split><n>.jsonl in the run_experiment record format plus
results/runs/medquad_trial_<split><n>_summary.json (counts by scenario, token usage, cost estimate).
LLM errors are retried a few times and otherwise abort the run; they are never written as abstentions.
"""
import argparse
import csv
import json
import statistics
import time
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from evrag.config import ROOT, RUNS_DIR, load_config
from evrag.llm import LLMCallError, generate
from evrag.methods import get_method
from evrag.retriever import Retriever

DATA_DIR = ROOT / "data" / "medquad"
# USD per million input / output tokens, OpenAI list price (2025); only for a rough estimate.
PRICE_PER_M = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60)}


def load_questions(split, per_scenario):
    with (DATA_DIR / "dataset.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    picked, taken = [], {"A": 0, "B": 0, "C": 0}
    for row in rows:
        if taken[row["scenario"]] < per_scenario:
            picked.append(row)
            taken[row["scenario"]] += 1
    return picked


def prime_cache(config, data_dir, cache_dir):
    """Embed every MedQuAD chunk once (corpus_all.jsonl, keyed by text) and write B's cache file
    for the current corpus.jsonl from it, so rebuilding the hold-outs never re-embeds 30k chunks."""
    import hashlib
    import numpy as np
    from evrag.retriever import load_corpus

    everything = data_dir / "corpus_all.jsonl"
    if not everything.exists():
        return
    model_name = config["retrieval"]["embedding_model"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    master = cache_dir / "all_chunks.npz"
    keys, vectors = [], None
    if master.exists():
        stored = np.load(master, allow_pickle=False)
        if str(stored["model"]) == model_name:
            keys, vectors = [str(k) for k in stored["keys"]], stored["vectors"]
    index = {k: i for i, k in enumerate(keys)}

    def digest(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    all_docs = load_corpus(everything)
    missing = list(dict.fromkeys(d["text"] for d in all_docs if digest(d["text"]) not in index))
    if missing:
        from sentence_transformers import SentenceTransformer
        print(f"embedding {len(missing)} chunks once with {model_name} ...", flush=True)
        encoder = SentenceTransformer(model_name)
        new = encoder.encode(missing, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        new = np.asarray(new, dtype=np.float32)
        for text in missing:
            index[digest(text)] = len(keys)
            keys.append(digest(text))
        vectors = new if vectors is None else np.vstack([vectors, new])
        np.savez(master, model=np.array(model_name), keys=np.array(keys), vectors=vectors)
    documents = load_corpus(data_dir / "corpus.jsonl")
    fingerprint = hashlib.sha256(json.dumps({"format": 1, "model": model_name, "documents": documents},
                                            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = cache_dir / f"{fingerprint}.npy"
    if not target.exists():
        np.save(target, vectors[[index[digest(d["text"])] for d in documents]], allow_pickle=False)
    print(f"embedding cache: {len(keys)} chunks embedded in total, {len(missing)} new this run", flush=True)


def chunk_focus(text):
    """The page name from a chunk header '<Focus> — <Section>'."""
    return text.split("\n", 1)[0].rsplit(" — ", 1)[0].strip().lower()


def retrieval_report(searcher, questions, top_k):
    """Retrieval-only statistics per scenario, no LLM calls: gold rank for A, what B/C retrieve,
    and top-1 score quartiles (useful for tuning C's threshold)."""
    report = {}
    for scenario in ("A", "B", "C"):
        rows = [q for q in questions if q["scenario"] == scenario]
        if not rows:
            continue
        ranks, same_focus, top1 = [], [], []
        for q in rows:
            docs = searcher.search(q["question"], top_k)
            gold = set(json.loads(q["gold_doc_ids"]))
            ranks.append(next((i for i, d in enumerate(docs, 1) if d.doc_id in gold), None))
            same_focus.append(sum(chunk_focus(d.text) == q["focus"].lower() for d in docs))
            top1.append(docs[0].score if docs else 0.0)
        scores = sorted(top1)
        entry = {"n": len(rows),
                 "same_page_chunks_in_top_k_mean": round(statistics.mean(same_focus), 2),
                 "questions_with_same_page_chunk": sum(s > 0 for s in same_focus),
                 "top1_score_quartiles": [round(scores[int(p * (len(scores) - 1))], 3) for p in (0, .25, .5, .75, 1)]}
        if scenario == "A":
            found = [r for r in ranks if r]
            entry["gold_chunk_in_top_k"] = len(found)
            entry["gold_rank_1"] = sum(r == 1 for r in found)
        report[scenario] = entry
    return report


def estimate_cost(calls):
    prompt = sum(c["usage"].get("prompt_tokens", 0) for c in calls)
    completion = sum(c["usage"].get("completion_tokens", 0) for c in calls)
    model = calls[0]["model"] if calls else ""
    price = next((p for name, p in PRICE_PER_M.items() if model and model.startswith(name)), None)
    cost = None if price is None else round((prompt * price[0] + completion * price[1]) / 1e6, 4)
    return {"calls": len(calls), "model": model, "prompt_tokens": prompt, "completion_tokens": completion,
            "estimated_cost_usd": cost}


def summarise(records, questions, calls):
    by_id = {q["question_id"]: q for q in questions}
    out = {}
    for scenario in ("A", "B", "C"):
        rows = [r for r in records if by_id[r["question_id"]]["scenario"] == scenario]
        if not rows:
            continue
        latency = sorted(r["latency_ms"] for r in rows)
        entry = {"n": len(rows), "answered": sum(not r["abstained"] for r in rows),
                 "abstained": sum(r["abstained"] for r in rows),
                 "latency_ms_mean": round(statistics.mean(latency), 1),
                 "latency_ms_p95": round(latency[min(len(latency) - 1, int(0.95 * len(latency)))], 1)}
        if scenario == "A":
            hits = 0
            for r in rows:
                gold = set(json.loads(by_id[r["question_id"]]["gold_doc_ids"]))
                hits += any(d["doc_id"] in gold for d in r["retrieved_docs"])
            entry["gold_chunk_in_top_k"] = hits
        if rows and "verdict" in rows[0]["meta"]:
            entry["verdicts"] = dict(sorted(
                ((v, sum(r["meta"]["verdict"] == v for r in rows)) for v in {r["meta"]["verdict"] for r in rows})))
            entry["parse_failures"] = sum(r["meta"]["parse_error"] for r in rows)
            entry["snippet_in_evidence"] = sum(r["meta"]["snippet_in_evidence"] is True for r in rows)
            entry["llm_calls"] = sum(r["meta"]["n_llm_calls"] for r in rows)
        out[scenario] = entry
    out["usage"] = estimate_cost(calls)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--per-scenario", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=["baseline", "evidence"])
    parser.add_argument("--judge-prompt", type=Path, default=ROOT / "configs" / "prompts" / "evidence_judge_v2.txt")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="no LLM calls: write retrieval statistics for the selected questions")
    args = parser.parse_args()
    if args.per_scenario < 1:
        parser.error("--per-scenario must be positive")

    config = load_config()
    questions = load_questions(args.split, args.per_scenario)
    if not questions:
        raise ValueError(f"No {args.split} questions in {DATA_DIR / 'dataset.csv'}")
    cache_dir = ROOT / "models" / "retrieval_medquad"
    prime_cache(config, DATA_DIR, cache_dir)
    searcher = Retriever(DATA_DIR / "corpus.jsonl", config=config, cache_dir=cache_dir)
    print(f"corpus: {len(searcher.documents)} chunks, cache_hit={searcher.cache_hit}; "
          f"{len(questions)} {args.split} questions")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.retrieval_only:
        report = {"split": args.split, "n": len(questions), "embedding_model": searcher.model_name,
                  "top_k": config["retrieval"]["top_k"],
                  "scenarios": retrieval_report(searcher, questions, config["retrieval"]["top_k"])}
        target = RUNS_DIR / f"retrieval_medquad_{args.split}{len(questions)}.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    calls = []

    def tracked_transport(request, timeout):
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        body = json.loads(raw)
        calls.append({"model": body.get("model"), "usage": body.get("usage", {}),
                      "finish_reason": body.get("choices", [{}])[0].get("finish_reason")})
        return BytesIO(raw)

    def generator(prompt):
        for attempt in range(args.retries + 1):
            try:
                return generate(prompt, config=config, transport=tracked_transport)
            except LLMCallError:
                if attempt == args.retries:
                    raise
                time.sleep(2 ** attempt)

    summary = {"split": args.split, "per_scenario": args.per_scenario, "embedding_model": searcher.model_name,
               "top_k": config["retrieval"]["top_k"], "methods": {}}
    for name in args.methods:
        kwargs = {"search_fn": searcher.search, "generate_fn": generator, "config": config}
        if name == "evidence":
            kwargs["judge_prompt"] = args.judge_prompt
        method = get_method(name, **kwargs)
        first_call = len(calls)
        records = []
        target = RUNS_DIR / f"{name}_medquad_{args.split}{len(questions)}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for k, q in enumerate(questions, 1):
                record = method.run(q["question"], q["question_id"]).to_dict()
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if k % 10 == 0:
                    print(f"  {name}: {k}/{len(questions)}")
        summary["methods"][name] = summarise(records, questions, calls[first_call:])
        if name == "evidence":
            summary["methods"][name]["prompt_version"] = method.prompt_version
        print(f"Wrote {len(records)} outputs to {target}")
    summary["usage_total"] = estimate_cost(calls)
    target = RUNS_DIR / f"medquad_trial_{args.split}{len(questions)}_summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
