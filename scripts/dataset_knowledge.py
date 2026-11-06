"""Check whether models already know MedQuAD, LiveQA-Med and Symptom2Disease without any evidence.

    python scripts/dataset_knowledge.py build
    python scripts/dataset_knowledge.py run --model qwen3:4b-instruct-2507-q4_K_M
    python scripts/dataset_knowledge.py run --model gemma4:e4b --no-think --probes closed classify
    python scripts/dataset_knowledge.py grade --grader gemma4:e4b
    python scripts/dataset_knowledge.py summarize

Two different questions, so two kinds of probe:

  closed    knows the answer: the question alone, graded against the reference answer
  classify  knows the answer: a symptom description, choosing one of the 24 labels
  continue  has seen the text: raw completion of a reference text's first half, against a control

Reads data/medquad, data/raw/liveqa_med and data/raw/symptom2disease. MedQuAD uses the dev split only. Nothing here tunes a method.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

from evrag import llm
from evrag.config import DATA_DIR, RESULTS_DIR, SEED, load_config
from llm_screen import (
    OLLAMA_URL, REFUSAL, TIMEOUT_SECONDS, bootstrap_ci, chat_no_think, complete_raw, content_tokens,
    ngram_hit, recall, result_path, rouge_l, split_note,
)

OUT_DIR = RESULTS_DIR / "dataset_knowledge"
SAMPLES = OUT_DIR / "samples.json"
GRADES = OUT_DIR / "grades.jsonl"
RAW_DIR = DATA_DIR / "raw"
PROBES = ["closed", "classify", "continue"]

N_CONTINUE = 40
PER_LABEL = 10
MIN_CONTINUE_WORDS = 60
MIN_SYMPTOM_WORDS = 24
N_GRADER_CONTROL = 40
MAX_REFERENCE_WORDS = 400

CLOSED_PROMPT = (
    "Answer this health question in at most three sentences, from your own knowledge. "
    "Give your best answer.\n\nQUESTION:\n{question}\n\nANSWER:\n"
)
CLASSIFY_PROMPT = (
    "A person describes their symptoms below. Choose the single most likely condition from this list:\n"
    "{labels}\n\nReply with only the condition's name, exactly as written.\n\n"
    "DESCRIPTION:\n{text}\n\nCONDITION:\n"
)
GRADE_PROMPT = (
    "You grade an answer to a health question against a reference answer.\n"
    "CORRECT: it gives the reference's main point and does not contradict it.\n"
    "PARTIAL: it gives some of the reference's points, or is relevant but much less specific.\n"
    "WRONG: it contradicts the reference, misses its main point, or does not answer.\n"
    "Judge only agreement with the reference, not writing style.\n\n"
    "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}\n\n"
    "Reply with one word: CORRECT, PARTIAL or WRONG.\n"
)
GRADE_WORD = re.compile(r"\b(CORRECT|PARTIAL|WRONG)\b")


def parse_label(reply: str, labels: list[str]) -> str | None:
    """The label named in the reply, or None if it names none or several."""
    text = " ".join(reply.lower().split())
    found = [label for label in labels if re.search(rf"\b{re.escape(label.lower())}\b", text)]
    return found[0] if len(found) == 1 else None


def parse_grade(reply: str) -> str | None:
    match = GRADE_WORD.search(reply or "")
    return match.group(1) if match else None


def clip_words(text: str, limit: int = MAX_REFERENCE_WORDS) -> str:
    return " ".join(text.split()[:limit])


def load_medquad() -> list[dict]:
    with open(DATA_DIR / "medquad" / "dataset.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    return [{"item_id": r["question_id"], "question": r["question"], "scenario": r["scenario"],
             "group": r["qtype"], "reference": json.loads(r["gold_answers"])[0]} for r in rows]


def load_liveqa() -> list[dict]:
    """Questions as the public wrote them, with their reference answers joined."""
    items = []
    for node in ET.parse(RAW_DIR / "liveqa_med" / "test.xml").getroot():
        original = node.find("Original-Question")
        subject = " ".join((original.findtext("SUBJECT") or "").split())
        message = " ".join((original.findtext("MESSAGE") or "").split())
        answers = [" ".join(a.text.split()) for a in node.iter("ANSWER") if a.text and a.text.strip()]
        items.append({"item_id": node.attrib["qid"], "question": f"{subject}\n{message}".strip(),
                      "scenario": "", "group": "liveqa", "reference": " ".join(answers)})
    return items


def load_symptoms() -> list[dict]:
    """Distinct descriptions only: the file repeats 47 texts."""
    with open(RAW_DIR / "symptom2disease" / "Symptom2Disease.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen, items = set(), []
    for i, row in enumerate(rows):
        if row["text"] not in seen:
            seen.add(row["text"])
            items.append({"item_id": f"S2D_{i}", "text": row["text"], "label": row["label"]})
    return items


def continuation_items(dataset: str, items: list[dict], field: str, min_words: int,
                       rng: random.Random) -> list[dict]:
    """Split long texts in half; the control is the second half of a text from the same group."""
    pool = [i for i in items if len(i[field].split()) >= min_words]
    picked = []
    for item in rng.sample(pool, min(N_CONTINUE, len(pool))):
        group = item.get("group") or item.get("label")
        others = [o for o in pool if o["item_id"] != item["item_id"] and (o.get("group") or o.get("label")) == group]
        prefix, suffix = split_note(clip_words(item[field]))
        control = split_note(clip_words(rng.choice(others)[field]))[1]
        picked.append({"dataset": dataset, "item_id": item["item_id"], "prefix": prefix,
                       "suffix": suffix, "control_suffix": control})
    return picked


def build_samples(medquad: list[dict], liveqa: list[dict], symptoms: list[dict], seed: int = SEED) -> dict:
    rng = random.Random(seed)
    closed = [{"dataset": "medquad", **i} for i in medquad] + [{"dataset": "liveqa", **i} for i in liveqa]
    by_label = collections.defaultdict(list)
    for item in symptoms:
        by_label[item["label"]].append(item)
    labels = sorted(by_label)
    classify = [{"dataset": "symptom2disease", **item}
                for label in labels for item in rng.sample(by_label[label], PER_LABEL)]
    continuation = (continuation_items("medquad", medquad, "reference", MIN_CONTINUE_WORDS, rng)
                    + continuation_items("liveqa", liveqa, "reference", MIN_CONTINUE_WORDS, rng)
                    + continuation_items("symptom2disease", symptoms, "text", MIN_SYMPTOM_WORDS, rng))
    return {"seed": seed, "labels": labels, "closed": closed, "classify": classify, "continue": continuation}


def build() -> None:
    samples = build_samples(load_medquad(), load_liveqa(), load_symptoms())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES.write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in samples.items() if k != "seed"})


def chat_fn(model: str, no_think: bool, max_tokens: int | None = None):
    os.environ["LLM_MODEL"] = model
    os.environ.setdefault("LLM_BASE_URL", f"{OLLAMA_URL}/v1")
    os.environ.setdefault("LLM_API_KEY", "ollama")
    config = copy.deepcopy(load_config())
    config["llm"]["timeout_seconds"] = TIMEOUT_SECONDS
    if max_tokens:
        config["llm"]["max_tokens"] = max_tokens
    if no_think:
        return lambda prompt: chat_no_think(model, prompt, config)
    return lambda prompt: llm.generate(prompt, config=config)


def run(model: str, probes: list[str], limit: int | None, no_think: bool) -> None:
    label = f"{model}-no-think" if no_think else model
    chat = chat_fn(model, no_think)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    labels = "\n".join(samples["labels"])
    path = OUT_DIR / result_path(label).name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():       # resume after a crash
        with open(path, encoding="utf-8") as f:
            done = {(r["probe"], r["item_id"]) for r in map(json.loads, f) if not r["error"]}

    chat("Reply with the word ready.")      # load the model outside the timings
    with open(path, "a", encoding="utf-8") as f:
        for probe in probes:
            for item in samples[probe][:limit]:
                if (probe, item["item_id"]) in done:
                    continue
                reply, error = None, None
                start = time.perf_counter()
                try:
                    if probe == "continue":
                        reply = complete_raw(model, item["prefix"])
                    elif probe == "classify":
                        reply = chat(CLASSIFY_PROMPT.format(labels=labels, text=item["text"]))
                    else:
                        reply = chat(CLOSED_PROMPT.format(question=item["question"]))
                except (llm.LLMCallError, OSError, KeyError) as exc:
                    error = str(exc)
                f.write(json.dumps({"model": label, "probe": probe, "dataset": item["dataset"],
                                    "item_id": item["item_id"], "reply": reply, "error": error,
                                    "latency_ms": (time.perf_counter() - start) * 1000},
                                   ensure_ascii=False) + "\n")
                f.flush()
            print(f"{label} {probe} done", flush=True)


def load_runs() -> list[dict]:
    records = []
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        if path != GRADES:
            with open(path, encoding="utf-8") as f:
                records += list({(r["probe"], r["item_id"]): r for r in map(json.loads, f)}.values())
    return records


def grade(grader: str, control_model: str) -> None:
    """Grade closed-book replies; mismatched references on one model measure the grader's false positives."""
    chat = chat_fn(grader, no_think=True, max_tokens=8)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    items = {i["item_id"]: i for i in samples["closed"]}
    done = set()
    if GRADES.exists():
        with open(GRADES, encoding="utf-8") as f:
            done = {(g["model"], g["item_id"], g["control"]) for g in map(json.loads, f)}
    records = [r for r in load_runs() if r["probe"] == "closed" and r["reply"]]

    rng = random.Random(SEED)
    jobs = [(r, items[r["item_id"]]["reference"], False) for r in records]
    for dataset in ("medquad", "liveqa"):
        pool = [r for r in records if r["model"] == control_model and r["dataset"] == dataset]
        for record in rng.sample(pool, min(N_GRADER_CONTROL, len(pool))):
            others = [i for i in samples["closed"]
                      if i["dataset"] == dataset and i["item_id"] != record["item_id"]]
            jobs.append((record, rng.choice(others)["reference"], True))

    with open(GRADES, "a", encoding="utf-8") as f:
        for record, reference, control in jobs:
            if (record["model"], record["item_id"], control) in done:
                continue
            reply = chat(GRADE_PROMPT.format(question=items[record["item_id"]]["question"],
                                             reference=clip_words(reference), answer=record["reply"]))
            f.write(json.dumps({"model": record["model"], "item_id": record["item_id"], "control": control,
                                "dataset": record["dataset"], "grade": parse_grade(reply), "raw": reply},
                               ensure_ascii=False) + "\n")
            f.flush()
    print(f"Graded {len(jobs)} jobs with {grader}", flush=True)


def rate(values) -> dict:
    low, high = bootstrap_ci(values)
    return {"value": float(np.mean(values)), "lo": low, "hi": high, "n": len(values)}


def summarize() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    closed = {i["item_id"]: i for i in samples["closed"]}
    classify = {i["item_id"]: i for i in samples["classify"]}
    text = {i["item_id"]: i for i in samples["continue"]}
    grades = {}
    if GRADES.exists():
        with open(GRADES, encoding="utf-8") as f:
            for g in map(json.loads, f):
                grades[(g["model"], g["item_id"], g["control"])] = g

    table = []
    records = load_runs()
    for model in sorted({r["model"] for r in records}):
        mine = [r for r in records if r["model"] == model]
        strata = [("medquad", ""), ("medquad", "A"), ("medquad", "B"), ("medquad", "C"), ("liveqa", "")]
        for dataset, scenario in strata:
            rows = [r for r in mine if r["probe"] == "closed" and r["dataset"] == dataset
                    and (not scenario or closed[r["item_id"]]["scenario"] == scenario)]
            found = [grades[(model, r["item_id"], False)]["grade"] for r in rows
                     if (model, r["item_id"], False) in grades]
            if not rows:
                continue
            row = {"model": model, "dataset": dataset, "scenario": scenario, "probe": "closed", "n": len(rows)}
            if found:
                result = rate([g == "CORRECT" for g in found])
                row.update(correct=result["value"], correct_lo=result["lo"], correct_hi=result["hi"],
                           partial=float(np.mean([g == "PARTIAL" for g in found])),
                           wrong=float(np.mean([g == "WRONG" for g in found])),
                           ungraded=float(np.mean([g is None for g in found])))
            row["refusal"] = float(np.mean([bool(REFUSAL.search(r["reply"] or "")) for r in rows]))
            row["word_recall"] = float(np.mean([
                recall(content_tokens(closed[r["item_id"]]["reference"], closed[r["item_id"]]["question"]),
                       r["reply"] or "") for r in rows]))
            table.append(row)

        rows = [r for r in mine if r["probe"] == "classify"]
        if rows:
            predicted = [parse_label(r["reply"] or "", samples["labels"]) for r in rows]
            result = rate([p == classify[r["item_id"]]["label"] for p, r in zip(predicted, rows)])
            table.append({"model": model, "dataset": "symptom2disease", "scenario": "", "probe": "classify",
                          "n": len(rows), "correct": result["value"], "correct_lo": result["lo"],
                          "correct_hi": result["hi"], "ungraded": float(np.mean([p is None for p in predicted]))})

        for dataset in ("medquad", "liveqa", "symptom2disease"):
            rows = [r for r in mine if r["probe"] == "continue" and r["dataset"] == dataset]
            if not rows:
                continue
            pairs = [(text[r["item_id"]], r["reply"] or "") for r in rows]
            gap = rate([rouge_l(reply, " ".join(i["suffix"].split()[:64]))
                        - rouge_l(reply, " ".join(i["control_suffix"].split()[:64])) for i, reply in pairs])
            table.append({"model": model, "dataset": dataset, "scenario": "", "probe": "continue", "n": len(rows),
                          "hit": float(np.mean([ngram_hit(reply, i["suffix"], i["prefix"]) for i, reply in pairs])),
                          "hit_control": float(np.mean([ngram_hit(reply, i["control_suffix"], i["prefix"])
                                                        for i, reply in pairs])),
                          "rouge_gap": gap["value"], "rouge_gap_lo": gap["lo"], "rouge_gap_hi": gap["hi"]})

    for dataset in ("medquad", "liveqa"):
        found = [g["grade"] for k, g in grades.items() if k[2] and g["dataset"] == dataset]
        if found:
            table.append({"model": "grader-control", "dataset": dataset, "scenario": "", "probe": "closed",
                          "n": len(found), "correct": float(np.mean([g == "CORRECT" for g in found])),
                          "partial": float(np.mean([g == "PARTIAL" for g in found])),
                          "wrong": float(np.mean([g == "WRONG" for g in found]))})

    columns = list(dict.fromkeys(key for row in table for key in row))
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in table])
    print(f"Wrote {len(table)} rows to {OUT_DIR / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("summarize")
    runner = commands.add_parser("run")
    runner.add_argument("--model", required=True)
    runner.add_argument("--probes", nargs="+", default=PROBES, choices=PROBES)
    runner.add_argument("--limit", type=int, default=None)
    runner.add_argument("--no-think", action="store_true")
    grader = commands.add_parser("grade")
    grader.add_argument("--grader", required=True)
    grader.add_argument("--control-model", default="qwen3:4b-instruct-2507-q4_K_M")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "summarize":
        summarize()
    elif args.command == "grade":
        grade(args.grader, args.control_model)
    else:
        run(args.model, args.probes, args.limit, args.no_think)


if __name__ == "__main__":
    main()
