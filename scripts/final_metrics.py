"""Tables for the final results, from results/final/runs only (no model is run). Owner: E.

    python scripts/final_metrics.py

1. threshold_rag runs: basic_rag outputs with the dev-tuned tau applied (threshold_rag.apply).
2. metrics.csv: evaluation.metrics per dataset, split, method and scenario, plus BLEU-4 and METEOR
   (nltk) over answered answerable questions. metrics_unseen.csv: the same without the answers
   flagged in data/processed/<set>/eval/train_overlap.csv.
3. paired.csv: paired bootstrap of evidence_rag against every other method on the same test
   questions, for answer_correctness and reliability.
4. risk_coverage.csv: coverage and selective accuracy of basic_rag as tau sweeps the top-1 score.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from evrag.config import FINAL_DIR, dataset_paths, load_thresholds
from evrag.data.unanswerable import SETS
from evrag.evaluation import metrics as M
from evrag.methods import threshold_rag

RUNS = FINAL_DIR / "runs"
METHODS = ["nlp_only", "basic_rag", "graph_rag", "threshold_rag", "evidence_rag"]
SPLITS = ["dev", "test"]


def write_threshold_runs() -> None:
    for dataset in SETS:
        tau = load_thresholds(dataset)["threshold_rag"]["tau"]
        for split in SPLITS:
            path = RUNS / dataset / f"basic_rag_{split}.jsonl"
            if not path.exists():
                continue
            rows = M.load_runs([path])
            with open(RUNS / dataset / f"threshold_rag_{split}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(threshold_rag.apply(r, tau)) + "\n" for r in rows)


def add_bleu_meteor(rows: list[dict], runs: list[dict], dataset: dict) -> None:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    nltk.download("wordnet", quiet=True)
    smooth = SmoothingFunction().method1
    answers = {(o["question_id"], o["method"]): o["answer"] for o in runs}
    for r in rows:
        r["bleu4"] = r["meteor"] = None
        if r["answered"] and r["answerable"]:
            hyp = M.normalize(answers[(r["question_id"], r["method"])])
            refs = [M.normalize(a) for a in dataset[r["question_id"]]["gold_answers"]]
            r["bleu4"] = sentence_bleu(refs, hyp, smoothing_function=smooth)
            r["meteor"] = meteor_score(refs, hyp)


def table(rows: list[dict], **keys) -> list[dict]:
    out = []
    for t in M.metrics_table(rows):
        subset = [r for r in rows if r["method"] == t["method"] and t["scenario"] in ("all", r["scenario"])]
        out.append({**keys, **t, "bleu4": M._mean(r["bleu4"] for r in subset if r["answered"]),
                    "meteor": M._mean(r["meteor"] for r in subset if r["answered"])})
    return out


def risk_coverage(rows: list[dict], runs: list[dict], **keys) -> list[dict]:
    top = {o["question_id"]: (o["retrieved_docs"][0]["score"] if o["retrieved_docs"] else -1.0)
           for o in runs if o["method"] == "basic_rag"}
    base = [r for r in rows if r["method"] == "basic_rag"]
    scores = np.array([top[r["question_id"]] for r in base])
    correct = np.array([r["correct"] for r in base])
    answerable = np.array([r["answerable"] for r in base])
    out = []
    for tau in np.quantile(scores, np.linspace(0, 1, 41)):
        answered = scores >= tau
        out.append({**keys, "tau": float(tau), "coverage": float(answered.mean()),
                    "selective_accuracy": float(correct[answered].mean()) if answered.any() else None,
                    "answer_correctness": float(correct[answered].sum() / len(base)),
                    "abstention_recall": float((~answered[~answerable]).mean()),
                    "over_refusal_rate": float((~answered[answerable]).mean())})
    return out


def write(rows: list[dict], name: str) -> None:
    M.write_csv(rows, FINAL_DIR / name)
    print(f"Wrote {len(rows)} rows to {FINAL_DIR / name}")


def main() -> None:
    write_threshold_runs()
    tables, unseen, paired, curves = [], [], [], []
    for dataset in SETS:
        paths = dataset_paths(dataset)
        with open(paths["corpus"].parent / "train_overlap.csv", encoding="utf-8", newline="") as f:
            seen = {r["question_id"] for r in csv.DictReader(f) if r["seen"] == "1"}
        for split in SPLITS:
            questions = M.load_dataset(paths["dataset"], split)
            runs = M.load_runs([p for m in METHODS if (p := RUNS / dataset / f"{m}_{split}.jsonl").exists()])
            if not runs:
                continue
            rows = M.score_rows(runs, questions)
            add_bleu_meteor(rows, runs, questions)
            tables += table(rows, dataset=dataset, split=split)
            unseen += table([r for r in rows if r["question_id"] not in seen], dataset=dataset, split=split)
            curves += risk_coverage(rows, runs, dataset=dataset, split=split)
            if split != "test":
                continue
            by = {(r["method"], r["question_id"]): r for r in rows}
            ids = sorted({r["question_id"] for r in rows})
            for other in METHODS:
                if other == "evidence_rag" or ("evidence_rag", ids[0]) not in by or (other, ids[0]) not in by:
                    continue
                for metric in ("correct", "reliable"):
                    a = [float(by[("evidence_rag", q)][metric]) for q in ids]
                    b = [float(by[(other, q)][metric]) for q in ids]
                    paired.append({"dataset": dataset, "baseline": other,
                                   "metric": {"correct": "answer_correctness", "reliable": "reliability"}[metric],
                                   **M.paired_bootstrap(a, b)})
    write(tables, "metrics.csv")
    write(unseen, "metrics_unseen.csv")
    write(paired, "paired.csv")
    write(curves, "risk_coverage.csv")


if __name__ == "__main__":
    main()
