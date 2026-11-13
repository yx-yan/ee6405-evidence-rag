"""Tune the abstention thresholds per dataset on dev only. Owner: E.

    python -m evrag.evaluation.tune         # -> results/final/thresholds.json, results/final/signals/<dataset>_<split>.csv

Retrieval only, no generation: evidence_rag's two retrievals and signals are computed once per
question (EvidenceRAG.trace with always_retry), so every threshold pair is scored from the same
table. Objective, for both methods: Youden's J = abstention_recall - over_refusal_rate, i.e. how
much better than chance the answer/abstain decision separates answerable from unanswerable
questions. It does not depend on the automatic F1 >= 0.5 correctness rule, which is unreliable for
COVID-QA's short span answers, and both degenerate policies (always answer, always abstain) score
0. Ties go to the lower over-refusal rate. Test signals are written for the notebook's curves only.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from evrag.config import FINAL_DIR
from evrag.data.unanswerable import SETS
from evrag.evaluation.metrics import parse_answers
from evrag.evaluation.run_experiment import load_questions
from evrag.methods.evidence_rag import EvidenceRAG

SIGNALS = ["score", "coverage", "retry_score", "retry_coverage"]
COVERAGE_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def signal_table(dataset: str, split: str) -> list[dict]:
    method = EvidenceRAG(dataset)
    rows = []
    for q in load_questions(split, dataset=dataset):
        t = method.trace(q["question"], always_retry=True)
        rows.append({"question_id": q["question_id"], "scenario": q["scenario"],
                     "answerable": int(bool(parse_answers(q["gold_answers"]))),
                     "score": t["initial"]["score"], "coverage": t["initial"]["coverage"],
                     "retry_score": t["retry"]["score"], "retry_coverage": t["retry"]["coverage"],
                     "top_doc": t["docs"][0].doc_id, "retry_top_doc": t["retry_docs"][0].doc_id,
                     "expanded_query": t["expanded_query"]})
    return rows


def youden(answered: np.ndarray, answerable: np.ndarray) -> tuple[float, float]:
    """(J, over_refusal_rate) of an answer/abstain decision."""
    abstention_recall = (~answered[~answerable]).mean() if (~answerable).any() else 0.0
    over_refusal = (~answered[answerable]).mean() if answerable.any() else 0.0
    return abstention_recall - over_refusal, over_refusal


def evidence_answers(s: dict, tau_score: float, tau_coverage: float) -> np.ndarray:
    first = (s["score"] >= tau_score) & (s["coverage"] >= tau_coverage)
    return first | ((s["retry_score"] >= tau_score) & (s["retry_coverage"] >= tau_coverage))


def tune(rows: list[dict]) -> dict:
    s = {k: np.array([r[k] for r in rows], dtype=float) for k in SIGNALS}
    answerable = np.array([r["answerable"] for r in rows], dtype=bool)
    score_grid = np.unique(np.round(s["score"], 3))
    best_tau = max(score_grid, key=lambda tau: (lambda j, o: (j, -o))(*youden(s["score"] >= tau, answerable)))
    pairs = [(ts, tc) for ts in score_grid for tc in COVERAGE_GRID]
    best_pair = max(pairs, key=lambda p: (lambda j, o: (j, -o))(*youden(evidence_answers(s, *p), answerable)))
    return {
        "threshold_rag": {"tau": float(best_tau), "dev_youden": youden(s["score"] >= best_tau, answerable)[0]},
        "evidence_rag": {"tau_score": float(best_pair[0]), "tau_coverage": float(best_pair[1]),
                         "dev_youden": youden(evidence_answers(s, *best_pair), answerable)[0]},
    }


def write_rows(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    thresholds = {}
    for dataset in SETS:
        for split in ("dev", "test"):
            rows = signal_table(dataset, split)
            write_rows(rows, FINAL_DIR / "signals" / f"{dataset}_{split}.csv")
            if split == "dev":
                thresholds[dataset] = tune(rows)
                print(dataset, json.dumps(thresholds[dataset]), flush=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")


if __name__ == "__main__":
    main()
