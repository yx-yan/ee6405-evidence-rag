"""Metrics on a hand-checked toy dataset and runs."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from evrag.evaluation import metrics as M

DATASET = {
    "q1": {"question_id": "q1", "question": "Symptoms of eczema?", "scenario": "A",
           "gold_answers": ["dry skin, itching"], "gold_doc_ids": ["d1"]},
    "q2": {"question_id": "q2", "question": "Cure for eczema?", "scenario": "B",
           "gold_answers": [], "gold_doc_ids": []},
    "q3": {"question_id": "q3", "question": "Symptoms of migraine?", "scenario": "C",
           "gold_answers": ["headache"], "gold_doc_ids": ["d3"]},
}


def out(qid, answer, docs, method="m", latency=10.0):
    return {"question_id": qid, "method": method, "answer": answer, "abstained": answer is None,
            "retrieved_docs": [{"doc_id": d, "score": 0.5, "text": t} for d, t in docs], "latency_ms": latency}


RUNS = [
    out("q1", "Dry skin and itching.", [("d1", "Eczema causes dry skin and itching.")]),       # correct, supported
    out("q2", None, [("d2", "Eczema is common.")]),                                              # right to abstain
    out("q3", "Fever.", [("d9", "Migraine: headache and sometimes fever.")], latency=30.0),      # wrong, gold doc missed
    out("q1", "Itchy red patches.", [], method="closed"),                                        # no evidence
]


def test_answer_level_scores():
    assert M.token_f1("Dry skin and itching.", "dry skin, itching") == pytest.approx(6 / 7)
    assert M.token_f1("fever", "headache") == 0.0
    assert M.support("Dry skin and itching.", ["Eczema causes dry skin and itching."]) == 1.0
    assert M.support("Itchy red patches.", []) == 0.0
    assert M.parse_answers('["a", "b "]') == ["a", "b"] == M.parse_answers("a | b")
    assert M.parse_answers("") == []


def test_table():
    table = {(r["method"], r["scenario"]): r for r in M.metrics_table(M.score_rows(RUNS, DATASET))}
    m = table[("m", "all")]
    assert m["n"] == 3
    assert m["coverage"] == pytest.approx(2 / 3)
    assert m["answer_correctness"] == pytest.approx(1 / 3)
    assert m["selective_accuracy"] == pytest.approx(1 / 2)
    assert m["abstention_recall"] == 1.0
    assert m["over_refusal_rate"] == 0.0
    assert m["unsupported_answer_rate"] == 0.0
    assert m["recall_at_k"] == pytest.approx(1 / 2)
    assert m["latency_mean_ms"] == pytest.approx(50 / 3)
    assert table[("m", "B")]["abstention_recall"] == 1.0 and table[("m", "B")]["over_refusal_rate"] is None
    closed = table[("closed", "all")]
    assert closed["unsupported_answer_rate"] == 1.0 and closed["recall_at_k"] is None


def test_manual_labels_override_f1(tmp_path):
    path = tmp_path / "adj.csv"
    n = M.write_adjudication(M.score_rows(RUNS, DATASET), RUNS, DATASET, path)
    assert n == 3                                    # every answered output, each (method, scenario) has < 30
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = "correct" if r["question_id"] == "q3" else ""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    labels = M.load_labels(path)
    assert labels == {("q3", "m"): True}
    m = [r for r in M.metrics_table(M.score_rows(RUNS, DATASET, labels)) if r["method"] == "m"][-1]
    assert m["selective_accuracy"] == 1.0


def test_evidence_reliance_and_bootstrap():
    r = M.evidence_reliance(np.array([[1.0], [-1.0], [2.0]]), [-1, 0, 1], gold={1})
    assert r == {"evidence_reliance": pytest.approx(3 / 4), "gold_share": pytest.approx(2 / 3)}
    assert M.paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0]) == {"diff": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    low, high = M.bootstrap_ci([0, 1] * 50)
    assert low < 0.5 < high


def test_write_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    M.write_csv(M.metrics_table(M.score_rows(RUNS, DATASET)), path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["method"], r["scenario"]) for r in rows][:2] == [("closed", "A"), ("closed", "all")]
    assert rows[0]["recall_at_k"] == ""
