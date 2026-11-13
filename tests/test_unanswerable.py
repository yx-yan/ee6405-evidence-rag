"""Unanswerable scenarios: removed gold passages, dropped orphans, out-of-domain questions."""
from __future__ import annotations

import json

from evrag.data import unanswerable as U


def toy_set(prefix: str, n: int = 6) -> tuple[list[dict], list[dict]]:
    corpus = [{"doc_id": f"{prefix}_{i}", "text": f"passage {i}"} for i in range(n)]
    # Questions 0 and 1 share passage 0, so removing one may orphan the other.
    questions = [{"question_id": f"{prefix}_Q{i}", "question": f"q{i}?", "gold_answers": json.dumps([f"a{i}"]),
                  "gold_doc_ids": json.dumps([f"{prefix}_{0 if i < 2 else i}"]), "split": "dev" if i % 2 else "test"}
                 for i in range(n)]
    return corpus, questions


def test_build_eval(monkeypatch):
    monkeypatch.setattr(U, "N_MISSING", 1)
    monkeypatch.setattr(U, "N_OUT_OF_DOMAIN", 2)
    sets = {name: toy_set(name[:3]) for name in U.SETS}
    corpus, rows, counts = U.build_eval("coronavirus", sets)
    by = {s: [r for r in rows if r["scenario"] == s] for s in ("answerable", "missing_evidence", "out_of_domain")}
    assert len(by["missing_evidence"]) == 2 and len(by["out_of_domain"]) == 8
    assert all(r["gold_answers"] == "[]" for r in by["missing_evidence"] + by["out_of_domain"])
    removed = {d["doc_id"] for d in sets["coronavirus"][0]} - {d["doc_id"] for d in corpus}
    assert counts["passages_removed"] == len(removed)
    for r in by["answerable"]:                      # every answerable question keeps a gold passage
        assert set(json.loads(r["gold_doc_ids"])) - removed
    assert len(by["answerable"]) + len(by["missing_evidence"]) + counts["answerable_dropped"] == 6
    for split in ("dev", "test"):                   # out-of-domain questions come from the same split
        assert all(r["split"] == split for r in by["out_of_domain"] if r["question_id"].endswith(("1", "3", "5")) == (split == "dev"))
    assert {r["source_set"] for r in by["out_of_domain"]} == {"cancer", "neuro"}


def test_train_overlap():
    from evrag.data.train_overlap import ngrams, overlap
    seen = ngrams("one two three four five six seven eight nine ten")
    assert overlap(["two three four five six seven eight nine"], seen) == 1.0
    assert overlap(["two three four five six seven eight nine eleven"], seen) == 0.5
    assert overlap(["too short"], seen) == 0.0
