"""Offline tests for the WikiDoc knowledge-base builder."""
from __future__ import annotations

import json

from evrag.data import wikidoc


def test_chunk_words_overlap():
    text = " ".join(str(i) for i in range(200))
    chunks = wikidoc.chunk_words(text, size=150, overlap=30)
    assert [c.split()[0] for c in chunks] == ["0", "120"]
    assert chunks[0].split()[-1] == "149" and chunks[1].split()[-1] == "199"
    assert wikidoc.chunk_words("short answer") == ["short answer"]


def test_build_records_filters_and_links():
    pairs = [
        {"input": "What is a cold?", "output": "A viral infection."},
        {"input": "What is asthma?", "output": " ".join(["word"] * 200)},
        {"input": "", "output": "Diabetes facts."},
        {"input": "Diabetes and blood pressure?", "output": "Linked to hypertension."},
    ]
    corpus, questions = wikidoc.build_records(pairs)
    assert [q["question"] for q in questions] == ["What is asthma?", "Diabetes and blood pressure?"]
    assert [q["split"] for q in questions] == ["dev", "test"]
    assert json.loads(questions[0]["gold_doc_ids"]) == ["doc_consumer_0", "doc_consumer_1"]
    assert corpus[2]["disease"] == "Diabetes_Hypertension"
    assert corpus[2]["article_id"] == "art_consumer_3_chunk_0"
