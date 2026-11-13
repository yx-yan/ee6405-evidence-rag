"""Tests for graph retrieval and evidence-grounded answers."""
from __future__ import annotations

import json

from evrag.methods.graph_rag import GraphRAG, MedicalKnowledgeGraph


def _corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "doc_id": "asthma-triggers",
            "title": "Asthma triggers",
            "disease": "Asthma",
            "text": "Common asthma triggers include exercise and pollen.",
        },
        {
            "doc_id": "asthma-symptoms",
            "title": "Asthma symptoms",
            "disease": "Asthma",
            "text": "Exercise may worsen asthma symptoms. Wheezing is common.",
        },
        {
            "doc_id": "diabetes",
            "title": "Diabetes care",
            "disease": "Diabetes",
            "text": "Insulin helps control blood sugar.",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows


def test_graph_retrieval_matches_entities_and_expands_to_related_documents(tmp_path):
    path, _ = _corpus(tmp_path)
    graph = MedicalKnowledgeGraph(path)

    docs, meta = graph.search("What are common asthma triggers?", top_k=3)

    assert docs[0].doc_id == "asthma-triggers"
    assert "term:triggers" in meta["matched_entities"]
    assert any("disease:asthma" in path for path in meta["graph_paths"])
    assert any(doc.doc_id == "asthma-symptoms" for doc in docs)
    assert all(doc.doc_id != "diabetes" for doc in docs)


def test_graph_rag_answer_is_extracted_from_source_and_abstains_without_evidence(tmp_path):
    path, rows = _corpus(tmp_path)
    method = GraphRAG(corpus_path=path)

    answer, docs, _ = method._answer("What are common asthma triggers?")

    assert "exercise and pollen" in answer.lower()
    assert any(doc.doc_id == "asthma-triggers" for doc in docs)

    answer, docs, _ = method._answer("Which brand of air purifier is recommended?")

    assert answer is None
    assert docs == []
