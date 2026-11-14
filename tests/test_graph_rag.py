"""Tests for graph retrieval and evidence-grounded answers."""
from __future__ import annotations

import json

from evrag.methods.graph_rag import GraphRAG, MedicalKnowledgeGraph
from evrag.schema import RetrievedDoc


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


class FakeSemanticRetriever:
    model_name = "test-roberta"

    def __init__(self, rows):
        self.documents = rows

    def search(self, query, top_k=None):
        if "purifier" in query.lower():
            scores = {"asthma-triggers": 0.2, "asthma-symptoms": 0.1, "diabetes": 0.05}
        elif "provokes" in query.lower():
            scores = {"asthma-triggers": 0.9, "asthma-symptoms": 0.8, "diabetes": 0.1}
        else:
            scores = {"asthma-triggers": 0.95, "asthma-symptoms": 0.75, "diabetes": 0.1}
        ranked = sorted(self.documents, key=lambda row: -scores[row["doc_id"]])
        return [
            RetrievedDoc(row["doc_id"], scores[row["doc_id"]], row["text"])
            for row in ranked[:top_k]
        ]


def test_graph_retrieval_matches_entities_and_expands_to_related_documents(tmp_path):
    path, rows = _corpus(tmp_path)
    graph = MedicalKnowledgeGraph(path, FakeSemanticRetriever(rows))

    docs, meta = graph.search("What are common asthma triggers?", top_k=3)

    assert docs[0].doc_id == "asthma-triggers"
    assert "term:triggers" in meta["matched_entities"]
    assert any("disease:asthma" in path for path in meta["graph_paths"])
    assert any(doc.doc_id == "asthma-symptoms" for doc in docs)
    assert all(doc.doc_id != "diabetes" for doc in docs)


def test_roberta_semantic_retrieval_finds_paraphrases_without_shared_keywords(tmp_path):
    path, rows = _corpus(tmp_path)
    graph = MedicalKnowledgeGraph(path, FakeSemanticRetriever(rows))

    docs, meta = graph.search("What provokes breathing attacks?", top_k=2)

    assert docs[0].doc_id == "asthma-triggers"
    assert meta["matched_entities"] == []
    assert meta["semantic_model"] == "test-roberta"
    assert any("RoBERTa semantic match" in path for path in meta["graph_paths"])


def test_semantic_threshold_filters_weak_matches(tmp_path):
    path, rows = _corpus(tmp_path)
    graph = MedicalKnowledgeGraph(path, FakeSemanticRetriever(rows))

    docs, meta = graph.search("Which brand of air purifier is recommended?", top_k=3)

    assert docs == []
    assert meta["best_semantic_score"] == 0.2


def test_graph_rag_answer_is_extracted_from_source_and_abstains_without_evidence(tmp_path):
    path, rows = _corpus(tmp_path)
    method = GraphRAG(corpus_path=path, semantic_retriever=FakeSemanticRetriever(rows))

    answer, docs, _ = method._answer("What are common asthma triggers?")

    assert "exercise and pollen" in answer.lower()
    assert any(doc.doc_id == "asthma-triggers" for doc in docs)

    answer, docs, _ = method._answer("Which brand of air purifier is recommended?")

    assert answer is None
    assert docs == []
