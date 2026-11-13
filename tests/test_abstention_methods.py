"""threshold_rag and evidence_rag decisions on toy retrievals."""
from __future__ import annotations

import pytest

from evrag import generator, retriever
from evrag.methods import evidence_rag, threshold_rag
from evrag.schema import RetrievedDoc

ON_TOPIC = RetrievedDoc("d1", 0.8, "Asthma: wheezing and coughing are symptoms of asthma.")
OFF_TOPIC = RetrievedDoc("d2", 0.3, "Diabetes: insulin controls blood sugar.")


@pytest.fixture(autouse=True)
def toy_backends(monkeypatch):
    monkeypatch.setattr(generator, "fit_contexts", lambda q, docs: (docs[:1], 10))
    monkeypatch.setattr(generator, "generate", lambda q, contexts=None: "an answer")
    monkeypatch.setattr(evidence_rag, "idf_table", lambda dataset: ({}, 1.0))


def test_threshold_rag(monkeypatch):
    monkeypatch.setattr(retriever, "search", lambda q, top_k=None, dataset=None: [OFF_TOPIC])
    assert threshold_rag.ThresholdRAG(tau=0.5).run("What is asthma?").abstained
    assert threshold_rag.ThresholdRAG(tau=0.2).run("What is asthma?").answer == "an answer"
    row = {"retrieved_docs": [{"doc_id": "d2", "score": 0.3, "text": "x"}], "answer": "a", "abstained": False,
           "method": "basic_rag", "meta": {"context_doc_ids": ["d2"]}}
    assert threshold_rag.apply(row, 0.5)["abstained"] and not threshold_rag.apply(row, 0.2)["abstained"]


def test_evidence_rag_answer_retry_abstain(monkeypatch):
    method = evidence_rag.EvidenceRAG(tau_score=0.5, tau_coverage=0.5)
    monkeypatch.setattr(retriever, "search", lambda q, top_k=None, dataset=None: [ON_TOPIC])
    out = method.run("What are the symptoms of asthma?")
    assert out.meta["action"] == "answer" and not out.meta["retried"]

    # The first retrieval misses; the expanded query (it carries feedback terms) finds the passage.
    monkeypatch.setattr(retriever, "search", lambda q, top_k=None, dataset=None:
                        [ON_TOPIC] if q != "What are the symptoms of asthma?" else [OFF_TOPIC])
    out = method.run("What are the symptoms of asthma?")
    assert out.meta["action"] == "retry_answer" and out.meta["expansion_terms"]
    assert out.meta["signals"]["retry"]["coverage"] == pytest.approx(1.0)

    monkeypatch.setattr(retriever, "search", lambda q, top_k=None, dataset=None: [OFF_TOPIC])
    out = method.run("What are the symptoms of asthma?")
    assert out.abstained and out.meta["action"] == "abstain"


def test_coverage_and_expansion():
    idf = {"asthma": 2.0, "symptoms": 1.0}
    assert evidence_rag.coverage("symptoms of asthma?", "asthma only", idf, 1.0) == pytest.approx(2 / 3)
    assert evidence_rag.expansion_terms("asthma?", ["asthma wheezing wheezing cough"], {"wheezing": 2, "cough": 3}) \
        == ["wheezing", "cough"]
