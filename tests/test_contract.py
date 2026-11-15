"""Every registered method must honour the RAGOutput contract."""

import pytest

from evrag import generator, retriever
from evrag.methods import REGISTRY, get_method
from evrag.schema import RAGOutput, RetrievedDoc

QUESTIONS = [
    "What is the project submission deadline?",
    "What was last year's exact percentage of A grades?",
    "Who wrote the course handout?",
]


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_run_returns_valid_output(name, monkeypatch):
    monkeypatch.setattr(retriever, "search", lambda q, top_k=None, dataset=None: [RetrievedDoc("d0", 0.8, "evidence")])
    monkeypatch.setattr(generator, "generate", lambda q, contexts=None: "test answer")
    monkeypatch.setattr(generator, "load", lambda: (lambda text: {"input_ids": text.split()}, None, "cpu"))
    method = get_method(name)
    for i, q in enumerate(QUESTIONS):
        out = method.run(q, question_id=f"q{i}")
        assert isinstance(out, RAGOutput)
        assert out.method == name
        assert out.question_id == f"q{i}"
        assert out.abstained == (out.answer is None)
        assert out.latency_ms >= 0
        assert all(isinstance(d, RetrievedDoc) for d in out.retrieved_docs)
        assert isinstance(out.to_dict(), dict)


def test_inconsistent_abstention_rejected():
    with pytest.raises(ValueError):
        RAGOutput("q", "m", answer="x", abstained=True, retrieved_docs=[], latency_ms=0.0)
