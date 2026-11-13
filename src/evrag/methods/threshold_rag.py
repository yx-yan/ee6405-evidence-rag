"""Basic RAG that abstains when the top-1 retrieval score is below tau. Owner: E.

tau is tuned per dataset on dev (evrag.evaluation.tune) and read from results/final/thresholds.json.
Experiments do not rerun it: `apply` turns a basic_rag output into a threshold_rag output, since
the retrieval and the generated answer are the same whenever it answers.
"""
from __future__ import annotations

from evrag import retriever
from evrag.config import dataset_name, load_thresholds
from evrag.methods.base import register
from evrag.methods.basic_rag import BasicRAG


def decide(top_score: float | None, tau: float) -> bool:
    """Answer only when there is a retrieved passage scoring at least tau."""
    return top_score is not None and top_score >= tau


def apply(output: dict, tau: float) -> dict:
    """A basic_rag run row (RAGOutput.to_dict()) as threshold_rag would have returned it."""
    docs = output["retrieved_docs"]
    top = docs[0]["score"] if docs else None
    meta = dict(output["meta"], top_score=top, tau=tau)
    if decide(top, tau):
        return dict(output, method=ThresholdRAG.name, meta=meta)
    return dict(output, method=ThresholdRAG.name, answer=None, abstained=True, meta=dict(meta, context_doc_ids=[]))


@register
class ThresholdRAG(BasicRAG):
    name = "threshold_rag"

    def __init__(self, dataset: str | None = None, tau: float | None = None):
        super().__init__(dataset)
        self.tau = load_thresholds(dataset_name(dataset))["threshold_rag"]["tau"] if tau is None else tau

    def _answer(self, question):
        docs = retriever.search(question, dataset=self.dataset)
        top = docs[0].score if docs else None
        if not decide(top, self.tau):
            return None, docs, {"top_score": top, "tau": self.tau, "context_doc_ids": []}
        answer, docs, meta = super()._answer(question)
        return answer, docs, dict(meta, top_score=top, tau=self.tau)
