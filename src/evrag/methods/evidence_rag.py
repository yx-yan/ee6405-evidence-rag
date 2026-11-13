"""Evidence-aware RAG: answer, retrieve again, or abstain, from two evidence signals. Owner: E.

  s1  score      top-1 dense retrieval score (cosine)
  s2  coverage   IDF-weighted share of the question's content words found in the top passage

Both at or above their thresholds: answer. Otherwise expand the query once with Rocchio pseudo
relevance feedback (the EXPANSION_TERMS terms with the highest TF-IDF over the top FEEDBACK_DOCS
passages, not already in the question), retrieve again and check again; still below: abstain.
Coverage is always measured against the original question, so the expansion terms cannot raise it.
Thresholds are tuned per dataset on dev (evrag.evaluation.tune) and read from
results/final/thresholds.json. meta records the action, every signal and both retrievals.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from evrag import generator, retriever
from evrag.config import dataset_name, dataset_paths, load_thresholds
from evrag.evaluation.metrics import STOPWORDS
from evrag.methods.base import Method, register

FEEDBACK_DOCS = 3
EXPANSION_TERMS = 5
_WORD = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    """Lower-case content words (no stopwords, no single characters)."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]


@lru_cache(maxsize=4)
def idf_table(dataset: str) -> tuple[dict[str, float], float]:
    """Smoothed IDF of every corpus term, and the IDF of an unseen term."""
    docs = retriever.load_corpus(dataset_paths(dataset)["corpus"])
    df = Counter(w for d in docs for w in set(terms(d["text"])))
    n = len(docs)
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}, math.log(n + 1) + 1


def coverage(question: str, passage: str, idf: dict[str, float], unseen: float) -> float:
    """IDF-weighted share of the question's content words that occur in the passage."""
    q = set(terms(question))
    if not q:
        return 0.0
    found = set(terms(passage))
    return sum(idf.get(w, unseen) for w in q & found) / sum(idf.get(w, unseen) for w in q)


def expansion_terms(question: str, passages: list[str], idf: dict[str, float], k: int = EXPANSION_TERMS) -> list[str]:
    """Rocchio feedback in text form: the top-k TF-IDF terms of the feedback passages."""
    tf = Counter(w for p in passages for w in terms(p))
    skip = set(terms(question))
    weights = {w: c * idf.get(w, 0.0) for w, c in tf.items() if w not in skip and not w.isdigit()}
    return sorted(weights, key=lambda w: (-weights[w], w))[:k]


@register
class EvidenceRAG(Method):
    name = "evidence_rag"

    def __init__(self, dataset: str | None = None, tau_score: float | None = None, tau_coverage: float | None = None):
        super().__init__(dataset)
        tuned = load_thresholds(dataset_name(dataset))["evidence_rag"]
        self.tau_score = tuned["tau_score"] if tau_score is None else tau_score
        self.tau_coverage = tuned["tau_coverage"] if tau_coverage is None else tau_coverage

    def signals(self, question: str, docs) -> dict:
        idf, unseen = idf_table(dataset_name(self.dataset))
        if not docs:
            return {"score": None, "coverage": 0.0}
        return {"score": docs[0].score, "coverage": coverage(question, docs[0].text, idf, unseen)}

    def passes(self, s: dict) -> bool:
        return s["score"] is not None and s["score"] >= self.tau_score and s["coverage"] >= self.tau_coverage

    def trace(self, question: str, always_retry: bool = False) -> dict:
        """Both retrievals and their signals, without generating. Tuning sets always_retry so every
        threshold pair can be evaluated from one trace."""
        docs = retriever.search(question, dataset=self.dataset)
        out = {"docs": docs, "initial": self.signals(question, docs)}
        if always_retry or not self.passes(out["initial"]):
            idf, _ = idf_table(dataset_name(self.dataset))
            extra = expansion_terms(question, [d.text for d in docs[:FEEDBACK_DOCS]], idf)
            query = f"{question} {' '.join(extra)}".strip()
            out["expansion_terms"], out["expanded_query"] = extra, query
            out["retry_docs"] = retriever.search(query, dataset=self.dataset)
            out["retry"] = self.signals(question, out["retry_docs"])
        return out

    def decide(self, trace: dict) -> str:
        if self.passes(trace["initial"]):
            return "answer"
        return "retry_answer" if self.passes(trace["retry"]) else "abstain"

    def _answer(self, question):
        t = self.trace(question)
        action = self.decide(t)
        docs = t["retry_docs"] if "retry_docs" in t else t["docs"]
        meta = {"action": action, "retried": "retry" in t, "tau_score": self.tau_score,
                "tau_coverage": self.tau_coverage, "signals": {k: t[k] for k in ("initial", "retry") if k in t},
                "initial_doc_ids": [d.doc_id for d in t["docs"]],
                "expanded_query": t.get("expanded_query"), "expansion_terms": t.get("expansion_terms"),
                "context_doc_ids": []}
        if action == "abstain":
            return None, docs, meta
        contexts, meta["input_tokens"] = generator.fit_contexts(question, docs)
        meta["context_doc_ids"] = [d.doc_id for d in contexts]
        return generator.generate(question, contexts=contexts).strip() or None, docs, meta
