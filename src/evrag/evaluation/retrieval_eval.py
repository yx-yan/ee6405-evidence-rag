"""Retrieval-only comparison: TF-IDF vs BM25 vs MiniLM vs the graph retriever. Owner: C.

    python -m evrag.evaluation.retrieval_eval     # -> results/final/retrieval.csv, results/final/recall_curve.csv

Answerable test questions of each dataset, ranked over that dataset's eval corpus. A hit is a gold
passage; Recall@k counts questions with a gold passage in the top k, MRR uses the first gold
passage in the top MAX_K (0 when there is none). No answers are generated, so this compares
traditional term weighting (Week 3) with dense embeddings directly.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter

import numpy as np

from evrag import retriever
from evrag.config import FINAL_DIR, dataset_paths
from evrag.data.unanswerable import SETS
from evrag.evaluation.run_experiment import load_questions
from evrag.methods.graph_rag import GraphRAG

MAX_K = 20
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25: idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.k1, self.b, self.avg = k1, b, self.lengths.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
        for term in set(tokenize(query)):
            if term in self.idf:
                tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
                out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


class TfidfRanker:
    """Cosine similarity of sublinear TF-IDF vectors (scikit-learn)."""

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()


def rankers(dataset: str) -> dict:
    """name -> function(question) returning the top MAX_K doc ids."""
    docs = retriever.load_corpus(dataset_paths(dataset)["corpus"])
    ids = [d["doc_id"] for d in docs]
    texts = [d["text"] for d in docs]
    tfidf, bm25 = TfidfRanker(texts), BM25(texts)
    graph = GraphRAG(dataset).graph

    def top(scores):
        return [ids[i] for i in np.argsort(-scores, kind="stable")[:MAX_K]]

    return {
        "tfidf": lambda q: top(tfidf.scores(q)),
        "bm25": lambda q: top(bm25.scores(q)),
        "minilm": lambda q: [d.doc_id for d in retriever.search(q, MAX_K, dataset=dataset)],
        "graph": lambda q: [d.doc_id for d in graph.search(q, MAX_K)[0]],
    }


def first_hit(ranked: list[str], gold: set[str]) -> int | None:
    """1-based rank of the first gold passage, or None."""
    return next((i for i, d in enumerate(ranked, 1) if d in gold), None)


def evaluate(split: str = "test") -> tuple[list[dict], list[dict]]:
    table, curve = [], []
    for dataset in SETS:
        questions = [q for q in load_questions(split, dataset=dataset) if q["scenario"] == "answerable"]
        for name, rank in rankers(dataset).items():
            hits = [first_hit(rank(q["question"]), set(json.loads(q["gold_doc_ids"]))) for q in questions]
            recall = lambda k: float(np.mean([h is not None and h <= k for h in hits]))
            table.append({"dataset": dataset, "retriever": name, "n": len(hits), "recall_at_1": recall(1),
                          "recall_at_5": recall(5), "mrr": float(np.mean([1 / h if h else 0.0 for h in hits]))})
            curve += [{"dataset": dataset, "retriever": name, "k": k, "recall": recall(k)} for k in range(1, MAX_K + 1)]
            print(table[-1], flush=True)
    return table, curve


def main() -> None:
    table, curve = evaluate()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for rows, name in ((table, "retrieval.csv"), (curve, "recall_curve.csv")):
        with open(FINAL_DIR / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)


if __name__ == "__main__":
    main()
