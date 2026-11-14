"""Lightweight graph RAG over the project's medical passage corpus."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from evrag.config import ROOT, load_config
from evrag.methods.base import Method, register
from evrag.retriever import Retriever
from evrag.schema import RetrievedDoc

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be",
    "been", "before", "being", "but", "by", "can", "could", "do", "does", "for",
    "from", "get", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "me", "of", "on", "or", "should", "that", "the", "their", "there", "these",
    "this", "those", "to", "under", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would",
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2 and token not in _STOP_WORDS}


class MedicalKnowledgeGraph:
    """A document/entity graph ranked with RoBERTa sentence embeddings."""

    def __init__(
        self,
        corpus_path: str | Path,
        semantic_retriever: Retriever,
        *,
        semantic_min_score: float = 0.3,
        semantic_weight: float = 0.75,
        graph_weight: float = 0.25,
    ):
        if not -1.0 <= semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between -1 and 1")
        if semantic_weight < 0 or graph_weight < 0 or not math.isclose(
            semantic_weight + graph_weight, 1.0
        ):
            raise ValueError("semantic_weight and graph_weight must be non-negative and sum to 1")
        self.corpus_path = Path(corpus_path)
        self.semantic_retriever = semantic_retriever
        self.semantic_min_score = semantic_min_score
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        if not self.corpus_path.is_absolute():
            self.corpus_path = ROOT / self.corpus_path
        self.documents: list[dict[str, str]] = []
        self.doc_entities: dict[str, set[str]] = {}
        self.entity_docs: dict[str, set[str]] = defaultdict(set)

        seen: set[str] = set()
        with self.corpus_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id", row.get("id"))
                text = row.get("text", row.get("context"))
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(f"Missing document ID on line {line_number}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing evidence text on line {line_number}")
                if doc_id in seen:
                    raise ValueError(f"Duplicate document ID: {doc_id}")
                seen.add(doc_id)
                disease = row.get("disease", "")
                if isinstance(disease, list):
                    diseases = disease
                else:
                    diseases = str(disease).replace("_", " ").split()
                title = str(row.get("title", "")).strip()
                section = str(row.get("section", "")).strip()
                document = {
                    "doc_id": doc_id,
                    "text": text.strip(),
                    "title": title,
                    "disease": " ".join(str(item) for item in diseases),
                    "section": section,
                }
                self.documents.append(document)
                entities = {f"term:{token}" for token in _tokens(f"{title} {section} {text}")}
                entities.update(f"disease:{token}" for item in diseases for token in _tokens(str(item)))
                self.doc_entities[doc_id] = entities
                for entity in entities:
                    self.entity_docs[entity].add(doc_id)

        if not self.documents:
            raise ValueError("The graph corpus is empty")
        self.by_id = {document["doc_id"]: document for document in self.documents}
        self.entity_frequency = {entity: len(doc_ids) for entity, doc_ids in self.entity_docs.items()}

    def search(self, question: str, top_k: int = 5) -> tuple[list[RetrievedDoc], dict]:
        """Blend RoBERTa semantic matches with graph-linked lexical evidence."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        semantic_docs = self.semantic_retriever.search(question, top_k=len(self.documents))
        best_semantic_score = max((doc.score for doc in semantic_docs), default=-1.0)
        if best_semantic_score < self.semantic_min_score:
            return [], {
                "matched_entities": [],
                "graph_paths": [],
                "semantic_model": self.semantic_retriever.model_name,
                "best_semantic_score": best_semantic_score,
            }
        semantic_docs = [
            doc for doc in semantic_docs if doc.score >= self.semantic_min_score
        ]
        semantic_scores = {doc.doc_id: doc.score for doc in semantic_docs}

        query_terms = _tokens(question)
        query_entities = {
            entity
            for term in query_terms
            for entity in (f"term:{term}", f"disease:{term}")
            if entity in self.entity_docs
        }
        direct_scores: dict[str, float] = defaultdict(float)
        for entity in query_entities:
            frequency = self.entity_frequency[entity]
            weight = math.log1p((len(self.documents) + 1) / (frequency + 1))
            for doc_id in self.entity_docs[entity]:
                direct_scores[doc_id] += weight
        if query_entities:
            total_weight = sum(
                math.log1p((len(self.documents) + 1) / (self.entity_frequency[entity] + 1))
                for entity in query_entities
            )
            for doc_id in direct_scores:
                direct_scores[doc_id] /= total_weight

        paths: dict[str, list[str]] = {}
        for doc_id in direct_scores:
            matched = sorted(entity for entity in query_entities if doc_id in self.entity_docs[entity])
            paths[doc_id] = [f"question -> {entity} -> doc:{doc_id}" for entity in matched]
        for doc in semantic_docs:
            paths.setdefault(
                doc.doc_id,
                [f"question -> RoBERTa semantic match -> doc:{doc.doc_id}"],
            )

        max_expansion_frequency = max(2, int(len(self.documents) * 0.25))
        semantic_seeds = [doc.doc_id for doc in semantic_docs[: min(5, len(semantic_docs))]]
        lexical_seeds = sorted(
            (doc_id for doc_id in direct_scores if doc_id in semantic_scores),
            key=lambda doc_id: -direct_scores[doc_id],
        )[:5]
        seed_ids = list(dict.fromkeys(lexical_seeds + semantic_seeds))
        graph_scores = dict(direct_scores)
        for seed_id in seed_ids:
            seed_score = max(direct_scores.get(seed_id, 0.0), semantic_scores[seed_id])
            for entity in self.doc_entities[seed_id]:
                frequency = self.entity_frequency[entity]
                if frequency > max_expansion_frequency:
                    continue
                related_docs = self.entity_docs[entity]
                relation_weight = 1 / math.log2(frequency + 1)
                for related_id in related_docs:
                    if related_id == seed_id:
                        continue
                    expanded_score = seed_score * 0.35 * relation_weight
                    if expanded_score > graph_scores.get(related_id, 0.0):
                        graph_scores[related_id] = expanded_score
                    paths.setdefault(related_id, []).append(
                        f"doc:{seed_id} -> {entity} -> doc:{related_id}"
                    )

        scores = {
            doc_id: self.semantic_weight * max(semantic_score, 0.0)
            + self.graph_weight * graph_scores.get(doc_id, 0.0)
            for doc_id, semantic_score in semantic_scores.items()
        }
        ranked_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
        retrieved = [
            RetrievedDoc(doc_id, float(scores[doc_id]), self.by_id[doc_id]["text"])
            for doc_id in ranked_ids
        ]
        meta = {
            "matched_entities": sorted(query_entities),
            "graph_paths": [path for doc_id in ranked_ids for path in paths.get(doc_id, [])],
            "semantic_model": self.semantic_retriever.model_name,
            "best_semantic_score": best_semantic_score,
        }
        return retrieved, meta


@lru_cache(maxsize=2)
def _load_graph(
    corpus_path: str,
    embedding_model: str,
    cache_dir: str | None,
    semantic_min_score: float,
    semantic_weight: float,
    graph_weight: float,
) -> MedicalKnowledgeGraph:
    config = load_config()
    retrieval_config = dict(config["retrieval"])
    retrieval_config["embedding_model"] = embedding_model
    config["retrieval"] = retrieval_config
    semantic_retriever = Retriever(
        corpus_path=corpus_path,
        config=config,
        cache_dir=cache_dir,
        backend="numpy",
    )
    return MedicalKnowledgeGraph(
        corpus_path,
        semantic_retriever,
        semantic_min_score=semantic_min_score,
        semantic_weight=semantic_weight,
        graph_weight=graph_weight,
    )


def _extractive_answer(question: str, docs: list[RetrievedDoc]) -> str | None:
    """Return up to three source sentences that overlap the question's content terms."""
    query_terms = _tokens(question)
    candidates: list[tuple[int, int, str]] = []
    for doc_index, doc in enumerate(docs):
        fragments = re.split(r"(?<=[.!?])\s+|\n+", doc.text)
        sentences: list[str] = []
        index = 0
        while index < len(fragments):
            sentence = fragments[index].strip()
            if sentence.endswith(":") and index + 1 < len(fragments):
                index += 1
                sentence = f"{sentence} {fragments[index].strip()}"
            if sentence:
                sentences.append(sentence)
            index += 1
        minimum_overlap = min(2, len(query_terms))
        for sentence_index, sentence in enumerate(sentences):
            clean_sentence = " ".join(sentence.split())
            overlap = len(query_terms & _tokens(clean_sentence))
            if overlap >= minimum_overlap:
                candidates.append((overlap, -doc_index * 1000 - sentence_index, clean_sentence))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    selected: list[str] = []
    for _, _, sentence in candidates:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 3:
            break
    return " ".join(selected)


@register
class GraphRAG(Method):
    """Retrieve linked passages and answer only with sentences found in those passages."""

    name = "graph_rag"

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        top_k: int | None = None,
        semantic_retriever: Retriever | None = None,
    ):
        config = load_config()
        settings = config.get("graph_rag", {})
        configured_path = corpus_path or config["data"]["corpus"]
        if configured_path == "TODO":
            configured_path = "data/corpus.jsonl"
        path = Path(configured_path)
        if not path.is_absolute():
            path = ROOT / path
        if semantic_retriever is None:
            self.graph = _load_graph(
                str(path.resolve()),
                settings.get("embedding_model", "sentence-transformers/all-distilroberta-v1"),
                settings.get("cache_dir", config["retrieval"].get("cache_dir")),
                float(settings.get("semantic_min_score", 0.3)),
                float(settings.get("semantic_weight", 0.75)),
                float(settings.get("graph_weight", 0.25)),
            )
        else:
            self.graph = MedicalKnowledgeGraph(
                path,
                semantic_retriever,
                semantic_min_score=float(settings.get("semantic_min_score", 0.3)),
                semantic_weight=float(settings.get("semantic_weight", 0.75)),
                graph_weight=float(settings.get("graph_weight", 0.25)),
            )
        self.top_k = config["retrieval"]["top_k"] if top_k is None else top_k

    def _answer(self, question: str) -> tuple[str | None, list[RetrievedDoc], dict]:
        docs, meta = self.graph.search(question, self.top_k)
        answer = _extractive_answer(question, docs)
        meta["answer_mode"] = "extractive"
        if answer is None:
            return None, [], meta
        return answer, docs, meta
