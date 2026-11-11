"""Shared exact cosine retrieval. Only evidence text is embedded, never QA labels."""
from __future__ import annotations

import hashlib
import os
import json
from functools import lru_cache
from pathlib import Path

from evrag.config import ROOT, load_config
from evrag.schema import RetrievedDoc


def load_corpus(path: str | Path) -> list[dict[str, str]]:
    """Read the project schema or A's id/context export without modifying it."""
    documents = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
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
            documents.append({"doc_id": doc_id, "text": text.strip()})
    if not documents:
        raise ValueError("The retrieval corpus is empty")
    return documents


class Retriever:
    """Sentence Transformer vectors, normalized for exact cosine search.

    ``backend='numpy'`` is an exact-cosine alternative for small corpora and
    deterministic tests. Production defaults to FAISS IndexFlatIP. An injected
    encoder must expose encode(list[str], **kwargs) returning a 2-D array.
    """

    def __init__(self, corpus_path=None, *, config=None, encoder=None,
                 cache_dir=None, backend=None):
        import numpy as np

        self.config = load_config() if config is None else config
        settings = self.config["retrieval"]
        self.model_name = os.environ.get("EVRAG_EMBEDDING_MODEL") or settings["embedding_model"]
        self.top_k = settings["top_k"]
        self.backend = backend or settings.get("backend", "faiss")
        if self.backend not in {"faiss", "numpy"}:
            raise ValueError("retrieval.backend must be faiss or numpy")
        path = Path(corpus_path or self.config["data"]["corpus"])
        self.corpus_path = path if path.is_absolute() else ROOT / path
        self.documents = load_corpus(self.corpus_path)
        # A custom encoder may have different weights under the same name:
        # disable persistent cache for injected encoders.
        custom_encoder = encoder is not None
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install requirements-b.txt for dense retrieval") from exc
            encoder = SentenceTransformer(self.model_name)
        self.encoder = encoder
        cache = cache_dir if cache_dir is not None else settings.get("cache_dir")
        self.cache_path = None
        if cache and not custom_encoder:
            directory = Path(cache)
            directory = directory if directory.is_absolute() else ROOT / directory
            fingerprint = hashlib.sha256(json.dumps(
                {"format": 1, "model": self.model_name, "documents": self.documents},
                sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=True)
            self.cache_path = directory / f"{fingerprint}.npy"
        self.cache_hit = self.cache_path is not None and self.cache_path.exists()
        if self.cache_hit:
            vectors = np.load(self.cache_path, allow_pickle=False)
        else:
            vectors = self._encode([d["text"] for d in self.documents])
        self.vectors = self._normalize(vectors)
        if len(self.vectors) != len(self.documents):
            raise ValueError("Encoder returned a different number of document vectors")
        if self.cache_path is not None and not self.cache_hit:
            np.save(self.cache_path, self.vectors, allow_pickle=False)
        self.index = None
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("Install faiss-cpu or explicitly use backend='numpy'") from exc
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def _encode(self, texts):
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def _normalize(vectors):
        import numpy as np

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be a finite two-dimensional array")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Encoder returned a zero-length embedding")
        return np.ascontiguousarray(vectors / norms)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        """Return descending cosine scores; scores are similarities, not probabilities."""
        import numpy as np

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = self.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("top_k must be a positive integer")
        k = min(k, len(self.documents))
        vector = self._normalize(self._encode([query]))
        if vector.shape != (1, self.vectors.shape[1]):
            raise ValueError("Query embedding has an incompatible shape")
        if self.index is not None:
            scores, positions = self.index.search(vector, k)
            pairs = zip(positions[0], scores[0])
        else:
            scores = (self.vectors @ vector[0]).reshape(-1)
            positions = np.argsort(-scores, kind="stable")[:k]
            pairs = ((i, scores[i]) for i in positions)
        return [RetrievedDoc(self.documents[int(i)]["doc_id"],
                             float(np.clip(score, -1.0, 1.0)),
                             self.documents[int(i)]["text"])
                for i, score in pairs]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Reuse one corpus/model/index in this process; clear cache after config changes."""
    return Retriever()


def search(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """Shared entry point for B/C/D: search(query, top_k)."""
    return get_retriever().search(query, top_k)
