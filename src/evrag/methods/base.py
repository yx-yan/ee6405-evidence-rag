"""Method protocol and registry. Owner: E.

To add a method: subclass Method, set `name`, implement `_answer`, and decorate
the class with @register. The contract test then picks it up automatically.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from evrag.schema import RAGOutput, RetrievedDoc

REGISTRY: dict[str, type["Method"]] = {}


def register(cls: type["Method"]) -> type["Method"]:
    REGISTRY[cls.name] = cls
    return cls


def get_method(name: str, **kwargs) -> "Method":
    return REGISTRY[name](**kwargs)


class Method(ABC):
    name: str = "base"

    def __init__(self, dataset: str | None = None):
        self.dataset = dataset          # evaluation set name; None means config.dataset_name()

    @abstractmethod
    def _answer(self, question: str) -> tuple[str | None, list[RetrievedDoc], dict]:
        """Return (answer or None to abstain, retrieved docs, meta)."""

    def run(self, question: str, question_id: str = "") -> RAGOutput:
        """Shared entry point: times the call and packs the result into RAGOutput."""
        start = time.perf_counter()
        answer, docs, meta = self._answer(question)
        latency_ms = (time.perf_counter() - start) * 1000
        return RAGOutput(
            question_id=question_id,
            method=self.name,
            answer=answer,
            abstained=answer is None,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            meta=meta,
        )
