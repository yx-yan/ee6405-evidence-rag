"""Interface contract shared by every method. Owner: E.

Every method's `run()` returns a RAGOutput. Changing a field here affects all
methods and the evaluation code — propose it in a PR and get E's approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    text: str


@dataclass
class RAGOutput:
    question_id: str
    method: str
    answer: str | None              # None iff abstained
    abstained: bool
    retrieved_docs: list[RetrievedDoc]
    latency_ms: float               # end-to-end time of run(), including any judge / retry calls
    meta: dict = field(default_factory=dict)   # method-specific: threshold score, judge verdict, retried, ...

    def __post_init__(self) -> None:
        if self.abstained != (self.answer is None):
            raise ValueError("abstained must be True exactly when answer is None")

    def to_dict(self) -> dict:
        return asdict(self)
