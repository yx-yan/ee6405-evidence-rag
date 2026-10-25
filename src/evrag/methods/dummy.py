"""Placeholder method so evaluation and the demo can be built before real methods exist. Owner: E."""

import random

from evrag.config import SEED
from evrag.methods.base import Method, register
from evrag.schema import RetrievedDoc


@register
class DummyMethod(Method):
    name = "dummy"

    def __init__(self, abstain_rate: float = 0.3, seed: int = SEED):
        self.abstain_rate = abstain_rate
        self.rng = random.Random(seed)

    def _answer(self, question):
        docs = [RetrievedDoc(doc_id="d0", score=self.rng.random(), text="placeholder passage")]
        if self.rng.random() < self.abstain_rate:
            return None, docs, {}
        return "placeholder answer", docs, {}
