"""Question-only baseline using the same BART checkpoint and decoding settings."""

from evrag import generator
from evrag.methods.base import Method, register


@register
class NLPOnly(Method):
    name = "nlp_only"

    def _answer(self, question):
        answer = generator.generate(question).strip()
        return answer or None, [], {"context_doc_ids": []}
