"""Basic dense retrieval with evidence bounded by the shared BART input budget. Owner: B."""

from evrag import generator, retriever
from evrag.methods.base import Method, register


@register
class BasicRAG(Method):
    name = "basic_rag"

    def _answer(self, question):
        docs = retriever.search(question, dataset=self.dataset)
        if not docs:
            return None, [], {"context_doc_ids": []}
        contexts, tokens = generator.fit_contexts(question, docs)
        answer = generator.generate(question, contexts=contexts).strip()
        return answer or None, docs, {"context_doc_ids": [doc.doc_id for doc in contexts], "input_tokens": tokens}
