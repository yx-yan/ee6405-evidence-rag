"""Basic dense retrieval with evidence bounded by the shared BART input budget."""

from evrag import generator, retriever
from evrag.methods.base import Method, register


@register
class BasicRAG(Method):
    name = "basic_rag"

    def _answer(self, question):
        docs = retriever.search(question)
        if not docs:
            return None, [], {"context_doc_ids": [], "sources": []}
        tokenizer, _, _ = generator.load()
        budget = int(generator.settings().get("max_source_tokens", 1024))
        contexts = []
        for doc in docs:
            candidate = contexts + [doc]
            size = len(tokenizer(generator.format_input(question, candidate))["input_ids"])
            if size <= budget:
                contexts.append(doc)
            if len(contexts) == 4:
                break
        if not contexts:
            raise ValueError("No retrieved passage fits the generator input budget")
        answer = generator.generate(question, contexts=contexts).strip()
        return answer or None, docs, {
            "context_doc_ids": [doc.doc_id for doc in contexts],
            "input_tokens": len(tokenizer(generator.format_input(question, contexts))["input_ids"]),
            "max_source_tokens": budget,
        }
