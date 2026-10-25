"""Shared vector retriever (sentence-transformers + FAISS / cosine). Owner: B.

TODO(B): build index from data/processed/corpus.jsonl; expose
`search(query: str, top_k: int) -> list[RetrievedDoc]`. Read model and top_k from load_config().
"""
