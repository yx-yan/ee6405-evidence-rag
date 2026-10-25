"""Standard RAG: retrieve top-k -> prompt LLM -> answer. No explicit gate. Owner: B.

TODO(B): subclass evrag.methods.base.Method, implement _answer(), add @register,
and import this module in evrag/methods/__init__.py.
"""
