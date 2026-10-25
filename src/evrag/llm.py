"""Thin wrapper around the single shared LLM. Owner: B.

TODO(B): `generate(prompt: str) -> str`, reading model / temperature from load_config()
and the key from .env (LLM_API_KEY). All methods call the LLM only through this module.
"""
