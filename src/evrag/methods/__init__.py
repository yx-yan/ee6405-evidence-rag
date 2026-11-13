"""Importing a method module registers it. Add each new method here."""

from evrag.methods import basic_rag, dummy, evidence_rag, graph_rag, nlp_only, threshold_rag  # noqa: F401
from evrag.methods.base import REGISTRY, Method, get_method, register  # noqa: F401
