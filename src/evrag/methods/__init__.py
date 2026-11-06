"""Importing a method module registers it. Add each new method here."""

from evrag.methods import dummy  # noqa: F401
from evrag.methods import baseline  # noqa: F401
from evrag.methods import evidence_rag  # noqa: F401
from evrag.methods.base import REGISTRY, Method, get_method, register  # noqa: F401
