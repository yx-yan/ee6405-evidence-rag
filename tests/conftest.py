"""Run E's unchanged contract test offline with an explicit retriever mock."""
import pytest
from evrag import retriever
from evrag.schema import RetrievedDoc


@pytest.fixture(autouse=True)
def offline_contract_backends(request, monkeypatch):
    if request.node.path.name == 'test_contract.py':
        monkeypatch.setattr(retriever, 'search', lambda q, top_k=None, dataset=None: [RetrievedDoc('toy', 0.8, 'Toy evidence')])
