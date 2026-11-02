"""Run E's unchanged contract test offline with explicit shared-backend mocks."""
import pytest
from evrag import llm, retriever
from evrag.schema import RetrievedDoc


@pytest.fixture(autouse=True)
def offline_contract_backends(request, monkeypatch):
    if request.node.path.name == 'test_contract.py':
        monkeypatch.setattr(retriever, 'search', lambda q, k: [RetrievedDoc('toy', 0.8, 'Toy evidence')])
        monkeypatch.setattr(llm, 'generate', lambda prompt: 'MOCK contract answer')
