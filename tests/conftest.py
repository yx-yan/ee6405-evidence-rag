"""Run E's unchanged contract test offline with an explicit retriever mock."""
import pytest
from evrag import retriever
from evrag.schema import RetrievedDoc


@pytest.fixture(autouse=True)
def offline_contract_backends(request, monkeypatch):
    if request.node.path.name == 'test_contract.py':
        monkeypatch.setattr(retriever, 'search', lambda q, k: [RetrievedDoc('toy', 0.8, 'Toy evidence')])
        from evrag.methods import graph_rag

        class OfflineGraph:
            def search(self, question, top_k):
                return [RetrievedDoc('toy', 0.8, 'Toy evidence')], {
                    'matched_entities': [],
                    'graph_paths': [],
                }

        monkeypatch.setattr(graph_rag, '_load_graph', lambda *args: OfflineGraph())
