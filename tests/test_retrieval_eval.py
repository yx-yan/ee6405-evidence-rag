"""BM25 and the ranking metrics of the retrieval comparison."""
import pytest

from evrag.evaluation.retrieval_eval import BM25, TfidfRanker, first_hit


def test_bm25_prefers_rare_matching_terms():
    texts = ["asthma asthma wheezing", "asthma is common", "diabetes insulin"]
    scores = BM25(texts).scores("asthma wheezing")
    assert scores.argmax() == 0 and scores[2] == 0


def test_tfidf():
    pytest.importorskip("sklearn")
    assert TfidfRanker(["asthma wheezing", "diabetes insulin"]).scores("insulin").argmax() == 1


def test_first_hit():
    assert first_hit(["a", "b", "c"], {"c"}) == 3
    assert first_hit(["a"], {"z"}) is None
