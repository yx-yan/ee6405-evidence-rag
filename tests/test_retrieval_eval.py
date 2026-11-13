"""BM25 and the ranking metrics of the retrieval comparison."""
from evrag.evaluation.retrieval_eval import BM25, TfidfRanker, first_hit


def test_bm25_prefers_rare_matching_terms():
    texts = ["asthma asthma wheezing", "asthma is common", "diabetes insulin"]
    scores = BM25(texts).scores("asthma wheezing")
    assert scores.argmax() == 0 and scores[2] == 0
    assert TfidfRanker(texts).scores("insulin").argmax() == 2


def test_first_hit():
    assert first_hit(["a", "b", "c"], {"c"}) == 3
    assert first_hit(["a"], {"z"}) is None
