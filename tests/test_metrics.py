import pytest
import numpy as np
from src.evrag.evaluation.metrics import calculate_token_f1, calculate_rouge_l, bootstrap_ci

def test_token_f1_exact_match():
    assert calculate_token_f1("Common asthma triggers include cold air.", "Common asthma triggers include cold air.") == 1.0

def test_token_f1_partial_match():
    score = calculate_token_f1("asthma triggers cold air", "asthma triggers exercise")
    assert 0.4 < score < 0.7

def test_rouge_l_lcs():
    assert calculate_rouge_l("chronic type 2 diabetes treatment", "type 2 diabetes guidelines") > 0.5

def test_bootstrap_ci_bounds():
    mock_data = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    low, high = bootstrap_ci(mock_data, num_bootstraps=100)
    assert 0.1 <= low <= 0.9
    assert 0.1 <= high <= 0.9
