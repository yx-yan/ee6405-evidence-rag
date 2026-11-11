#!/usr/bin/env python3
import pytest
import numpy as np
import pandas as pd
from src.evrag.evaluation.metrics import (
    calculate_token_f1,
    calculate_rouge_l,
    bootstrap_ci,
    parse_gold_answers
)

def test_calculate_token_f1_exact():
    """Verifies that an exact string token match yields a perfect score of 1.0."""
    gold = "Common asthma triggers include cold air and exercise."
    pred = "Common asthma triggers include cold air and exercise."
    assert calculate_token_f1(gold, pred) == 1.0

def test_calculate_token_f1_partial():
    """Verifies that overlapping tokens compute precise precision/recall balance metrics."""
    gold = "high blood pressure symptoms"
    pred = "high glucose symptoms"
    score = calculate_token_f1(gold, pred)
    # 3 common tokens out of 4 total unique slots -> expected around 0.5
    assert 0.4 <= score <= 0.6

def test_calculate_token_f1_empty():
    """Handles boundary anomalies like empty strings or whitespace characters gracefully."""
    assert calculate_token_f1("", "") == 1.0
    assert calculate_token_f1("diabetes", "") == 0.0

def test_calculate_rouge_l_identical():
    """Guarantees that identical multi-word strings yield absolute maximum LCS scores."""
    gold = "Metformin lowers blood sugar levels for type 2 diabetes."
    pred = "Metformin lowers blood sugar levels for type 2 diabetes."
    assert calculate_rouge_l(gold, pred) == 1.0

def test_calculate_rouge_l_subset():
    """Tests if longest common subsequence pattern mapping matches sub-strings accurately."""
    gold = "lifestyle modifications control hypertension"
    pred = "modifications control hypertension naturally"
    score = calculate_rouge_l(gold, pred)
    assert score > 0.5

def test_bootstrap_ci_variance():
    """Ensures the zero-dependency NumPy bootstrapper bounds means within true data vectors."""
    # Fixed dummy data to keep resampling boundaries consistent
    mock_data = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
    low, high = bootstrap_ci(mock_data, num_bootstraps=500)
    assert 0.0 <= low <= high <= 1.0
    assert 0.2 <= np.mean([low, high]) <= 0.8

def test_parse_gold_answers_json_list():
    """Confirms the parser handles JSON-formatted answers correctly."""
    json_str = '["High blood pressure", "Hypertension symptoms"]'
    parsed = parse_gold_answers(json_str)
    assert len(parsed) == 2
    assert parsed[0] == "High blood pressure"

def test_parse_gold_answers_pipe_separated():
    """Confirms the parser splits pipe-separated target arrays cleanly."""
    pipe_str = "Shortness of breath | Chronic coughing | Wheezing"
    parsed = parse_gold_answers(pipe_str)
    assert len(parsed) == 3
    assert parsed[1] == "Chronic coughing"

def test_parse_gold_answers_plain_text():
    """Guarantees that raw text strings are encapsulated into flat lookup lists safely."""
    plain_str = "Insulin therapy guidelines"
    parsed = parse_gold_answers(plain_str)
    assert len(parsed) == 1
    assert parsed[0] == "Insulin therapy guidelines"
