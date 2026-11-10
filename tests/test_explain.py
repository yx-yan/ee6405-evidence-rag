"""Offline tests for the attribution methods, with a toy scorer in place of the generator."""
from __future__ import annotations

import numpy as np
import pytest

from evrag import explain


def toy_scorer(key: int, other: int):
    """Two 'answer tokens': the first depends on feature `key` (plus a little on `other`), the
    second only on `other`."""
    def f(masks):
        masks = np.atleast_2d(masks)
        return np.stack([2.0 * masks[:, key] + 0.5 * masks[:, other], 1.0 * masks[:, other]], axis=1)
    return f


def test_features_and_masked_input():
    feats = explain.features("What is asthma?", ["Asthma is a lung disease. It causes wheezing.", "Other."])
    assert [f.text for f in feats[:3]] == ["What", "is", "asthma?"]
    assert {f.segment for f in feats} == {explain.QUESTION, 0, 1}
    keep = [1] * len(feats)
    assert explain.masked_input(feats, keep) == (
        "question: What is asthma? context: [1] Asthma is a lung disease. It causes wheezing. [2] Other.")
    keep[2] = keep[3] = 0
    assert explain.masked_input(feats, keep) == (
        "question: What is <mask> context: [1] <mask> is a lung disease. It causes wheezing. [2] Other.")
    sentences = explain.features("Q?", ["Asthma is a lung disease. It causes wheezing."], unit="sentence")
    assert [f.text for f in sentences[1:]] == ["Asthma is a lung disease.", "It causes wheezing."]


@pytest.mark.parametrize("method", ["Occlusion", "LIME", "SHAP"])
def test_methods_rank_the_key_feature_first(method):
    pytest.importorskip({"LIME": "lime", "SHAP": "shap"}.get(method, "numpy"))
    m, key, other = 8, 5, 2
    values = explain.METHODS[method](toy_scorer(key, other), m)
    assert values.shape == (m, 2)
    assert values[:, 0].argmax() == key
    assert values[:, 1].argmax() == other
    assert values[:, 0][key] == pytest.approx(2.0, abs=0.1)


def test_shap_is_additive():
    pytest.importorskip("shap")
    f = toy_scorer(1, 3)
    values = explain.shap_values(f, 6)
    np.testing.assert_allclose(values.sum(0), f(np.ones(6))[0] - f(np.zeros(6))[0], atol=1e-6)
