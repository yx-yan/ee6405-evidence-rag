"""Input-token attributions for the shared generator's answers (Week 9: LIME and SHAP). Owner: E.

The generator writes text, so we first fix what is explained: the answer `y` it generated. A
perturbed input x' is scored by teacher forcing, log p(y_t | y_<t, x') for every answer token t;
an attribution says how much a feature (a word, or a sentence of evidence) raises those log
probabilities. All three methods share that scoring function so they can be compared directly:

    feats = features(question, contexts)
    f = make_scorer(question, contexts, answer, feats)    # f(masks[n, M]) -> log probs [n, T]
    occlusion(f, len(feats)); lime(f, len(feats)); shap_values(f, len(feats))   # each [M, T]

Masked features become BART's <mask> token; the 'question:' / 'context: [i]' frame of
generator.format_input is always kept. torch, lime and shap are imported only when used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from evrag import generator

QUESTION = -1          # segment id of the question; evidence passages are 0, 1, ...


@dataclass
class Feature:
    segment: int       # QUESTION or the index of the evidence passage
    text: str


def features(question: str, contexts=None, unit: str = "word") -> list[Feature]:
    """Question words, then each passage split into words or sentences (`unit`)."""
    if unit not in {"word", "sentence"}:
        raise ValueError("unit must be word or sentence")
    feats = [Feature(QUESTION, w) for w in str(question).split()]
    for i, c in enumerate(generator._text(c) for c in (contexts or [])):
        parts = re.split(r"(?<=[.!?])\s+", c) if unit == "sentence" else c.split()
        feats += [Feature(i, p) for p in parts if p]
    return feats


def masked_input(feats: list[Feature], keep, mask_token: str = "<mask>") -> str:
    """The generator input with every feature where keep is false replaced by `mask_token`."""
    words = [f.text if k else mask_token for f, k in zip(feats, keep)]
    question = " ".join(w for f, w in zip(feats, words) if f.segment == QUESTION)
    n_docs = max((f.segment for f in feats), default=QUESTION) + 1
    contexts = [" ".join(w for f, w in zip(feats, words) if f.segment == i) for i in range(n_docs)]
    return generator.format_input(question or mask_token, contexts)


@lru_cache(maxsize=1)
def _scoring_model(model_dir: str | None = None):
    """The generator's model for teacher forcing; copied to Apple's GPU when there is one, since
    scoring many inputs at once is about twice as fast there (generation stays on the CPU)."""
    import copy

    import torch

    tokenizer, model, device = generator.load(model_dir or generator.settings()["model_dir"])
    if device == "cpu" and torch.backends.mps.is_available():
        model, device = copy.deepcopy(model).to("mps"), "mps"
    return tokenizer, model, device


def answer_tokens(answer: str, model_dir: str | None = None) -> list[str]:
    """The answer tokens that the columns of every attribution matrix refer to."""
    tokenizer, _, _ = _scoring_model(model_dir)
    ids = tokenizer(text_target=answer).input_ids
    return [tokenizer.decode([i]) for i in ids if i not in tokenizer.all_special_ids]


def make_scorer(question: str, contexts, answer: str, feats: list[Feature], *,
                batch_size: int = 32, model_dir: str | None = None):
    """f(masks) -> log p of each answer token (special tokens dropped) for every 0/1 mask row."""
    import torch

    tokenizer, model, device = _scoring_model(model_dir)
    max_len = int(generator.settings().get("max_source_tokens", 1024))
    labels = tokenizer(text_target=answer, return_tensors="pt").input_ids.to(device)
    keep_cols = [j for j, i in enumerate(labels[0].tolist()) if i not in tokenizer.all_special_ids]

    def f(masks) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks)) > 0.5
        out = []
        for start in range(0, len(masks), batch_size):
            rows = [masked_input(feats, m, tokenizer.mask_token) for m in masks[start:start + batch_size]]
            batch = tokenizer(rows, return_tensors="pt", padding=True, truncation=True,
                              max_length=max_len).to(device)
            with torch.no_grad():
                logits = model(**batch, labels=labels.repeat(len(rows), 1)).logits
            logp = logits.log_softmax(-1).gather(-1, labels.repeat(len(rows), 1)[..., None])[..., 0]
            out.append(logp[:, keep_cols].float().cpu().numpy())
        return np.concatenate(out)

    return f


def occlusion(f, m: int) -> np.ndarray:
    """Leave one out: f(all kept) - f(feature i masked), shape [m, T]."""
    masks = np.ones((m + 1, m))
    masks[np.arange(1, m + 1), np.arange(m)] = 0
    scores = f(masks)
    return scores[0] - scores[1:]


def lime(f, m: int, n_samples: int = 300, seed: int = 0, kernel_width: float = 25) -> np.ndarray:
    """LIME as LimeTextExplainer samples it (drop a random number of features, cosine distance,
    exponential kernel), with a weighted ridge regression per answer token, shape [m, T]."""
    from lime.lime_base import LimeBase

    rng = np.random.default_rng(seed)
    masks = np.ones((n_samples, m))
    for row in masks[1:]:
        row[rng.choice(m, rng.integers(1, m) if m > 1 else 1, replace=False)] = 0
    distances = 100 * (1 - masks.sum(1) / np.sqrt(masks.sum(1) * m))   # cosine distance to all-ones
    scores = f(masks)
    base = LimeBase(lambda d: np.sqrt(np.exp(-d ** 2 / kernel_width ** 2)), random_state=seed)
    weights = np.zeros((m, scores.shape[1]))
    for t in range(scores.shape[1]):
        _, exp, _, _ = base.explain_instance_with_data(masks, scores, distances, t, m, feature_selection="none")
        for i, w in exp:
            weights[i, t] = w
    return weights


def shap_values(f, m: int, permutations: int = 2, seed: int = 0) -> np.ndarray:
    """Shapley values from shap's Permutation explainer; masked = 0 is the baseline, so each
    column sums to f(all kept) - f(all masked). Shape [m, T]."""
    import shap

    masker = shap.maskers.Independent(np.zeros((1, m)), max_samples=1)
    explainer = shap.explainers.Permutation(f, masker, seed=seed)
    return explainer(np.ones((1, m)), max_evals=permutations * (2 * m + 1), batch_size=64).values[0]


METHODS = {"Occlusion": occlusion, "LIME": lime, "SHAP": shap_values}
