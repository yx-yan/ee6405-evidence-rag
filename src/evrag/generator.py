"""Shared answer generator: a BART model we fine-tune on medical QA. Owner: D.

Every route calls the same model and differs only in the evidence it passes:

    from evrag import generator
    generator.generate("What are the symptoms of eczema?")                  # no retrieval
    generator.generate("What are the symptoms of eczema?", contexts=docs)   # RAG / Graph RAG

`contexts` may hold strings or RetrievedDoc objects. The input format is defined once here
(`format_input`) and used by training (scripts/train_generator.py) and inference alike.
The model directory comes from configs/default.yaml (generator.model_dir) or the
EVRAG_GENERATOR_DIR environment variable. If it is missing, the trained weights are downloaded
once from the repo's GitHub Release (generator.release in the config) with the GitHub CLI.
transformers/torch are imported only when a model is loaded, so the rest of the package and the
tests do not need them.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from evrag.config import ROOT, load_config

QUESTION_PREFIX = "question: "
CONTEXT_PREFIX = " context:"


def _text(doc) -> str:
    return " ".join(str(getattr(doc, "text", doc)).split())


def format_input(question: str, contexts=None) -> str:
    """'question: …' alone, or followed by ' context: [1] … [2] …'. The question comes first so
    truncating a long input only ever cuts evidence."""
    question = " ".join(str(question).split())
    if not question:
        raise ValueError("question must be a non-empty string")
    parts = [_text(c) for c in (contexts or [])]
    parts = [p for p in parts if p]
    if not parts:
        return QUESTION_PREFIX + question
    evidence = " ".join(f"[{i}] {p}" for i, p in enumerate(parts, 1))
    return f"{QUESTION_PREFIX}{question}{CONTEXT_PREFIX} {evidence}"


def settings(config: dict | None = None) -> dict:
    """The generator section of the shared config, with the env override for model_dir."""
    section = dict((config or load_config()).get("generator") or {})
    model_dir = os.environ.get("EVRAG_GENERATOR_DIR") or section.get("model_dir")
    if model_dir and model_dir != "TODO":
        path = Path(model_dir)
        section["model_dir"] = str(path if path.is_absolute() else ROOT / path)
    else:
        section["model_dir"] = None
    return section


def download(model_dir: str, release: dict) -> None:
    """Fetch the zipped checkpoint from the repo's GitHub Release and unzip it into model_dir."""
    target = Path(model_dir)
    page = f"https://github.com/{release['repo']}/releases/tag/{release['tag']}"
    manual = (f"or download {release['asset']} from {page} and unzip it into {target}")
    if not shutil.which("gh"):
        raise RuntimeError(f"Generator model not found at {target}. Install the GitHub CLI and run "
                           f"`gh auth login` so it downloads automatically, {manual}")
    print(f"Downloading the generator model from {page} (about 0.5 GB, once) ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(["gh", "release", "download", release["tag"], "-R", release["repo"],
                            "-p", release["asset"], "-D", tmp], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Downloading the generator model failed (is `gh auth login` done?); {manual}") from e
        archive = Path(tmp) / release["asset"]
        if release.get("sha256"):
            digest = hashlib.sha256()
            with archive.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != release["sha256"]:
                raise RuntimeError(f"Downloaded {release['asset']} is corrupt (SHA-256 mismatch); try again")
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(partial)
        partial.rename(target)           # only a complete download ever appears at model_dir


@lru_cache(maxsize=2)
def load(model_dir: str | None = None):
    """Return (tokenizer, model, device) for a fine-tuned checkpoint directory. The configured
    directory is downloaded from the GitHub Release on first use if it is missing."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = settings()
    model_dir = model_dir or cfg["model_dir"]
    if not model_dir:
        raise RuntimeError("No generator model: set generator.model_dir in configs/default.yaml "
                           "or EVRAG_GENERATOR_DIR to a fine-tuned checkpoint directory")
    if not Path(model_dir).exists():
        if model_dir != cfg["model_dir"] or not cfg.get("release"):
            raise RuntimeError(f"Generator model not found at {model_dir}")
        download(model_dir, cfg["release"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    return tokenizer, model, device


def generate(question: str, contexts=None, *, model_dir: str | None = None, **overrides) -> str:
    """Answer one question, with or without evidence. Decoding settings come from the config's
    generator section; keyword overrides (e.g. num_beams=1) win."""
    import torch

    cfg = settings()
    tokenizer, model, device = load(model_dir or cfg["model_dir"])
    batch = tokenizer(format_input(question, contexts), return_tensors="pt", truncation=True,
                      max_length=int(cfg.get("max_source_tokens", 1024))).to(device)
    gen = decoding(cfg)
    gen.update(overrides)
    if gen.get("num_beams", 1) <= 1:
        gen.pop("early_stopping", None)
    with torch.no_grad():
        output = model.generate(**batch, **gen)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def decoding(cfg: dict | None = None) -> dict:
    """Shared decoding settings from the generator section of the config (default: the shared config)."""
    cfg = settings() if cfg is None else cfg
    gen = {"num_beams": int(cfg.get("num_beams", 4)), "max_new_tokens": int(cfg.get("max_new_tokens", 200)),
           "no_repeat_ngram_size": int(cfg.get("no_repeat_ngram_size", 3))}
    if gen["num_beams"] > 1:
        gen["early_stopping"] = True
    return gen


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on lower-cased whitespace tokens (no extra dependency)."""
    p, r = str(prediction).lower().split(), str(reference).lower().split()
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    m = prev[-1]
    if not m:
        return 0.0
    precision, recall = m / len(p), m / len(r)
    return 2 * precision * recall / (precision + recall)
