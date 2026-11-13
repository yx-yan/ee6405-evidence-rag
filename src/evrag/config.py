"""Project-wide paths and settings. Import these instead of hard-coding paths or model names."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / os.environ.get("EVRAG_RESULTS_DIR", "results")
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the shared experiment config (configs/default.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)


SEED = load_config()["seed"]
DATASETS = list(load_config()["datasets"])


def dataset_name(name: str | None = None) -> str:
    """The evaluation set to use: `name`, else EVRAG_DATASET, else data.default in the config."""
    name = name or os.environ.get("EVRAG_DATASET") or load_config()["data"]["default"]
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")
    return name


def dataset_paths(name: str | None = None) -> dict[str, Path]:
    """Absolute {'corpus', 'dataset'} paths of an evaluation set (see dataset_name)."""
    entry = load_config()["datasets"][dataset_name(name)]
    return {key: ROOT / entry[key] for key in ("corpus", "dataset")}
