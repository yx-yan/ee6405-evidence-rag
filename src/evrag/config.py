"""Project-wide paths and settings. Import these instead of hard-coding paths or model names."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the shared experiment config (configs/default.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)


SEED = load_config()["seed"]
