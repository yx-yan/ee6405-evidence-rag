"""Basic RAG v1 (WikiDoc appendix): both routes on dev and test, then metrics and the paired summary.

    python scripts/run_basic_v1.py

Writes to results/basic_v1 (or EVRAG_RESULTS_DIR). Uses the generator in
models/generator/bart-base/best unless EVRAG_GENERATOR_DIR is set.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ)
    env.setdefault("EVRAG_RESULTS_DIR", "results/basic_v1")
    env["EVRAG_DATASET"] = "wikidoc"
    env.setdefault("EVRAG_GENERATOR_DIR", str(ROOT / "models/generator/bart-base/best"))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    env.setdefault("OMP_NUM_THREADS", "4")
    results = env["EVRAG_RESULTS_DIR"]

    def run(*args: str) -> None:
        subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)

    for split in ("dev", "test"):
        for method in ("nlp_only", "basic_rag"):
            run("-u", "-m", "evrag.evaluation.run_experiment", "--method", method, "--split", split)
        run("-m", "evrag.evaluation.metrics", "--split", split,
            "--runs", f"{results}/runs/wikidoc/nlp_only_{split}.jsonl", f"{results}/runs/wikidoc/basic_rag_{split}.jsonl",
            "--out", f"{results}/metrics_{split}.csv")
    run("scripts/summarize_basic.py")


if __name__ == "__main__":
    main()
