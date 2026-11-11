"""Run a method over the evaluation set and save raw outputs. Owner: E.

    python -m evrag.evaluation.run_experiment --method dummy --split dev --limit 20

Writes results/runs/<method>_<split>[<limit>].jsonl. Metrics are computed from these
files only, so nobody needs to rerun a method to recompute a table.
"""

import argparse
import csv
import json

from evrag.config import ROOT, RUNS_DIR, load_config
from evrag.methods import get_method


def load_questions(split: str, limit: int | None = None) -> list[dict]:
    # The question set comes from configs/default.yaml (data.dataset), like the corpus does.
    config = load_config()
    dataset_relative_path = config["data"]["dataset"]
    dataset_full_path = ROOT / dataset_relative_path
    
    if not os.path.exists(dataset_full_path):
        raise FileNotFoundError(f"❌ Target evaluation matrix data missing at tracked directory configuration layout path: {dataset_full_path}")
        
    with open(dataset_full_path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    method = get_method(args.method)
    questions = load_questions(args.split, args.limit)
    out = RUNS_DIR / f"{args.method}_{args.split}{args.limit or ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for q in questions:
            f.write(json.dumps(method.run(q["question"], q["question_id"]).to_dict()) + "\n")
    print(f"Wrote {len(questions)} outputs to {out}")


if __name__ == "__main__":
    main()
