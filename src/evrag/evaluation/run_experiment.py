"""Run a method over the evaluation set and save raw outputs. Owner: E.

    python -m evrag.evaluation.run_experiment --dataset cancer --method basic_rag --split dev --limit 20

Writes results/runs/<dataset>/<method>_<split>[<limit>].jsonl. Metrics are computed from these
files only, so nobody needs to rerun a method to recompute a table. scripts/run_final.py runs
every dataset and method, resuming where it stopped.
"""

import argparse
import csv
import json

from evrag.config import DATASETS, RUNS_DIR, dataset_name, dataset_paths
from evrag.methods import get_method


def load_questions(split: str, limit: int | None = None, dataset: str | None = None) -> list[dict]:
    """Questions of one split of an evaluation set (configs/default.yaml, datasets)."""
    with open(dataset_paths(dataset)["dataset"], encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return rows[:limit] if limit else rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, help="default: EVRAG_DATASET or data.default")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = dataset_name(args.dataset)
    method = get_method(args.method, dataset=dataset)
    questions = load_questions(args.split, args.limit, dataset)
    out = RUNS_DIR / dataset / f"{args.method}_{args.split}{args.limit or ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for index, q in enumerate(questions, 1):
            f.write(json.dumps(method.run(q["question"], q["question_id"]).to_dict()) + "\n")
            f.flush()
            if index % 10 == 0 or index == len(questions):
                print(f"{args.method} {args.split}: {index}/{len(questions)}", flush=True)
    print(f"Wrote {len(questions)} outputs to {out}")


if __name__ == "__main__":
    main()
