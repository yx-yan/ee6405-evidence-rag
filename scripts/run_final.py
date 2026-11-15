"""Final experiment: every dataset x method x split, resumable. Owner: E.

    python scripts/run_final.py                    # everything; rows already in a run file are skipped
    python scripts/run_final.py --limit 10         # the first 10 questions of each group (timing)
    python scripts/run_final.py --datasets cancer --methods basic_rag --splits dev

Writes results/final/runs/<dataset>/<method>_<split>.jsonl (RAGOutput rows). threshold_rag is not
run: scripts/final_metrics.py derives it from basic_rag with the tuned tau. Run
python -m evrag.evaluation.tune first so evidence_rag uses the dev thresholds.
"""
from __future__ import annotations

import argparse
import json
import time

from evrag.config import FINAL_DIR
from evrag.data.unanswerable import SETS
from evrag.evaluation.run_experiment import load_questions
from evrag.methods import get_method

METHODS = ["nlp_only", "basic_rag", "graph_rag", "evidence_rag"]


def done_ids(path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["question_id"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="*", default=SETS)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--splits", nargs="*", default=["dev", "test"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    groups = [(d, m, s) for s in args.splits for d in args.datasets for m in args.methods]
    todo = {}
    for d, m, s in groups:
        path = FINAL_DIR / "runs" / d / f"{m}_{s}.jsonl"
        seen = done_ids(path)
        todo[(d, m, s)] = (path, [q for q in load_questions(s, args.limit, d) if q["question_id"] not in seen])
    total = sum(len(qs) for _, qs in todo.values())
    print(f"{total} questions to run in {len(groups)} groups", flush=True)
    finished, start = 0, time.perf_counter()
    for (d, m, s), (path, questions) in todo.items():
        if not questions:
            continue
        method = get_method(m, dataset=d)
        path.parent.mkdir(parents=True, exist_ok=True)
        group_start = time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(method.run(q["question"], q["question_id"]).to_dict()) + "\n")
                f.flush()
                finished += 1
        elapsed = time.perf_counter() - start
        eta = elapsed / finished * (total - finished)
        print(f"{d}/{m}_{s}: {len(questions)} in {time.perf_counter() - group_start:.0f}s "
              f"({finished}/{total}, ETA {eta / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
