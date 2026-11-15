"""Evaluation versions of the disease sets, with unanswerable questions. Owner: E.

    python -m evrag.data.unanswerable       # writes data/processed/<set>/eval/{corpus.jsonl,dataset.csv}

The original files are not changed. Each eval dataset has a `scenario` column:

  answerable        the set's own questions, unchanged
  missing_evidence  N_MISSING questions per split of this set whose gold passages are removed from
                    the eval corpus; their gold answers are emptied, since the corpus no longer holds
                    them. Answerable questions left without any gold passage are dropped (counted).
  out_of_domain     N_OUT_OF_DOMAIN questions per split from each of the other two sets (same split),
                    gold emptied: the corpus covers another disease category.

Questions are drawn with the config seed, per split, so dev and test never share a question.
"""
from __future__ import annotations

import argparse
import csv
import json
import random

from evrag.config import PROCESSED_DIR, SEED
from evrag.data.disease_sets import MEDQUAD_SITES
from evrag.data.wikidoc import write_jsonl

SETS = ["coronavirus", *MEDQUAD_SITES]
N_MISSING = 25
N_OUT_OF_DOMAIN = 25
FIELDS = ["question_id", "question", "gold_answers", "gold_doc_ids", "split", "scenario", "source_set"]


def read_set(name: str) -> tuple[list[dict], list[dict]]:
    folder = PROCESSED_DIR / name
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    with open(folder / "dataset.csv", encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))
    return corpus, questions


def sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    return rng.sample(rows, min(n, len(rows)))


def build_eval(name: str, sets: dict[str, tuple[list[dict], list[dict]]],
               seed: int = SEED) -> tuple[list[dict], list[dict], dict]:
    """Eval corpus, eval questions and counts for set `name`; `sets` maps every set name to
    (corpus, questions)."""
    rng = random.Random(f"{seed}-{name}")
    corpus, questions = sets[name]
    removed, missing, others = set(), [], []
    for split in ("dev", "test"):
        picked = sample([q for q in questions if q["split"] == split], N_MISSING, rng)
        missing += picked
        for q in picked:
            removed.update(json.loads(q["gold_doc_ids"]))
        for other in SETS:
            if other != name:
                pool = [q for q in sets[other][1] if q["split"] == split]
                others += [dict(q, source_set=other) for q in sample(pool, N_OUT_OF_DOMAIN, rng)]
    missing_ids = {q["question_id"] for q in missing}

    rows, dropped = [], 0
    for q in questions:
        if q["question_id"] in missing_ids:
            rows.append(dict(q, gold_answers="[]", gold_doc_ids="[]", scenario="missing_evidence"))
            continue
        kept = [d for d in json.loads(q["gold_doc_ids"]) if d not in removed]
        if not kept:
            dropped += 1
            continue
        rows.append(dict(q, gold_doc_ids=json.dumps(kept), scenario="answerable"))
    rows += [dict(q, gold_answers="[]", gold_doc_ids="[]", scenario="out_of_domain") for q in others]
    for row in rows:
        row.setdefault("source_set", name)
    eval_corpus = [d for d in corpus if d["doc_id"] not in removed]
    counts = {"passages_removed": len(removed), "answerable_dropped": dropped}
    return eval_corpus, rows, counts


def write_eval(name: str, corpus: list[dict], rows: list[dict]) -> None:
    out = PROCESSED_DIR / name / "eval"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    with open(out / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    sets = {name: read_set(name) for name in SETS}
    for name in SETS:
        corpus, rows, counts = build_eval(name, sets)
        write_eval(name, corpus, rows)
        table = {(r["split"], r["scenario"]): 0 for r in rows}
        for r in rows:
            table[(r["split"], r["scenario"])] += 1
        print(f"{name}: {len(corpus)} passages ({counts['passages_removed']} removed), "
              f"{counts['answerable_dropped']} answerable dropped, "
              + ", ".join(f"{s}/{c}={n}" for (s, c), n in sorted(table.items())))


if __name__ == "__main__":
    main()
