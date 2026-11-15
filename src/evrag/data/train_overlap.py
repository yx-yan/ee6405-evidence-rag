"""Which evaluation answers the shared generator may have seen in training. Owner: E.

    python -m evrag.data.train_overlap      # writes data/processed/<set>/eval/train_overlap.csv

An answerable question is flagged `seen` when at least SEEN_SHARE of the word 8-grams of one of its
gold answers occur in the generator's training sources (generator_data.SOURCES: questions, answers
and context sections, all splits, so this is an upper bound). MEDIQA-AnS takes its sections from
NIH pages, including cancer.gov and NINDS, which MedQuAD also uses. Results notebooks report the
main table with and without the flagged questions.
"""
from __future__ import annotations

import csv
import json
import re

from evrag.config import PROCESSED_DIR
from evrag.data.generator_data import RAW, SOURCES, download
from evrag.data.unanswerable import SETS

N = 8
SEEN_SHARE = 0.5
_WORD = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    words = _WORD.findall(str(text).lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def training_texts(paths: dict) -> list[str]:
    texts = []
    for name, path in paths.items():
        if name == "mashqa":
            for file in sorted(path.glob("*_webmd_squad_v2_full.json")):
                for article in json.loads(file.read_text(encoding="utf-8"))["data"]:
                    for para in article["paragraphs"]:
                        texts += [para["context"]] + [qa["question"] for qa in para["qas"]]
        else:
            texts += [f"{r.get('input', '')} {r.get('output', '')}" for r in json.loads(path.read_text(encoding="utf-8"))]
    return texts


def overlap(answers: list[str], seen: set) -> float:
    """Largest share of an answer's 8-grams found in `seen` (0 for answers under 8 words)."""
    return max((len(g & seen) / len(g) for g in map(ngrams, answers) if g), default=0.0)


def main() -> None:
    seen = set().union(*map(ngrams, training_texts(download(RAW, SOURCES))))
    for name in SETS:
        path = PROCESSED_DIR / name / "eval"
        with open(path / "dataset.csv", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["scenario"] == "answerable"]
        with open(path / "train_overlap.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "overlap", "seen"])
            flagged = 0
            for r in rows:
                share = overlap(json.loads(r["gold_answers"]), seen)
                flagged += share >= SEEN_SHARE
                writer.writerow([r["question_id"], f"{share:.3f}", int(share >= SEEN_SHARE)])
        print(f"{name}: {flagged} of {len(rows)} answerable questions seen in generator training")


if __name__ == "__main__":
    main()
