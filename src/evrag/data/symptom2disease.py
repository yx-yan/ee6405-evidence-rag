"""Symptom2Disease: colloquial symptom descriptions labelled with a disease, as a test set over the
MedQuAD corpus. Owner: D.

    python -m evrag.data.symptom2disease        # reads data/raw/symptom2disease/Symptom2Disease.csv

The Kaggle file (1,200 rows, 24 labels, 1,153 distinct texts) has columns label,text. Each
description becomes one question (scenario "S"): gold_answers = [label]; focus = the MedQuAD
page(s) for that disease (PAGES below, kept only if present in the corpus); gold_doc_ids = those
pages' Symptoms chunks when they exist, otherwise all their chunks. Labels with no MedQuAD page
(Cervical spondylosis) keep the row with evidence_scenario_status=no_page, so answer grading can
still use them while retrieval metrics skip them. Rows are sampled per label with the shared
seed; the dev/test split is stratified per label.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evrag.config import DATA_DIR, RAW_DIR, load_config

DEFAULT_SOURCE = RAW_DIR / "symptom2disease" / "Symptom2Disease.csv"
DEFAULT_OUT = DATA_DIR / "symptom2disease"
DEFAULT_CORPUS = DATA_DIR / "medquad" / "corpus.jsonl"
# Kaggle label -> MedQuAD page names (focus, case-insensitive) that describe the same condition.
PAGES = {
    "Acne": ["Acne"], "Arthritis": ["Arthritis", "Osteoarthritis", "Rheumatoid Arthritis"],
    "Bronchial Asthma": ["Asthma"], "Cervical spondylosis": [], "Chicken pox": ["Chickenpox"],
    "Common Cold": ["Common Cold"], "Dengue": ["Dengue", "Dengue fever"], "Dimorphic Hemorrhoids": ["Hemorrhoids"],
    "Fungal infection": ["Fungal Infections", "Tinea Infections"], "Hypertension": ["High Blood Pressure"],
    "Impetigo": ["Impetigo"], "Jaundice": ["Jaundice"], "Malaria": ["Malaria"], "Migraine": ["Migraine"],
    "Pneumonia": ["Pneumonia"], "Psoriasis": ["Psoriasis"], "Typhoid": ["Typhoid Fever"],
    "Varicose Veins": ["Varicose Veins"], "allergy": ["Allergy", "Food Allergy"],
    "diabetes": ["Diabetes", "Type 1 Diabetes", "Type 2 Diabetes", "Diabetes Type 1", "Diabetes Type 2"],
    "drug reaction": ["Drug Reactions"], "gastroesophageal reflux disease": ["GERD", "Heartburn"],
    "peptic ulcer disease": ["Peptic Ulcer"], "urinary tract infection": ["Urinary Tract Infections", "Urinary Tract Infection In Adults"],
}
COLUMNS = ("question_id", "question", "gold_answers", "scenario", "split", "gold_doc_ids", "source_doc_id",
           "evidence_scenario_status", "focus", "qtype", "label")


def corpus_pages(corpus_path: Path) -> dict[str, list[tuple[str, str]]]:
    """lower-cased page name -> [(chunk id, section)]."""
    pages = defaultdict(list)
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                header = row["text"].split("\n", 1)[0]
                focus, _, section = header.rpartition(" — ")
                pages[focus.strip().lower()].append((row["doc_id"], section.strip()))
    return pages


def gold_for(label: str, pages: dict) -> tuple[list[str], list[str]]:
    """(page names present in the corpus, gold chunk ids) for a label."""
    present, gold = [], []
    for name in PAGES.get(label, []):
        chunks = pages.get(name.lower())
        if not chunks:
            continue
        present.append(name)
        symptoms = [cid for cid, section in chunks if section.lower() == "symptoms"]
        gold.extend(cid for cid in (symptoms or [cid for cid, _ in chunks]) if cid not in gold)
    return present, gold


def build(source: Path = DEFAULT_SOURCE, out: Path = DEFAULT_OUT, corpus: Path = DEFAULT_CORPUS, *,
          per_label: int = 10, test_fraction: float = 0.5, seed: int | None = None) -> dict:
    seed = load_config()["seed"] if seed is None else seed
    rng = random.Random(seed)
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    unknown = sorted({r["label"] for r in rows} - set(PAGES))
    if unknown:
        raise ValueError(f"Labels without a page mapping: {unknown}")
    pages = corpus_pages(corpus)
    seen, by_label = set(), defaultdict(list)
    for i, r in enumerate(rows):
        text = " ".join(r["text"].split())
        if text and text not in seen:
            seen.add(text)
            by_label[r["label"]].append((i, text))
    items = []
    for label in sorted(by_label):
        present, gold = gold_for(label, pages)
        sample = rng.sample(by_label[label], min(per_label, len(by_label[label])))
        rng.shuffle(sample)
        n_test = int(len(sample) * test_fraction + 0.5)
        for k, (i, text) in enumerate(sample):
            items.append({"question_id": f"S2D_{i}", "question": text, "gold_answers": [label], "scenario": "S",
                          "split": "test" if k < n_test else "dev", "gold_doc_ids": gold, "source_doc_id": "",
                          "evidence_scenario_status": "mapped" if present else "no_page",
                          "focus": "; ".join(present), "qtype": "symptoms", "label": label})
    items.sort(key=lambda r: (r["split"] != "dev", r["label"], r["question_id"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in items:
            writer.writerow([r["question_id"], r["question"], json.dumps(r["gold_answers"], ensure_ascii=False),
                             r["scenario"], r["split"], json.dumps(r["gold_doc_ids"]), r["source_doc_id"],
                             r["evidence_scenario_status"], r["focus"], r["qtype"], r["label"]])
    manifest = {
        "note": "Symptom2Disease descriptions over the MedQuAD corpus. gold_answers is the disease label; focus "
                "lists the MedQuAD pages for it; gold_doc_ids are those pages' Symptoms chunks (or all chunks).",
        "parameters": {"seed": seed, "per_label": per_label, "test_fraction": test_fraction, "corpus": str(corpus)},
        "counts": {"source_rows": len(rows), "distinct_texts": len(seen), "items": len(items),
                   "labels": len(by_label), "status": dict(Counter(r["evidence_scenario_status"] for r in items)),
                   "split": dict(Counter(r["split"] for r in items))},
        "pages": {label: {"pages_present": gold_for(label, pages)[0], "gold_chunks": len(gold_for(label, pages)[1]),
                          "descriptions": len(by_label[label])} for label in sorted(by_label)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build(args.source, args.out, args.corpus, per_label=args.per_label,
                     test_fraction=args.test_fraction, seed=args.seed)
    print(json.dumps(manifest["counts"], indent=2))
    print({label: v["pages_present"] for label, v in manifest["pages"].items()})


if __name__ == "__main__":
    main()
