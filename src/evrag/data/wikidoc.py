"""RAG knowledge base and evaluation questions from WikiDoc patient information. Owner: A.

    python -m evrag.data.wikidoc            # downloads into data/raw/wikidoc, writes data/processed

Source: medalpaca/medical_meadow_wikidoc_patient_information (HF), ~5.9k question/answer pairs.
Keeps every pair whose question or answer mentions asthma, diabetes or hypertension. Each answer is
cut into CHUNK_WORDS-word passages with an OVERLAP-word overlap; its question becomes an evaluation
question whose gold answer is the full answer and whose gold documents are its passages. Questions
alternate between dev and test. Output (see data/README.md):

  processed/corpus.jsonl            doc_id, article_id, title, section, text, disease, source_url
  processed/dataset.csv             question_id, question, gold_answers, gold_doc_ids, split
  processed/by_disease/<d>.jsonl    the corpus split by the diseases a passage mentions
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evrag.config import PROCESSED_DIR, RAW_DIR

RAW = RAW_DIR / "wikidoc"
HF_REPO = "medalpaca/medical_meadow_wikidoc_patient_information"
HF_FILE = "medical_meadow_wikidoc_patient_info.json"
TARGET_DISEASES = ("Asthma", "Diabetes", "Hypertension")
CHUNK_WORDS = 150
OVERLAP = 30
SOURCE_URL = "https://huggingface.co"
DATASET_FIELDS = ["question_id", "question", "gold_answers", "gold_doc_ids", "split"]


def download(raw: Path = RAW) -> list[dict]:
    """The WikiDoc QA pairs as {instruction, input, output} dicts, in the dataset's order."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", cache_dir=raw / "hf")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP) -> list[str]:
    """Fixed windows of `size` words, each starting `size - overlap` words after the last."""
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[start:start + size]) for start in range(0, len(words), size - overlap)]


def build_records(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Corpus passages and evaluation questions, with sequential doc and question ids."""
    corpus, questions = [], []
    for idx, item in enumerate(pairs):
        question, answer = item.get("input", ""), item.get("output", "")
        if not question.strip() or not answer.strip():
            continue
        combined = f"{question} {answer}".lower()
        diseases = [d for d in TARGET_DISEASES if d.lower() in combined]
        if not diseases:
            continue
        doc_ids = []
        for chunk_idx, text in enumerate(chunk_words(answer)):
            doc_id = f"doc_consumer_{len(corpus)}"
            doc_ids.append(doc_id)
            corpus.append({
                "doc_id": doc_id,
                "article_id": f"art_consumer_{idx}_chunk_{chunk_idx}",
                "title": question[:50] + "...",
                "section": f"Patient Leaflet Section {chunk_idx + 1}",
                "text": text,
                "disease": "_".join(diseases),
                "source_url": SOURCE_URL,
            })
        n = len(questions)
        questions.append({
            "question_id": f"Q_eval_{n}",
            "question": question,
            "gold_answers": json.dumps([answer]),
            "gold_doc_ids": json.dumps(doc_ids),
            "split": "dev" if n % 2 == 0 else "test",
        })
    return corpus, questions


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(raw: Path = RAW, out: Path = PROCESSED_DIR) -> None:
    corpus, questions = build_records(download(raw))
    (out / "by_disease").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)

    by_disease = defaultdict(list)
    for passage in corpus:
        by_disease[passage["disease"]].append(passage)
    for disease, passages in by_disease.items():
        write_jsonl(out / "by_disease" / f"{disease.lower()}.jsonl", passages)

    with open(out / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(questions)

    print(f"{len(corpus)} passages, {len(questions)} questions -> {out}")
    for disease, passages in sorted(by_disease.items()):
        print(f"  by_disease/{disease.lower()}.jsonl: {len(passages)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW, help="download cache")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="output folder")
    args = parser.parse_args(argv)
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
