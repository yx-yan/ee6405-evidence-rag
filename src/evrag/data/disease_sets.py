"""Three single-disease RAG sets from three sources: coronavirus, cancer, neurological. Owner: A.

    python -m evrag.data.disease_sets       # downloads into data/raw, writes data/processed/<set>/

| Set | Disease category | Source | Licence |
|---|---|---|---|
| coronavirus | coronaviruses (SARS, MERS, COVID-19) | COVID-QA (Möller et al., 2020): CORD-19 papers | Apache 2.0 |
| cancer | cancer | MedQuAD (Ben Abacha & Demner-Fushman, 2019): cancer.gov (NCI) | CC BY 4.0 |
| neuro | neurological disorders | MedQuAD: ninds.nih.gov (NINDS) | CC BY 4.0 |

COVID-QA covers 147 papers, but most are about other viruses (HIV, hepatitis B, tuberculosis...);
only the 39 papers whose title names a coronavirus are kept. A paper is cut into CHUNK_WORDS-word
passages with an OVERLAP-word overlap; a question's gold documents are the passages that hold its
answer span. In MedQuAD every answer is one section of a page; it is cut the same way and its
question's gold documents are its passages. Passage text starts with the paper title or page focus
(e.g. "Adult Acute Lymphoblastic Leukemia: ..."), as a page heading would, since later passages of
an answer often never name the disease.

Every passage goes into the corpus. Evaluation questions skip MedQuAD's "research" type (NINDS and
NCI boilerplate about their own research programmes) and answers over MAX_ANSWER_WORDS words.
Questions asked more than once (the same question about several papers or pages) are merged: their
gold answers and gold documents are combined. At most MAX_QUESTIONS are drawn with the config seed;
they alternate between dev and test. Output per set, in the same schema as data/README.md:

  processed/<set>/corpus.jsonl      doc_id, article_id, title, section, text, disease, source_url
  processed/<set>/dataset.csv       question_id, question, gold_answers, gold_doc_ids, split
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from evrag.config import PROCESSED_DIR, RAW_DIR, SEED
from evrag.data.wikidoc import CHUNK_WORDS, DATASET_FIELDS, OVERLAP, write_jsonl

COVID_QA_URL = ("https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/"
                "data/question-answering/COVID-QA.json")
MEDQUAD_REPO = "https://github.com/abachaa/MedQuAD"
CORONAVIRUS = re.compile(r"corona|covid|sars|mers|ncov|wuhan", re.IGNORECASE)
MEDQUAD_SITES = {  # set name -> (MedQuAD folder, doc id prefix, disease label)
    "cancer": ("1_CancerGov_QA", "can", "Cancer"),
    "neuro": ("6_NINDS_QA", "neu", "Neurological"),
}
SKIP_QTYPES = {"research"}
MAX_ANSWER_WORDS = 300
MAX_QUESTIONS = 300


def chunk_spans(n_words: int, size: int = CHUNK_WORDS, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """Word ranges [start, end) of the windows wikidoc.chunk_words cuts from an n-word text."""
    if n_words <= size:
        return [(0, n_words)]
    return [(start, min(start + size, n_words)) for start in range(0, n_words, size - overlap)]


def answer_word_span(context: str, start: int, text: str) -> tuple[int, int]:
    """Word range [first, last + 1) of the answer at character `start` in `context`."""
    before = context[:start]
    first = len(before.split()) - (1 if before and not before[-1].isspace() else 0)
    return first, len(context[:start + len(text)].split())


def gold_chunks(spans: list[tuple[int, int]], first: int, end: int) -> list[int]:
    """Indices of the windows holding the whole answer, or else those overlapping it."""
    inside = [i for i, (s, e) in enumerate(spans) if s <= first and end <= e]
    return inside or [i for i, (s, e) in enumerate(spans) if s < end and first < e]


def add_passages(corpus: list[dict], prefix: str, heading: str, words: list[str], **fields) -> list[str]:
    """Append the windows of `words` to the corpus, each headed by `heading`; return their ids."""
    doc_ids = []
    for chunk_idx, (start, end) in enumerate(chunk_spans(len(words))):
        doc_id = f"{prefix}_{len(corpus)}"
        doc_ids.append(doc_id)
        corpus.append({"doc_id": doc_id, "text": f"{heading}: {' '.join(words[start:end])}",
                       **{key: value.format(chunk=chunk_idx + 1) for key, value in fields.items()}})
    return doc_ids


def merge_and_sample(candidates: list[dict], prefix: str, limit: int = MAX_QUESTIONS,
                     seed: int = SEED) -> list[dict]:
    """Merge repeated questions, draw at most `limit`, and alternate them between dev and test.

    Each candidate is {question, answer, doc_ids}; the result rows follow DATASET_FIELDS.
    """
    merged: dict[str, dict] = {}
    for c in candidates:
        key = " ".join(re.sub(r"[^\w\s]", " ", c["question"].lower()).split())
        row = merged.setdefault(key, {"question": c["question"].strip(), "answers": [], "doc_ids": []})
        row["answers"].append(c["answer"])
        row["doc_ids"] += [d for d in c["doc_ids"] if d not in row["doc_ids"]]
    rows = list(merged.values())
    if len(rows) > limit:
        keep = set(random.Random(seed).sample(range(len(rows)), limit))
        rows = [row for i, row in enumerate(rows) if i in keep]
    return [{
        "question_id": f"{prefix}_Q{n}",
        "question": row["question"],
        "gold_answers": json.dumps(row["answers"], ensure_ascii=False),
        "gold_doc_ids": json.dumps(row["doc_ids"]),
        "split": "dev" if n % 2 == 0 else "test",
    } for n, row in enumerate(rows)]


def covid_qa_records(squad: dict, limit: int = MAX_QUESTIONS) -> tuple[list[dict], list[dict]]:
    """Corpus and questions from the coronavirus papers of COVID-QA (SQuAD format)."""
    corpus, candidates = [], []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            lines = [line.strip() for line in context.split("\n") if line.strip()]
            title = lines[0]
            if not CORONAVIRUS.search(title):
                continue
            url = next((line for line in lines if line.startswith("http")), "")
            words = context.split()
            spans = chunk_spans(len(words))
            doc_ids = add_passages(corpus, "cov", title, words, article_id=f"cov_paper_{para['document_id']}",
                                   title=title, section="Chunk {chunk}", disease="Coronavirus", source_url=url)
            for qa in para["qas"]:
                answer = qa["answers"][0]
                first, end = answer_word_span(context, answer["answer_start"], answer["text"])
                candidates.append({"question": qa["question"], "answer": answer["text"].strip(),
                                   "doc_ids": [doc_ids[i] for i in gold_chunks(spans, first, end)]})
    return corpus, merge_and_sample(candidates, "cov", limit)


def medquad_records(site_dir: Path, prefix: str, disease: str,
                    limit: int = MAX_QUESTIONS) -> tuple[list[dict], list[dict]]:
    """Corpus and questions from one MedQuAD site folder of XML pages."""
    corpus, candidates = [], []
    for path in sorted(site_dir.glob("*.xml")):
        page = ET.parse(path).getroot()
        focus = " ".join((page.findtext("Focus") or "").split())
        for pair in page.iter("QAPair"):
            question = pair.find("Question")
            answer = " ".join((pair.findtext("Answer") or "").split())
            if not answer:
                continue
            words = answer.split()
            doc_ids = add_passages(corpus, prefix, focus, words, article_id=f"{prefix}_{question.get('qid')}",
                                   title=focus, section=question.get("qtype"), disease=disease,
                                   source_url=page.get("url", ""))
            if question.get("qtype") not in SKIP_QTYPES and len(words) <= MAX_ANSWER_WORDS:
                candidates.append({"question": question.text, "answer": answer, "doc_ids": doc_ids})
    return corpus, merge_and_sample(candidates, prefix, limit)


def download_covid_qa(raw: Path = RAW_DIR) -> dict:
    path = raw / "covid_qa" / "COVID-QA.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COVID_QA_URL, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def download_medquad(raw: Path = RAW_DIR) -> Path:
    path = raw / "MedQuAD"
    if not path.exists():
        subprocess.run(["git", "clone", "--depth", "1", MEDQUAD_REPO, str(path)], check=True)
    return path


def write_set(out: Path, corpus: list[dict], questions: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    with open(out / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(questions)
    print(f"{out.name}: {len(corpus)} passages, {len(questions)} questions -> {out}")


def build(raw: Path = RAW_DIR, out: Path = PROCESSED_DIR, limit: int = MAX_QUESTIONS) -> None:
    write_set(out / "coronavirus", *covid_qa_records(download_covid_qa(raw), limit))
    medquad = download_medquad(raw)
    for name, (folder, prefix, disease) in MEDQUAD_SITES.items():
        write_set(out / name, *medquad_records(medquad / folder, prefix, disease, limit))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="download cache")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="output folder")
    parser.add_argument("--limit", type=int, default=MAX_QUESTIONS, help="questions per set")
    args = parser.parse_args(argv)
    build(args.raw, args.out, args.limit)


if __name__ == "__main__":
    main()
