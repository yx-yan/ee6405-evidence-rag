"""TREC 2017 LiveQA-Med: real consumer health questions as a test set over the MedQuAD corpus. Owner: D.

    python -m evrag.data.liveqa                 # reads data/raw/liveqa_med, writes data/liveqa

Inputs (data/raw/liveqa_med, from https://github.com/abachaa/LiveQA_MedicalTask_TREC2017 and the
judged-answers zip in https://github.com/abachaa/MedQuAD):
  test.xml                                   104 questions: the public's message, an NLM summary,
                                             focus/type annotations and NLM reference answers
  All-qrels_LiveQAMed2017-...-Judged-Answers.txt   "<qid> <1-Incorrect|2-Related|3-Incomplete|4-Excellent> <MedQuAD answer file>"
  All-2479-Answers-retrieved-from-MedQuAD.csv      the text of those MedQuAD answers

Each question becomes one row in the project schema. gold_answers = the NLM reference answers.
gold_doc_ids = the chunks, in our corpus, of MedQuAD answers that NLM judged 3-Incomplete or
4-Excellent for that question, verified by text against the corpus; questions whose judged answers
all come from the MedlinePlus sets without answers (ADAM, drugs, herbs) get no gold chunks and
evidence_scenario_status=reference_only. scenario is "R" (real question): nothing is held out.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from evrag.config import DATA_DIR, RAW_DIR, load_config

DEFAULT_SOURCE = RAW_DIR / "liveqa_med"
DEFAULT_OUT = DATA_DIR / "liveqa"
DEFAULT_CORPUS = DATA_DIR / "medquad" / "corpus.jsonl"
GOOD = ("3", "4")                      # NLM judgement codes that count as a usable answer
SITES = {"GHR": "GHR", "GARD": "GARD", "NIDDK": "NIDDK", "NINDS": "NINDS", "NHLBI": "NHLBI", "CDC": "CDC",
         "CancerGov": "CancerGov", "MPlusHealthTopics": "MPlus_Health_Topics", "NIHSeniorHealth": "SeniorHealth"}
COLUMNS = ("question_id", "question", "gold_answers", "scenario", "split", "gold_doc_ids", "source_doc_id",
           "evidence_scenario_status", "focus", "qtype", "summary")
_ANSWER_FILE = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<doc>\d+(?:_\d+)?)_Sec(?P<sec>\d+)\.txt$")
_WORD = re.compile(r"[a-z0-9]+")


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_questions(xml_path: Path) -> list[dict]:
    items = []
    for node in ET.parse(xml_path).getroot():
        original = node.find("Original-Question")
        subject = _clean(original.findtext("SUBJECT") if original is not None else "")
        message = _clean(original.findtext("MESSAGE") if original is not None else "")
        answers = [_clean(a.text) for a in node.iter("ANSWER") if _clean(a.text)]
        items.append({
            "qid": node.attrib["qid"], "question": f"{subject}\n{message}".strip(), "answers": answers,
            "summary": _clean(node.findtext("NLM-Summary")),
            "focus": "; ".join(_clean(f.text) for f in node.iter("FOCUS") if _clean(f.text)),
            "qtype": "; ".join(_clean(t.text) for t in node.iter("TYPE") if _clean(t.text)),
        })
    if not items:
        raise ValueError(f"No questions in {xml_path}")
    return items


def medquad_pair_id(answer_file: str) -> str | None:
    """'GHR_0000804_Sec2.txt' -> 'GHR_0000804-2'; None for sites whose answers are not in the corpus."""
    match = _ANSWER_FILE.match(answer_file)
    if not match or match["site"] not in SITES:
        return None
    return f"{SITES[match['site']]}_{match['doc']}-{match['sec']}"


def load_judgements(qrels_path: Path) -> dict[str, list[tuple[str, str]]]:
    """question number -> [(judgement code, answer file)]."""
    judged = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            judged.setdefault(parts[0], []).append((parts[1][0], parts[2]))
    return judged


def load_answer_texts(csv_path: Path) -> dict[str, str]:
    """answer file -> the answer text (after the 'Answer:' marker)."""
    texts = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            body = row.get("Answer") or ""
            marker = body.find("Answer:")
            texts[row["AnswerID"]] = _clean(body[marker + len("Answer:"):] if marker >= 0 else body)
    return texts


def corpus_chunks(corpus_path: Path) -> dict[str, list[tuple[str, str]]]:
    """MedQuAD pair id -> [(chunk id, chunk text)] in order."""
    pairs = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs.setdefault(row["doc_id"].rsplit("#", 1)[0], []).append((row["doc_id"], row["text"]))
    return pairs


def _starts_alike(answer: str, chunk_text: str, words: int = 12) -> bool:
    body = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else chunk_text
    return _WORD.findall(answer.lower())[:words] == _WORD.findall(body.lower())[:words]


def build(source: Path = DEFAULT_SOURCE, out: Path = DEFAULT_OUT, corpus: Path = DEFAULT_CORPUS, *,
          test_fraction: float = 0.5, seed: int | None = None) -> dict:
    seed = load_config()["seed"] if seed is None else seed
    questions = parse_questions(source / "test.xml")
    qrels = next(source.glob("All-qrels*.txt"))
    judged = load_judgements(qrels)
    answer_texts = load_answer_texts(next(source.glob("All-*Answers*.csv")))
    pairs = corpus_chunks(corpus)
    rows, stats = [], Counter()
    for item in questions:
        number = item["qid"].lstrip("TQ")
        gold, good_files = [], []
        for code, answer_file in judged.get(number, []):
            if code not in GOOD:
                continue
            good_files.append(answer_file)
            pair_id = medquad_pair_id(answer_file)
            if pair_id is None:
                stats["good_answers_outside_corpus"] += 1
                continue
            chunks = pairs.get(pair_id)
            if not chunks:
                stats["good_answers_missing_pair"] += 1
                continue
            if not _starts_alike(answer_texts.get(answer_file, ""), chunks[0][1]):
                stats["good_answers_text_mismatch"] += 1
                continue
            stats["good_answers_in_corpus"] += 1
            gold.extend(cid for cid, _ in chunks if cid not in gold)
        rows.append({"question_id": f"LQ_{item['qid']}", "question": item["question"], "gold_answers": item["answers"],
                     "scenario": "R", "gold_doc_ids": gold, "source_doc_id": "",
                     "evidence_scenario_status": "judged" if gold else "reference_only",
                     "focus": item["focus"], "qtype": item["qtype"], "summary": item["summary"],
                     "judged_good_files": good_files})
    rng = random.Random(seed)
    for status in ("judged", "reference_only"):
        group = [r for r in rows if r["evidence_scenario_status"] == status]
        rng.shuffle(group)
        n_test = int(len(group) * test_fraction + 0.5)
        for k, r in enumerate(group):
            r["split"] = "test" if k < n_test else "dev"
    rows.sort(key=lambda r: (r["split"] != "dev", int(r["question_id"][5:])))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([r["question_id"], r["question"], json.dumps(r["gold_answers"], ensure_ascii=False),
                             r["scenario"], r["split"], json.dumps(r["gold_doc_ids"]), r["source_doc_id"],
                             r["evidence_scenario_status"], r["focus"], r["qtype"], r["summary"]])
    manifest = {
        "note": "TREC 2017 LiveQA-Med test questions over the MedQuAD corpus. gold_doc_ids are corpus chunks "
                "of MedQuAD answers NLM judged 3-Incomplete or 4-Excellent for the question; reference_only "
                "rows have NLM reference answers but no judged answer inside the corpus.",
        "parameters": {"seed": seed, "test_fraction": test_fraction, "good_codes": list(GOOD), "corpus": str(corpus)},
        "counts": {"questions": len(rows), **dict(stats),
                   "status": dict(Counter(r["evidence_scenario_status"] for r in rows)),
                   "split": {s: dict(Counter(r["split"] for r in rows if r["evidence_scenario_status"] == s))
                             for s in ("judged", "reference_only")}},
        "judged": [{"question_id": r["question_id"], "status": r["evidence_scenario_status"],
                    "judged_good_files": r["judged_good_files"], "gold_doc_ids": r["gold_doc_ids"]} for r in rows],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build(args.source, args.out, args.corpus, test_fraction=args.test_fraction, seed=args.seed)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
