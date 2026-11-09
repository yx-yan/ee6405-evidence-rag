"""Training data for the shared answer generator (src/evrag/generator.py). Owner: D.

    python -m evrag.data.generator_data            # downloads into data/raw/generator, writes data/generator

Sources (general medical QA, kept apart from the RAG knowledge base and its evaluation questions):

  wikidoc_patient   medalpaca/medical_meadow_wikidoc_patient_information (HF, CC)   ~5.9k QA, patient-information pages
  wikidoc_textbook  medalpaca/medical_meadow_wikidoc (HF, CC)                       ~10k QA, WikiDoc living textbook
  mashqa            MASH-QA (Zhu et al., Findings of EMNLP 2020; WebMD articles)    ~35k QA with the source article
  mediqa            medalpaca/medical_meadow_mediqa (MEDIQA-AnS)                     consumer questions + source sections

Each example is either "closed" (question -> answer) or "context" (question + evidence chunks
-> answer), so one model serves every route. Context examples mimic retrieval: the article is cut
into <= CHUNK_WORDS-word chunks; the chunks holding the answer are mixed with chunks from the same
article and from other articles, in a seeded random order. Output: data/generator/{train,val,test}.jsonl
with id, source, mode, question, contexts, answer, plus manifest.json. MASH-QA keeps its official
splits; the others are split by a hash of the question. Questions in val/test never appear in
train, and --exclude drops questions of a later evaluation set (e.g. the RAG questions) everywhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from evrag.config import DATA_DIR, RAW_DIR, load_config

RAW = RAW_DIR / "generator"
OUT = DATA_DIR / "generator"
HF_FILES = {
    "wikidoc_patient": ("medalpaca/medical_meadow_wikidoc_patient_information", "medical_meadow_wikidoc_patient_info.json"),
    "wikidoc_textbook": ("medalpaca/medical_meadow_wikidoc", "medical_meadow_wikidoc.json"),
    "mediqa": ("medalpaca/medical_meadow_mediqa", "medical_meadow_mediqa.json"),
}
MASHQA_DRIVE_ID = "1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK"     # from github.com/mingzhu0527/MASHQA
SOURCES = ("wikidoc_patient", "wikidoc_textbook", "mashqa", "mediqa")
CHUNK_WORDS = 150
CONTEXT_CHUNKS = 4               # evidence chunks per context example (~600 words, fits BART's 1024 tokens)
ANSWER_WORDS = (5, 400)
HELD_OUT = {"val": 0.05, "test": 0.05}

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_JS_MARKERS = ("s_context", "webmd.", "$(", "function(", "require([")


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def clean(text: str) -> str:
    return " ".join(str(text).split())


def strip_webmd_script(context: str) -> str:
    """Some MASH-QA contexts start with WebMD's inline video-player JavaScript; drop that prefix."""
    head = context[:4000]
    if not any(m in head for m in _JS_MARKERS):
        return context
    ends = [m.end() for m in re.finditer(r"\}\s*\)\s*;?|\}", head)]
    cut = max(ends) if ends else 0
    return re.sub(r"^[\s;})\]]+", "", context[cut:])


def chunk_spans(text: str, max_words: int = CHUNK_WORDS) -> list[tuple[int, int]]:
    """Character spans of consecutive sentences packed into chunks of at most max_words words."""
    spans, start, count, pos = [], None, 0, 0
    for sentence in _SENTENCE.split(text):
        begin = text.find(sentence, pos)
        end = begin + len(sentence)
        pos = end
        n = len(sentence.split())
        if not n:
            continue
        if start is not None and count + n > max_words:
            spans.append((start, prev_end))
            start, count = None, 0
        if start is None:
            start = begin
        count += n
        prev_end = end
    if start is not None:
        spans.append((start, prev_end))
    return spans


def split_of(question: str) -> str:
    """Deterministic val/test/train assignment from the question text."""
    bucket = int(hashlib.sha1(norm(question).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < HELD_OUT["test"]:
        return "test"
    if bucket < HELD_OUT["test"] + HELD_OUT["val"]:
        return "val"
    return "train"


def example(source: str, mode: str, split: str, question: str, answer: str, contexts=()) -> dict:
    key = hashlib.sha1(f"{source}|{mode}|{norm(question)}|{norm(answer)}".encode()).hexdigest()[:12]
    return {"id": f"{source}-{mode}-{key}", "source": source, "mode": mode, "split": split,
            "question": clean(question), "contexts": [clean(c) for c in contexts], "answer": clean(answer)}


# ---------- loaders ----------

def download(raw: Path, sources) -> dict:
    """Fetch whatever is missing; return local paths. MASH-QA needs gdown (or a manual download)."""
    paths = {}
    for name in sources:
        if name in HF_FILES:
            from huggingface_hub import hf_hub_download
            repo, filename = HF_FILES[name]
            paths[name] = Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=raw / "hf"))
        elif name == "mashqa":
            folder = raw / "mashqa" / "mashqa_data"
            if not folder.exists():
                archive = raw / "mashqa" / "mashqa.zip"
                if not archive.exists():
                    try:
                        import gdown
                    except ImportError as exc:
                        raise RuntimeError(f"pip install gdown, or download MASH-QA from Google Drive id "
                                           f"{MASHQA_DRIVE_ID} to {archive}") from exc
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    gdown.download(id=MASHQA_DRIVE_ID, output=str(archive), quiet=True)
                with zipfile.ZipFile(archive) as z:
                    z.extractall(archive.parent, members=[m for m in z.namelist()
                                                          if m.startswith("mashqa_data/") and m.endswith(".json")])
            paths[name] = folder
    return paths


def load_alpaca_qa(path: Path, source: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        question, answer = clean(r.get("input", "")), clean(r.get("output", ""))
        if len(question.split()) >= 3 and question.endswith("?"):
            out.append(example(source, "closed", split_of(question), question, answer))
    return out


def load_mediqa(path: Path, rng: random.Random) -> list[dict]:
    """Rows share a question and summary answer across several source sections: one context
    example per (question, answer) with those sections chunked."""
    groups = defaultdict(list)
    for r in json.loads(path.read_text(encoding="utf-8")):
        groups[(clean(r["instruction"]), clean(r["output"]))].append(clean(r["input"]))
    out = []
    for (question, answer), sections in groups.items():
        chunks = [s[a:b] for s in sections for a, b in chunk_spans(s)][:CONTEXT_CHUNKS]
        if chunks and len(question.split()) >= 3:
            out.append(example("mediqa", "context", split_of(question), question, answer, chunks))
    return out


def load_mashqa(folder: Path, rng: random.Random) -> list[dict]:
    """Closed and context examples from MASH-QA's official splits (the *_full files)."""
    out = []
    for split, name in (("train", "train"), ("val", "val"), ("test", "test")):
        data = json.loads((folder / f"{name}_webmd_squad_v2_full.json").read_text(encoding="utf-8"))
        articles = []
        for article in data["data"]:
            for para in article["paragraphs"]:
                context = strip_webmd_script(para["context"])
                spans = chunk_spans(context)
                articles.append((context, spans, para["qas"]))
        pool = [(i, context[a:b]) for i, (context, spans, _) in enumerate(articles) for a, b in spans]
        for i, (context, spans, qas) in enumerate(articles):
            for qa in qas:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                answer = clean(" ".join(a["text"] for a in qa["answers"]))
                question = clean(qa["question"])
                out.append(example("mashqa", "closed", split, question, answer))
                start = context.find(qa["answers"][0]["text"].strip()[:80])
                if start < 0:
                    continue
                end = start + len(qa["answers"][0]["text"].strip())
                gold = [context[a:b] for a, b in spans if a < end and b > start][:CONTEXT_CHUNKS - 1]
                same = [context[a:b] for a, b in spans if not (a < end and b > start)]
                rng.shuffle(same)
                chunks = gold + same[:max(0, CONTEXT_CHUNKS - 1 - len(gold))]
                while len(chunks) < CONTEXT_CHUNKS:
                    j, text = pool[rng.randrange(len(pool))]
                    if j != i:
                        chunks.append(text)
                rng.shuffle(chunks)
                out.append(example("mashqa", "context", split, question, answer, chunks))
    return out


# ---------- assembly ----------

def read_exclusions(path: Path | None) -> set:
    if not path:
        return set()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".csv":
        return {norm(r["question"]) for r in csv.DictReader(text.splitlines()) if r.get("question")}
    return {norm(line) for line in text.splitlines() if line.strip()}


def assemble(examples: list[dict], exclude: set) -> tuple[dict, Counter]:
    """Filter, de-duplicate, keep held-out questions out of train, and drop excluded questions."""
    dropped = Counter()
    held_out = {norm(e["question"]) for e in examples if e["split"] != "train"}
    seen, splits = set(), {"train": [], "val": [], "test": []}
    for e in examples:
        words = len(e["answer"].split())
        q = norm(e["question"])
        if not ANSWER_WORDS[0] <= words <= ANSWER_WORDS[1]:
            dropped["answer_length"] += 1
        elif q in exclude:
            dropped["excluded_question"] += 1
        elif e["split"] == "train" and q in held_out:
            dropped["question_in_val_or_test"] += 1
        elif (e["mode"], q, norm(e["answer"])) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add((e["mode"], q, norm(e["answer"])))
            splits[e["split"]].append(e)
    return splits, dropped


def build(raw: Path = RAW, out: Path = OUT, *, sources=SOURCES, exclude: Path | None = None,
          seed: int | None = None, paths: dict | None = None) -> dict:
    seed = load_config()["seed"] if seed is None else seed
    rng = random.Random(seed)
    paths = paths or download(raw, sources)
    examples = []
    for name in sources:
        if name in ("wikidoc_patient", "wikidoc_textbook"):
            examples += load_alpaca_qa(paths[name], name)
        elif name == "mediqa":
            examples += load_mediqa(paths[name], rng)
        elif name == "mashqa":
            examples += load_mashqa(paths[name], rng)
    splits, dropped = assemble(examples, read_exclusions(exclude))
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        random.Random(f"{seed}-{split}").shuffle(rows)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps({k: v for k, v in e.items() if k != "split"}, ensure_ascii=False) + "\n")
    counts = {split: dict(Counter(f"{e['source']}/{e['mode']}" for e in rows)) for split, rows in splits.items()}
    manifest = {"seed": seed, "sources": list(sources), "chunk_words": CHUNK_WORDS, "context_chunks": CONTEXT_CHUNKS,
                "answer_words": list(ANSWER_WORDS), "held_out": HELD_OUT,
                "exclude_file": str(exclude) if exclude else None, "dropped": dict(dropped),
                "totals": {split: len(rows) for split, rows in splits.items()}, "counts": counts}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    parser.add_argument("--exclude", type=Path, default=None,
                        help="CSV with a 'question' column, or a text file with one question per line")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build(args.raw, args.out, sources=args.sources, exclude=args.exclude, seed=args.seed)
    print(json.dumps({k: manifest[k] for k in ("totals", "counts", "dropped")}, indent=2))


if __name__ == "__main__":
    main()
