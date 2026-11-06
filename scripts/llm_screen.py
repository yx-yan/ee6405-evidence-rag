"""Screen candidate local LLMs for memorisation of MTS-Dialog and for the answer/judge tasks.

    python scripts/llm_screen.py build
    python scripts/llm_screen.py run --model qwen3:4b-instruct-2507-q4_K_M
    python scripts/llm_screen.py run --model llama2:7b --probes p1 p2 p3
    python scripts/llm_screen.py summarize

Dev split only. Raw replies go to results/llm_screen/<model>.jsonl and every number in
summary.csv is recomputed from those files, so nobody needs to re-call a model.

  p1  closed-book: question only; recall of gold-specific words, minus a mismatched gold answer
  p2  note continuation: greedy raw completion of a note's first half, minus another note's suffix
  p3  counterfactual: one term in the note is swapped; does the answer follow the note?
  p4a answering with the source note among four distractors
  p4b judging SUFFICIENT with the source note, INSUFFICIENT with hard negatives only
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import os
import random
import re
import time
from urllib.request import Request, urlopen

import numpy as np

from evrag import llm
from evrag.config import PROCESSED_DIR, RESULTS_DIR, ROOT, SEED, load_config
from evrag.methods.baseline import build_prompt
from evrag.methods.evidence_rag import (
    INSUFFICIENT, SUFFICIENT, build_judge_prompt, parse_judgement,
)
from evrag.retriever import Retriever
from evrag.schema import RetrievedDoc

OUT_DIR = RESULTS_DIR / "llm_screen"
SAMPLES = OUT_DIR / "samples.json"
MTS_JUDGE_PROMPT = ROOT / "configs" / "prompts" / "evidence_judge_v1.txt"   # the screen ran on MTS-Dialog
OLLAMA_URL = "http://localhost:11434"
TIMEOUT_SECONDS = 120          # the shared 30 s does not cover loading a model into memory
PROBES = ["p1", "p2", "p3", "p4a", "p4b"]

MIN_SUPPORT = 0.7              # share of gold words found in the source note
MIN_JUDGE_SUPPORT = 0.8        # stricter: the judge must find every part of the answer
MAX_ANCHOR_NOTES = 5           # a word in at most this many notes and questions identifies the case
MIN_NOTE_CHARS = 400
NGRAM = 8
CONTINUATION_TOKENS = 64
MAX_PER_TERM = 8               # keeps "hypertension" from dominating p3
N_P1_P4A, N_P2, N_P3, N_P4B = 60, 40, 40, 30

STOPWORDS = set(
    "a an the of and or to in on for with is was were are be been has have had his her he she "
    "patient patients who that this it at by as from not no their they any did does do what which "
    "reported states denies".split())
TERM_CLASSES = {
    "drug": ("aspirin lisinopril metformin atenolol metoprolol simvastatin lipitor coumadin warfarin "
             "prednisone albuterol insulin penicillin codeine sulfa morphine tylenol ibuprofen lasix "
             "synthroid zoloft prozac nexium prilosec plavix norvasc amoxicillin vicodin percocet "
             "ambien").split(),
    "condition": ("hypertension diabetes asthma cancer stroke depression arthritis migraines emphysema "
                  "glaucoma gout anemia osteoporosis hypothyroidism pneumonia epilepsy").split(),
}
# Items removed after reading samples.json, before any model was run. See docs/llm_selection.md.
EXCLUDED_P3 = {"QA_MED_62F77E7D6197863A"}     # the swap contradicts the question (thyroid treatment)
EXCLUDED_P4B_NEGATIVES = {       # generic despite an anchor: another patient's note can answer
    "QA_MED_DE0023E398111D43",
    "QA_MED_EDBAD80A93ADB583",
    "QA_MED_4EC9599FC203D176",
    "QA_MED_6A97982DCCF77DD3",
    "QA_MED_79D6EAA2676189EB",
    "QA_MED_C49B60838C5C6DDF",
    "QA_MED_0EF962215CC05578",
    "QA_MED_10716564F7BEA470",
    "QA_MED_2C7D5490E6050836",
    "QA_MED_33512007840CED1B",
    "QA_MED_46D4C7C3285E2D04",
    "QA_MED_47FEC9F491173C57",
    "QA_MED_5283F1B4E6646761",
    "QA_MED_6B86B273FF34FCE1",
    "QA_MED_6E4001871C0CF27C",
    "QA_MED_83F814F7A92E365C",
    "QA_MED_A4E00D7E6AA82111",
    "QA_MED_B281BC2C616CB3C3",
    "QA_MED_C6D537E112156BE1",
    "QA_MED_0F0B82FAE280AE9F",
    "QA_MED_3D914F9348C9CC0F",
    "QA_MED_4299DA7466DF0951",
    "QA_MED_4BE84111A613654B",
    "QA_MED_5B7C4E75C9485E2E",
    "QA_MED_61DBEC1D67AFE651",
    "QA_MED_6D976934BE74941F",
    "QA_MED_736E537F0F664A3D",
    "QA_MED_75F7313C20144E39",
    "QA_MED_9284D61C8C2BA62B",
    "QA_MED_B064BDBA19113968",
    "QA_MED_C9A5DA075F9E5C3E",
    "QA_MED_D48FF4B2F68A10FD",
}
REFUSAL = re.compile(r"\b(can't|cannot|can not|unable to|sorry|don't have|do not have)\b", re.IGNORECASE)
P1_PROMPT = (
    "The question below is about one de-identified clinical note from the public MTSamples collection. "
    "Recall that note and answer with your best guess of what it says, in one or two sentences. "
    "Do not ask for more information.\n\nQUESTION:\n{question}\n\nANSWER:\n"
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str, exclude: str = "") -> set[str]:
    """Distinct non-stopwords of text, leaving out any word that appears in `exclude`."""
    return set(tokens(text)) - STOPWORDS - set(tokens(exclude))


def recall(reference: set[str], reply: str) -> float:
    return len(reference & set(tokens(reply))) / len(reference) if reference else 0.0


def support(gold: str, note: str) -> float:
    """Share of the gold answer's content words that the note contains."""
    words = [t for t in tokens(gold) if t not in STOPWORDS]
    return sum(t in set(tokens(note)) for t in words) / len(words) if words else 0.0


def token_f1(reply: str, gold: str) -> float:
    a, b = collections.Counter(tokens(reply)), collections.Counter(tokens(gold))
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision, rec = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * rec / (precision + rec)


def rouge_l(reply: str, reference: str) -> float:
    a, b = tokens(reply), tokens(reference)
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], previous + 1 if x == y else max(row[j], row[j - 1])
    precision, rec = row[-1] / len(a), row[-1] / len(b)
    return 2 * precision * rec / (precision + rec) if row[-1] else 0.0


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_hit(reply: str, reference: str, prefix: str) -> bool:
    """True if the reply shares an n-gram with the reference that it could not copy from the prefix."""
    return bool((ngrams(reply) & ngrams(reference)) - ngrams(prefix))


def split_note(note: str) -> tuple[str, str]:
    """Cut a note in half at a word boundary, keeping the original spacing."""
    spans = list(re.finditer(r"\S+", note))
    cut = spans[len(spans) // 2].start()
    return note[:cut], note[cut:]


def swap_term(note: str, term: str, replacement: str) -> str:
    """Replace every whole-word `term`, matching a leading capital or all capitals."""
    def match_case(match):
        found = match.group(0)
        if found.isupper():
            return replacement.upper()
        return replacement.capitalize() if found[0].isupper() else replacement
    return re.sub(rf"\b{re.escape(term)}\b", match_case, note, flags=re.IGNORECASE)


def find_swap(note: str, gold: str, question: str, rng: random.Random) -> dict | None:
    """Pick a term the answer depends on and a same-class term absent from the note, or None."""
    lowered_note, lowered_gold, lowered_question = note.lower(), gold.lower(), question.lower()
    for kind, terms in TERM_CLASSES.items():
        for term in terms:
            pattern = rf"\b{term}\b"
            if not (re.search(pattern, lowered_note) and re.search(pattern, lowered_gold)):
                continue
            if term in lowered_question:
                continue
            options = [t for t in terms if t not in lowered_note and t not in lowered_gold
                       and t not in lowered_question]
            replacement = rng.choice(options)
            swapped = swap_term(note, term, replacement)
            if term in swapped.lower():     # e.g. "cancers" survives a whole-word swap
                continue
            return {"term": term, "replacement": replacement, "kind": kind, "note": swapped}
    return None


def section(note: str) -> str:
    return note.split("\n", 1)[0]


def load_dev() -> tuple[list[dict], dict[str, str]]:
    """Dev questions with their gold answer and source note, plus the dev-only notes."""
    with open(PROCESSED_DIR / "corpus.jsonl", encoding="utf-8") as f:
        corpus = {d["doc_id"]: d["text"] for d in map(json.loads, f)}
    with open(PROCESSED_DIR / "dataset.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    for row in rows:
        row["gold"] = json.loads(row["gold_answers"])[0]
        row["note"] = corpus[row["source_doc_id"]]
    return rows, {r["source_doc_id"]: r["note"] for r in rows}


def anchors(question: str, note_frequency: collections.Counter) -> set[str]:
    """Rare question words. Without one, another patient's note can answer the question just as well."""
    return {t for t in content_tokens(question) if t.isalpha() and len(t) >= 4
            and note_frequency[t] <= MAX_ANCHOR_NOTES}


def distractors(row: dict, search, dev_notes: dict[str, str], n: int,
                forbidden: frozenset[str] = frozenset()) -> list[dict]:
    """Top retrieved distinct dev notes that are not the source note and cannot answer the question."""
    gold_words = content_tokens(row["gold"], row["question"])
    picked, seen = [], {row["note"]}
    for doc in search(row["question"], 150):
        if (doc.doc_id == row["source_doc_id"] or doc.doc_id not in dev_notes or doc.text in seen
                or recall(gold_words, doc.text) >= 0.5 or forbidden & set(tokens(doc.text))):
            continue
        seen.add(doc.text)
        picked.append({"doc_id": doc.doc_id, "text": doc.text})
        if len(picked) == n:
            return picked
    raise ValueError(f"Too few distractors for {row['question_id']}")


def with_source(row: dict, docs: list[dict], rng: random.Random) -> list[dict]:
    """Insert the source note at a random rank so position does not give it away."""
    merged = list(docs)
    merged.insert(rng.randrange(len(docs) + 1), {"doc_id": row["source_doc_id"], "text": row["note"]})
    return merged


def build_samples(rows: list[dict], dev_notes: dict[str, str], search, seed: int = SEED) -> dict:
    """Draw the fixed item lists every model is tested on."""
    rng = random.Random(seed)
    supported = [r for r in rows if support(r["gold"], r["note"]) >= MIN_SUPPORT]
    counts = collections.Counter(r["question"] for r in rows)
    texts = list(dev_notes.values()) + [r["question"] for r in rows]
    note_frequency = collections.Counter(t for text in texts for t in set(tokens(text)))
    by_section = collections.defaultdict(list)
    for row in rows:
        by_section[section(row["note"])].append(row)

    closed_book, answering = [], []
    for row in rng.sample(supported, N_P1_P4A):
        others = [r for r in by_section[section(row["note"])] if r["source_doc_id"] != row["source_doc_id"]]
        base = {"question_id": row["question_id"], "question": row["question"], "gold": row["gold"]}
        closed_book.append({**base, "control_gold": rng.choice(others or rows)["gold"]})
        answering.append({**base, "docs": with_source(row, distractors(row, search, dev_notes, 4), rng)})

    long_notes = sorted(((i, n) for i, n in dev_notes.items() if len(n) >= MIN_NOTE_CHARS))
    continuation = []
    for doc_id, note in rng.sample(long_notes, N_P2):
        others = [n for i, n in long_notes if i != doc_id and section(n) == section(note)]
        prefix, suffix = split_note(note)
        control = split_note(rng.choice(others or [n for i, n in long_notes if i != doc_id]))[1]
        continuation.append({"question_id": doc_id, "prefix": prefix, "suffix": suffix, "control_suffix": control})

    swaps, per_term = [], collections.Counter()
    for row in rng.sample(rows, len(rows)):
        found = find_swap(row["note"], row["gold"], row["question"], rng)
        if not found or per_term[found["term"]] >= MAX_PER_TERM or row["question_id"] in EXCLUDED_P3:
            continue
        per_term[found["term"]] += 1
        swaps.append({"question_id": row["question_id"], "question": row["question"], "gold": row["gold"],
                      "docs": [{"doc_id": row["source_doc_id"], "text": found.pop("note")}], **found})
        if len(swaps) == N_P3:
            break

    # Own generator and fixed walk order: excluding one item leaves the other picks in place.
    judge_rng = random.Random(seed + 1)
    usable = [r for r in rows if counts[r["question"]] == 1 and len(r["note"]) >= 80]
    well_supported = [r for r in usable if support(r["gold"], r["note"]) >= MIN_JUDGE_SUPPORT]
    judging = []
    for row in judge_rng.sample(well_supported, N_P4B):
        docs = with_source(row, distractors(row, search, dev_notes, 4), judge_rng)
        judging.append({"question_id": row["question_id"], "question": row["question"], "gold": row["gold"],
                        "label": SUFFICIENT, "docs": docs})
    taken = {item["question_id"] for item in judging}
    anchored = [(r, anchors(r["question"], note_frequency)) for r in usable if r["question_id"] not in taken]
    anchored = [(r, words) for r, words in anchored if words]
    negatives = [(r, w) for r, w in judge_rng.sample(anchored, len(anchored))
                 if r["question_id"] not in EXCLUDED_P4B_NEGATIVES]
    for row, words in negatives[:N_P4B]:
        docs = distractors(row, search, dev_notes, 5, forbidden=frozenset(words))
        judging.append({"question_id": row["question_id"], "question": row["question"], "gold": row["gold"],
                        "label": INSUFFICIENT, "anchors": sorted(words), "docs": docs})
    return {"seed": seed, "p1": closed_book, "p2": continuation, "p3": swaps,
            "p4a": answering, "p4b": judging}


def retrieved(item: dict) -> list[RetrievedDoc]:
    return [RetrievedDoc(d["doc_id"], 0.0, d["text"]) for d in item["docs"]]


def prompt_for(probe: str, item: dict, judge_template: str) -> str:
    if probe == "p1":
        return P1_PROMPT.format(question=item["question"])
    if probe == "p4b":
        return build_judge_prompt(item["question"], retrieved(item), judge_template)
    return build_prompt(item["question"], retrieved(item))


def complete_raw(model: str, prefix: str) -> str:
    """Plain text completion with no chat template; llm.generate only speaks chat."""
    payload = {"model": model, "prompt": prefix, "raw": True, "stream": False,
               "options": {"temperature": 0, "seed": SEED, "num_predict": CONTINUATION_TOKENS}}
    request = Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))["response"]


def chat_no_think(model: str, prompt: str, config: dict) -> str:
    """Same request as llm.generate plus reasoning_effort, which llm.py does not send."""
    settings = config["llm"]
    payload = {"model": model, "temperature": settings["temperature"], "max_tokens": settings["max_tokens"],
               "reasoning_effort": "none", "messages": [{"role": "user", "content": prompt}]}
    request = Request(f"{OLLAMA_URL}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        reply = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    if not reply or not reply.strip():
        raise llm.LLMCallError("LLM endpoint returned a malformed or empty completion")
    return reply.strip()


def result_path(model: str, smoke: bool = False):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", model) + ".jsonl"
    return (OUT_DIR / "smoke" if smoke else OUT_DIR) / name


def run(model: str, probes: list[str], limit: int | None, no_think: bool = False) -> None:
    os.environ["LLM_MODEL"] = model
    os.environ.setdefault("LLM_BASE_URL", f"{OLLAMA_URL}/v1")
    os.environ.setdefault("LLM_API_KEY", "ollama")
    config = copy.deepcopy(load_config())
    config["llm"]["timeout_seconds"] = TIMEOUT_SECONDS
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    judge_template = MTS_JUDGE_PROMPT.read_text(encoding="utf-8")

    label = f"{model}-no-think" if no_think else model
    chat = ((lambda prompt: chat_no_think(model, prompt, config)) if no_think
            else (lambda prompt: llm.generate(prompt, config=config)))
    path = result_path(label, smoke=limit is not None)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if limit is None and path.exists():     # resume after a crash
        with open(path, encoding="utf-8") as f:
            done = {(r["probe"], r["question_id"]) for r in map(json.loads, f) if not r["error"]}

    chat("Reply with the word ready.")      # load the model outside the timings
    with open(path, "w" if limit is not None else "a", encoding="utf-8") as f:
        for probe in probes:
            for item in samples[probe][:limit]:
                if (probe, item["question_id"]) in done:
                    continue
                reply, error = None, None
                start = time.perf_counter()
                try:
                    if probe == "p2":
                        reply = complete_raw(model, item["prefix"])
                    else:
                        reply = chat(prompt_for(probe, item, judge_template))
                except (llm.LLMCallError, OSError, KeyError) as exc:
                    error = str(exc)
                f.write(json.dumps({"model": label, "probe": probe, "question_id": item["question_id"],
                                    "reply": reply, "error": error,
                                    "latency_ms": (time.perf_counter() - start) * 1000},
                                   ensure_ascii=False) + "\n")
                f.flush()
            print(f"{label} {probe} done", flush=True)


def bootstrap_ci(values, seed: int = SEED, resamples: int = 2000) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(data, (resamples, data.size)).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(records: list[dict], samples: dict) -> dict:
    """Metrics for one model. Failed calls score zero and are also counted in error_rate."""
    by_probe = collections.defaultdict(dict)
    for record in records:
        by_probe[record["probe"]][record["question_id"]] = record
    out = {}

    def add_mean(name, values):
        low, high = bootstrap_ci(values)
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = float(np.mean(values)), low, high

    def replies(probe):
        return [(item, by_probe[probe][item["question_id"]].get("reply") or "")
                for item in samples[probe] if item["question_id"] in by_probe[probe]]

    if by_probe["p1"]:
        rows = replies("p1")
        true_scores = [recall(content_tokens(i["gold"], i["question"]), r) for i, r in rows]
        control = [recall(content_tokens(i["control_gold"], i["question"]), r) for i, r in rows]
        out["p1_recall"], out["p1_control"] = float(np.mean(true_scores)), float(np.mean(control))
        add_mean("p1_gap", np.subtract(true_scores, control))
        out["p1_refusal"] = float(np.mean([bool(REFUSAL.search(r)) for _, r in rows]))
    if by_probe["p2"]:
        rows = replies("p2")
        out["p2_hit"] = float(np.mean([ngram_hit(r, i["suffix"], i["prefix"]) for i, r in rows]))
        out["p2_hit_control"] = float(np.mean([ngram_hit(r, i["control_suffix"], i["prefix"]) for i, r in rows]))
        # Score against as much reference as the model was allowed to write.
        cut = lambda text: " ".join(text.split()[:CONTINUATION_TOKENS])
        true_scores = [rouge_l(r, cut(i["suffix"])) for i, r in rows]
        control = [rouge_l(r, cut(i["control_suffix"])) for i, r in rows]
        out["p2_rouge"], out["p2_rouge_control"] = float(np.mean(true_scores)), float(np.mean(control))
        add_mean("p2_gap", np.subtract(true_scores, control))
    if by_probe["p3"]:
        rows = replies("p3")
        add_mean("p3_follow", [i["replacement"] in r.lower() for i, r in rows])
        out["p3_original"] = float(np.mean([i["term"] in r.lower() for i, r in rows]))
    if by_probe["p4a"]:
        add_mean("p4a_f1", [token_f1(r, i["gold"]) for i, r in replies("p4a")])
    if by_probe["p4b"]:
        rows = replies("p4b")
        parsed = [(i["label"], parse_judgement(r)) for i, r in rows]
        positive = [j.verdict == SUFFICIENT for label, j in parsed if label == SUFFICIENT]
        negative = [j.verdict == INSUFFICIENT for label, j in parsed if label == INSUFFICIENT]
        out["p4b_tpr"], out["p4b_tnr"] = float(np.mean(positive)), float(np.mean(negative))
        out["p4b_balanced_acc"] = (out["p4b_tpr"] + out["p4b_tnr"]) / 2
        out["p4b_balanced_lo"], out["p4b_balanced_hi"] = balanced_ci(positive, negative)
        out["p4b_over_refusal"] = 1 - out["p4b_tpr"]
        out["p4b_parse_fail"] = float(np.mean([j.status == "failed" for _, j in parsed]))
    chat = [r for r in records if r["probe"] != "p2" and not r["error"]]
    if chat:
        latencies = [r["latency_ms"] / 1000 for r in chat]
        out["latency_p50_s"] = float(np.percentile(latencies, 50))
        out["latency_p95_s"] = float(np.percentile(latencies, 95))
    out["n_calls"], out["error_rate"] = len(records), float(np.mean([bool(r["error"]) for r in records]))
    return out


def balanced_ci(positive, negative, seed: int = SEED, resamples: int = 2000) -> tuple[float, float]:
    """Resample each class separately so the 95% interval respects the balanced design."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    values = (rng.choice(pos, (resamples, pos.size)).mean(axis=1)
              + rng.choice(neg, (resamples, neg.size)).mean(axis=1)) / 2
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    table = []
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            records = list(map(json.loads, f))
        latest = {(r["probe"], r["question_id"]): r for r in records}   # a resumed retry replaces the failure
        table.append({"model": records[0]["model"], **score(list(latest.values()), samples)})
    columns = list(dict.fromkeys(key for row in table for key in row))
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                          for row in table])
    print(f"Wrote {len(table)} models to {OUT_DIR / 'summary.csv'}")


def build() -> None:
    config = copy.deepcopy(load_config())
    local_weights = ROOT / "models" / "all-MiniLM-L6-v2"     # from scripts/b_download_model.py
    if local_weights.exists():
        config["retrieval"]["embedding_model"] = str(local_weights)
    retriever = Retriever(config=config)
    rows, dev_notes = load_dev()
    samples = build_samples(rows, dev_notes, retriever.search)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES.write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in samples.items() if k != "seed"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("summarize")
    runner = commands.add_parser("run")
    runner.add_argument("--model", required=True)
    runner.add_argument("--probes", nargs="+", default=PROBES, choices=PROBES)
    runner.add_argument("--limit", type=int, default=None, help="smoke test: first N items, written to smoke/")
    runner.add_argument("--no-think", action="store_true", help="supplementary: turn off a thinking model's reasoning")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "summarize":
        summarize()
    else:
        run(args.model, args.probes, args.limit, args.no_think)


if __name__ == "__main__":
    main()
