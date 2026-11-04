"""MedQuAD knowledge base and A/B/C evaluation questions in the project format. Draft by D for A.

    python -m evrag.data.medquad                      # clones MedQuAD into data/raw/MedQuAD if needed
    python -m evrag.data.medquad --per-scenario 60 --out data/medquad

MedQuAD (Ben Abacha & Demner-Fushman, 2019; CC BY 4.0) holds question-answer pairs taken
from NIH sites, one page per condition. Every answered pair becomes knowledge-base chunks;
questions about common conditions become evaluation items whose scenario says what the
knowledge base holds for them after hold-outs:

  A  sufficient   the answer's chunks are in the corpus                     -> should answer
  B  missing      every chunk about that condition is held out             -> should abstain
  C  distracting  only the asked section is held out; the other sections   -> should abstain
                  about the same condition stay in the corpus

A "no answer" label is only valid against the whole corpus, so: pages about one concept
(same focus name or same specific UMLS CUI set, across sites) and their neighbours (pages
sharing a specific CUI) are held out together; exact-duplicate chunks are merged within a
concept before labelling; and B/C candidates whose gold answer still overlaps a remaining
chunk (word 8-grams) are rejected.
The labels are automatic, so dataset.csv marks them evidence_scenario_status=auto until A
audits them.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from evrag.config import DATA_DIR, RAW_DIR, load_config

MEDQUAD_GIT = "https://github.com/abachaa/MedQuAD.git"
DEFAULT_SOURCE = RAW_DIR / "MedQuAD"
DEFAULT_OUT = DATA_DIR / "medquad"

# Sites about common conditions. GARD and GHR (rare and genetic diseases) stay corpus-only.
QUESTION_SOURCES = ("CancerGov", "NIDDK", "NINDS", "SeniorHealth", "NHLBI", "CDC")
# Question types a user would ask a health chatbot. "information" (What is X?) is corpus-only
# and is the usual section left behind in scenario C.
QUESTION_TYPES = ("symptoms", "treatment", "causes", "exams and tests", "outlook", "prevention",
                  "susceptibility", "complications", "considerations", "stages")
SECTION_NAMES = {"information": "Overview", "exams and tests": "Exams and tests",
                 "genetic changes": "Genetic changes", "support groups": "Support groups"}
HOLDOUT_CUI_DOCS = 40     # a CUI on more pages than this is an umbrella term, not a related page
CHUNK_WORDS = 150         # MiniLM reads 256 word pieces and silently truncates the rest
SHINGLE_WORDS = 8
LEAK_MAX = 0.10           # reject a B/C candidate if a remaining chunk shares >= 10% of its gold 8-grams
ANSWER_WORDS = (20, 400)  # gold answers outside this range are too thin or too long to grade
COLUMNS = ("question_id", "question", "gold_answers", "scenario", "split", "gold_doc_ids",
           "source_doc_id", "evidence_scenario_status", "focus", "qtype", "source", "leak_overlap")

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Doc:
    key: str                # e.g. NIDDK_0000001 (MedQuAD ids repeat across sites)
    source: str
    url: str
    focus: str
    cuis: tuple
    pairs: list = field(default_factory=list)


@dataclass
class Pair:
    qid: str                # e.g. NIDDK_0000001-2
    qtype: str
    question: str
    answer: str
    doc: Doc


@dataclass
class Corpus:
    chunk_ids: list                   # canonical chunk ids in answer order
    texts: dict                       # chunk id -> header + body
    owners: dict                      # chunk id -> qids whose answer contains this chunk
    pair_chunks: dict                 # qid -> canonical chunk ids, in answer order
    duplicates: int = 0


def short_source(dirname: str) -> str:
    """'5_NIDDK_QA' -> 'NIDDK', '8_NHLBI_QA_XML' -> 'NHLBI'."""
    return re.sub(r"_QA(_XML)?$", "", re.sub(r"^\d+_", "", dirname))


def section_name(qtype: str) -> str:
    return SECTION_NAMES.get(qtype, qtype.capitalize()) or "Information"


def clean_text(text: str) -> str:
    """Collapse whitespace inside paragraphs; keep paragraph breaks as single newlines."""
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n".join(p for p in paragraphs if p)


def clean_question(text: str) -> str:
    """MedQuAD writes 'What is (are) X ?' and sometimes 'X? ?'; drop the space and repeats."""
    return re.sub(r"\?{2,}", "?", re.sub(r"\s+([?.!,])", r"\1", " ".join(text.split())))


def ensure_source(source: Path) -> str | None:
    """Clone MedQuAD if it is not there; return the checked-out commit when known."""
    if not any(source.glob("*/*.xml")):
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", MEDQUAD_GIT, str(source)], check=True)
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_medquad(source: Path) -> list[Doc]:
    """Read every answered QA pair; the three MedlinePlus sets whose answers were removed drop out.

    Ids come from the file name and the pair's pid, not from the id/qid attributes: CancerGov
    files 0000013_2_1.xml ... 0000013_2_6.xml all carry id="0000013_2" and repeat its qids.
    For every other file the result is the same as the attributes (<stem>-<pid>).
    """
    docs = []
    for path in sorted(source.glob("*/*.xml")):
        root = ET.parse(path).getroot()
        site = short_source(path.parent.name)
        focus = " ".join((root.findtext("Focus") or "").split())
        cuis = tuple(sorted({c.text.strip() for c in root.iter("CUI") if c.text and c.text.strip()}))
        doc = Doc(f"{site}_{path.stem}", site, root.get("url") or "", focus, cuis)
        for index, qa in enumerate(root.iter("QAPair"), 1):
            q = qa.find("Question")
            answer = clean_text(qa.findtext("Answer") or "")
            question = clean_question((q.text if q is not None else None) or "")
            if not answer or not question:
                continue
            qid = f"{doc.key}-{(qa.get('pid') or '').strip() or index}"
            doc.pairs.append(Pair(qid, (q.get("qtype") or "").strip().lower(), question, answer, doc))
        if doc.pairs:
            docs.append(doc)
    if not docs:
        raise ValueError(f"No answered MedQuAD QA pairs under {source}")
    qids = Counter(p.qid for d in docs for p in d.pairs)
    if any(n > 1 for n in qids.values()):
        raise ValueError(f"Repeated QA pair ids: {[q for q, n in qids.items() if n > 1][:5]}")
    return docs


def concept_groups(docs: list[Doc], max_cui_docs: int = HOLDOUT_CUI_DOCS) -> dict[str, str]:
    """Map each doc key to a group id: pages with the same focus name, or with the same CUI set
    that holds at least one specific CUI (listed on at most max_cui_docs pages), are one concept.

    MedQuAD's CUIs come from mapping the focus name, so a multi-CUI page such as "Branchial
    arch syndrome X-linked" also lists fragments like Syndrome (C0039082) that would chain
    unrelated conditions together if any shared CUI counted, and dozens of unrelated pages
    carry exactly the set ("C0039082",). Related pages are handled by neighbourhoods().
    """
    cui_docs = Counter(c for d in docs for c in set(d.cuis))
    parent = {d.key: d.key for d in docs}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first = {}
    for doc in docs:
        tags = [("cuis", doc.cuis)] if any(cui_docs[c] <= max_cui_docs for c in doc.cuis) else []
        if doc.focus:
            tags.append(("focus", doc.focus.lower()))
        for tag in tags:
            if tag in first:
                a, b = find(doc.key), find(first[tag])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[tag] = doc.key
    return {d.key: find(d.key) for d in docs}


def neighbourhoods(docs: list[Doc], groups: dict, max_cui_docs: int = HOLDOUT_CUI_DOCS) -> dict:
    """Groups held out together with each group: itself plus every group sharing a CUI that is
    listed on at most max_cui_docs pages (one hop, no chaining). A CUI on more pages is an
    umbrella term such as Syndrome or Disease and does not make pages neighbours."""
    cui_docs = Counter(c for d in docs for c in set(d.cuis))
    by_cui = defaultdict(set)
    for doc in docs:
        for cui in doc.cuis:
            if cui_docs[cui] <= max_cui_docs:
                by_cui[cui].add(groups[doc.key])
    hood = {g: {g} for g in set(groups.values())}
    for doc in docs:
        for cui in doc.cuis:
            if cui_docs[cui] <= max_cui_docs:
                hood[groups[doc.key]] |= by_cui[cui]
    return hood


def chunk_answer(answer: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Pack sentences into chunks of at most max_words words; a longer sentence is cut into pieces."""
    chunks, current, count = [], [], 0
    for paragraph in answer.split("\n"):
        for sentence in _SENTENCE.split(paragraph):
            words = sentence.split()
            for start in range(0, len(words), max_words):
                piece = words[start:start + max_words]
                if current and count + len(piece) > max_words:
                    chunks.append(" ".join(current))
                    current, count = [], 0
                current.append(" ".join(piece))
                count += len(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _norm_question(question: str) -> str:
    return " ".join(_WORD.findall(question.lower()))


def shingles(text: str, n: int = SHINGLE_WORDS) -> set:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_chunks(docs: list[Doc], groups: dict, max_words: int = CHUNK_WORDS) -> Corpus:
    """Chunk every answer. Identical chunk bodies are kept once per concept group (the same page
    on two sites), never across concepts: a paragraph shared by two conditions stays under each
    condition's own header, so holding one condition out leaves no chunk named after it."""
    canonical, corpus = {}, Corpus([], {}, defaultdict(set), {})
    for doc in docs:
        for pair in doc.pairs:
            header = f"{doc.focus} — {section_name(pair.qtype)}" if doc.focus else section_name(pair.qtype)
            ids = []
            for i, body in enumerate(chunk_answer(pair.answer, max_words)):
                norm = (groups[doc.key], " ".join(_WORD.findall(body.lower())))
                cid = canonical.get(norm)
                if cid is None:
                    cid = canonical[norm] = f"{pair.qid}#{i}"
                    corpus.texts[cid] = f"{header}\n\n{body}"
                    corpus.chunk_ids.append(cid)
                else:
                    corpus.duplicates += 1
                corpus.owners[cid].add(pair.qid)
                if cid not in ids:
                    ids.append(cid)
            corpus.pair_chunks[pair.qid] = ids
    return corpus


@dataclass
class Item:
    scenario: str
    pair: Pair
    group: str
    held: set                       # qids whose chunks leave the corpus because of this item
    gold_doc_ids: list
    leak: float = 0.0
    split: str = ""


def _overlap_counts(pool, corpus):
    """For each pool answer, how many of its 8-grams each corpus chunk shares (one pass over the corpus)."""
    index, sizes = defaultdict(list), []
    for i, (_, pair) in enumerate(pool):
        grams = shingles(pair.answer)
        sizes.append(max(len(grams), 1))
        for gram in grams:
            index[gram].append(i)
    counts = defaultdict(Counter)
    for cid in corpus.chunk_ids:
        for gram in shingles(corpus.texts[cid]):
            for i in index.get(gram, ()):
                counts[i][cid] += 1
    return counts, sizes


def select_items(docs, groups, corpus, *, per_scenario, rng, question_sources=QUESTION_SOURCES,
                 question_types=QUESTION_TYPES, answer_words=ANSWER_WORDS, leak_max=LEAK_MAX,
                 holdout_cui_docs=HOLDOUT_CUI_DOCS):
    """Pick one question per concept group and label it B, C or A (in that order, so hold-outs
    never delete an A item's gold chunks); return (items, held qids, stats).

    B holds out the whole neighbourhood of the group; C holds out the asked section across the
    neighbourhood and keeps the group's other sections; a group touched by any hold-out is not
    used again.
    """
    group_pairs = defaultdict(list)
    for doc in docs:
        for pair in doc.pairs:
            group_pairs[groups[doc.key]].append(pair)
    hood = neighbourhoods(docs, groups, holdout_cui_docs)

    def eligible(pair):
        return (pair.qtype in question_types and pair.doc.source in question_sources
                and answer_words[0] <= len(pair.answer.split()) <= answer_words[1])

    order = sorted(group_pairs)
    rng.shuffle(order)
    pool = []
    for group in order:
        pairs = group_pairs[group]
        candidates = [p for p in pairs if eligible(p)]
        if candidates:
            pool.append((group, rng.choice(candidates)))
    counts, sizes = _overlap_counts(pool, corpus)

    def leak_after(i, held):
        remaining = (c for cid, c in counts[i].items() if corpus.owners[cid] - held)
        return max(remaining, default=0) / sizes[i]

    quotas = {"B": per_scenario, "C": per_scenario, "A": per_scenario}
    items, used, skipped = [], set(), Counter()
    for scenario in ("B", "C", "A"):
        for i, (group, pair) in enumerate(pool):
            if not quotas[scenario]:
                break
            if group in used:
                continue
            own = group_pairs[group]
            if scenario == "A":
                # Gold = the asked pair plus the identical question on the concept's other pages
                # (another site); not a page's other questions of the same type (CDC pages have
                # several "prevention" questions) and not a sibling page that shares the CUI set.
                asked = _norm_question(pair.question)
                gold = []
                for p in sorted(own, key=lambda p: p.qid):
                    if p.qtype == pair.qtype and _norm_question(p.question) == asked:
                        gold.extend(c for c in corpus.pair_chunks[p.qid] if c not in gold)
                items.append(Item("A", pair, group, set(), gold))
                used.add(group)
            else:
                if scenario == "C" and not any(p.qtype != pair.qtype for p in own):
                    skipped["C_no_other_section"] += 1
                    continue
                if hood[group] & used:
                    skipped[f"{scenario}_neighbour_used"] += 1
                    continue
                held = {p.qid for g in hood[group] for p in group_pairs[g]
                        if scenario == "B" or p.qtype == pair.qtype}
                leak = leak_after(i, held)
                if leak >= leak_max:
                    skipped[f"{scenario}_leak"] += 1
                    continue
                items.append(Item(scenario, pair, group, held, [], leak))
                used |= hood[group]
            quotas[scenario] -= 1
    held = set().union(*(item.held for item in items)) if items else set()
    for item in items:                              # final leak against the corpus everybody searches
        if item.scenario != "A":
            i = next(k for k, entry in enumerate(pool) if entry[1] is item.pair)
            item.leak = leak_after(i, held)
    stats = {"pool_groups": len(pool), "unfilled": {s: q for s, q in quotas.items() if q},
             "skipped": dict(skipped),
             "neighbourhood_groups": {"median": sorted(len(h) for h in hood.values())[len(hood) // 2],
                                      "max": max(len(h) for h in hood.values())}}
    return items, held, stats


def assign_splits(items, rng, test_fraction):
    """Stratified dev/test split inside each scenario."""
    for scenario in ("A", "B", "C"):
        group = [item for item in items if item.scenario == scenario]
        rng.shuffle(group)
        n_test = round(len(group) * test_fraction)
        for k, item in enumerate(group):
            item.split = "test" if k < n_test else "dev"
    rng.shuffle(items)
    items.sort(key=lambda item: item.split != "dev")


def build(source: Path, out: Path, *, per_scenario=60, test_fraction=0.25, seed=None,
          chunk_words=CHUNK_WORDS, leak_max=LEAK_MAX, question_sources=QUESTION_SOURCES) -> dict:
    commit = ensure_source(source)
    seed = load_config()["seed"] if seed is None else seed
    rng = random.Random(seed)
    docs = parse_medquad(source)
    groups = concept_groups(docs)
    corpus = build_chunks(docs, groups, chunk_words)
    items, held, stats = select_items(docs, groups, corpus, per_scenario=per_scenario, rng=rng,
                                      question_sources=tuple(question_sources), leak_max=leak_max)
    assign_splits(items, rng, test_fraction)
    kept = [cid for cid in corpus.chunk_ids if corpus.owners[cid] - held]
    doc_of = {p.qid: d for d in docs for p in d.pairs}
    if len(set(kept)) != len(kept) or len({item.pair.qid for item in items}) != len(items):
        raise ValueError("Chunk ids or question ids are not unique; the corpus would be rejected by load_corpus")
    for item in items:                              # B pages must leave nothing behind
        if item.scenario == "B" and any(corpus.owners[cid] & item.held for cid in kept):
            raise ValueError(f"Held-out page chunks survived for {item.pair.qid}")
    for item in items:
        if item.scenario == "A" and not set(item.gold_doc_ids) <= set(kept):
            raise ValueError(f"Gold chunks missing from the corpus for {item.pair.qid}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for cid in kept:
            f.write(json.dumps({"doc_id": cid, "text": corpus.texts[cid]}, ensure_ascii=False) + "\n")
    with (out / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for item in items:
            pair = item.pair
            writer.writerow([f"MQ_{item.scenario}_{pair.qid}", pair.question,
                             json.dumps([pair.answer], ensure_ascii=False), item.scenario, item.split,
                             json.dumps(item.gold_doc_ids), pair.qid, "auto", pair.doc.focus, pair.qtype,
                             pair.doc.source, f"{item.leak:.3f}"])
    manifest = {
        "medquad_commit": commit,
        "parameters": {"seed": seed, "per_scenario": per_scenario, "test_fraction": test_fraction,
                       "chunk_words": chunk_words, "shingle_words": SHINGLE_WORDS, "leak_max": leak_max,
                       "holdout_cui_docs": HOLDOUT_CUI_DOCS,
                       "answer_words": list(ANSWER_WORDS), "question_sources": list(question_sources),
                       "question_types": list(QUESTION_TYPES)},
        "counts": {"docs": len(docs), "answered_pairs": sum(len(d.pairs) for d in docs),
                   "pairs_by_source": dict(Counter(d.source for d in docs for _ in d.pairs)),
                   "concept_groups": len(set(groups.values())),
                   "chunks_before_dedupe": len(corpus.chunk_ids) + corpus.duplicates,
                   "duplicate_chunks_merged": corpus.duplicates, "chunks_total": len(corpus.chunk_ids),
                   "chunks_held_out": len(corpus.chunk_ids) - len(kept), "corpus_chunks": len(kept),
                   **stats},
        "scenarios": {s: dict(Counter(item.split for item in items if item.scenario == s))
                      for s in ("A", "B", "C")},
        "held_out": [{"question_id": f"MQ_{item.scenario}_{item.pair.qid}", "scenario": item.scenario,
                      "focus": item.pair.doc.focus, "qtype": item.pair.qtype,
                      "held_pages": sorted({f"{doc_of[q].key} ({doc_of[q].focus})" for q in item.held}),
                      "held_pairs": sorted(item.held),
                      "held_chunks": sum(1 for cid in corpus.chunk_ids
                                         if not (corpus.owners[cid] - held) and corpus.owners[cid] & item.held),
                      "leak_overlap": round(item.leak, 3)}
                     for item in items if item.scenario != "A"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="MedQuAD checkout")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-scenario", type=int, default=60)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None, help="defaults to configs/default.yaml seed")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--leak-max", type=float, default=LEAK_MAX)
    parser.add_argument("--question-sources", nargs="+", default=list(QUESTION_SOURCES))
    args = parser.parse_args(argv)
    if args.per_scenario < 1 or not 0 <= args.test_fraction < 1:
        parser.error("--per-scenario must be positive and --test-fraction in [0, 1)")
    manifest = build(args.source, args.out, per_scenario=args.per_scenario, test_fraction=args.test_fraction,
                     seed=args.seed, chunk_words=args.chunk_words, leak_max=args.leak_max,
                     question_sources=args.question_sources)
    print(json.dumps({"counts": manifest["counts"], "scenarios": manifest["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
