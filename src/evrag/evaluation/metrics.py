"""Metrics computed from results/runs/*.jsonl joined with dataset.csv. Owner: E.

    python -m evrag.evaluation.metrics --dataset cancer --split test  # -> results/metrics_cancer_test.csv
    python -m evrag.evaluation.metrics --split test --adjudication   # -> results/adjudication_test.csv to label
    python -m evrag.evaluation.metrics --split test --attribution 50 # + SHAP evidence reliance on 50 answers per run

A question is answerable when its `gold_answers` are non-empty (JSON list, or answers separated
by '|'). An answer counts as correct when a manual label in results/adjudication_<split>.csv says
so, otherwise when its best token F1 against the gold answers reaches CORRECT_F1.

Per method, per scenario (when dataset.csv has a `scenario` column) and for all questions:
  coverage                answered / all
  answer_correctness      correct answers / all questions
  selective_accuracy      correct answers / answered questions
  reliability             (correct answers + abstained unanswerable questions) / all questions
  abstention_recall       abstained unanswerable questions / all unanswerable questions
  over_refusal_rate       abstained answerable questions / all answerable questions
  unsupported_answer_rate answers whose content words are mostly absent from the evidence / answers with evidence
  token_f1, rouge_l       mean over answered answerable questions (best gold answer)
  gold_token_recall       share of the gold answer's tokens found in the answer (best gold), the
                          containment check used for long answers to short-span questions (COVID-QA)
  recall_at_k             a gold document is among the retrieved ones (questions with gold_doc_ids)
  latency_mean_ms, latency_p95_ms
  evidence_reliance       share of SHAP attribution on the evidence rather than the question (--attribution)
  gold_share              share of the evidence attribution on gold documents (--attribution)
answer_correctness also gets a 95% bootstrap interval; `paired_bootstrap` compares two methods.
Support is reported on its own (unsupported_answer_rate) rather than folded into correctness, so
a closed-book route, which has no evidence to be supported by, can still be compared on correctness.
The evidence is what the generator was given: the retrieved documents listed in
meta["context_doc_ids"] when a method records it, otherwise all retrieved documents.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

from evrag.config import DATASETS, RESULTS_DIR, RUNS_DIR, SEED, dataset_name, dataset_paths
from evrag.generator import rouge_l

CORRECT_F1 = 0.5        # automatic correctness threshold, until a manual label overrides it
SUPPORT_MIN = 0.5       # an answer is supported when at least this share of its content words is in the evidence
STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its
this that these those there their they them he she his her you your we our i not no can may might will would should
could do does did have has had which who whom what when where why how also such than then so very more most some any
all each other into about over after before during while because""".split())


# --- Per answer --------------------------------------------------------------------------------
def normalize(text: str) -> list[str]:
    """SQuAD normalisation: lower case, no punctuation or articles, whitespace tokens."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize(prediction), normalize(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Share of the reference tokens that the prediction contains (with multiplicity)."""
    p, r = normalize(prediction), normalize(reference)
    return sum((Counter(p) & Counter(r)).values()) / len(r) if r else 0.0


def best(metric, prediction: str, references: list[str]) -> float:
    return max((metric(prediction, r) for r in references), default=0.0)


def support(answer: str, evidence: list[str]) -> float:
    """Share of the answer's content words that occur in the evidence (0 with no evidence)."""
    words = [w for w in normalize(answer) if w not in STOPWORDS]
    if not words:
        return 1.0
    seen = set(normalize(" ".join(evidence)))
    return sum(w in seen for w in words) / len(words)


def evidence_reliance(values: np.ndarray, segments: list[int], gold: set[int] = frozenset()) -> dict:
    """From an attribution matrix [features, answer tokens] and each feature's segment (-1 for
    the question, else the evidence index): the share of absolute attribution on the evidence,
    and the share of that on the `gold` evidence indices."""
    mass = np.abs(np.asarray(values)).mean(1)
    segments = np.asarray(segments)
    evidence = mass[segments >= 0].sum()
    total = mass.sum()
    out = {"evidence_reliance": float(evidence / total) if total else 0.0}
    if gold:
        out["gold_share"] = float(mass[np.isin(segments, list(gold))].sum() / evidence) if evidence else 0.0
    return out


# --- Loading -----------------------------------------------------------------------------------
def parse_answers(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if cell.startswith("["):
        return [str(a).strip() for a in json.loads(cell) if str(a).strip()]
    return [a.strip() for a in cell.split("|") if a.strip()]


def load_dataset(path: Path, split: str) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    for r in rows:
        r["gold_answers"] = parse_answers(r.get("gold_answers", ""))
        r["gold_doc_ids"] = parse_answers(r.get("gold_doc_ids", ""))
    return {r["question_id"]: r for r in rows}


def load_runs(paths) -> list[dict]:
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs += [json.loads(line) for line in f if line.strip()]
    return runs


def load_labels(path: Path) -> dict[tuple[str, str], bool]:
    """Manual correctness labels: (question_id, method) -> correct, from a filled adjudication CSV."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["question_id"], r["method"]): r["label"].strip().lower() in {"correct", "1", "yes", "y"}
                for r in csv.DictReader(f) if r.get("label", "").strip()}


def evidence_docs(out: dict) -> list[dict]:
    """The retrieved documents the generator was given (meta["context_doc_ids"], else all)."""
    docs = out.get("retrieved_docs") or []
    ids = (out.get("meta") or {}).get("context_doc_ids")
    return docs if ids is None else [d for d in docs if d["doc_id"] in ids]


# --- Scoring -----------------------------------------------------------------------------------
def score_rows(runs: list[dict], dataset: dict[str, dict], labels=None) -> list[dict]:
    """One scored row per run output whose question is in `dataset`."""
    labels = labels or {}
    rows = []
    for out in runs:
        q = dataset.get(out["question_id"])
        if q is None:
            continue
        docs = out.get("retrieved_docs") or []
        answered = not out["abstained"]
        answerable = bool(q["gold_answers"])
        row = {"question_id": q["question_id"], "method": out["method"], "scenario": q.get("scenario", ""),
               "answered": answered, "answerable": answerable, "latency_ms": float(out["latency_ms"]),
               "f1": None, "rouge_l": None, "gold_recall": None, "support": None, "recall": None}
        if q["gold_doc_ids"] and docs:
            row["recall"] = float(any(d["doc_id"] in q["gold_doc_ids"] for d in docs))
        if answered:
            evidence = evidence_docs(out)
            row["support"] = support(out["answer"], [d["text"] for d in evidence]) if evidence else None
            if answerable:
                row["f1"] = best(token_f1, out["answer"], q["gold_answers"])
                row["rouge_l"] = best(rouge_l, out["answer"], q["gold_answers"])
                row["gold_recall"] = best(token_recall, out["answer"], q["gold_answers"])
        label = labels.get((row["question_id"], row["method"]))
        row["correct"] = answered and answerable and (label if label is not None else row["f1"] >= CORRECT_F1)
        row["reliable"] = row["correct"] or (not answered and not answerable)
        rows.append(row)
    return rows


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _rate(hits, total) -> float | None:
    return hits / total if total else None


def bootstrap_ci(values, n: int = 2000, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n: int = 2000, seed: int = SEED) -> dict:
    """Mean of a - b over the same questions, with a 95% interval; the gap is reliable when the
    interval excludes 0."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(diff, n, seed)
    return {"diff": float(diff.mean()), "ci_low": low, "ci_high": high}


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["answered"]]
    with_evidence = [r for r in answered if r["support"] is not None]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    good = [float(r["correct"]) for r in rows]
    latency = [r["latency_ms"] for r in rows]
    low, high = bootstrap_ci(good)
    return {
        "n": len(rows),
        "coverage": _rate(len(answered), len(rows)),
        "answer_correctness": _mean(good),
        "answer_correctness_ci_low": low if good else None,
        "answer_correctness_ci_high": high if good else None,
        "selective_accuracy": _rate(sum(r["correct"] for r in answered), len(answered)),
        "reliability": _mean(float(r["reliable"]) for r in rows),
        "abstention_recall": _rate(sum(not r["answered"] for r in unanswerable), len(unanswerable)),
        "over_refusal_rate": _rate(sum(not r["answered"] for r in answerable), len(answerable)),
        "unsupported_answer_rate": _rate(sum(r["support"] < SUPPORT_MIN for r in with_evidence), len(with_evidence)),
        "token_f1": _mean(r["f1"] for r in answered),
        "rouge_l": _mean(r["rouge_l"] for r in answered),
        "gold_token_recall": _mean(r["gold_recall"] for r in answered),
        "recall_at_k": _mean(r["recall"] for r in rows),
        "latency_mean_ms": float(np.mean(latency)) if latency else None,
        "latency_p95_ms": float(np.percentile(latency, 95)) if latency else None,
        "evidence_reliance": _mean(r.get("evidence_reliance") for r in rows),
        "gold_share": _mean(r.get("gold_share") for r in rows),
    }


def metrics_table(rows: list[dict]) -> list[dict]:
    """One summary row per (method, scenario), plus scenario 'all' per method. Questions without a
    scenario only count towards 'all'."""
    table = []
    for method in sorted({r["method"] for r in rows}):
        mine = [r for r in rows if r["method"] == method]
        for scenario in sorted({r["scenario"] for r in mine if r["scenario"]}) + ["all"]:
            subset = mine if scenario == "all" else [r for r in mine if r["scenario"] == scenario]
            table.append({"method": method, "scenario": scenario, **summarize(subset)})
    return table


# --- Optional steps ----------------------------------------------------------------------------
def add_attribution(rows: list[dict], runs: list[dict], dataset: dict[str, dict], limit: int,
                    unit: str = "sentence", permutations: int = 2) -> None:
    """SHAP evidence reliance for up to `limit` answered questions with evidence per method (needs
    the generator; it explains how the shared generator scores each method's answer)."""
    from evrag import explain

    by_key = {(o["question_id"], o["method"]): o for o in runs}
    done = Counter()
    for row in rows:
        out = by_key[(row["question_id"], row["method"])]
        docs = evidence_docs(out)
        if not row["answered"] or not docs or done[row["method"]] >= limit:
            continue
        q = dataset[row["question_id"]]
        contexts = [d["text"] for d in docs]
        feats = explain.features(q["question"], contexts, unit)
        f = explain.make_scorer(q["question"], contexts, out["answer"], feats)
        values = explain.shap_values(f, len(feats), permutations)
        gold = {i for i, d in enumerate(docs) if d["doc_id"] in q["gold_doc_ids"]}
        row.update(evidence_reliance(values, [ft.segment for ft in feats], gold))
        done[row["method"]] += 1


def write_adjudication(rows: list[dict], runs: list[dict], dataset: dict[str, dict], path: Path,
                       per_scenario: int = 30, seed: int = SEED) -> int:
    """A stratified sample of answered questions (per method and scenario) to label by hand."""
    answers = {(o["question_id"], o["method"]): o["answer"] for o in runs}
    rng = np.random.default_rng(seed)
    picked = []
    for key in sorted({(r["method"], r["scenario"]) for r in rows}):
        pool = [r for r in rows if (r["method"], r["scenario"]) == key and r["answered"]]
        picked += [pool[i] for i in sorted(rng.permutation(len(pool))[:per_scenario])]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "method", "scenario", "question", "gold_answers", "answer", "token_f1", "label"])
        for r in picked:
            q = dataset[r["question_id"]]
            w.writerow([r["question_id"], r["method"], r["scenario"], q["question"], " | ".join(q["gold_answers"]),
                        answers[(r["question_id"], r["method"])], "" if r["f1"] is None else f"{r['f1']:.2f}", ""])
    return len(picked)


def write_csv(table: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for row in table:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v) for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, help="default: EVRAG_DATASET or data.default")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--runs", nargs="*", help="run files (default: results/runs/<dataset>/*_<split>.jsonl)")
    parser.add_argument("--adjudication", action="store_true", help="write a sample to label by hand and stop")
    parser.add_argument("--attribution", type=int, default=0, metavar="N",
                        help="add SHAP evidence reliance for N answered questions per method")
    parser.add_argument("--out", help="default: results/metrics_<dataset>_<split>.csv")
    args = parser.parse_args()

    name = dataset_name(args.dataset)
    dataset = load_dataset(dataset_paths(name)["dataset"], args.split)
    runs = load_runs(args.runs or sorted((RUNS_DIR / name).glob(f"*_{args.split}.jsonl")))
    labels_path = RESULTS_DIR / f"adjudication_{name}_{args.split}.csv"
    args.out = args.out or str(RESULTS_DIR / f"metrics_{name}_{args.split}.csv")
    rows = score_rows(runs, dataset, load_labels(labels_path))
    if not rows:
        raise SystemExit(f"No run outputs match the {args.split} questions")
    if args.adjudication:
        n = write_adjudication(rows, runs, dataset, labels_path)
        print(f"Wrote {n} answers to label (column 'label': correct / incorrect) to {labels_path}")
        return
    if args.attribution:
        add_attribution(rows, runs, dataset, args.attribution)
    table = metrics_table(rows)
    write_csv(table, Path(args.out))
    print(f"Wrote {len(table)} rows ({len(runs)} outputs) to {args.out}")


if __name__ == "__main__":
    main()
