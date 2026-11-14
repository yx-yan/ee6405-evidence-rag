"""Figures for the project video, from results/final only (no model is run). Owner: E.

    python scripts/video_figures.py

Writes results/figures/video/*.png. The notebook figures in results/figures/ are reused as they are;
these add the overview and per-member slides:

    pipeline.png             the five methods as one pipeline (opening)
    datasets.png             test questions per set and scenario (A)
    basic_rag_errors.png     what basic RAG gets right and wrong (B)
    closed_vs_open_book.png  the generator with and without evidence (D)
    graph_example.png        how the graph retriever finds the evidence for "What is MVO?" (C)
    retriever_heatmap.png    Recall@5 of the four retrievers (C)
    coronavirus_quality.png  gold token recall on coronavirus (C)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from evrag.config import FINAL_DIR, RESULTS_DIR, dataset_paths
from evrag.evaluation.metrics import load_dataset, load_runs

OUT = RESULTS_DIR / "figures" / "video"
SETS = ["coronavirus", "cancer", "neuro"]
METHODS = ["nlp_only", "basic_rag", "graph_rag", "threshold_rag", "evidence_rag"]
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SCENARIO_COLORS = {"answerable": "#2a78d6", "missing_evidence": "#eb6834", "out_of_domain": "#1baf7a"}
INK, MUTED, LINE = "#2b2a27", "#6b6a66", "#c9c8c3"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "font.size": 12,
                     "figure.constrained_layout.use": True})

metrics = pd.read_csv(FINAL_DIR / "metrics.csv")
test = metrics[metrics.split == "test"]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT / name}.png")


def value(dataset: str, method: str, column: str, scenario: str = "all") -> float:
    row = test[(test.dataset == dataset) & (test.method == method) & (test.scenario == scenario)]
    return float(row[column].iloc[0])


def box(ax, x, y, text, color="#f3f2ee", edge=LINE, w=0.17, h=0.12, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=color, edgecolor=edge, linewidth=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color=INK, weight=weight)


def arrow(ax, a, b, color=MUTED):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5})


def pipeline() -> None:
    """Question -> evidence -> decision -> generator, one row per method."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.1, 1))
    ax.axis("off")
    for x, title in [(0.09, "Question"), (0.33, "Evidence"), (0.6, "Decision"), (0.87, "Output")]:
        ax.text(x, 0.97, title, ha="center", fontsize=13, weight="bold", color=MUTED)
    rows = [("nlp_only", "none (closed book)", "always answer"),
            ("basic_rag", "MiniLM top 5", "always answer"),
            ("graph_rag", "term graph + 1 hop", "always answer"),
            ("threshold_rag", "MiniLM top 5", "top-1 score ≥ τ ?"),
            ("evidence_rag", "MiniLM top 5\n(+ expanded query)", "score & coverage ok?\nelse retry, else abstain")]
    for i, (m, evidence, decision) in enumerate(rows):
        y = 0.84 - i * 0.18
        box(ax, 0.09, y, m, color="white", edge=COLORS[m], weight="bold")
        box(ax, 0.33, y, evidence, w=0.2)
        box(ax, 0.6, y, decision, w=0.22,
            color="#fdf1d8" if m in ("threshold_rag", "evidence_rag") else "#f3f2ee")
        box(ax, 0.87, y, "BART answer\nor abstain" if m in ("threshold_rag", "evidence_rag") else "BART answer",
            w=0.17)
        arrow(ax, (0.18, y), (0.23, y))
        arrow(ax, (0.43, y), (0.49, y))
        arrow(ax, (0.71, y), (0.78, y))
    ax.annotate("", xy=(0.33, 0.055), xytext=(0.6, 0.055),
                arrowprops={"arrowstyle": "-|>", "color": COLORS["evidence_rag"], "lw": 1.5,
                            "connectionstyle": "arc3,rad=-0.3"})
    ax.text(0.465, -0.07, "not enough evidence: retrieve again once with an expanded query", ha="center",
            fontsize=10, color=MUTED)
    save(fig, "pipeline")


def datasets() -> None:
    counts = {d: load_dataset(dataset_paths(d)["dataset"], "test") for d in SETS}
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, d in enumerate(SETS):
        left = 0
        for s, c in SCENARIO_COLORS.items():
            n = sum(q["scenario"] == s for q in counts[d].values())
            ax.barh(i, n, left=left, color=c, edgecolor="white", linewidth=2, label=s.replace("_", " ") if i == 0 else None)
            ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white", weight="bold")
            left += n
    ax.set(yticks=range(3), yticklabels=SETS, xlabel="test questions")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("Three knowledge bases, three kinds of question", loc="left", weight="bold")
    save(fig, "datasets")


def basic_rag_errors() -> None:
    """Of all test questions: right answers, wrong answers, and answers to unanswerable questions."""
    parts = {"correct answer": "#1baf7a", "wrong answer": "#eda100", "answered an unanswerable question": "#eb6834"}
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, d in enumerate(SETS):
        n = {s: value(d, "basic_rag", "n", s) for s in ("all", "answerable")}
        right = value(d, "basic_rag", "answer_correctness", "answerable") * n["answerable"]
        shares = [right, n["answerable"] - right, n["all"] - n["answerable"]]
        left = 0
        for (label, c), x in zip(parts.items(), np.array(shares) / n["all"]):
            ax.barh(i, x, left=left, color=c, edgecolor="white", linewidth=2, label=label if i == 0 else None)
            ax.text(left + x / 2, i, f"{x:.0%}", ha="center", va="center", color="white", weight="bold")
            left += x
    ax.set(yticks=range(3), yticklabels=SETS, xlim=(0, 1), xlabel="share of test questions")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("basic_rag always answers, even when it should not", loc="left", weight="bold")
    save(fig, "basic_rag_errors")


def closed_vs_open_book() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(3)
    for i, m in enumerate(["nlp_only", "basic_rag"]):
        v = [value(d, m, "gold_token_recall", "answerable") for d in SETS]
        ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=COLORS[m],
               label={"nlp_only": "no evidence (nlp_only)", "basic_rag": "with retrieved evidence (basic_rag)"}[m])
        for xi, vi in zip(x + (i - 0.5) * 0.36, v):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", color=INK)
    ax.set(xticks=x, xticklabels=SETS, ylim=(0, 0.6), ylabel="gold token recall")
    ax.legend(loc="upper left")
    ax.set_title("Same generator: evidence makes the difference", loc="left", weight="bold")
    save(fig, "closed_vs_open_book")


def graph_example(qid: str = "cov_Q43") -> None:
    """Question -> matched term -> passages, and one hop to related passages, for one test question."""
    run = {r["question_id"]: r for r in load_runs([FINAL_DIR / "runs" / "coronavirus" / "graph_rag_test.jsonl"])}[qid]
    question = load_dataset(dataset_paths("coronavirus")["dataset"], "test")[qid]
    gold = set(question["gold_doc_ids"])
    g = nx.DiGraph()
    for path in dict.fromkeys(run["meta"]["graph_paths"]):
        a, via, b = (p.strip() for p in path.split("->"))
        g.add_edge(a, via)
        g.add_edge(via, b)
    layer = {n: 0 if n == "question" else 1 if n.startswith("term:") and g.has_edge("question", n) else
             2 if any(g.has_edge(t, n) for t in g.successors("question")) else 3 if n.startswith("doc:") else 2.5
             for n in g}
    pos = {}
    for k in set(layer.values()):
        nodes = sorted(n for n in g if layer[n] == k)
        for j, n in enumerate(nodes):
            pos[n] = (k, (len(nodes) - 1) / 2 - j)
    colors = ["#2a78d6" if n == "question" else "#eda100" if n.startswith("term:") else
              "#1baf7a" if n[4:] in gold else "#d9d8d3" for n in g]
    labels = {n: (f'"{question["question"]}"' if n == "question" else n.replace("term:", "")
                  if n.startswith("term:") else n[4:] + ("  ✓ gold" if n[4:] in gold else "")) for n in g}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=LINE, arrows=True, arrowsize=14, width=1.5,
                           node_size=1800)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=1800, edgecolors="white", linewidths=2)
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.32, labels[n], ha="center", va="top", fontsize=11, color=INK)
    for k, title in [(0, "question"), (1, "matched rare term"), (2, "direct match"), (2.5, "shared term"),
                     (3, "one hop")]:
        if k in layer.values():
            ax.text(k, max(y for n, (x, y) in pos.items() if layer[n] == k) + 0.45, title, ha="center",
                    fontsize=11, color=MUTED, weight="bold")
    ax.set_ylim(min(y for _, y in pos.values()) - 0.8, max(y for _, y in pos.values()) + 0.8)
    dense = {r["question_id"]: r for r in load_runs([FINAL_DIR / "runs" / "coronavirus" / "basic_rag_test.jsonl"])}[qid]
    hit = gold & {d["doc_id"] for d in dense["retrieved_docs"]}
    ax.text(0, ax.get_ylim()[0] + 0.1, f"Same question, MiniLM top 5: {'found' if hit else 'no'} gold passage",
            fontsize=11, color=MUTED)
    ax.set_title("graph_rag: from a rare term straight to the right passage", loc="left", weight="bold")
    save(fig, "graph_example")


def retriever_heatmap() -> None:
    r = pd.read_csv(FINAL_DIR / "retrieval.csv")
    names = ["tfidf", "bm25", "minilm", "graph"]
    grid = r.pivot(index="retriever", columns="dataset", values="recall_at_5").loc[names, SETS]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(SETS)):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else INK, weight="bold")
    ax.set(xticks=range(3), xticklabels=SETS, yticks=range(4),
           yticklabels=["TF-IDF", "BM25 (ours)", "MiniLM (dense)", "graph (C)"])
    ax.grid(False)
    ax.set_title("Recall@5: no single retriever wins everywhere", loc="left", weight="bold")
    save(fig, "retriever_heatmap")


def coronavirus_quality() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    v = [value("coronavirus", m, "gold_token_recall", "answerable") for m in METHODS]
    ax.barh(range(5), v, color=[COLORS[m] for m in METHODS], height=0.6)
    for i, vi in enumerate(v):
        ax.text(vi + 0.01, i, f"{vi:.2f}", va="center", color=INK)
    ax.set(yticks=range(5), yticklabels=METHODS, xlim=(0, 0.7), xlabel="gold token recall (answerable questions)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("coronavirus: the graph route covers the most of the gold answer", loc="left", weight="bold")
    save(fig, "coronavirus_quality")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (pipeline, datasets, basic_rag_errors, closed_vs_open_book, graph_example, retriever_heatmap,
                 coronavirus_quality):
        make()


if __name__ == "__main__":
    main()
