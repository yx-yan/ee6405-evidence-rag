"""Figures for the project video, from results/final only (no model is run). Owner: E.

    python scripts/video_figures.py           # add --guide to outline the camera corner

Writes results/figures/video/*.png. The notebook figures in results/figures/ are reused as they are;
these add the overview and per-member slides. All figures share the style in src/evrag/figures.py
(1920x1080, top-right corner left empty for the camera):

    pipeline.png             the five methods as one pipeline (opening)
    datasets.png             test questions per set and scenario (A)
    basic_rag_errors.png     what basic RAG gets right and wrong (B)
    closed_vs_open_book.png  the generator with and without evidence (D)
    graph_example.png        how the graph retriever finds the evidence for "What is MVO?" (C)
    retriever_heatmap.png    Recall@5 of the four retrievers (C)
    coronavirus_quality.png  gold token recall on coronavirus (C)

Theory slides (docs/video_script_v2.md), one formula per row with a one-line intuition under it:

    theory_task          answer or abstain, reliability, the three scenarios (opening)
    theory_explain       occlusion, LIME and SHAP attributions (demo)
    theory_chunking      sliding-window passages and the unanswerable scenarios (A)
    theory_dense         bi-encoder and cosine similarity (B)
    theory_sparse_graph  TF-IDF, BM25 and the term graph's one hop (C)
    theory_bart          BART's input and fine-tuning loss (D)
    theory_abstain       the two evidence signals, the retry and Youden's J (E)
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from evrag import figures as F
from evrag.config import FINAL_DIR, RESULTS_DIR, dataset_paths
from evrag.evaluation.metrics import load_dataset, load_runs

OUT = RESULTS_DIR / "figures" / "video"
SETS = ["coronavirus", "cancer", "neuro"]
METHODS = F.METHODS

metrics = pd.read_csv(FINAL_DIR / "metrics.csv")
test = metrics[metrics.split == "test"]


def save(fig, name: str) -> None:
    F.save(fig, OUT / f"{name}.png")
    plt.close(fig)


def value(dataset: str, method: str, column: str, scenario: str = "all") -> float:
    row = test[(test.dataset == dataset) & (test.method == method) & (test.scenario == scenario)]
    return float(row[column].iloc[0])


def box(ax, x, y, text, color=F.PANEL, edge=None, w=0.17, h=0.13, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                facecolor=color, edgecolor=edge or color, linewidth=2, mutation_aspect=0.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=12, color=F.INK, weight=weight, linespacing=1.3)


def arrow(ax, a, b, color=F.MUTED):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5,
                                                "shrinkA": 0, "shrinkB": 0})


def hbar_labels(ax, i, left, width, color, text):
    if width > 0.04 * ax.get_xlim()[1]:
        ax.text(left + width / 2, i, text, ha="center", va="center", color=F.text_on(color), weight="bold")


def pipeline() -> None:
    """Question -> evidence -> decision -> BART, one row per method."""
    fig, ax = F.figure("Five ways to answer a medical question",
                       "Same question, same BART generator; the methods differ in evidence and in when they answer",
                       diagram=True)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for x, title in [(0.1, "Method"), (0.43, "Evidence"), (0.8, "Decision")]:
        ax.text(x, 0.97, title.upper(), ha="center", fontsize=11, weight="bold", color=F.MUTED)
    rows = [("nlp_only", "none (closed book)", "always answer"),
            ("basic_rag", "MiniLM top 5", "always answer"),
            ("graph_rag", "term graph + 1 hop", "always answer"),
            ("threshold_rag", "MiniLM top 5", "answer if top-1 score ≥ τ,\nelse abstain"),
            ("evidence_rag", "MiniLM top 5\n(+ expanded query on retry)", "score and coverage ok?\nelse retry, else abstain")]
    for i, (m, evidence, decision) in enumerate(rows):
        y = 0.83 - i * 0.185
        box(ax, 0.1, y, m, color="white", edge=F.METHOD_COLORS[m], w=0.2, h=0.13, weight="bold")
        box(ax, 0.43, y, evidence, w=0.32, h=0.13)
        box(ax, 0.8, y, decision, w=0.32, h=0.13,
            color="#fdf1d8" if m in ("threshold_rag", "evidence_rag") else F.PANEL)
        arrow(ax, (0.2, y), (0.27, y))
        arrow(ax, (0.59, y), (0.64, y))
    y = 0.83 - 4 * 0.185 - 0.065
    ax.annotate("", xy=(0.47, y), xytext=(0.76, y),
                arrowprops={"arrowstyle": "-|>", "color": F.METHOD_COLORS["evidence_rag"], "lw": 2,
                            "shrinkA": 0, "shrinkB": 0, "connectionstyle": "arc3,rad=-0.3"})
    F.legend(fig, [Patch(color="#fdf1d8", label="may abstain"), Patch(color=F.PANEL, label="always answers")],
             title="DECISION")
    F.note(fig, "Pink arrow: evidence_rag\nretrieves once more with an\nexpanded query before it\ngives up.")
    save(fig, "pipeline")


def datasets() -> None:
    counts = {d: load_dataset(dataset_paths(d)["dataset"], "test") for d in SETS}
    fig, ax = F.figure("Three knowledge bases, three kinds of question",
                       "Test questions per set: answerable, and two kinds the system should refuse")
    for i, d in enumerate(SETS):
        left = 0
        for s, c in F.SCENARIO_COLORS.items():
            n = sum(q["scenario"] == s for q in counts[d].values())
            ax.barh(i, n, left=left, color=c, edgecolor="white", linewidth=2, height=0.6)
            hbar_labels(ax, i, left, n, c, str(n))
            left += n
    ax.set(yticks=range(3), yticklabels=SETS, xlabel="test questions")
    ax.tick_params(axis="y", labelsize=13)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    F.legend(fig, [Patch(color=c, label=s.replace("_", " ")) for s, c in F.SCENARIO_COLORS.items()],
             title="SCENARIO")
    F.note(fig, "Missing evidence: the gold\npassages are removed.\nOut of domain: questions\nfrom the other two sets.")
    save(fig, "datasets")


def basic_rag_errors() -> None:
    """Of all test questions: right answers, wrong answers, and answers to unanswerable questions."""
    parts = {"correct answer": F.STATUS["good"], "wrong answer": F.STATUS["warning"],
             "answered an\nunanswerable question": F.STATUS["critical"]}
    fig, ax = F.figure("basic_rag always answers, even when it should not",
                       "Share of all test questions, by what basic_rag did with them")
    for i, d in enumerate(SETS):
        n = {s: value(d, "basic_rag", "n", s) for s in ("all", "answerable")}
        right = value(d, "basic_rag", "answer_correctness", "answerable") * n["answerable"]
        shares = np.array([right, n["answerable"] - right, n["all"] - n["answerable"]]) / n["all"]
        left = 0
        for c, x in zip(parts.values(), shares):
            ax.barh(i, x, left=left, color=c, edgecolor="white", linewidth=2, height=0.6)
            hbar_labels(ax, i, left, x, c, f"{x:.0%}")
            left += x
    ax.set(yticks=range(3), yticklabels=SETS, xlim=(0, 1), xlabel="share of test questions")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="y", labelsize=13)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    F.legend(fig, [Patch(color=c, label=label) for label, c in parts.items()], title="OUTCOME")
    save(fig, "basic_rag_errors")


def closed_vs_open_book() -> None:
    fig, ax = F.figure("Same generator: evidence makes the difference",
                       "Gold token recall on answerable test questions")
    x = np.arange(3)
    labels = {"nlp_only": "no evidence\n(nlp_only)", "basic_rag": "retrieved evidence\n(basic_rag)"}
    for i, m in enumerate(labels):
        v = [value(d, m, "gold_token_recall", "answerable") for d in SETS]
        xs = x + (i - 0.5) * 0.34
        ax.bar(xs, v, 0.32, color=F.METHOD_COLORS[m], edgecolor="white", linewidth=2)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", color=F.INK, fontsize=12)
    ax.set(xticks=x, xticklabels=SETS, ylim=(0, 0.6), ylabel="gold token recall")
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", visible=False)
    F.legend(fig, [Patch(color=F.METHOD_COLORS[m], label=label) for m, label in labels.items()], title="GENERATOR INPUT")
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
             2 if any(g.has_edge(t, n) for t in g.successors("question")) else 4 if n.startswith("doc:") else 3
             for n in g}
    pos = {}
    for k in set(layer.values()):
        nodes = sorted(n for n in g if layer[n] == k)
        for j, n in enumerate(nodes):
            pos[n] = (k, (len(nodes) - 1) / 2 - j)
    kinds = {"question": F.INK, "rare term": F.METHOD_COLORS["threshold_rag"], "gold passage": F.METHOD_COLORS["graph_rag"],
             "other passage": "#d6d5d0"}
    kind = {n: "question" if n == "question" else "rare term" if n.startswith("term:") else
            "gold passage" if n[4:] in gold else "other passage" for n in g}
    labels = {n: "question" if n == "question" else n.replace("term:", "") if n.startswith("term:") else n[4:]
              for n in g}
    fig, ax = F.figure("graph_rag: from a rare term straight to the right passage",
                       f'Coronavirus test question: "{question["question"]}"', diagram=True)
    ax.axis("off")
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=F.LINE, arrows=True, arrowsize=16, width=1.8, node_size=1500)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[kinds[kind[n]] for n in g], node_size=1500,
                           edgecolors="white", linewidths=2)
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.3, labels[n] + ("  ✓" if kind[n] == "gold passage" else ""), ha="center", va="top",
                fontsize=12, color=F.INK, weight="bold" if kind[n] == "gold passage" else "normal")
    top = max(y for _, y in pos.values())
    for k, title in [(0, "question"), (1, "matched term"), (2, "direct match"), (3, "shared term"), (4, "one hop")]:
        if k in layer.values():
            ax.text(k, top + 0.5, title.upper(), ha="center", fontsize=11, color=F.MUTED, weight="bold")
    ax.set(xlim=(-0.45, max(layer.values()) + 0.45), ylim=(min(y for _, y in pos.values()) - 0.7, top + 0.75))
    F.legend(fig, [Line2D([], [], ls="", marker="o", ms=13, mec="white", mfc=c, label=k) for k, c in kinds.items()],
             title="NODE")
    dense = {r["question_id"]: r for r in load_runs([FINAL_DIR / "runs" / "coronavirus" / "basic_rag_test.jsonl"])}[qid]
    hit = gold & {d["doc_id"] for d in dense["retrieved_docs"]}
    F.note(fig, f"Same question with MiniLM\ntop 5: {'found' if hit else 'no'} gold passage.")
    save(fig, "graph_example")


def retriever_heatmap() -> None:
    r = pd.read_csv(FINAL_DIR / "retrieval.csv")
    names = list(F.RETRIEVER_LABELS)
    grid = r.pivot(index="retriever", columns="dataset", values="recall_at_5").loc[names, SETS]
    fig, ax = F.figure("Recall@5: no single retriever wins everywhere",
                       "Share of answerable test questions with a gold passage in the top 5")
    ax.imshow(grid.values, cmap=F.BLUES, vmin=0.3, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(SETS)):
            v = grid.values[i, j]
            best = v == grid.values[:, j].max()
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=15 if best else 13,
                    color=F.text_on(F.BLUES((v - 0.3) / 0.7)), weight="bold" if best else "normal")
    ax.set(xticks=range(3), xticklabels=SETS, yticks=range(4),
           yticklabels=F.RETRIEVER_LABELS.values())
    ax.tick_params(labelsize=13, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    F.note(fig, "Bold: best retriever\nfor that set.")
    save(fig, "retriever_heatmap")


def coronavirus_quality() -> None:
    fig, ax = F.figure("coronavirus: graph_rag covers the most of the gold answer",
                       "Gold token recall on answerable coronavirus test questions")
    v = [value("coronavirus", m, "gold_token_recall", "answerable") for m in METHODS]
    ax.barh(range(5), v, color=[F.METHOD_COLORS[m] for m in METHODS], height=0.6, edgecolor="white", linewidth=2)
    for i, vi in enumerate(v):
        ax.text(vi + 0.008, i, f"{vi:.2f}", va="center", color=F.INK, fontsize=12,
                weight="bold" if vi == max(v) else "normal")
    ax.set(yticks=range(5), yticklabels=METHODS, xlim=(0, 0.7), xlabel="gold token recall")
    ax.tick_params(axis="y", labelsize=13)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    F.note(fig, "COVID-QA gold answers are\nshort spans, so token F1\nunderstates every method.")
    save(fig, "coronavirus_quality")


def theory(name: str, title: str, subtitle: str, rows: list[tuple[str, str]], code: str) -> None:
    """A theory slide: each row is a formula (mathtext) in large type with its intuition under it, stacked
    in the left column and centred in the space under the title; the code it matches goes in the right
    column, under the camera corner. Type shrinks until everything fits."""
    for size in range(32, 17, -2):
        fig, ax = F.figure(title, subtitle, diagram=True)
        ax.remove()
        renderer = fig.canvas.get_renderer()
        w, h = fig.bbox.width, fig.bbox.height
        x, y, placed = 0.06, 0.76, []
        for formula, intuition in rows:
            f = fig.text(x, y, formula, fontsize=size, color=F.INK, va="top")
            fb = f.get_window_extent(renderer)
            i = fig.text(x, fb.y0 / h - 0.022, intuition, fontsize=max(15, size * 0.56), color=F.TEXT, va="top",
                         linespacing=1.35)
            ib = i.get_window_extent(renderer)
            placed.append((f, i, y, ib.y0 / h))
            y = ib.y0 / h - 0.06
        right = max(max(f.get_window_extent(renderer).x1, i.get_window_extent(renderer).x1) for f, i, *_ in placed)
        bottom = placed[-1][3]
        if bottom >= 0.07 and right <= 0.72 * w:
            break
        plt.close(fig)
    shift = (bottom - 0.07) / 2                     # centre the stack between the subtitle and the bottom
    for f, i, top, low in placed:
        for t in (f, i):
            t.set_y(t.get_position()[1] - shift)
        fig.add_artist(Line2D([x - 0.018] * 2, [low - shift, top - shift], transform=fig.transFigure,
                              color=F.METHOD_COLORS["evidence_rag"], lw=4, solid_capstyle="round"))
    F.note(fig, "In the code:\n" + code)
    save(fig, name)


def theory_slides() -> None:
    theory("theory_task", "The task: answer, or abstain",
           "One decision per question, scored on questions that should and should not be answered",
           [(r"$f(q, D) \in \{\, y,\ \perp \,\}$",
             "Given question q and retrieved passages D, output an answer y,\n"
             "or ⊥: \"not enough evidence\"."),
            (r"$\mathrm{reliability} = \dfrac{\mathrm{correct\ answers} + \mathrm{correct\ refusals}}"
             r"{\mathrm{all\ questions}}$",
             "Answering what it can and refusing what it can't both count as right."),
            ("answerable  ·  missing evidence  ·  out of domain",
             "The three test scenarios: only the first should be answered.")],
           "evaluation/metrics.py")
    theory("theory_explain", "Why did it say that? Word attributions",
           "Three ways to score how much each input word drove the same answer y",
           [(r"$a_i = \log p(y \mid x) - \log p(y \mid x_{\setminus i})$",
             "Occlusion: how much the answer's probability drops when word i is masked."),
            (r"LIME:  $g(z) = w_0 + \sum_i w_i z_i$",
             "A linear model fitted on random masks z; its weights are the attributions."),
            (r"SHAP:  $\phi_i$ = Shapley value of word $i$",
             "Each word's contribution averaged over mask combinations. All three score y\n"
             "by teacher forcing.")],
           "explain.py")
    theory("theory_chunking", "Cutting texts into passages",
           "A sliding window, and two ways to make questions that should not be answered",
           [(r"passage $k$ = words $[\,120k,\ 120k + 150\,)$",
             "Passages of 150 words, a new one every 120 words: neighbours overlap by 30."),
            (r"$|\mathrm{answer}| \leq 30\ \Rightarrow$  one passage holds it whole",
             "A short answer is never split in half at a border."),
            ("missing evidence  ·  out of domain",
             "Missing evidence: delete the gold passages from the corpus.\n"
             "Out of domain: take questions from the other two sets.")],
           "data/disease_sets.py\nchunk_spans")
    theory("theory_dense", "Dense retrieval: one vector per passage",
           "A bi-encoder: question and passage are encoded separately, then compared",
           [(r"$\mathbf{q} = E(q), \quad \mathbf{d} = E(d)$",
             "MiniLM turns a text into one vector (mean pooling over its tokens)."),
            (r"$\cos(\mathbf{q}, \mathbf{d}) = \dfrac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|\,\|\mathbf{d}\|}$",
             "Passages pointing the same way as the question score highest; the top 5 are kept."),
            (r"$\|\mathbf{q}\| = \|\mathbf{d}\| = 1\ \Rightarrow$  FAISS inner product = cosine",
             "Passages are encoded once; a question costs one encoding.")],
           "retriever.py")
    theory("theory_sparse_graph", "Word-based retrieval: BM25 and a term graph",
           "Exact words, weighted by how rare they are",
           [(r"TF-IDF:  $\mathrm{tf}(t,d) \cdot \mathrm{idf}(t)$",
             "A word counts more the more often it appears here, and the rarer it is overall."),
            (r"BM25:  $\sum_t \mathrm{idf}(t)\,\dfrac{\mathrm{tf}\,(k_1+1)}"
             r"{\mathrm{tf} + k_1(1 - b + b\,\frac{|d|}{\mathrm{avgdl}})}$",
             "k₁ = 1.5, b = 0.75: repeated words saturate, long passages are normalised."),
            (r"one hop:  $\mathrm{score} \times 0.35 \,/\, \log_2(\mathrm{df} + 1)$",
             "Graph: passages ↔ words. Direct score = share of the question's IDF weight a passage\n"
             "contains; a hop through rare words spreads, through common words it does not.")],
           "evaluation/retrieval_eval.py BM25\nmethods/graph_rag.py")
    theory("theory_bart", "The generator: BART, an encoder–decoder",
           "Pretrained as a denoiser, fine-tuned on medical question answering",
           [(r"$x$ = question: … context: [1] … [2] …",
             "The encoder reads the question and the passages; the decoder writes the answer\n"
             "while attending to them (cross-attention)."),
            (r"$\mathcal{L} = -\sum_t \log p(y_t \mid y_{<t}, x)$",
             "Fine-tuning: predict each next word of the right answer."),
            ("beam search, 4 beams",
             "Training: 66k examples, 3 epochs, lr 5e-5, batch 16.")],
           "generator.py\ndocs/generator.md")
    theory("theory_abstain", "When to abstain: two signals, one retry",
           "evidence_rag answers only when the evidence passes both checks",
           [(r"$s_1$ = top-1 cosine,   $s_2 = \frac{\sum_{w \in q \cap d} \mathrm{idf}(w)}{\sum_{w \in q} \mathrm{idf}(w)}$",
             "s₂ (coverage): how much of the question's important words the best passage holds."),
            (r"answer if $s_1 \geq \tau_1$ and $s_2 \geq \tau_2$",
             "Else expand once (Rocchio: q′ = q + the 5 top TF-IDF words of the top 3 passages),\n"
             "retrieve again, check again; still failing → abstain."),
            (r"$J$ = abstention recall $-$ over-refusal rate",
             "τ₁ and τ₂ are tuned on dev only, by Youden's J.")],
           "methods/evidence_rag.py\nevaluation/tune.py")


def main() -> None:
    F.GUIDE = "--guide" in sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (pipeline, datasets, basic_rag_errors, closed_vs_open_book, graph_example, retriever_heatmap,
                 coronavirus_quality, theory_slides):
        make()


if __name__ == "__main__":
    main()
