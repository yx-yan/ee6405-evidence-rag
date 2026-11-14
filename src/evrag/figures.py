"""One look for every figure (notebook and video): a 1920x1080 (16:9) white canvas, title and
subtitle top left, the plot below them, legend and note in a column on the right. The top-right
corner is left empty so the presenters' camera can be placed there in the video; `save` refuses a
figure that draws into it.

    fig, ax = figures.figure("Title", "subtitle")
    ...
    figures.legend(fig, handles)
    figures.save(fig, path)

Set GUIDE = True (or EVRAG_FIGURE_GUIDE=1) to outline the camera corner while checking a layout.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox

SIZE, DPI = (12.8, 7.2), 150                    # 1920 x 1080 px
CORNER = (0.74, 0.60)                           # kept empty: x > 0.74 and y > 0.60 (figure fraction)
LEFT, RIGHT, BOTTOM, TOP = 0.13, 0.70, 0.11, 0.76
SIDE_X, SIDE_TOP = 0.765, 0.56                  # legend / note column under the corner
GUIDE = os.environ.get("EVRAG_FIGURE_GUIDE") == "1"

INK, TEXT, MUTED = "#1f1e1c", "#52514e", "#8a8984"
SURFACE, PANEL, LINE, GRID = "#ffffff", "#f4f3ef", "#c9c8c3", "#e9e8e4"

# Colour follows the entity in every figure (validated reference palette, categorical slots 1-5).
METHODS = ["nlp_only", "basic_rag", "graph_rag", "threshold_rag", "evidence_rag"]
METHOD_COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
# MiniLM is basic_rag's retriever and graph is graph_rag's, so they take those colours.
RETRIEVER_COLORS = {"tfidf": "#a3a29d", "bm25": "#6250d6", "minilm": "#eb6834", "graph": "#1baf7a"}
RETRIEVER_LABELS = {"tfidf": "TF-IDF", "bm25": "BM25 (ours)", "minilm": "MiniLM (dense)", "graph": "graph (C)"}
SCENARIOS = ["answerable", "missing_evidence", "out_of_domain"]
SCENARIO_COLORS = dict(zip(SCENARIOS, ["#52514e", "#a3a29d", "#d6d5d0"]))
# evidence_rag's three actions, as shades of evidence_rag's colour.
ACTION_COLORS = {"answer": "#a83a66", "retry_answer": "#e87ba4", "abstain": "#f6d3e0"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#eef5fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": TEXT, "axes.labelsize": 12, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.edgecolor": LINE, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT, "ytick.color": TEXT, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "lines.linewidth": 2.2, "legend.frameon": False, "legend.fontsize": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def figure(title: str, subtitle: str | None = None, nrows: int = 1, ncols: int = 1, diagram: bool = False,
           wspace: float = 0.2, hspace: float = 0.35, **subplot_kw):
    """A titled 16:9 figure with its axes in the plot area; a diagram (no axis labels) starts at the title's edge."""
    fig = plt.figure(figsize=SIZE, dpi=DPI)
    grid = fig.add_gridspec(nrows, ncols, left=0.045 if diagram else LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
                            wspace=wspace, hspace=hspace)
    axes = grid.subplots(squeeze=True, **subplot_kw)
    fig.text(0.045, 0.915, title, fontsize=22, weight="bold", color=INK, va="baseline")
    if subtitle:
        fig.text(0.045, 0.862, subtitle, fontsize=13, color=TEXT, va="baseline", linespacing=1.4)
    return fig, axes


def legend(fig, handles, title: str | None = None, y: float = SIDE_TOP):
    """Legend in the right column, under the camera corner."""
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(SIDE_X - 0.008, y),
                     bbox_transform=fig.transFigure, title=title, alignment="left", handlelength=1.2,
                     handleheight=1.0, labelspacing=0.75, borderaxespad=0)
    if title:
        leg.get_title().set(color=MUTED, fontsize=11, weight="bold")
    return leg


def method_legend(fig, methods=METHODS, y: float = SIDE_TOP):
    return legend(fig, [Patch(color=METHOD_COLORS[m], label=m) for m in methods], title="METHOD", y=y)


def note(fig, text: str, y: float = BOTTOM) -> None:
    """A short takeaway at the bottom of the right column."""
    fig.text(SIDE_X, y, text, fontsize=12, color=TEXT, va="bottom", ha="left", linespacing=1.45, wrap=False)


def text_on(color: str) -> str:
    """Ink or white, whichever reads on a fill of this colour."""
    r, g, b = to_rgb(color)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def _check_corner(fig) -> None:
    renderer = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    corner = Bbox.from_extents(CORNER[0] * w + 1, CORNER[1] * h + 1, w, h)
    for artist in [*fig.texts, *fig.legends, *fig.axes]:
        box = artist.get_tightbbox(renderer) if artist.get_visible() else None
        if box is not None and box.overlaps(corner):
            raise ValueError(f"{artist!r} draws into the camera corner")


def save(fig, path: Path) -> None:
    _check_corner(fig)
    if GUIDE:
        fig.add_artist(Rectangle(CORNER, 1 - CORNER[0], 1 - CORNER[1], transform=fig.transFigure,
                                 fill=False, ls="--", lw=1.5, ec=MUTED))
        fig.text((1 + CORNER[0]) / 2, (1 + CORNER[1]) / 2, "camera", ha="center", va="center", color=MUTED)
    fig.savefig(path, dpi=DPI)
    print(f"Wrote {path}")
