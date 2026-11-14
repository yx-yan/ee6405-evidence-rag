"""One look for every figure (notebook and video): a 1920x1080 (16:9) white canvas divided into the
same zones the video uses (pixels, origin top left):

    MAIN        x   60-1400, y   40-940    title, subtitle and the plot (or formulas, code, demo)
    CAMERA      x 1421-1920, y    0-432    left empty: the presenter's camera goes here
    NAME        x 1421-1880, y  440-500    the speaker's name tag, under the camera (video only)
    SIDE        x 1421-1880, y  510-940    legend at the top, note at the bottom
    SUBTITLES   x  140-1780, y  950-1030   video only
    REFERENCES  x   60-1880, y 1040-1070   video only

`save` refuses a figure that draws outside MAIN and SIDE.

    fig, ax = figures.figure("Title", "subtitle")
    ...
    figures.legend(fig, handles)
    figures.save(fig, path)

Set GUIDE = True (or EVRAG_FIGURE_GUIDE=1) to outline the zones while checking a layout.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle

SIZE, DPI = (12.8, 7.2), 150                    # 1920 x 1080 px
W, H = 1920, 1080
ZONES = {                                       # (x0, y0, x1, y1) in pixels, origin top left
    "MAIN": (60, 40, 1400, 940),
    "CAMERA": (1421, 0, 1920, 432),
    "NAME": (1421, 440, 1880, 500),
    "SIDE": (1421, 510, 1880, 940),
    "SUBTITLES": (140, 950, 1780, 1030),
    "REFERENCES": (60, 1040, 1880, 1070),
}
MAIN, CAMERA, NAME, SIDE = ZONES["MAIN"], ZONES["CAMERA"], ZONES["NAME"], ZONES["SIDE"]
SUBTITLES, REFERENCES = ZONES["SUBTITLES"], ZONES["REFERENCES"]


def fx(x: float) -> float:
    """Pixel column -> figure fraction."""
    return x / W


def fy(y: float) -> float:
    """Pixel row (from the top) -> figure fraction (from the bottom)."""
    return 1 - y / H


# The plot area inside MAIN: room on the left for tick labels, under the title for the subtitle, and at
# the bottom for tick labels and the axis label, so nothing reaches the subtitle band.
LEFT, RIGHT, TOP, BOTTOM = fx(MAIN[0] + 215), fx(MAIN[2] - 35), fy(MAIN[1] + 140), fy(MAIN[3] - 95)
SIDE_X, SIDE_TOP, SIDE_BOTTOM = fx(SIDE[0]), fy(SIDE[1]), fy(SIDE[3])
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
           wspace: float = 0.2, hspace: float = 0.35, top: float = TOP, **subplot_kw):
    """A titled 16:9 figure with its axes filling MAIN under the title; a diagram (no axis labels) uses all
    of MAIN's width and height."""
    fig = plt.figure(figsize=SIZE, dpi=DPI)
    grid = fig.add_gridspec(nrows, ncols, left=fx(MAIN[0] + 15) if diagram else LEFT, right=RIGHT,
                            bottom=fy(MAIN[3]) if diagram else BOTTOM, top=top, wspace=wspace, hspace=hspace)
    axes = grid.subplots(squeeze=True, **subplot_kw)
    fig.text(fx(MAIN[0]), fy(MAIN[1]), title, fontsize=22, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(fx(MAIN[0]), fy(MAIN[1] + 62), subtitle, fontsize=13, color=TEXT, va="top", linespacing=1.4)
    return fig, axes


def legend(fig, handles, title: str | None = None, y: float = SIDE_TOP):
    """Legend at the top of the right column, under the camera and the name tag."""
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(SIDE_X, y),
                     bbox_transform=fig.transFigure, title=title, alignment="left", handlelength=1.2,
                     handleheight=1.0, labelspacing=0.75, borderaxespad=0, borderpad=0)
    if title:
        leg.get_title().set(color=MUTED, fontsize=11, weight="bold")
    return leg


def method_legend(fig, methods=METHODS, y: float = SIDE_TOP):
    return legend(fig, [Patch(color=METHOD_COLORS[m], label=m) for m in methods], title="METHOD", y=y)


def note(fig, text: str, y: float = SIDE_BOTTOM) -> None:
    """A short takeaway at the bottom of the right column."""
    fig.text(SIDE_X, y, text, fontsize=12, color=TEXT, va="bottom", ha="left", linespacing=1.45, wrap=False)


def text_on(color: str) -> str:
    """Ink or white, whichever reads on a fill of this colour."""
    r, g, b = to_rgb(color)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def _check_zones(fig) -> None:
    """Every text, legend and axes (with its labels) must lie inside MAIN or SIDE."""
    renderer = fig.canvas.get_renderer()
    scale = W / fig.bbox.width
    for artist in [*fig.texts, *fig.legends, *fig.axes]:
        box = artist.get_tightbbox(renderer) if artist.get_visible() else None
        if box is None or not artist.get_visible():
            continue
        x0, x1 = box.x0 * scale, box.x1 * scale
        y0, y1 = H - box.y1 * scale, H - box.y0 * scale
        if not any(x0 >= z[0] - 2 and x1 <= z[2] + 2 and y0 >= z[1] - 2 and y1 <= z[3] + 2 for z in (MAIN, SIDE)):
            raise ValueError(f"{artist!r} at x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f} is outside MAIN and SIDE")


def draw_guide(fig) -> None:
    for name, (x0, y0, x1, y1) in ZONES.items():
        fig.add_artist(Rectangle((fx(x0), fy(y1)), fx(x1 - x0), (y1 - y0) / H, transform=fig.transFigure,
                                 fill=False, ls="--", lw=1.2, ec="#d03b3b"))
        fig.text(fx(x0 + 6), fy(y0 + 4), name.lower(), fontsize=8, color="#d03b3b", va="top")


def save(fig, path: Path) -> None:
    _check_zones(fig)
    if GUIDE:
        draw_guide(fig)
    fig.savefig(path, dpi=DPI)
    print(f"Wrote {path}")
