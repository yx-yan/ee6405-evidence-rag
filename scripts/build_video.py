"""Render the project video from a timeline (configs/video_timeline*.yaml). Owner: E.

    python scripts/build_video.py --timeline configs/video_timeline_v2.yaml   # -> build/video/draft_v2.*
    python scripts/build_video.py ... --record     # also re-record the demo clips (otherwise cached)
    python scripts/build_video.py ... --no-burn    # subtitles only in the .srt, not in the picture
    python scripts/build_video.py ... --final      # refuse unless every block has real voice and camera

Every block of the timeline becomes one clip laid out in the screen zones of src/evrag/figures.py:
its main items (figure, code, demo, title, endcard or footage) in MAIN, the presenter's camera in
CAMERA (top right), the speaker's name tag right under it in NAME, subtitles in their band and the
block's references on one line at the bottom. Figures and slides are drawn for these zones and go in as
they are; recordings are fitted into MAIN. Blocks cross-fade into each other without changing the total
length, so a change of speaker also cross-fades the name tag. A block without `audio` gets a scratch
voice from macOS `say`, one sentence at a time, so the subtitles follow it, and a "DRAFT - scratch
voice" mark; a block with `audio` spreads its subtitles over the block by character count.

    python scripts/build_video.py ... --guide      # outline the zones on every frame (draft_v2_guide.mp4)

The course rules out generated voices, so the scratch voice is only for timing: --final stops with a
list of every block that still uses it, has no camera, or has an empty footage slot.

Demo items are recorded from the Streamlit app with Playwright, zoomed by the timeline's demo.zoom
(CSS zoom) and cached in build/video/demo/; a `zoom` step pushes in on an element afterwards
(smooth zoom in and out). An item that fails to record becomes a "DEMO: ..." placeholder card.

Needs ffmpeg (brew install ffmpeg) and, for demos, `playwright install chromium`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import yaml
from evrag import figures as F
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.styles import get_style_by_name

ROOT = Path(__file__).resolve().parents[1]
TIMELINE = ROOT / "configs" / "video_timeline.yaml"
OUT = ROOT / "build" / "video"

W, H, FPS, RATE = F.W, F.H, 30, 48000
MAIN, CAMERA, NAME = F.MAIN, F.CAMERA, F.NAME
SUBTITLES, REFERENCES = F.SUBTITLES, F.REFERENCES
INK, TEXT, MUTED, ACCENT = "#1f1e1c", "#52514e", "#8a8984", "#e87ba4"
HIGHLIGHT = "#fdf0c8"                           # background of the code lines the narration is about
VOICE, WPM = "Samantha", 165                    # scratch voice; the script is paced at ~150 words a minute
LEAD, GAP = 0.4, 0.3                            # scratch voice: silence before the first sentence, between sentences
SUB_CHARS = 90                                  # a longer sentence is split at a comma into several subtitles
# Demo recordings: the part of the 1920x1080 recording around the app's column (x 432-1488 at zoom 1.5),
# with MAIN's aspect, fitted into MAIN; a push-in zooms within it (ZOOM_RAMP seconds in and out, at most
# ZOOM_MAX), never so far that a full-width element is cut.
DEMO_CROP = (216, 30, 1704, 1030)
ZOOM_RAMP, ZOOM_MAX = 0.7, 1.6

SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO if mono else SANS, size, index=1 if bold else 0)


def seconds(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    m, s = str(t).split(":")
    return int(m) * 60 + float(s)


def stamp(t: float, sep: str = ",") -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def path_of(p: str | None) -> Path | None:
    return (ROOT / p if not Path(p).is_absolute() else Path(p)) if p else None


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=f) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def repo_url() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    return re.sub(r"\.git$", "", url)


# --- Pictures ---------------------------------------------------------------------------------

def line_ranges(spec) -> set[int]:
    """'96-97, 109' -> {96, 97, 109}."""
    out = set()
    for part in str(spec or "").replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def break_at(line: list[tuple[str, str]], room: int) -> int:
    """Where to wrap a code line that is longer than room: after the last comma in the last 40% of the
    room, else an operator there, else the last space, else anywhere."""
    for marks, start in ((",", room * 6 // 10), ("([{:=+-*/", room * 6 // 10), (" ", room // 4)):
        for i in range(min(room, len(line)) - 1, start, -1):
            if line[i][0] in marks:
                return i + 1
    return room


def render_code(file: str, lines: str, out: Path, highlight=None) -> None:
    """A 1920x1080 code shot in MAIN: file name and line range top left, then the numbered, coloured lines,
    centred vertically, in the largest font from 36 down to 30 px. A line too long for MAIN wraps after a
    comma or operator and continues indented. Lines in `highlight` get a light background."""
    a, b = (int(x) for x in lines.split("-"))
    src = (ROOT / file).read_text().splitlines()
    b = min(b, len(src))
    while b > a and not src[b - 1].strip():
        b -= 1
    style = get_style_by_name("friendly")
    colors = {}
    chars: list[list[tuple[str, str]]] = [[]]
    for ttype, value in lex("\n".join(src), get_lexer_for_filename(file)):
        if ttype not in colors:
            t = ttype
            while not style.style_for_token(t)["color"] and t.parent:
                t = t.parent
            colors[ttype] = "#" + (style.style_for_token(t)["color"] or INK.lstrip("#"))
        for ch in value:
            if ch == "\n":
                chars.append([])
            else:
                chars[-1].append((ch, colors[ttype]))
    indent = min((len(l) - len(l.lstrip()) for l in src[a - 1:b] if l.strip()), default=0)
    x0, top, right, bottom = MAIN[0], MAIN[1] + 150, MAIN[2], MAIN[3] - 10

    def layout(size: int):
        """[(line number, first row of the line, chars)] and the sizes, wrapping what does not fit."""
        cw, lh = font(size, mono=True).getlength("M"), round(size * 1.45)
        gutter = cw * (len(str(b)) + 2)
        room = int((right - x0 - gutter) / cw)
        rows = []
        for n in range(a, b + 1):
            line = chars[n - 1][indent:] if n - 1 < len(chars) else []
            lead = len(line) - len("".join(c for c, _ in line).lstrip())
            first = True
            while len(line) > room:
                cut = break_at(line, room)
                rows.append((n, first, line[:cut]))
                rest = line[cut:]
                while rest and rest[0][0] == " ":
                    rest = rest[1:]
                line, first = [(" ", INK)] * (lead + 4) + rest, False
            rows.append((n, first, line))
        return rows, cw, lh, gutter

    for size in range(36, 29, -1):                  # largest font that needs the fewest wrapped rows and fits
        rows, cw, lh, gutter = layout(size)
        if len(rows) * lh <= bottom - top and len(rows) == b - a + 1:
            break
    else:
        size = 30
        rows, cw, lh, gutter = layout(size)
        if len(rows) * lh > bottom - top:
            sys.exit(f"{file} {lines}: too long for one code shot (30 px); split the range")
    y0 = top + (bottom - top - len(rows) * lh) / 2
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((x0, MAIN[1]), file, font=font(44, bold=True), fill=INK)
    d.text((x0, MAIN[1] + 66), f"lines {a}–{b}", font=font(28), fill=TEXT)
    marked = line_ranges(highlight)
    f = font(size, mono=True)
    widest = x0 + gutter + max(len(row) for *_, row in rows) * cw + 16
    for k, (n, first, row) in enumerate(rows):
        y = y0 + k * lh
        if n in marked:
            d.rectangle((x0 + gutter - 12, y - lh * 0.16, min(widest, right), y + lh * 0.84), fill=HIGHLIGHT)
        if first:
            d.text((x0, y), f"{n:>{len(str(b))}}", font=f, fill=MUTED)
        x = x0 + gutter
        for ch, color in row:
            d.text((x, y), ch, font=f, fill=color)
            x += cw
    img.save(out)


def render_card(text: str, subtitle: str, out: Path) -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 400
    for line in wrap(d, text, font(64, bold=True), MAIN[2] - MAIN[0]):
        d.text((MAIN[0], y), line, font=font(64, bold=True), fill=INK)
        y += 82
    for line in wrap(d, subtitle or "", font(32), MAIN[2] - MAIN[0]):
        d.text((MAIN[0], y + 16), line, font=font(32), fill=TEXT)
        y += 44
    img.save(out)


def render_endcard(refs: list[str], out: Path) -> None:
    """Every reference, numbered in order of appearance, in two top-aligned columns in MAIN, under the
    title and the repository link."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((MAIN[0], MAIN[1]), "References", font=font(52, bold=True), fill=INK)
    d.text((MAIN[0], MAIN[1] + 78), repo_url(), font=font(34, bold=True), fill=ACCENT)
    d.text((MAIN[0], MAIN[1] + 124), "Code, data and a notebook that reproduces every number.", font=font(28),
           fill=TEXT)
    top, gap = MAIN[1] + 210, 40
    width = (MAIN[2] - MAIN[0] - gap) // 2
    half = (len(refs) + 1) // 2
    for size in range(30, 17, -1):
        f, lh = font(size), round(size * 1.3)
        placed, fits = [], True
        for c, part in enumerate((refs[:half], refs[half:])):
            x, y = MAIN[0] + c * (width + gap), top
            for i, ref in enumerate(part, 1 + c * half):
                lines = wrap(d, f"[{i}]  {ref}", f, width)
                placed.append((x, y, lines))
                y += lh * len(lines) + size * 0.6
            fits &= y <= MAIN[3]
        if fits:
            break
    for x, y, lines in placed:
        for k, line in enumerate(lines):
            d.text((x + (0 if k == 0 else size * 1.6), y + k * lh), line, font=f, fill=INK)
    img.save(out)


def draw_references(d: ImageDraw.ImageDraw, refs: list[str]) -> None:
    """The block's references on one line in REFERENCES, on a light translucent band."""
    x0, y0, x1, y1 = REFERENCES
    text = "References: " + " · ".join(refs)
    size = next((n for n in range(24, 15, -1) if d.textlength(text, font=font(n)) <= x1 - x0 - 24), 16)
    f = font(size)
    w = d.textlength(text, font=f)
    d.rounded_rectangle((x0 - 8, y0 - 4, x0 + w + 12, y1 + 4), radius=6, fill=(255, 255, 255, 215))
    d.text((x0 + 2, (y0 + y1) / 2), text, font=f, fill=TEXT, anchor="lm")


def draw_subtitle(d: ImageDraw.ImageDraw, text: str) -> None:
    """One line at 34 px, or two at 27 px, centred in the subtitle band."""
    x0, y0, x1, y1 = SUBTITLES
    lines, size = wrap(d, text, font(34), x1 - x0 - 40), 34
    if len(lines) > 1:
        lines, size = wrap(d, text, font(27), x1 - x0 - 40)[:2], 27
    f, lh = font(size), round(size * 1.22)
    box_w = max(d.textlength(l, font=f) for l in lines) + 40
    box_h = lh * len(lines) + 16
    top = (y0 + y1 - box_h) / 2
    d.rounded_rectangle(((W - box_w) / 2, top, (W + box_w) / 2, top + box_h), radius=10, fill=(20, 20, 20, 190))
    for i, l in enumerate(lines):
        d.text((W / 2, top + 8 + i * lh), l, font=f, fill="white", anchor="mt")


def draw_name_tag(d: ImageDraw.ImageDraw, name: str, role: str, alpha: float = 1.0) -> None:
    """The speaker's name and part, right-aligned in NAME, right under the camera, like a name plate."""
    x0, y0, x1, y1 = NAME
    a = round(255 * alpha)
    fn, fr = font(26, bold=True), font(18)
    w = max(d.textlength(name, font=fn), d.textlength(role, font=fr)) + 28
    d.rounded_rectangle((x1 - w, y0, x1, y1), radius=6, fill=(255, 255, 255, round(225 * alpha)))
    d.rectangle((x1 - 4, y0 + 8, x1, y1 - 8), fill=ACCENT[:7] + f"{a:02x}")
    d.text((x1 - 14, y0 + 5), name, font=fn, fill=(31, 30, 28, a), anchor="ra")
    d.text((x1 - 14, y1 - 6), role, font=fr, fill=(82, 81, 78, a), anchor="rd")


def draw_title(d: ImageDraw.ImageDraw, t: dict) -> None:
    """The opening title over the first seconds, on an opaque card that covers MAIN."""
    x0, y0, x1, y1 = MAIN
    d.rectangle((x0 - 20, y0 - 20, x1, y1 + 5), fill=(255, 255, 255, 255))
    d.rectangle((x0, y0 + 160, x0 + 12, y0 + 700), fill=ACCENT)
    x = x0 + 70
    d.text((x, y0 + 170), t["course"], font=font(40), fill=TEXT)
    d.text((x, y0 + 240), t["project"], font=font(120, bold=True), fill=INK)
    if t.get("subtitle"):
        d.text((x, y0 + 400), t["subtitle"], font=font(40), fill=TEXT)
    d.text((x, y0 + 500), t["group"], font=font(54, bold=True), fill=ACCENT)
    names = "  ·  ".join(t["names"])
    size = next(n for n in range(40, 20, -1) if d.textlength(names, font=font(n)) <= x1 - x - 20)
    d.text((x, y0 + 600), names, font=font(size), fill=INK)


def draw_watermark(d: ImageDraw.ImageDraw) -> None:
    f = font(20, bold=True)
    text = "DRAFT – scratch voice"
    w = d.textlength(text, font=f)
    d.rounded_rectangle((6, 5, 6 + w + 18, 33), radius=6, fill=(208, 59, 59, 200))
    d.text((15, 19), text, font=f, fill="white", anchor="lm")


def draw_guide(d: ImageDraw.ImageDraw) -> None:
    """The screen zones, outlined, for checking a layout."""
    for name, (x0, y0, x1, y1) in F.ZONES.items():
        d.rectangle((x0, y0, x1, y1), outline=(208, 59, 59, 255), width=2)
        d.text((x0 + 6, y0 + 3), name.lower(), font=font(16), fill=(208, 59, 59, 255))


def overlay_track(layers: list[tuple[float, float, object]], dur: float, stem: str) -> Path:
    """One transparent PNG per stretch of time in which the same layers are on, as an ffconcat list."""
    cuts = sorted({0.0, dur} | {min(max(t, 0.0), dur) for t0, t1, _ in layers for t in (t0, t1)})
    entries, cache = [], {}
    for a, b in zip(cuts, cuts[1:]):
        on = tuple(i for i, (t0, t1, _) in enumerate(layers) if t0 < b and t1 > a)
        if on not in cache:
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            for i in on:
                layers[i][2](d)
            cache[on] = OUT / "work" / f"{stem}_ov{len(cache):02d}.png"
            img.save(cache[on])
        entries.append((cache[on], b - a))
    listing = OUT / "work" / f"{stem}_overlay.txt"
    listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{p}'\nduration {t:.3f}\n" for p, t in entries)
                       + f"file '{entries[-1][0]}'\n")
    return listing


# --- Narration: sentences, subtitles, scratch voice -----------------------------------------

def sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.?!])\s+|(?<=[.?!]")\s+', " ".join(text.split())) if s]


def chunks(sentence: str) -> list[str]:
    """Split a long sentence at the comma (or colon, semicolon) nearest its middle, recursively."""
    if len(sentence) <= SUB_CHARS:
        return [sentence]
    cuts = [m.end() for m in re.finditer(r"[,;:]\s", sentence)]
    if not cuts:
        return [sentence]
    cut = min(cuts, key=lambda c: abs(c - len(sentence) / 2))
    return chunks(sentence[:cut].strip()) + chunks(sentence[cut:].strip())


def spread(parts: list[str], t0: float, t1: float) -> list[tuple[float, float, str]]:
    """(start, end, text) for parts sharing [t0, t1] by character count."""
    total, out, t = sum(len(p) for p in parts), [], t0
    for p in parts:
        dt = (t1 - t0) * len(p) / total
        out.append((t, t + dt, p))
        t += dt
    return out


def say(text: str, cache: Path) -> Path:
    spoken = text.replace("_", " ")
    key = hashlib.sha1(f"{VOICE}|{WPM}|{spoken}".encode()).hexdigest()[:16]
    wav = cache / f"{key}.wav"
    if not wav.exists():
        aiff = cache / f"{key}.aiff"
        subprocess.run(["say", "-v", VOICE, "-r", str(WPM), "-o", str(aiff), spoken], check=True)
        run(["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(wav)])
        aiff.unlink()
    return wav


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def write_pcm(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def silence(t: float) -> bytes:
    return b"\0\0" * round(t * RATE)


def narration_track(seg: dict, dur: float, work: Path, warnings: list[str]):
    """Write the block's audio (exactly dur seconds) and return its subtitles, relative to the block."""
    audio = work / f"{seg['id']}.wav"
    sents = sentences(seg.get("narration") or "")
    real = path_of(seg.get("audio"))
    if real:
        run(["ffmpeg", "-y", "-i", str(real), "-af", f"apad,atrim=0:{dur}", "-ac", "1", "-ar", str(RATE),
             "-c:a", "pcm_s16le", str(audio)])
        if duration(real) > dur + 0.05:
            warnings.append(f"{seg['id']}: audio is {duration(real):.1f}s, cut to {dur:.1f}s")
        subs = [c for s, e, text in spread(sents, 0, dur) for c in spread(chunks(text), s, e)] if sents else []
        return audio, subs
    cache = OUT / "voice"
    cache.mkdir(parents=True, exist_ok=True)
    pcm, subs, t = silence(LEAD), [], LEAD
    for s in sents:
        clip = read_pcm(say(s, cache))
        length = len(clip) / 2 / RATE
        subs += spread(chunks(s), t, t + length)
        pcm += clip + silence(GAP)
        t += length + GAP
    spoken = t - GAP
    if spoken > dur:
        warnings.append(f"{seg['id']}: scratch voice runs {spoken:.1f}s, slot is {dur:.1f}s "
                        f"(over by {spoken - dur:.1f}s; the end is cut)")
    n = round(dur * RATE) * 2
    write_pcm(audio, (pcm + silence(dur))[:n])
    return audio, [(s, min(e, dur), text) for s, e, text in subs if s < dur]


# --- Demo recording ---------------------------------------------------------------------------

CURSOR_JS = """
document.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M3 2l7 19 2.5-7.5L20 11z" '
              + 'fill="#1f1e1c" stroke="white" stroke-width="1.5"/></svg>';
  Object.assign(c.style, {position: 'fixed', left: '-40px', top: '-40px', zIndex: 2147483647,
                          pointerEvents: 'none', transition: 'transform 0.08s'});
  document.body.appendChild(c);
  // Under CSS zoom the cursor's own position is zoomed too, so undo it.
  const z = window.__evragZoom || 1;
  document.addEventListener('mousemove', e => { c.style.left = e.clientX / z + 'px'; c.style.top = e.clientY / z + 'px'; });
  document.addEventListener('mousedown', () => c.style.transform = 'scale(0.8)');
  document.addEventListener('mouseup', () => c.style.transform = '');
});
"""
# The app's toolbar and status widget sit in the camera corner.
# (The sidebar button lives in stToolbar too, so the toolbar itself stays.)
HIDE_CSS = ('[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],'
            ' [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {display: none !important;}'
            ' * {scroll-margin-top: 90px;}'           # scroll targets stop below the fixed header
            # Under CSS zoom the dataframe's canvas is drawn wider than the column: clip it there.
            ' [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {overflow: hidden !important;}')


class Demo:
    def __init__(self, port: int, zoom: float = 1.0):
        self.url, self.port, self.proc = f"http://localhost:{port}", port, None
        # CSS zoom on the page (a device scale factor would only record a 1280x720 corner of the frame).
        self.zoom = zoom
        self.view = {"width": W, "height": H}
        self.mouse = (W / 2, H / 2)

    def __enter__(self):
        from urllib.request import urlopen

        from playwright.sync_api import sync_playwright
        try:                                         # reuse a running app: its caches are already warm
            urlopen(self.url, timeout=2)
            print(f"using the app already running at {self.url}")
        except OSError:
            log = open(OUT / "streamlit.log", "w")
            self.proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.headless",
                                          "true", "--server.port", str(self.port), "--server.fileWatcherType",
                                          "none"], cwd=ROOT, stdout=log, stderr=log)
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch()
        return self

    def __exit__(self, *exc):
        self.browser.close()
        self.pw.stop()
        if self.proc:
            self.proc.terminate()

    def idle(self, page, timeout: float = 900) -> None:
        """Wait for the app's script run to finish; a second run started on top of it can crash torch on MPS."""
        time.sleep(0.4)
        page.wait_for_selector("[data-test-script-state=notRunning]", timeout=timeout * 1000)

    def open(self, page) -> None:
        for _ in range(60):
            try:
                page.goto(self.url)
                break
            except Exception:
                time.sleep(2)
        page.add_style_tag(content=HIDE_CSS + f" html {{zoom: {self.zoom};}}")
        page.get_by_role("button", name="Generate").wait_for(timeout=120000)
        self.idle(page)

    def locate(self, page, target: str):
        kind, _, name = target.partition(":")
        if kind == "radio":
            return page.get_by_role("radio", name=name, exact=True)
        if kind == "button":
            return page.get_by_role("button", name=name, exact=True)
        if kind == "label":
            return page.get_by_label(name)
        if kind == "testid":
            return page.locator(f'[data-testid="{name}"]').first
        if kind == "css":
            return page.locator(name).first
        if kind == "sidebar":
            return page.locator('[data-testid="stExpandSidebarButton"]')
        return page.get_by_text(name).first

    def settled_box(self, loc) -> dict:
        """The element's box once a smooth scroll has stopped moving it."""
        box = loc.bounding_box(timeout=20000)
        for _ in range(30):
            time.sleep(0.1)
            again = loc.bounding_box(timeout=20000)
            if abs(again["y"] - box["y"]) < 1 and abs(again["x"] - box["x"]) < 1:
                return again
            box = again
        return box

    def glide(self, page, loc, seconds_: float = 0.7) -> None:
        for _ in range(3):                           # aim again if the page moved during the glide
            box = self.settled_box(loc)
            x, y = box["x"] + min(box["width"] / 2, 60), box["y"] + box["height"] / 2
            (x0, y0), n = self.mouse, 24
            for i in range(1, n + 1):
                k = i / n
                k = k * k * (3 - 2 * k)
                page.mouse.move(x0 + (x - x0) * k, y0 + (y - y0) * k)
                time.sleep(seconds_ / n)
            self.mouse = (x, y)
            box = loc.bounding_box(timeout=20000)
            if box["x"] <= x <= box["x"] + box["width"] and box["y"] <= y <= box["y"] + box["height"]:
                return

    def zoom_event(self, page, s: dict) -> dict:
        """A push-in on the step's target(s): their box in video pixels, from `at` to `until`."""
        targets = s["target"] if isinstance(s["target"], list) else [s["target"]]
        boxes = [self.settled_box(self.locate(page, t)) for t in targets]
        x0, y0 = min(b["x"] for b in boxes), min(b["y"] for b in boxes)
        x1 = max(b["x"] + b["width"] for b in boxes)
        y1 = max(b["y"] + b["height"] for b in boxes)
        return {"t0": float(s["at"]), "t1": float(s["until"]), "box": [round(v, 1) for v in (x0, y0, x1, y1)],
                "scale": float(s.get("scale", ZOOM_MAX))}

    def step(self, page, s: dict, paced: bool) -> None:
        do = s["do"]
        if do == "zoom":
            return
        loc = self.locate(page, s.get("target", "text:Evidence-aware RAG"))
        if do == "scroll":
            loc.evaluate(f"e => e.scrollIntoView({{behavior: '{'smooth' if paced else 'instant'}', "
                         f"block: '{s.get('block', 'center')}'}})")
            if paced:
                self.settled_box(loc)
            return
        if paced:
            loc.scroll_into_view_if_needed(timeout=20000)
            self.glide(page, loc)
        if do == "hover":
            if not paced:
                loc.hover(timeout=20000)
        elif do == "click":
            if paced:
                page.mouse.down()
                time.sleep(0.08)
                page.mouse.up()
            else:
                loc.click(timeout=20000)
            self.idle(page)
        elif do == "type":
            loc.click(timeout=20000)
            loc.press_sequentially(s["text"], delay=90 if paced else 0)
            loc.press("Enter")
            self.idle(page)

    def warm(self, steps: list[dict]) -> None:
        """Run an item's steps once without recording, so every model and answer is cached in the app."""
        page = self.browser.new_page(viewport=self.view)
        self.open(page)
        for s in steps:
            self.step(page, s, paced=False)
        page.close()

    def record(self, steps: list[dict], dur: float, out: Path) -> None:
        """Record the steps in real time (each at its `at`), cut to dur; zoom events go to out.json."""
        raw = OUT / "demo" / "raw"
        ctx = self.browser.new_context(viewport=self.view, record_video_dir=str(raw),
                                       record_video_size={"width": W, "height": H})
        ctx.add_init_script(f"window.__evragZoom = {self.zoom};" + CURSOR_JS)
        page = ctx.new_page()
        began = time.monotonic()
        self.mouse = (W / 2, H / 2)
        zooms = []
        try:
            self.open(page)
            page.mouse.move(*self.mouse)
            t0 = time.monotonic()
            for s in steps:
                time.sleep(max(0.0, t0 + float(s["at"]) - time.monotonic()))
                if s["do"] == "zoom":
                    zooms.append(self.zoom_event(page, s))
                else:
                    self.step(page, s, paced=True)
            late = time.monotonic() - t0 - dur
            time.sleep(max(0.0, -late) + 0.5)
        finally:
            video = page.video
            ctx.close()
        if late > 0:
            print(f"  ! demo ran {late:.1f}s past its slot; the end is cut")
        run(["ffmpeg", "-y", "-ss", f"{t0 - began:.2f}", "-i", str(video.path()), "-t", f"{dur:.2f}", "-r", str(FPS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)])
        out.with_suffix(".json").write_text(json.dumps(zooms, indent=1))
        Path(video.path()).unlink()


def push_in(z: dict) -> tuple[float, float, float]:
    """(zoom, pivot x, pivot y) for one push-in, in DEMO_CROP fractions: as close as the element's
    width and height allow, centred; one still wider than the view is shown from its left edge."""
    cx0, cy0, cx1, cy1 = DEMO_CROP
    cw, ch = cx1 - cx0, cy1 - cy0
    x0, y0, x1, y1 = z["box"]
    x0, x1, y0, y1 = x0 - cx0, x1 - cx0, y0 - cy0, y1 - cy0
    bw, bh = x1 - x0, y1 - y0
    k = min(z["scale"], 0.85 * ch / bh, 0.96 * cw / bw)
    k = max(k, 1.05)
    vw, vh = cw / k, ch / k                          # the part of the crop that stays in view
    vx = x0 - 30 if bw > vw - 60 else (x0 + x1 - vw) / 2
    vx = min(max(vx, 0), cw - vw)
    vy = min(max((y0 + y1 - vh) / 2, 0), ch - vh)
    # zoompan keeps the pivot P fixed: the view's corner is P * (1 - 1/k).
    return k, vx / (1 - 1 / k) / cw, vy / (1 - 1 / k) / ch


def zoompan(zooms: list[dict]) -> str:
    """ffmpeg filter for the push-ins inside DEMO_CROP: zoom about each one's pivot, eased in and out."""
    cw, ch = DEMO_CROP[2] - DEMO_CROP[0], DEMO_CROP[3] - DEMO_CROP[1]
    t = f"(in/{FPS})"
    zoom, xs, ys = "1", [], []
    for z in zooms:
        k, px, py = push_in(z)
        p = f"clip(({t}-{z['t0']})/{ZOOM_RAMP},0,1)*clip(({z['t1']}-{t})/{ZOOM_RAMP},0,1)"
        zoom += f"+{k - 1:.4f}*(({p})*({p})*(3-2*({p})))"
        on = f"between({t},{z['t0']},{z['t1']})"
        xs.append(f"{on}*{px:.4f}*iw*(1-1/zoom)")
        ys.append(f"{on}*{py:.4f}*ih*(1-1/zoom)")
    # Upscale first so the pan moves in sub-pixel steps of the output.
    return (f"scale={2 * cw}:{2 * ch},zoompan=z='{zoom}':x='{'+'.join(xs) or 0}':y='{'+'.join(ys) or 0}'"
            f":d=1:s={cw}x{ch}:fps={FPS}")


# --- Assembly ---------------------------------------------------------------------------------

def main_items(seg: dict) -> list[dict]:
    """The block's main items with absolute start and end times."""
    items = seg["main"] if isinstance(seg["main"], list) else [seg["main"]]
    start, end = seconds(seg["start"]), seconds(seg["end"])
    out = []
    for i, item in enumerate(items):
        t0 = seconds(item["at"]) if i and "at" in item else start
        out.append({**item, "t0": t0})
    for i, item in enumerate(out):
        item["t1"] = out[i + 1]["t0"] if i + 1 < len(out) else end
        if item["t1"] <= item["t0"]:
            sys.exit(f"{seg['id']}: main item {i} has no time (at {item['t0']}s)")
    return out


def picture(seg: dict, i: int, item: dict, ctx: dict, placeholders: list[str]) -> tuple[Path, str, list]:
    """(file, kind, zoom events) for one main item; kind is picture, demo or footage."""
    frames = OUT / "frames"
    kind = item["type"]
    if kind == "figure":
        return path_of(item["path"]), "picture", []
    if kind == "footage":
        return path_of(item["path"]), "footage", []
    out = frames / f"{seg['id']}_{i}.png"
    if kind == "code":
        render_code(item["file"], str(item["lines"]), out, item.get("highlight"))
    elif kind == "title":
        render_card(item["text"], item.get("subtitle", ""), out)
    elif kind == "endcard":
        render_endcard(ctx["references"], out)
    elif kind == "demo":
        clip = ctx["demo_clips"].get((seg["id"], i))
        if clip and clip.exists():
            meta = clip.with_suffix(".json")
            return clip, "demo", json.loads(meta.read_text()) if meta.exists() else []
        placeholders.append(f"{seg['id']} ({item['label']})")
        render_card(f"DEMO: {item['label']}", "placeholder: the recording failed or is missing", out)
    else:
        sys.exit(f"{seg['id']}: unknown main type {kind!r}")
    return out, "picture", []


def render_block(seg: dict, items: list[tuple], overlay: Path, camera: Path | None, dur: float, out: Path) -> None:
    """items: (file, kind, seconds, zoom events). Pictures are drawn for the zones and go in as they are;
    a demo recording is cut to DEMO_CROP, pushed in, and fitted into MAIN; other footage is fitted into MAIN."""
    args, chains, labels = ["ffmpeg", "-y"], [], []
    mw, mh = MAIN[2] - MAIN[0], MAIN[3] - MAIN[1]
    into_main = (f"scale={mw}:{mh}:force_original_aspect_ratio=decrease,"
                 f"pad={W}:{H}:{MAIN[0]}+({mw}-iw)/2:{MAIN[1]}:white,setsar=1")
    full = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1"
    for k, (path, kind, d, zooms) in enumerate(items):
        hold = f"tpad=stop_mode=clone:stop_duration={d},trim=duration={d},setpts=PTS-STARTPTS"
        if kind == "picture":
            args += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d}", "-i", str(path)]
            chains.append(f"[{k}:v]{full},trim=duration={d},setpts=PTS-STARTPTS[m{k}]")
        elif kind == "demo":
            x0, y0, x1, y1 = DEMO_CROP
            push = f",{zoompan(zooms)}" if zooms else ""
            args += ["-i", str(path)]
            chains.append(f"[{k}:v]fps={FPS},{full},crop={x1 - x0}:{y1 - y0}:{x0}:{y0}{push},{hold},{into_main}[m{k}]")
        else:
            args += ["-i", str(path)]
            chains.append(f"[{k}:v]fps={FPS},{hold},{into_main}[m{k}]")
        labels.append(f"[m{k}]")
    n = len(items)
    chains.append(f"{''.join(labels)}concat=n={n}:v=1:a=0,format=yuv420p[v0]")
    last, k = "v0", n
    if camera:
        cx0, cy0, cx1, cy1 = CAMERA
        cw, ch = cx1 - cx0, cy1 - cy0
        args += ["-i", str(camera)]
        chains.append(f"[{k}:v]fps={FPS},scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch},"
                      f"tpad=stop_mode=clone:stop_duration={dur},trim=duration={dur},setpts=PTS-STARTPTS[cam]")
        chains.append(f"[{last}][cam]overlay={cx0}:{cy0}[v{k}]")
        last, k = f"v{k}", k + 1
    args += ["-f", "concat", "-safe", "0", "-i", str(overlay)]
    chains.append(f"[{k}:v]fps={FPS},format=rgba,trim=duration={dur},setpts=PTS-STARTPTS[ov]")
    chains.append(f"[{last}][ov]overlay=0:0:eof_action=pass:format=auto[out]")
    script = OUT / "work" / f"{seg['id']}.filter"
    script.write_text(";\n".join(chains))
    args += ["-/filter_complex", str(script), "-map", "[out]", "-t", f"{dur}", "-r", str(FPS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(out)]
    try:
        run(args)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed on {seg['id']}:\n{e.stderr.decode()[-2000:]}")


def join(clips: list[Path], starts: list[float], fade: float, audio: Path, out: Path) -> None:
    """Concatenate the block clips, cross-fading at every boundary, and add the narration track. Each clip
    but the last runs `fade` seconds past its block, so the total length does not change."""
    if fade <= 0:
        listing = OUT / "work" / "clips.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio), "-map", "0:v",
             "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
        return
    args, chain, last = ["ffmpeg", "-y"], [], "0:v"
    for c in clips:
        args += ["-i", str(c)]
    for k in range(1, len(clips)):
        chain.append(f"[{last}][{k}:v]xfade=transition=fade:duration={fade}:offset={starts[k] - starts[0]:.3f}[x{k}]")
        last = f"x{k}"
    script = OUT / "work" / "join.filter"
    script.write_text(";\n".join(chain))
    args += ["-i", str(audio), "-/filter_complex", str(script), "-map", f"[{last}]", "-map", f"{len(clips)}:a",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    try:
        run(args)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed joining the clips:\n{e.stderr.decode()[-2000:]}")


def contact_sheet(segs: list[dict], shots: dict[str, list[float]], video: Path, out: Path) -> None:
    """One row per block: up to three frames (one per main item, or mid-zoom), camera box outlined in red."""
    tw, th, cols, label, gap = 560, 315, 3, 34, 12
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, len(segs) * (th + label + gap) + gap), "#e9e8e4")
    d = ImageDraw.Draw(sheet)
    tmp, origin = OUT / "work" / "frame.png", seconds(segs[0]["start"])
    for r, seg in enumerate(segs):
        y0 = gap + r * (th + label + gap)
        d.text((gap, y0 + 4), f"{seg['id']}  {seg['start']}–{seg['end']}", font=font(22, bold=True), fill=INK)
        for c, t in enumerate(shots[seg["id"]][:cols]):
            run(["ffmpeg", "-y", "-ss", f"{t - origin:.2f}", "-i", str(video), "-frames:v", "1", str(tmp)])
            frame = Image.open(tmp).convert("RGB")
            ImageDraw.Draw(frame).rectangle(CAMERA, outline="#d03b3b", width=5)
            x0 = gap + c * (tw + gap)
            sheet.paste(frame.resize((tw, th)), (x0, y0 + label))
            d.text((x0 + tw - 70, y0 + 8), stamp(t, ".")[3:8], font=font(18), fill=MUTED)
    sheet.save(out)


def layout_check(frames: list[tuple[str, float]], video: Path, out: Path) -> None:
    """One full-size frame per kind of picture, with the zones outlined, three to a row."""
    tmp, imgs = OUT / "work" / "frame.png", []
    for label, t in frames:
        run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(tmp)])
        img = Image.open(tmp).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        draw_guide(d)
        d.text((W - 20, H - 6), f"{label}  {stamp(t, '.')[3:10]}", font=font(28, bold=True), fill=(208, 59, 59, 255),
               anchor="rd")
        imgs.append(Image.alpha_composite(img, layer).convert("RGB"))
    cols = 3
    sheet = Image.new("RGB", (cols * W, -(-len(imgs) // cols) * H), "#e9e8e4")
    for i, img in enumerate(imgs):
        sheet.paste(img, ((i % cols) * W, (i // cols) * H))
    sheet.save(out)


def final_problems(segs: list[dict]) -> list[str]:
    """What still stops a final render: scratch voice, a missing camera, an empty footage slot."""
    out = []
    for seg in segs:
        why = []
        if not seg.get("audio"):
            why.append("scratch voice (audio is empty)")
        elif not path_of(seg["audio"]).exists():
            why.append(f"audio file missing: {seg['audio']}")
        if not seg.get("camera"):
            why.append("camera is empty")
        elif not path_of(seg["camera"]).exists():
            why.append(f"camera file missing: {seg['camera']}")
        for item in main_items(seg):
            if item["type"] == "footage" and not (item.get("path") and path_of(item["path"]).exists()):
                why.append(f"footage slot at {item['t0']:.0f}s is empty")
        if why:
            out.append(f"{seg['id']} ({seg['start']}–{seg['end']}): " + "; ".join(why))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", type=Path, default=TIMELINE)
    ap.add_argument("--record", action="store_true", help="re-record every demo clip")
    ap.add_argument("--no-burn", action="store_true", help="do not burn the subtitles into the picture")
    ap.add_argument("--final", action="store_true", help="stop unless every block has real voice and camera")
    ap.add_argument("--guide", action="store_true", help="outline the screen zones on every frame")
    ap.add_argument("--port", type=int, default=8599)
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found: brew install ffmpeg")
    cfg = yaml.safe_load(args.timeline.read_text())
    segs, speakers = cfg["segments"], cfg["speakers"]
    name = cfg.get("name", "draft")
    if args.final:
        problems = final_problems(segs)
        if problems:
            print("--final: not rendering. The course does not allow generated voices, and these blocks are "
                  "not ready:\n  " + "\n  ".join(problems), file=sys.stderr)
            sys.exit(1)
        name = name.replace("draft", "final") if "draft" in name else f"{name}_final"
    if args.guide:
        name += "_guide"
    fade = float(cfg.get("crossfade", 0))
    zoom = float(cfg.get("demo", {}).get("zoom", 1.0))
    for d in ("frames", "work", "demo", "clips"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for a, b in zip(segs, segs[1:]):
        if seconds(a["end"]) != seconds(b["start"]):
            sys.exit(f"gap or overlap between {a['id']} and {b['id']}")

    # Demo clips: warm up every item first, then record the ones not cached.
    demo_clips, todo = {}, []
    for seg in segs:
        for i, item in enumerate(main_items(seg)):
            if item["type"] == "demo":
                clip = OUT / "demo" / f"{seg['id']}_{i}_x{zoom:g}.mp4"
                demo_clips[(seg["id"], i)] = clip
                if args.record or not clip.exists():
                    todo.append((seg["id"], item, clip))
    if todo:
        with Demo(args.port, zoom) as demo:
            print(f"warming up {len(todo)} demo item(s) ...")
            for sid, item, _ in todo:
                try:
                    demo.warm(item["steps"])
                except Exception as e:
                    print(f"  ! warm-up of {sid} failed: {str(e).splitlines()[0]}")
            for sid, item, clip in todo:
                print(f"recording {sid} ({item['t1'] - item['t0']:.0f}s) ...")
                try:
                    demo.record(item["steps"], item["t1"] - item["t0"], clip)
                except Exception as e:
                    print(f"  ! recording {sid} failed: {str(e).splitlines()[0]}")
                    clip.unlink(missing_ok=True)

    refs = list(dict.fromkeys(r for seg in segs for r in seg.get("references") or []))
    ctx = {"demo_clips": demo_clips, "references": refs}
    warnings, placeholders, srt, clips, starts, audio = [], [], [], [], [], b""
    shots, checks = {}, {}
    for n, seg in enumerate(segs):
        start, end = seconds(seg["start"]), seconds(seg["end"])
        dur = end - start
        ext = fade if n + 1 < len(segs) else 0.0     # runs on under the next block's fade-in
        print(f"{seg['id']}: {seg['start']}–{seg['end']}")
        items, shots[seg["id"]] = [], []
        for i, item in enumerate(main_items(seg)):
            path, kind, zooms = picture(seg, i, item, ctx, placeholders)
            if not path.exists():
                sys.exit(f"{seg['id']}: missing {path}")
            d = item["t1"] - item["t0"] + (ext if item["t1"] == end else 0)
            items.append((path, kind, d, zooms))
            mid = [item["t0"] + (z["t0"] + z["t1"]) / 2 for z in zooms]
            shots[seg["id"]] += mid[:1] or [(item["t0"] + item["t1"]) / 2]
            label = ("theory slide" if "theory_" in str(item.get("path")) else "diagram"
                     if "pipeline" in str(item.get("path")) else "result figure"
                     if item["type"] == "figure" else item["type"])
            checks.setdefault(label, shots[seg["id"]][-1])
        if len(shots[seg["id"]]) == 1:              # one long item: show its start and end too
            shots[seg["id"]] = [start + dur * f for f in (0.2, 0.5, 0.85)]
        track, subs = narration_track(seg, dur, OUT / "work", warnings)
        audio += read_pcm(track)
        srt += [(start + s, start + e, text) for s, e, text in subs]

        layers = []
        if seg.get("references"):
            layers.append((0, dur + ext, lambda d, r=seg["references"]: draw_references(d, r)))
        # The name tag stays for the whole block; the cross-fade between blocks fades it on a change of
        # speaker. The first one fades in once the opening title is gone.
        who = speakers[seg["speaker"]]
        shown = float(cfg["title"].get("seconds", 5)) if n == 0 and cfg.get("title") else 0.0
        if shown:
            layers.append((0, shown, lambda d, t=cfg["title"]: draw_title(d, t)))
            checks["title card"] = start + shown / 2
            for j in range(4):
                layers.append((shown + j * 0.1, shown + (j + 1) * 0.1,
                               lambda d, w=who, a=(j + 1) / 5: draw_name_tag(d, w["name"], w["role"], a)))
            shown += 0.4
        layers.append((shown, dur + ext, lambda d, w=who: draw_name_tag(d, w["name"], w["role"])))
        if not seg.get("audio"):
            layers.append((0, dur + ext, draw_watermark))
        if not args.no_burn:
            layers += [(s, e - 0.04, lambda d, t=text: draw_subtitle(d, t)) for s, e, text in subs]
        if args.guide:
            layers.append((0, dur + ext, draw_guide))
        clip = OUT / "clips" / f"{seg['id']}.mp4"
        render_block(seg, items, overlay_track(layers, dur + ext, seg["id"]), path_of(seg.get("camera")),
                     dur + ext, clip)
        clips.append(clip)
        starts.append(start)

    write_pcm(OUT / "work" / "audio.wav", audio)
    video = OUT / f"{name}.mp4"
    print("joining ...")
    join(clips, starts, fade, OUT / "work" / "audio.wav", video)
    (OUT / f"{name}.srt").write_text("\n".join(f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}\n"
                                               for i, (s, e, text) in enumerate(srt, 1)))
    sheet = OUT / name.replace("draft", "contact_sheet").replace("final", "contact_sheet")
    contact_sheet(segs, shots, video, sheet.with_suffix(".png"))
    origin = seconds(segs[0]["start"])
    order = ["title card", "demo", "theory slide", "result figure", "code", "endcard"]
    layout_check([(k, checks[k] - origin) for k in order if k in checks], video, OUT / "layout_check.png")

    print(f"\n{video.relative_to(ROOT)}: {duration(video):.2f}s "
          f"(timeline {seconds(segs[-1]['end']) - seconds(segs[0]['start']):.0f}s)")
    print("placeholders:", ", ".join(placeholders) or "none")
    print("warnings:" if warnings else "warnings: none")
    for w in warnings:
        print("  " + w)
    (OUT / f"{name}_report.json").write_text(json.dumps({"placeholders": placeholders, "warnings": warnings},
                                                        indent=2))


if __name__ == "__main__":
    main()
