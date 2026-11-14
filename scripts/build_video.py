"""Render the project video from a timeline (configs/video_timeline*.yaml). Owner: E.

    python scripts/build_video.py --timeline configs/video_timeline_v2.yaml   # -> build/video/draft_v2.*
    python scripts/build_video.py ... --record     # also re-record the demo clips (otherwise cached)
    python scripts/build_video.py ... --no-burn    # subtitles only in the .srt, not in the picture
    python scripts/build_video.py ... --final      # refuse unless every block has real voice and camera

Every block of the timeline becomes one clip: its main items (figure, code, demo, title, endcard or
footage) scaled into the area above the subtitle band, the presenter's camera in the top-right corner
(right 26% x top 40%, which every generated picture leaves empty), references in the bottom-left,
subtitles above them, and a lower-third whenever the speaker changes. Blocks cross-fade into each other
without changing the total length. A block without `audio` gets a scratch voice from macOS `say`, one
sentence at a time, so the subtitles follow it, and a "DRAFT - scratch voice" mark; a block with
`audio` spreads its subtitles over the block by character count.

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
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.styles import get_style_by_name

ROOT = Path(__file__).resolve().parents[1]
TIMELINE = ROOT / "configs" / "video_timeline.yaml"
OUT = ROOT / "build" / "video"

W, H, FPS, RATE = 1920, 1080, 30, 48000
CAM = (1421, 0, 499, 432)                       # x, y, w, h: right 26% x top 40%
MAIN_SCALE, MAIN_X = 0.85, 40                   # main picture: 1632x918 at (40, 0); the band below holds subtitles
MAIN_W, MAIN_H = round(W * MAIN_SCALE), round(H * MAIN_SCALE)
VIDEO_X = round(W / 2 * (1 - MAIN_SCALE))       # recordings stay centred, so the app's column lands mid-frame
INK, TEXT, MUTED, ACCENT = "#1f1e1c", "#52514e", "#8a8984", "#e87ba4"
HIGHLIGHT = "#fdf0c8"                           # background of the code lines the narration is about
VOICE, WPM = "Samantha", 165                    # scratch voice; the script is paced at ~150 words a minute
LEAD, GAP = 0.4, 0.3                            # scratch voice: silence before the first sentence, between sentences
SUB_CHARS = 90                                  # a longer sentence is split at a comma into several subtitles
ZOOM_RAMP, ZOOM_MAX = 0.7, 1.8                  # push-in: seconds to zoom in (and out), largest zoom
ZOOM_AT = (800, 600)                            # where a zoomed element's centre lands, clear of the camera

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

def free_x(y: float) -> int:
    """Rightmost x a picture may use at height y: clear of the camera, also after scaling into the main area."""
    if y < CAM[3]:
        return CAM[0] - 20
    if y < CAM[3] / MAIN_SCALE:
        return round((CAM[0] - MAIN_X) / MAIN_SCALE) - 20
    return W - 60


def line_ranges(spec) -> set[int]:
    """'96-97, 109' -> {96, 97, 109}."""
    out = set()
    for part in str(spec or "").replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def break_at(line: list[tuple[str, str]], room: int) -> int:
    """Where to wrap a code line that is longer than room: after the last space or punctuation that fits."""
    for i in range(min(room, len(line)) - 1, room // 3, -1):
        if line[i][0] in " ,([{:=+-*/":
            return i + 1
    return room


def render_code(file: str, lines: str, out: Path, highlight=None) -> None:
    """A 1920x1080 code shot: file name and line range as the title, numbered and coloured code, centred
    in the space under the title. The font is the largest at which no line wraps; when even the smallest
    needs it, long lines wrap at a space or punctuation and continue indented. Lines in `highlight` get a
    light background."""
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
    head, bottom, x0 = 190, H - 50, 86

    def layout(size: int, top: float, wrapping: bool):
        """[(line number, first row of the line, chars, y)] and the bottom, or None if a line does not fit."""
        cw, lh = font(size, mono=True).getlength("M"), round(size * 1.4)
        gutter = cw * 6
        rows, y = [], top
        for n in range(a, b + 1):
            line = chars[n - 1][indent:] if n - 1 < len(chars) else []
            lead = len(line) - len("".join(c for c, _ in line).lstrip())
            first = True
            while True:
                room = int((free_x(y) - x0 - gutter) / cw)
                if len(line) <= room:
                    rows.append((n, first, line, y))
                    y += lh
                    break
                if not wrapping:
                    return None
                cut = break_at(line, room)
                rows.append((n, first, line[:cut], y))
                rest = line[cut:]
                while rest and rest[0][0] == " ":
                    rest = rest[1:]
                line, first, y = [(" ", INK)] * (lead + 4) + rest, False, y + lh
        return (rows, y, cw, lh, gutter) if y <= bottom else None

    def fit(top: float):
        for wrapping, sizes in ((False, range(32, 19, -1)), (True, range(24, 15, -1))):
            for size in sizes:
                done = layout(size, top, wrapping)
                if done:
                    return size, wrapping, done
        sys.exit(f"{file} {lines}: too long for one code shot; split the range")

    size, wrapping, (rows, y_end, *_) = fit(head)
    top = head + (bottom - y_end) / 2                # centre vertically; keep the top layout if centring breaks it
    done = layout(size, top, wrapping) or layout(size, head, wrapping)
    rows, _, cw, lh, gutter = done
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 52), file, font=font(44, bold=True), fill=INK)
    d.text((86, 118), f"lines {a}–{b}", font=font(28), fill=TEXT)
    marked = line_ranges(highlight)
    f = font(size, mono=True)
    widest = x0 + gutter + max(len(row) for _, _, row, _ in rows) * cw + 16
    for n, first, row, y in rows:
        if n in marked:
            d.rectangle((x0 + gutter - 10, y - lh * 0.18, min(widest, free_x(y) + 10), y + lh * 0.82),
                        fill=HIGHLIGHT)
        if first:
            d.text((x0, y), f"{n:>4}", font=f, fill=MUTED)
        x = x0 + gutter
        for ch, color in row:
            d.text((x, y), ch, font=f, fill=color)
            x += cw
    img.save(out)


def render_card(text: str, subtitle: str, out: Path) -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 430
    for line in wrap(d, text, font(64, bold=True), 1300):
        d.text((86, y), line, font=font(64, bold=True), fill=INK)
        y += 82
    for line in wrap(d, subtitle or "", font(32), 1300):
        d.text((86, y + 16), line, font=font(32), fill=TEXT)
        y += 44
    img.save(out)


def render_endcard(refs: list[str], out: Path) -> None:
    """Every reference, numbered in order of appearance, in two columns, and the repository link."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((86, 52), "References", font=font(52, bold=True), fill=INK)
    d.text((86, 128), "Code, data and a notebook that reproduces every number:", font=font(30), fill=TEXT)
    d.text((86, 172), repo_url(), font=font(34, bold=True), fill=ACCENT)
    # Left column from under the link; right column starts below the camera corner. Half the list each.
    cols = [(86, 250, 920), (1000, round(CAM[3] / MAIN_SCALE) + 20, W - 60)]
    half = (len(refs) + 1) // 2
    for size in range(30, 17, -1):
        f, lh = font(size), round(size * 1.3)
        placed, fits = [], True
        for c, part in enumerate((refs[:half], refs[half:])):
            x, y, right = cols[c]
            for i, ref in enumerate(part, 1 + c * half):
                lines = wrap(d, f"[{i}]  {ref}", f, right - x)
                placed.append((x, y, lines))
                y += lh * len(lines) + size * 0.55
            fits &= y <= H - 70
        if fits:
            break
    for x, y, lines in placed:
        for k, line in enumerate(lines):
            d.text((x + (0 if k == 0 else size * 1.6), y + k * lh), line, font=f, fill=INK)
    img.save(out)


def draw_references(d: ImageDraw.ImageDraw, refs: list[str]) -> None:
    text = "References: " + " · ".join(refs)
    for size in range(26, 19, -1):
        f = font(size)
        if d.textlength(text, font=f) <= W - 100:
            break
    lines = wrap(d, text, f, W - 100)[:2]
    lh = round(size * 1.3)
    top = H - 12 - lh * len(lines) - 8
    w = max(d.textlength(l, font=f) for l in lines)
    d.rounded_rectangle((28, top - 6, 28 + w + 24, H - 10), radius=8, fill=(255, 255, 255, 215))
    for i, line in enumerate(lines):
        d.text((40, top + i * lh), line, font=f, fill=TEXT)


def draw_subtitle(d: ImageDraw.ImageDraw, text: str) -> None:
    f = font(34)
    lines = wrap(d, text, f, 1640)[:2]
    lh, pad = 44, 12
    box_w = max(d.textlength(l, font=f) for l in lines) + 2 * pad + 12
    bottom = H - 64
    top = bottom - lh * len(lines) - 2 * pad
    x = (W - box_w) / 2
    d.rounded_rectangle((x, top, x + box_w, bottom), radius=10, fill=(20, 20, 20, 190))
    for i, l in enumerate(lines):
        d.text(((W - d.textlength(l, font=f)) / 2, top + pad + i * lh), l, font=f, fill="white")


def draw_lower_third(d: ImageDraw.ImageDraw, name: str, role: str) -> None:
    x, y = 60, MAIN_H - 150
    w = max(d.textlength(name, font=font(40, bold=True)), d.textlength(role, font=font(26))) + 60
    d.rectangle((x, y, x + w, y + 110), fill=(255, 255, 255, 235))
    d.rectangle((x, y, x + 8, y + 110), fill=ACCENT)
    d.text((x + 30, y + 14), name, font=font(40, bold=True), fill=INK)
    d.text((x + 30, y + 66), role, font=font(26), fill=TEXT)


def draw_title(d: ImageDraw.ImageDraw, t: dict) -> None:
    """The opening title over the first seconds: project, course, group and members."""
    x, y, w, h = 60, 110, 1330, 600                 # left of the camera corner, over the app's own title
    d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(255, 255, 255, 250), outline=(230, 229, 225, 255))
    d.rectangle((x, y + 40, x + 10, y + h - 40), fill=ACCENT)
    d.text((x + 60, y + 60), t["course"], font=font(34), fill=TEXT)
    d.text((x + 60, y + 120), t["project"], font=font(96, bold=True), fill=INK)
    if t.get("subtitle"):
        d.text((x + 60, y + 265), t["subtitle"], font=font(36), fill=TEXT)
    d.text((x + 60, y + 350), t["group"], font=font(44, bold=True), fill=ACCENT)
    names = "  ·  ".join(t["names"])
    size = next(n for n in range(36, 20, -1) if d.textlength(names, font=font(n)) <= w - 100)
    d.text((x + 60, y + 430), names, font=font(size), fill=INK)


def draw_watermark(d: ImageDraw.ImageDraw) -> None:
    f = font(22, bold=True)
    text = "DRAFT – scratch voice"
    w = d.textlength(text, font=f)
    d.rounded_rectangle((8, 8, 8 + w + 20, 42), radius=6, fill=(208, 59, 59, 200))
    d.text((18, 12), text, font=f, fill="white")


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
        """A push-in on the step's target(s), in video pixels: where to zoom and how far."""
        targets = s["target"] if isinstance(s["target"], list) else [s["target"]]
        boxes = [self.settled_box(self.locate(page, t)) for t in targets]
        x0, y0 = min(b["x"] for b in boxes), min(b["y"] for b in boxes)
        x1 = max(b["x"] + b["width"] for b in boxes)
        y1 = max(b["y"] + b["height"] for b in boxes)
        col = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0     # video pixels
        col_right = col["x"] + col["width"] if col else W
        px, py = ZOOM_AT
        # Fill at most ~1350x640 px, and keep the app's column clear of the camera corner.
        z = min(float(s.get("scale", ZOOM_MAX)), 1350 / bw, 640 / bh,
                (round((CAM[0] - VIDEO_X) / MAIN_SCALE) - 15 - px) / max(1.0, col_right - cx))
        z = max(z, 1.05)
        # Zoom about a fixed pivot, chosen so the element's centre ends up at ZOOM_AT.
        pivot_x = min(max((cx - px / z) / (1 - 1 / z), 0), W)
        pivot_y = min(max((cy - py / z) / (1 - 1 / z), 0), H)
        return {"t0": float(s["at"]), "t1": float(s["until"]), "z": round(z, 3),
                "px": round(pivot_x / W, 4), "py": round(pivot_y / H, 4)}

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


def zoompan(zooms: list[dict]) -> str:
    """ffmpeg filter for the push-ins: zoom about each event's pivot, eased in and out over ZOOM_RAMP."""
    t = f"(in/{FPS})"
    ease = []
    for z in zooms:
        p = f"clip(({t}-{z['t0']})/{ZOOM_RAMP},0,1)*clip(({z['t1']}-{t})/{ZOOM_RAMP},0,1)"
        ease.append(f"(({p})*({p})*(3-2*({p})))")
    zoom = "1" + "".join(f"+{z['z'] - 1:.4f}*{e}" for z, e in zip(zooms, ease))
    on = [f"between({t},{z['t0']},{z['t1']})" for z in zooms]
    x = "+".join(f"{o}*{z['px']}*iw*(1-1/zoom)" for o, z in zip(on, zooms))
    y = "+".join(f"{o}*{z['py']}*ih*(1-1/zoom)" for o, z in zip(on, zooms))
    # Upscale first so the pan moves in sub-pixel steps of the output.
    return f"scale={2 * W}:{2 * H},zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={FPS}"


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


def picture(seg: dict, i: int, item: dict, ctx: dict, placeholders: list[str]) -> tuple[Path, bool, list]:
    """(file, is_video, zoom events) for one main item."""
    frames = OUT / "frames"
    kind = item["type"]
    if kind == "figure":
        return path_of(item["path"]), False, []
    if kind == "footage":
        return path_of(item["path"]), True, []
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
            return clip, True, json.loads(meta.read_text()) if meta.exists() else []
        placeholders.append(f"{seg['id']} ({item['label']})")
        render_card(f"DEMO: {item['label']}", "placeholder: the recording failed or is missing", out)
    else:
        sys.exit(f"{seg['id']}: unknown main type {kind!r}")
    return out, False, []


def render_block(seg: dict, items: list[tuple], overlay: Path, camera: Path | None, dur: float, out: Path) -> None:
    """items: (file, is_video, seconds, zoom events). Recordings sit centred, pictures at MAIN_X."""
    args, chains, labels = ["ffmpeg", "-y"], [], []
    fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1"
    for k, (path, is_video, d, zooms) in enumerate(items):
        place = f"scale={MAIN_W}:{MAIN_H},pad={W}:{H}:{VIDEO_X if is_video else MAIN_X}:0:white,setsar=1"
        if is_video:
            args += ["-i", str(path)]
            push = f"{zoompan(zooms)}," if zooms else ""
            chains.append(f"[{k}:v]fps={FPS},{fit},{push}tpad=stop_mode=clone:stop_duration={d},trim=duration={d},"
                          f"setpts=PTS-STARTPTS,{place}[m{k}]")
        else:
            args += ["-loop", "1", "-framerate", str(FPS), "-t", f"{d}", "-i", str(path)]
            chains.append(f"[{k}:v]{fit},trim=duration={d},setpts=PTS-STARTPTS,{place}[m{k}]")
        labels.append(f"[m{k}]")
    n = len(items)
    chains.append(f"{''.join(labels)}concat=n={n}:v=1:a=0,format=yuv420p[v0]")
    last, k = "v0", n
    if camera:
        cx, cy, cw, ch = CAM
        args += ["-i", str(camera)]
        chains.append(f"[{k}:v]fps={FPS},scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch},"
                      f"tpad=stop_mode=clone:stop_duration={dur},trim=duration={dur},setpts=PTS-STARTPTS[cam]")
        chains.append(f"[{last}][cam]overlay={cx}:{cy}[v{k}]")
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
            x, y, w, h = CAM
            ImageDraw.Draw(frame).rectangle((x, y, x + w - 1, y + h - 1), outline="#d03b3b", width=5)
            x0 = gap + c * (tw + gap)
            sheet.paste(frame.resize((tw, th)), (x0, y0 + label))
            d.text((x0 + tw - 70, y0 + 8), stamp(t, ".")[3:8], font=font(18), fill=MUTED)
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
    shots, previous = {}, None
    for n, seg in enumerate(segs):
        start, end = seconds(seg["start"]), seconds(seg["end"])
        dur = end - start
        ext = fade if n + 1 < len(segs) else 0.0     # runs on under the next block's fade-in
        print(f"{seg['id']}: {seg['start']}–{seg['end']}")
        items, shots[seg["id"]] = [], []
        for i, item in enumerate(main_items(seg)):
            path, is_video, zooms = picture(seg, i, item, ctx, placeholders)
            if not path.exists():
                sys.exit(f"{seg['id']}: missing {path}")
            d = item["t1"] - item["t0"] + (ext if item["t1"] == end else 0)
            items.append((path, is_video, d, zooms))
            mid = [item["t0"] + (z["t0"] + z["t1"]) / 2 for z in zooms]
            shots[seg["id"]] += mid[:1] or [(item["t0"] + item["t1"]) / 2]
        if len(shots[seg["id"]]) == 1:              # one long item: show its start and end too
            shots[seg["id"]] = [start + dur * f for f in (0.2, 0.5, 0.85)]
        track, subs = narration_track(seg, dur, OUT / "work", warnings)
        audio += read_pcm(track)
        srt += [(start + s, start + e, text) for s, e, text in subs]

        layers = []
        if seg.get("references"):
            layers.append((0, dur + ext, lambda d, r=seg["references"]: draw_references(d, r)))
        if seg["speaker"] != previous:
            who = speakers[seg["speaker"]]
            layers.append((0.5, min(6.5, dur), lambda d, w=who: draw_lower_third(d, w["name"], w["role"])))
        previous = seg["speaker"]
        if n == 0 and cfg.get("title"):
            layers.append((0, float(cfg["title"].get("seconds", 5)), lambda d, t=cfg["title"]: draw_title(d, t)))
        if not seg.get("audio"):
            layers.append((0, dur + ext, draw_watermark))
        if not args.no_burn:
            layers += [(s, e - 0.04, lambda d, t=text: draw_subtitle(d, t)) for s, e, text in subs]
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
