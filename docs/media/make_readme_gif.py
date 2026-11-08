"""Compose the English README GIF from per-controller SUMO frames.

Input frames and run records come from the thesis visualisation pipeline
(document/skrypty/sumo_porownanie.py, steps `nagraj` + `klatki`).

Usage:
    python docs/media/make_readme_gif.py runs/wizualizacja \
        docs/media/krakow_dead_detectors.gif /tmp/gif_frames [STEP]

STEP = take every STEP-th frame (one frame = 10 s of simulation; default 3).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REC = Path(sys.argv[1])  # tsc_marl/runs/wizualizacja
OUT = Path(sys.argv[2])  # output gif path
TMP = Path(sys.argv[3])  # scratch dir for frames
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 3  # take every STEP-th frame (10 s each)

INK, MUTED, GRID = "#1d2430", "#8a8f98", "#e6e8ec"
ARMS = (
    ("fixedtime", "FixedTime (Webster)", INK),
    ("advancedmp", "Advanced-MP", "#c28833"),
    ("gatppo", "GAT-PPO", "#3b79c8"),
    ("gatppo_fail_gate", "GAT-PPO + failure training + gate", "#3a9b74"),
)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"
f_title = ImageFont.truetype(FONT.format("-Bold"), 15)
f_big = ImageFont.truetype(FONT.format("-Bold"), 34)
f_text = ImageFont.truetype(FONT.format(""), 14)
f_small = ImageFont.truetype(FONT.format(""), 12)

recs = {k: json.loads((REC / f"{k}.json").read_text()) for k, _, _ in ARMS}
dirs = {k: REC / "klatki" / f"{k}__aleja" for k, _, _ in ARMS}
names = sorted(p.name for p in dirs["fixedtime"].glob("*.png"))
names = names[::STEP] + ([names[-1]] if (len(names) - 1) % STEP else [])

# common crop: bounding box of non-white pixels on the last frames
last = [np.asarray(Image.open(dirs[k] / names[-1]).convert("RGB")) for k, _, _ in ARMS]
rows, cols = np.where(np.any([(im < 250).any(-1) for im in last], axis=0))
side = max(rows.max() - rows.min(), cols.max() - cols.min()) + 1
box = (int(cols.min()), int(rows.min()), int(cols.min() + side), int(rows.min() + side))

IMG, PAD, TITLE = 290, 12, 24
W = 2 * IMG + 3 * PAD + 300
H = 2 * (IMG + TITLE) + 3 * PAD
chart = (2 * IMG + 3 * PAD + 10, 230, W - 50, H - 120)
vmax = max(max(r["vehicles"]) for r in recs.values())
n_dead = len(recs["fixedtime"]["dead"])


def to_xy(t, v):
    x0, y0, x1, y1 = chart
    return x0 + (x1 - x0) * t / 3600, y1 - (y1 - y0) * v / (vmax * 1.05)


TMP.mkdir(parents=True, exist_ok=True)
for old in TMP.glob("*.png"):
    old.unlink()

for i, name in enumerate(names):
    t = int(name[:4])
    frame = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(frame)
    for j, (key, label, color) in enumerate(ARMS):
        x = PAD + (j % 2) * (IMG + PAD)
        y = PAD + (j // 2) * (IMG + TITLE + PAD)
        im = Image.open(dirs[key] / name).convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(bg, im).convert("RGB").crop(box).resize((IMG, IMG), Image.LANCZOS)
        frame.paste(im, (x, y + TITLE))
        d.rectangle((x, y + TITLE, x + IMG - 1, y + TITLE + IMG - 1), outline=color, width=3)
        d.text((x + 2, y + 3), label, font=f_title if len(label) < 25 else f_text, fill=color)
        v = recs[key]["vehicles"][t - 1]
        bb = d.textbbox((x + IMG - 8, y + TITLE + 8), f"{v} vehicles", font=f_text, anchor="ra")
        d.rectangle((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3), fill="white", outline=GRID)
        d.text((x + IMG - 8, y + TITLE + 8), f"{v} vehicles", font=f_text, fill=INK, anchor="ra")

    tx = chart[0]
    d.text((tx, PAD + 4), f"{t // 60:02d}:{t % 60:02d}", font=f_big, fill=INK)
    d.text((tx + 112, PAD + 22), "sim time", font=f_small, fill=MUTED)
    lines = (
        "Kraków, al. Trzech Wieszczów (SUMO)",
        f"Red circles: dead detectors at {n_dead}/25",
        "intersections. Same traffic and same",
        "failures for every controller.",
        "Grey = moving; yellow → dark red =",
        "stopped (darker = longer, up to 3 min).",
    )
    for k, line in enumerate(lines):
        d.text((tx, 70 + 18 * k), line, font=f_small, fill=INK)

    x0, y0, x1, y1 = chart
    d.text((x0, y0 - 22), "vehicles in network", font=f_text, fill=INK)
    for m in range(0, 61, 20):
        xm, _ = to_xy(m * 60, 0)
        d.line((xm, y0, xm, y1), fill=GRID)
        d.text((xm, y1 + 6), str(m), font=f_small, fill=MUTED, anchor="mt")
    d.text(((x0 + x1) / 2, y1 + 22), "minutes", font=f_small, fill=MUTED, anchor="mt")
    d.line((x0, y1, x1, y1), fill=MUTED, width=1)
    ends = []
    for key, _, color in ARMS:
        v = recs[key]["vehicles"][:t]
        pts = [to_xy(s + 1, v[s]) for s in range(0, t, 10)] + [to_xy(t, v[-1])]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
        ends.append([pts[-1][1], pts[-1][0], v[-1], color])
    ends.sort()
    for a, b in zip(ends, ends[1:]):
        b[0] = max(b[0], a[0] + 15)
    for ye, xe, n, color in ends:
        d.text((xe + 5, ye), str(n), font=f_small, fill=color, anchor="lm")

    ly = y1 + 44
    for k, (key, label, color) in enumerate(ARMS):
        yy = ly + 17 * k
        d.line((x0, yy + 7, x0 + 22, yy + 7), fill=color, width=3)
        short = label.replace(" + failure training + gate", " + fail-train + gate")
        d.text((x0 + 28, yy), f"{short}: ATT {recs[key]['att']:.0f} s", font=f_small, fill=INK)
    frame.save(TMP / f"{i:04d}.png")

pal = TMP / "palette.png"
src = ["-framerate", "12", "-i", str(TMP / "%04d.png")]
pad = "tpad=stop_mode=clone:stop_duration=3"
subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf", f"{pad},palettegen=max_colors=96:stats_mode=diff", str(pal)], check=True)
subprocess.run(
    ["ffmpeg", "-v", "error", "-y", *src, "-i", str(pal), "-lavfi",
     f"{pad}[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(OUT)],
    check=True,
)
print(OUT, OUT.stat().st_size // 1024, "KB,", len(names), "frames,", f"{W}x{H}")
