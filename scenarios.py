# -*- coding: utf-8 -*-
"""Scenario animation engines. Each renders RGBA frames on a transparent canvas.

Scenarios (demo):
    working  - character at a desk, typing, monitor with scrolling code
    waiting  - idle breathing/sway with pulsing '...' dots
    create   - an AI-style painting derived from the input photo reveals itself
"""
import math
import random
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 512
FPS = 15
DURATION_S = 4
N_FRAMES = FPS * DURATION_S


def new_canvas():
    return Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))


def _paste(base, sprite, cx, cy, scale=1.0):
    w, h = sprite.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    s = sprite.resize((nw, nh), Image.LANCZOS)
    base.alpha_composite(s, (int(cx - nw / 2), int(cy - nh / 2)))


def _rounded(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


# ---------------------------------------------------------------- working ---
_CODE_GRID = None


def _code_grid():
    """Deterministic pseudo code-rain grid (cols x rows of brightness)."""
    global _CODE_GRID
    if _CODE_GRID is None:
        rng = random.Random(7)
        cols, rows = 9, 26
        _CODE_GRID = [[rng.random() for _ in range(rows)] for _ in range(cols)]
    return _CODE_GRID


def frame_working(sprite, i):
    t = i / N_FRAMES
    img = new_canvas()
    d = ImageDraw.Draw(img)
    rng = random.Random(42)

    # --- monitor glow behind ---
    glow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    flick = 0.85 + 0.15 * math.sin(t * math.pi * 8)
    gd.ellipse((300, 120, 500, 380),
               fill=(90, 200, 255, int(38 * flick)))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(30)))

    # --- character (behind desk), bobbing as if working ---
    bob = math.sin(t * math.pi * 4) * 7
    cx, cy = 178, 296 + bob
    _paste(img, sprite, cx, cy, scale=1.0)
    d = ImageDraw.Draw(img)

    # --- desk ---
    _rounded(d, (52, 372, 470, 492), 22, (58, 40, 28, 255))
    _rounded(d, (52, 372, 470, 398), 22, (84, 60, 40, 255))
    d.rectangle((52, 385, 470, 398), fill=(84, 60, 40, 255))

    # --- typing hands (alternate up/down) ---
    phase = int(t * 8) % 2
    hy1 = 352 + (0 if phase == 0 else 10)
    hy2 = 352 + (10 if phase == 0 else 0)
    _rounded(d, (110, hy1, 180, hy1 + 34), 14, (235, 200, 170, 255))
    _rounded(d, (200, hy2, 270, hy2 + 34), 14, (235, 200, 170, 255))

    # --- monitor ---
    _rounded(d, (300, 176, 476, 344), 12, (24, 26, 32, 255))
    _rounded(d, (310, 186, 466, 320), 8, (8, 14, 18, 255))
    # stand + base
    d.rectangle((380, 344, 396, 384), fill=(24, 26, 32, 255))
    _rounded(d, (344, 380, 432, 396), 8, (24, 26, 32, 255))

    # --- scrolling code on screen ---
    grid = _code_grid()
    cols, rows = len(grid), len(grid[0])
    cw, chh = (466 - 310) / cols, (320 - 186) / 14
    scroll = int(t * 30) % rows
    for c in range(cols):
        for r in range(14):
            v = grid[c][(r + scroll) % rows]
            if v < 0.35:
                continue
            x0 = 310 + c * cw + 2
            y0 = 186 + r * chh + 2
            a = int(90 + 165 * v)
            col = (80, 255, 140, a) if v > 0.7 else (60, 180, 200, a)
            d.rectangle((x0, y0, x0 + cw - 4, y0 + chh - 4), fill=col)

    # --- rising idea particles ---
    for _ in range(14):
        x0 = rng.uniform(60, 470)
        sp = rng.uniform(0.5, 1.1)
        ph = rng.random()
        sz = rng.uniform(3, 7)
        prog = (ph + t * sp) % 1.0
        y = 480 - prog * 380
        a = int(200 * math.sin(prog * math.pi))
        d.ellipse((x0 - sz, y - sz, x0 + sz, y + sz),
                  fill=(255, 214, 120, max(0, a)))
    return img


# ---------------------------------------------------------------- waiting ---
def frame_waiting(sprite, i):
    t = i / N_FRAMES
    img = new_canvas()
    d = ImageDraw.Draw(img)

    # --- pulsing glow ring behind character ---
    r = 148 + 14 * math.sin(t * math.pi * 2)
    a = 70 + 45 * math.sin(t * math.pi * 2)
    d.ellipse((256 - r, 268 - r, 256 + r, 268 + r),
              outline=(120, 200, 255, max(0, int(a))), width=7)

    # --- character: breathing + gentle sway ---
    cx = 256 + math.sin(t * math.pi * 2) * 10
    cy = 268 + math.sin(t * math.pi * 4) * 4
    scale = 1.0 + 0.018 * math.sin(t * math.pi * 2)
    _paste(img, sprite, cx, cy, scale=scale)
    d = ImageDraw.Draw(img)

    # --- animated '...' dots (drawn circles, no text) ---
    for k in range(3):
        ph = (t * 1.4 + k * 0.22) % 1.0
        bounce = -10 * abs(math.sin(ph * math.pi))
        alpha = int(90 + 165 * math.sin(ph * math.pi) ** 2)
        x = 216 + k * 40
        y = 448 + bounce
        d.ellipse((x - 11, y - 11, x + 11, y + 11),
                  fill=(150, 210, 255, max(0, alpha)))
    return img


# ------------------------------------------------------------------ create ---
_CREATE_CACHE = {}


def _create_plan(painting):
    """Precompute tile reveal order (spiral from center, seeded shuffle)."""
    key = id(painting)
    if key in _CREATE_CACHE:
        return _CREATE_CACHE[key]
    n = 12
    cells = [(x, y) for y in range(n) for x in range(n)]
    cx = cy = (n - 1) / 2
    cells.sort(key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    rng = random.Random(11)
    # slight shuffle within rings so it feels like brush strokes
    rng.shuffle(cells)
    cells.sort(key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2) // 9)
    spark = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.random() * 6.28)
             for _ in range(26)]
    plan = {"cells": cells, "n": n, "spark": spark}
    _CREATE_CACHE[key] = plan
    return plan


def frame_create(sprite, painting, i):
    t = i / N_FRAMES
    img = new_canvas()
    d = ImageDraw.Draw(img)
    plan = _create_plan(painting)
    n, cells = plan["n"], plan["cells"]

    S = 384                      # painting size
    ox, oy = (CANVAS - S) // 2, (CANVAS - S) // 2 - 10

    # --- wooden frame ---
    _rounded(d, (ox - 14, oy - 14, ox + S + 14, oy + S + 14), 16, (96, 66, 40, 255))
    _rounded(d, (ox - 6, oy - 6, ox + S + 6, oy + S + 6), 10, (60, 40, 24, 255))
    # --- blank canvas ---
    d.rectangle((ox, oy, ox + S, oy + S), fill=(22, 22, 30, 255))

    # --- progressive reveal (first 70% of loop) ---
    p = min(t / 0.70, 1.0)
    shown = int(p * len(cells))
    tile = S / n
    for idx in range(shown):
        gx, gy = cells[idx]
        x0, y0 = ox + gx * tile, oy + gy * tile
        piece = painting.crop((int(gx * tile), int(gy * tile),
                               int((gx + 1) * tile), int((gy + 1) * tile)))
        # fresh strokes pop slightly
        age = shown - idx
        if age < 8:
            sc = 1.0 + 0.25 * (1 - age / 8)
            w2 = int(tile * sc)
            piece = piece.resize((w2, w2), Image.LANCZOS)
            img.alpha_composite(piece, (int(x0 - (w2 - tile) / 2),
                                        int(y0 - (w2 - tile) / 2)))
        else:
            img.alpha_composite(piece, (int(x0), int(y0)))

    d = ImageDraw.Draw(img)

    # --- brush cursor at the latest stroke ---
    if shown < len(cells):
        gx, gy = cells[shown]
        bx, by = ox + (gx + 0.5) * tile, oy + (gy + 0.5) * tile
        for rr, aa in ((26, 60), (16, 110), (8, 220)):
            d.ellipse((bx - rr, by - rr, bx + rr, by + rr),
                      fill=(255, 220, 130, aa))
        d.ellipse((bx - 4, by - 4, bx + 4, by + 4), fill=(255, 255, 255, 255))

    # --- sparkles ---
    for sx, sy, ph in plan["spark"]:
        tw = math.sin(t * math.pi * 6 + ph)
        if tw > 0.55:
            x, y = ox + sx * S, oy + sy * S
            r = 3 + 3 * tw
            a = int(120 + 135 * tw)
            d.line((x - r * 2, y, x + r * 2, y), fill=(255, 255, 255, a), width=2)
            d.line((x, y - r * 2, x, y + r * 2), fill=(255, 255, 255, a), width=2)

    # --- shine sweep on the finished painting (last 30%) ---
    if t > 0.72:
        sp = (t - 0.72) / 0.28
        sx = ox - 80 + sp * (S + 160)
        for k in range(40):
            x = sx + k * 3
            a = max(0, int(90 - abs(k - 20) * 4.5))
            if a > 0:
                d.line((x, oy, x - 60, oy + S), fill=(255, 255, 255, a), width=3)
    return img


# --------------------------------------------------------------- dispatcher --
RENDERERS = {
    "working": lambda sprite, painting, i: frame_working(sprite, i),
    "waiting": lambda sprite, painting, i: frame_waiting(sprite, i),
    "create": frame_create,
}

SCENARIO_NAMES_FA = {
    "working": "در حال کار",
    "waiting": "در انتظار",
    "create": "ساخت عکس",
}


def render_all(sprite, painting, scenario):
    """Render the full frame list for a scenario."""
    fn = RENDERERS[scenario]
    return [fn(sprite, painting, i) for i in range(N_FRAMES)]
