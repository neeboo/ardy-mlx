#!/usr/bin/env python3
"""Skeleton preview renderer for ARDY clips -- pure numpy + system ffmpeg.

Why this file exists
--------------------
The user asked "does it look right / can I preview it". Answering that needs a
*visual* of the contract output (``docs/motion-contract.md`` S7), rendered from
real data, with no heavyweight dependency: no matplotlib, no mujoco, no smplx, no
torch. So this module rasterises the ``posed_joints`` world positions into PPM
frames itself and shells out to ``ffmpeg`` only to assemble/animate them.

What it draws
-------------
* the 27-joint skeleton, as a stick figure with depth-sorted bones;
* a ground grid on the world XZ plane (contract S2: Y up, metres, ground = XZ);
* the four ``foot_contacts`` markers, recoloured when the flag is true;
* a text HUD (prompt / model / frames / fps / provenance / git sha).

The HUD font is a hand-rolled 5x7 bitmap: this repo's ffmpeg has no libfreetype,
so there is no ``drawtext`` filter to lean on, and pulling in a font library
would defeat the "no heavy dependencies" rule. Text also stays in the frames,
so the PNG-sequence fallback is self-describing too.

Honest limits of this renderer
------------------------------
* It is a **skeleton preview**, not a character/skin render. There is no mesh, no
  skinning, no shading, no floor contact solving.
* The camera is a single fixed pinhole with a fixed palette shared by every clip
  it is given (see ``camera``), so two clips can be compared pixel for pixel.
* Nothing here is a substitute for the numeric contract checks in
  ``scripts/ardy_align.py``; the numbers this script prints are *pixel* numbers.

Usage
-----
    # 1. fit one camera over every clip that will be compared
    python scripts/ardy_preview.py camera --npz golden.npz --npz ours.npz \
        --out /tmp/cam.json --azimuth 38 --elevation 14

    # 2. render each clip with that camera
    python scripts/ardy_preview.py render --npz golden.npz --gif out.gif \
        --camera /tmp/cam.json --label "GOLDEN (reference run)" \
        --model ARDY-Core-RP-20FPS-Horizon40 --git-sha b4918ba

    # 3. pixel-compare two renders
    python scripts/ardy_preview.py compare --gif a.gif --gif b.gif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# skeleton (transcribed from ardy_mlx.model.skeleton; duplicated here on purpose
# so this renderer has no import-time dependency on mlx)
# --------------------------------------------------------------------------- #

JOINT_NAMES: Tuple[str, ...] = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)

JOINT_PARENTS: Tuple[int, ...] = (
    -1, 0, 1, 2, 3, 4, 5,
    4, 7, 8, 9, 10, 10,
    4, 13, 14, 15, 16, 16,
    0, 19, 20, 21,
    0, 23, 24, 25,
)

#: ``foot_contacts`` column order is ``(left heel, left toe, right heel, right toe)``
#: (contract S1.5) -- i.e. joints ``LeftFoot, LeftToeBase, RightFoot, RightToeBase``.
CONTACT_JOINTS: Tuple[int, ...] = (25, 26, 21, 22)
CONTACT_NAMES: Tuple[str, ...] = ("L-heel", "L-toe", "R-heel", "R-toe")

#: which colour a bone gets, keyed by the chain its *child* joint belongs to
_CHAIN: Dict[int, str] = {}
for _j in (7, 8, 9, 10, 11, 12):
    _CHAIN[_j] = "right_arm"
for _j in (13, 14, 15, 16, 17, 18):
    _CHAIN[_j] = "left_arm"
for _j in (19, 20, 21, 22):
    _CHAIN[_j] = "right_leg"
for _j in (23, 24, 25, 26):
    _CHAIN[_j] = "left_leg"
for _j in (1, 2, 3, 4, 5, 6):
    _CHAIN[_j] = "torso"

BONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "torso": (206, 214, 228),
    "right_arm": (255, 146, 92),
    "left_arm": (96, 190, 255),
    "right_leg": (255, 96, 148),
    "left_leg": (124, 236, 148),
    "root": (240, 240, 240),
}

COL_CONTACT = (70, 255, 130)
COL_NO_CONTACT = (255, 176, 64)
COL_GRID = (86, 100, 122)
COL_GRID_MAJOR = (126, 144, 170)
COL_AXIS_X = (198, 96, 96)
COL_AXIS_Z = (96, 140, 208)
COL_JOINT = (250, 250, 255)

# --------------------------------------------------------------------------- #
# 5x7 bitmap font: one glyph = 7 rows, low 5 bits of each row = columns L->R
# --------------------------------------------------------------------------- #

FONT: Dict[str, Tuple[int, ...]] = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "!": (0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04),
    '"': (0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00),
    "#": (0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A),
    "$": (0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04),
    "%": (0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13),
    "&": (0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D),
    "'": (0x0C, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "*": (0x00, 0x0A, 0x04, 0x1F, 0x04, 0x0A, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    ";": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "?": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04),
    "@": (0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x01, 0x01, 0x01, 0x01, 0x01, 0x11, 0x0E),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "[": (0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E),
    "\\": (0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01),
    "]": (0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E),
    "^": (0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    "a": (0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F),
    "b": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E),
    "c": (0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E),
    "d": (0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F),
    "e": (0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E),
    "f": (0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08),
    "g": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "h": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "i": (0x04, 0x00, 0x04, 0x04, 0x04, 0x04, 0x04),
    "j": (0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C),
    "k": (0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12),
    "l": (0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "m": (0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15),
    "n": (0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "o": (0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E),
    "p": (0x00, 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10),
    "q": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x01),
    "r": (0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10),
    "s": (0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E),
    "t": (0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06),
    "u": (0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D),
    "v": (0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "w": (0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A),
    "x": (0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11),
    "y": (0x00, 0x11, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "z": (0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F),
    "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "~": (0x00, 0x00, 0x09, 0x15, 0x12, 0x00, 0x00),
}

GLYPH_W, GLYPH_H, GLYPH_ADV = 5, 7, 6

def _renderer_id() -> str:
    """Short content hash of this file -- the render provenance in the HUD.

    A git sha for the *render* cannot be known before the render is committed, so
    the HUD carries a self-verifying digest of the renderer instead; the commit
    that contains it is recorded in ``docs/PREVIEW.md``.
    """
    try:
        blob = Path(__file__).read_bytes()
    except OSError:  # pragma: no cover - unreadable source
        return "unknown"
    return hashlib.sha256(blob).hexdigest()[:7]


#: ``ardy_preview.py #<7 hex of its own sha256>``
RENDERER_ID = f"ardy_preview.py #{_renderer_id()}"


def text_size(text: str, scale: int) -> Tuple[int, int]:
    lines = text.split("\n")
    width = max((len(ln) for ln in lines), default=0)
    return width * GLYPH_ADV * scale, len(lines) * (GLYPH_H + 1) * scale


def draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Optional[Tuple[int, int, int]] = None,
    bg_alpha: float = 0.55,
) -> None:
    """Blit ``text`` into ``img`` (H,W,3 uint8) at integer ``(x, y)`` top-left."""
    h, w = img.shape[:2]
    if bg is not None:
        tw, th = text_size(text, scale)
        pad = 2 * scale
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + tw + pad), min(h, y + th + pad)
        if x1 > x0 and y1 > y0:
            base = img[y0:y1, x0:x1].astype(np.float32)
            tint = np.array(bg, dtype=np.float32)
            img[y0:y1, x0:x1] = np.clip(
                base * (1.0 - bg_alpha) + tint * bg_alpha, 0, 255
            ).astype(np.uint8)

    col = np.array(color, dtype=np.uint8)
    cy = y
    for line in text.split("\n"):
        cx = x
        for ch in line:
            rows = FONT.get(ch)
            if rows is None:
                rows = FONT["?"]
            for ry, bits in enumerate(rows):
                if not bits:
                    continue
                for rx in range(GLYPH_W):
                    if bits & (1 << (GLYPH_W - 1 - rx)):
                        px, py = cx + rx * scale, cy + ry * scale
                        if 0 <= px < w and 0 <= py < h:
                            img[py : py + scale, px : px + scale] = col
            cx += GLYPH_ADV * scale
        cy += (GLYPH_H + 1) * scale


# --------------------------------------------------------------------------- #
# raster primitives
# --------------------------------------------------------------------------- #


def _blend(img: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0) -> None:
    """Alpha-blend ``color`` where ``mask`` (float HxW coverage) is nonzero."""
    if not mask.any():
        return
    a = np.clip(mask, 0.0, 1.0)[..., None] * alpha
    col = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    img[:] = np.clip(img.astype(np.float32) * (1.0 - a) + col * a, 0, 255).astype(
        np.uint8
    )


def draw_segment(
    img: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: Tuple[int, int, int],
    radius: float,
    softness: float = 0.9,
    alpha: float = 1.0,
) -> None:
    """Thick anti-aliased line segment using a distance field."""
    h, w = img.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    lo_x = max(0, int(math.floor(min(x0, x1) - radius - 2)))
    hi_x = min(w, int(math.ceil(max(x0, x1) + radius + 2)))
    lo_y = max(0, int(math.floor(min(y0, y1) - radius - 2)))
    hi_y = min(h, int(math.ceil(max(y0, y1) + radius + 2)))
    if hi_x <= lo_x or hi_y <= lo_y:
        return
    ys, xs = np.mgrid[lo_y:hi_y, lo_x:hi_x]
    px = xs + 0.5
    py = ys + 0.5
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-9:
        dist = np.hypot(px - x0, py - y0)
    else:
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / seg_len2, 0.0, 1.0)
        dist = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    cov = np.clip((radius + softness - dist) / (2.0 * softness), 0.0, 1.0)
    _blend(img[lo_y:hi_y, lo_x:hi_x], cov, color, alpha)


def draw_disc(
    img: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    color: Tuple[int, int, int],
    ring: bool = False,
    alpha: float = 1.0,
) -> None:
    h, w = img.shape[:2]
    cx, cy = center
    ext = radius + 2
    lo_x = max(0, int(math.floor(cx - ext)))
    hi_x = min(w, int(math.ceil(cx + ext)))
    lo_y = max(0, int(math.floor(cy - ext)))
    hi_y = min(h, int(math.ceil(cy + ext)))
    if hi_x <= lo_x or hi_y <= lo_y:
        return
    ys, xs = np.mgrid[lo_y:hi_y, lo_x:hi_x]
    dist = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
    if ring:
        cov = np.clip(1.5 - np.abs(dist - radius), 0.0, 1.0)
    else:
        cov = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    _blend(img[lo_y:hi_y, lo_x:hi_x], cov, color, alpha)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #

BG_TOP = np.array([16, 20, 30], dtype=np.float32)
BG_HORIZON = np.array([54, 64, 84], dtype=np.float32)
BG_BOTTOM = np.array([22, 26, 34], dtype=np.float32)


class Camera:
    """A single fixed pinhole camera, fit once and reused across clips."""

    def __init__(
        self,
        target,
        eye,
        focal,
        width,
        height,
        grid_half,
        margin_top=80.0,
        margin_bottom=62.0,
        margin_side=18.0,
    ):
        self.target = np.asarray(target, dtype=np.float64)
        self.eye = np.asarray(eye, dtype=np.float64)
        self.focal = float(focal)
        self.width = int(width)
        self.height = int(height)
        self.grid_half = float(grid_half)
        #: bands reserved for the HUD; the projection fit keeps joints out of them
        self.margin_top = float(margin_top)
        self.margin_bottom = float(margin_bottom)
        self.margin_side = float(margin_side)

        self.f = self.target - self.eye
        self.f /= np.linalg.norm(self.f)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(self.f, up)
        n = np.linalg.norm(right)
        if n < 1e-9:  # looking straight down
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / n
        self.r = right
        self.u = np.cross(self.r, self.f)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    # -- projection ------------------------------------------------------- #
    def project(self, pts: np.ndarray) -> np.ndarray:
        """(..., 3) world -> (..., 2) pixels (no clipping; z is guaranteed > 0)."""
        v = np.asarray(pts, dtype=np.float64) - self.eye
        x = v @ self.r
        y = v @ self.u
        z = v @ self.f
        z = np.maximum(z, 1e-6)
        sx = self.cx + self.focal * x / z
        sy = self.cy - self.focal * y / z
        return np.stack([sx, sy], axis=-1)

    def depth(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - self.eye) @ self.f

    # -- background ------------------------------------------------------- #
    def background(self) -> np.ndarray:
        """Sky/ground vertical gradient split at the true horizon row."""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        horizon = self.cy + self.focal * self.f[1] / self.u[1]
        horizon = float(np.clip(horizon, -10 * self.height, 10 * self.height))
        rows = np.arange(self.height, dtype=np.float64)
        for i, row in enumerate(rows):
            if row <= horizon:
                span = max(horizon, 1.0)
                t = float(np.clip(row / span, 0.0, 1.0))
                img[i] = BG_TOP * (1 - t) + BG_HORIZON * t
            else:
                span = max(self.height - horizon, 1.0)
                t = float(np.clip((row - horizon) / span, 0.0, 1.0))
                img[i] = BG_HORIZON * (1 - t) + BG_BOTTOM * t
        return img

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {
            "target": self.target.tolist(),
            "eye": self.eye.tolist(),
            "focal": self.focal,
            "width": self.width,
            "height": self.height,
            "grid_half": self.grid_half,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "margin_side": self.margin_side,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Camera":
        return cls(
            d["target"], d["eye"], d["focal"], d["width"], d["height"], d["grid_half"],
            d.get("margin_top", 80.0), d.get("margin_bottom", 62.0), d.get("margin_side", 18.0),
        )


def fit_camera(
    clips: Sequence[np.ndarray],
    width: int,
    height: int,
    azimuth_deg: float,
    elevation_deg: float,
    margin_top: float,
    margin_bottom: float,
    margin_side: float,
    pad: float = 1.35,
) -> Camera:
    """Fit one camera whose projection contains every joint of every clip.

    The frame is split into a reserved HUD band at the top and bottom (so text
    never lands on the figure) and the remaining rectangle must contain every
    joint of every supplied clip. Because the margin is asymmetric, the focal
    limit is taken separately for points above and below the optical axis.
    """
    all_pts = np.concatenate([c.reshape(-1, 3) for c in clips], axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    target = (lo + hi) / 2.0

    # ground grid just has to cover the horizontal footprint
    xz = all_pts[:, [0, 2]]
    mid_xz = (xz.min(axis=0) + xz.max(axis=0)) / 2.0
    half_span = float(np.max(np.abs(xz - mid_xz))) * pad
    grid_half = max(1.5, math.ceil(half_span * 2.0) / 2.0)

    size = float(np.linalg.norm(hi - lo))
    dist = max(2.0, 2.6 * size)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    back = np.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)]
    )
    eye = target + dist * back

    cam = Camera(
        target, eye, 1.0, width, height, grid_half, margin_top, margin_bottom, margin_side
    )

    # largest focal length that still keeps every point inside the free band
    v = all_pts - cam.eye
    x = v @ cam.r
    y = v @ cam.u
    z = np.maximum(v @ cam.f, 1e-6)
    yz = y / z  # screen-relative vertical slope (positive = above the axis)
    xz_ = x / z
    limits = []
    rx = width / 2.0 - margin_side
    if rx > 1.0 and np.abs(xz_).max() > 1e-12:
        limits.append(rx / np.abs(xz_).max())
    ry_up = height / 2.0 - margin_top
    ry_dn = height / 2.0 - margin_bottom
    up = yz > 1e-12
    dn = yz < -1e-12
    if ry_up > 1.0 and up.any():
        limits.append(ry_up / yz[up].max())
    if ry_dn > 1.0 and dn.any():
        limits.append(ry_dn / (-yz[dn].min()))
    cam.focal = min(limits) if limits else float(width)
    return cam


# --------------------------------------------------------------------------- #
# clip loading / rendering
# --------------------------------------------------------------------------- #


class Clip:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.path = path
        joints = np.asarray(z["posed_joints"], dtype=np.float64)
        self.joints = joints[0] if joints.ndim == 4 else joints  # (T, J, 3)
        self.frames = np.asarray(z["frames"]) if "frames" in z.files else None
        if self.frames is not None and self.frames.ndim == 3:
            self.frames = self.frames[0]
        fc = np.asarray(z["foot_contacts"])
        self.contacts = (fc[0] if fc.ndim == 3 else fc).astype(bool)  # (T, 4)
        self.fps = int(z["fps"])
        self.text = str(z["text"]) if "text" in z.files else path.stem
        self.offsets = (
            np.asarray(z["offsets"]).tolist() if "offsets" in z.files else [0, len(self.joints)]
        )
        #: True when the window plan had to be inferred rather than read from the
        #: npz. The golden npz files predate the ``offsets`` member, so a 4.0 s
        #: golden clip would otherwise be labelled "1 window" when it is really 2
        #: (contract S11 N16 / align-report cli_acceptance). ``--offsets`` fixes it.
        self.offsets_inferred = "offsets" not in z.files
        if self.joints.shape[-2] != len(JOINT_NAMES):
            raise ValueError(
                f"{path.name}: expected {len(JOINT_NAMES)} joints, got {self.joints.shape[-2]}"
            )

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])


def render_frame(clip: Clip, cam: Camera, t: int) -> np.ndarray:
    """Rasterise one frame: background, ground grid, skeleton, contact markers."""
    img = cam.background()

    # ---- ground grid ---------------------------------------------------- #
    g = cam.grid_half
    step = 0.5
    ticks = np.arange(-g, g + 1e-9, step)
    d_eye = cam.depth(cam.target)
    for v in ticks:
        major = abs(v % 1.0) < 1e-9
        col = COL_GRID_MAJOR if major else COL_GRID
        # lines parallel to Z (constant x)
        pts = np.array([[v, 0.0, -g], [v, 0.0, g]])
        a, b = cam.project(pts)
        d = cam.depth(pts).mean()
        alpha = float(np.clip(1.15 - 0.85 * (d / (d_eye + g)), 0.16, 0.95))
        draw_segment(img, a, b, col, 1.1 if major else 0.8, alpha=alpha * (0.95 if major else 0.6))
        # lines parallel to X (constant z)
        pts = np.array([[-g, 0.0, v], [g, 0.0, v]])
        a, b = cam.project(pts)
        d = cam.depth(pts).mean()
        alpha = float(np.clip(1.15 - 0.85 * (d / (d_eye + g)), 0.16, 0.95))
        draw_segment(img, a, b, col, 1.1 if major else 0.8, alpha=alpha * (0.95 if major else 0.6))

    # world axes, so screen orientation is readable
    for end, col in (((g, 0, 0), COL_AXIS_X), ((0, 0, g), COL_AXIS_Z)):
        a, b = cam.project(np.array([[0.0, 0.0, 0.0], list(end)]))
        draw_segment(img, a, b, col, 1.6, alpha=0.9)

    # ---- skeleton ------------------------------------------------------- #
    J = clip.joints[t]
    depth = cam.depth(J)
    bones = []
    for j in range(1, len(JOINT_NAMES)):
        p = JOINT_PARENTS[j]
        if p < 0:
            continue
        bones.append((0.5 * (depth[j] + depth[p]), p, j))
    bones.sort(key=lambda b: -b[0])  # far first (painter's algorithm)

    joints_2d = cam.project(J)

    # ---- ground shadow --------------------------------------------------- #
    # Cheap depth cue: the skeleton dropped onto y=0, drawn dark *before* the
    # figure. Not a lighting model -- just a footprint.
    shadow = J.copy()
    shadow[:, 1] = 0.0
    shadow_2d = cam.project(shadow)
    for _, p, j in sorted(bones, key=lambda b: -b[0]):
        draw_segment(img, tuple(shadow_2d[p]), tuple(shadow_2d[j]), (10, 12, 16), 5.0, alpha=0.30)

    for _, p, j in bones:
        chain = _CHAIN.get(j, "root")
        col = BONE_COLORS[chain]
        draw_segment(img, tuple(joints_2d[p]), tuple(joints_2d[j]), col, 4.0, alpha=0.96)

    for j in range(len(JOINT_NAMES)):
        r = 3.2 if j == 0 else 2.3
        draw_disc(img, tuple(joints_2d[j]), r, COL_JOINT, alpha=0.95)
    draw_disc(img, tuple(joints_2d[0]), 5.0, BONE_COLORS["torso"], ring=True, alpha=0.9)

    # ---- foot contacts -------------------------------------------------- #
    for col_i, j in enumerate(CONTACT_JOINTS):
        on = bool(clip.contacts[t, col_i])
        gx, gz = float(J[j, 0]), float(J[j, 2])
        gp = cam.project(np.array([[gx, 0.0, gz]]))[0]
        jp = joints_2d[j]
        if on:
            # a solid pad on the ground plus a stem up to the joint, so
            # "contact" reads as "this foot is planted here"
            draw_disc(img, tuple(gp), 5.0, COL_CONTACT, alpha=0.95)
            draw_disc(img, tuple(gp), 7.5, COL_CONTACT, ring=True, alpha=0.7)
            draw_segment(img, tuple(gp), tuple(jp), COL_CONTACT, 1.2, alpha=0.45)
        else:
            draw_disc(img, tuple(gp), 4.5, COL_NO_CONTACT, ring=True, alpha=0.85)
    return img


def _pick_scale(
    w: int, pad_x: int, left: Sequence[str], right: Sequence[str], cap: int = 2
) -> int:
    """Largest integer font scale at which a two-column strip cannot collide.

    The right column is short (one row here) and is drawn flush right starting at
    the first row, so only the left rows that share a row with it must stay clear
    of it; a long left line further down may use the full width.
    """
    if not left:
        return cap
    for scale in range(cap, 0, -1):
        adv = GLYPH_ADV * scale
        right_w = max((len(s) for s in right), default=0) * adv
        shared = min(len(right), len(left)) if right else 0
        need = max((len(s) for s in left), default=0) * adv
        if shared:
            need = max(need, right_w + max(len(left[i]) for i in range(shared)) * adv)
        if need + 2 * pad_x + (adv if right else 0) <= w:
            return scale
    return 1


def _strip(img: np.ndarray, y0: int, height: int, tint: float = 0.72) -> None:
    band = img[y0 : y0 + height]
    band[:] = np.clip(
        band.astype(np.float32) * (1.0 - tint) + np.array([10, 12, 18], np.float32) * tint,
        0,
        255,
    ).astype(np.uint8)


def compose_frame(
    clip: Clip,
    cam: Camera,
    t: int,
    label: str,
    model: str,
    git_sha: str,
    prompt: str,
    notes: Sequence[str],
    draw_labels: bool,
) -> np.ndarray:
    """One finished frame: skeleton raster, then (optionally) the text HUD."""
    img = render_frame(clip, cam, t)
    if not draw_labels:
        return img
    w, h = cam.width, cam.height
    pad_x = 6
    cnt = int(clip.contacts[t].sum())
    wins = len(clip.offsets) - 1

    top_left = [label, model, f'prompt "{prompt}"', *notes]
    top_right = [f"{t + 1}/{clip.n_frames}"]
    bot_left = [
        f"frames {clip.n_frames}  fps {clip.fps}  {clip.n_frames / clip.fps:.2f} s",
        f"27 joints  {wins} window{'s' if wins != 1 else ''}  no postprocess",
        f"gen git {git_sha}  render {RENDERER_ID}",
    ]
    bot_right = [f"contacts {cnt}/4"]

    scale = min(
        _pick_scale(w, pad_x, top_left, top_right),
        _pick_scale(w, pad_x, bot_left, bot_right),
    )
    line_h = (GLYPH_H + 1) * scale
    pad_y = 2 * scale
    bar_h = 3 * scale

    # --- top strip: provenance (left) + frame counter (right) ------------- #
    strip_h = max(len(top_left), len(top_right)) * line_h + 2 * pad_y
    _strip(img, 0, strip_h)
    draw_text(img, "\n".join(top_left), pad_x, pad_y, scale, (255, 255, 255))
    tw, _ = text_size("\n".join(top_right), scale)
    draw_text(
        img, "\n".join(top_right), w - tw - pad_x, pad_y, scale,
        COL_CONTACT if cnt == 4 else COL_NO_CONTACT,
    )

    # --- bottom strip: shape of the run (left) + contacts (right) --------- #
    strip_h = max(len(bot_left), len(bot_right)) * line_h + 2 * pad_y + bar_h
    y0 = h - strip_h
    _strip(img, y0, strip_h - bar_h)
    draw_text(img, "\n".join(bot_left), pad_x, y0 + pad_y, scale, (226, 232, 240))
    tw, th = text_size("\n".join(bot_right), scale)
    draw_text(
        img, "\n".join(bot_right), w - tw - pad_x, y0 + pad_y, scale,
        COL_CONTACT if cnt == 4 else COL_NO_CONTACT,
    )

    # --- timeline strip with autoregressive window boundaries ------------ #
    bar_y = h - bar_h
    frac = (t + 1) / clip.n_frames
    img[bar_y:h, :] = (48, 54, 68)
    img[bar_y:h, : int(w * frac)] = COL_CONTACT
    for off in clip.offsets[1:-1]:
        px = int(w * off / clip.n_frames)
        img[bar_y - 4 * scale : bar_y, max(0, px - 1) : px + 1] = (255, 224, 120)
    return img


# --------------------------------------------------------------------------- #
# output: ppm frames -> gif (ffmpeg) or png sequence
# --------------------------------------------------------------------------- #


def which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def write_ppm(path: Path, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    with path.open("wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (w, h))
        fh.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


def read_ppm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    parts = data.split(b"\n", 3)
    if parts[0] != b"P6":
        raise ValueError(f"{path}: not a binary PPM")
    w, h = (int(v) for v in parts[1].split())
    return np.frombuffer(parts[3], dtype=np.uint8).reshape(h, w, 3)


def encode_gif(
    frames_dir: Path,
    gif_path: Path,
    fps: int,
    max_colors: int = 256,
    loop: int = 0,
) -> Tuple[bool, str]:
    """Assemble ``frame_%04d.ppm`` into an animated GIF. Returns (ok, message)."""
    ffmpeg = which_ffmpeg()
    if ffmpeg is None:
        return False, "ffmpeg not found"
    pattern = str(frames_dir / "frame_%04d.ppm")
    filt = (
        f"[0:v]split[a][b];[a]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", pattern,
        "-filter_complex", filt,
        "-loop", str(loop), str(gif_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-2000:]
    return True, f"{ffmpeg} (palettegen/paletteuse, {max_colors} colours)"


def probe_gif(gif_path: Path) -> Dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"error": "ffprobe not found"}
    cmd = [
        ffprobe, "-hide_banner", "-loglevel", "error", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,duration",
        "-of", "json", str(gif_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    st = json.loads(proc.stdout)["streams"][0]
    n = int(st.get("nb_read_frames") or 0)
    rate = st.get("avg_frame_rate", "0/0")
    num, _, den = rate.partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "frames": n,
        "fps": fps,
        "duration_s": (n / fps) if fps else None,
        "bytes": gif_path.stat().st_size,
    }


def decode_gif(gif_path: Path) -> np.ndarray:
    """GIF -> (T,H,W,3) uint8 via ffmpeg rawvideo on stdout."""
    ffmpeg = which_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    info = probe_gif(gif_path)
    if "error" in info:
        raise RuntimeError(f"{gif_path}: {info['error']}")
    w, h = info["width"], info["height"]
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(gif_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[-2000:])
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h * 3)
    return buf[: n * w * h * 3].reshape(n, h, w, 3)


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_camera(args) -> int:
    clips = [Clip(Path(p)) for p in args.npz]
    cam = fit_camera(
        [c.joints for c in clips],
        args.width,
        args.height,
        args.azimuth,
        args.elevation,
        args.margin_top,
        args.margin_bottom,
        args.margin_side,
    )
    d = cam.to_dict()
    d["fitted_from"] = [str(Path(p)) for p in args.npz]
    d["fitted_from_frames"] = [c.n_frames for c in clips]
    d["azimuth_deg"] = args.azimuth
    d["elevation_deg"] = args.elevation
    Path(args.out).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ardy_preview] camera -> {args.out}  target={np.round(cam.target, 3).tolist()} "
        f"eye={np.round(cam.eye, 3).tolist()} focal={cam.focal:.1f}px grid_half={cam.grid_half}"
    )
    for c in clips:
        print(f"    {Path(c.path).name}: {c.n_frames} frames @ {c.fps} fps")
    return 0


def cmd_render(args) -> int:
    clip = Clip(Path(args.npz))
    if args.offsets:
        clip.offsets = [int(v) for v in args.offsets.split(",")]
        if clip.offsets[0] != 0 or clip.offsets[-1] != clip.n_frames:
            print(
                f"error: --offsets {args.offsets} must start at 0 and end at "
                f"{clip.n_frames}",
                file=sys.stderr,
            )
            return 2
        clip.offsets_inferred = False
    elif clip.offsets_inferred and clip.n_frames > 40:
        print(
            f"[ardy_preview] warning: {Path(args.npz).name} has no 'offsets' member; "
            f"showing '1 window' for a {clip.n_frames}-frame clip. Pass --offsets to "
            f"state the real autoregressive plan.",
            file=sys.stderr,
        )
    if args.camera:
        cam = Camera.from_dict(json.loads(Path(args.camera).read_text()))
    else:
        cam = fit_camera(
            [clip.joints], args.width, args.height, args.azimuth, args.elevation,
            args.margin_top, args.margin_bottom, args.margin_side,
        )
    if args.camera and (cam.width != args.width or cam.height != args.height):
        print(
            f"[ardy_preview] note: {args.camera} was fitted at {cam.width}x{cam.height}; "
            f"rendering at that size, not {args.width}x{args.height}",
            file=sys.stderr,
        )

    fps = args.fps or clip.fps
    step = max(1, int(round(clip.fps / fps)))
    idx = list(range(0, clip.n_frames, step))
    if idx[-1] != clip.n_frames - 1 and args.keep_last:
        idx.append(clip.n_frames - 1)

    out_gif = Path(args.gif) if args.gif else None
    out_png = Path(args.png_dir) if args.png_dir else None
    if out_gif is None and out_png is None:
        print("error: need --gif or --png-dir", file=sys.stderr)
        return 2

    git_sha = args.git_sha or "unknown"
    notes: List[str] = list(args.note)

    with tempfile.TemporaryDirectory(prefix="ardy-preview-") as td:
        frames_dir = Path(td)
        for i, t in enumerate(idx):
            img = compose_frame(
                clip, cam, t, args.label, args.model, git_sha, args.prompt,
                notes, not args.no_text,
            )
            write_ppm(frames_dir / f"frame_{i:04d}.ppm", img)

        if out_png is not None:
            out_png.mkdir(parents=True, exist_ok=True)
            for f in sorted(frames_dir.glob("frame_*.ppm")):
                shutil.copyfile(f, out_png / (f.stem + ".ppm"))
            print(f"[ardy_preview] {len(idx)} PPM frames -> {out_png}")

        if out_gif is not None:
            out_gif.parent.mkdir(parents=True, exist_ok=True)
            ok, msg = encode_gif(frames_dir, out_gif, fps, args.max_colors)
            if not ok:
                fallback = out_gif.with_suffix("")
                fallback.mkdir(parents=True, exist_ok=True)
                for f in sorted(frames_dir.glob("frame_*.ppm")):
                    shutil.copyfile(f, fallback / (f.stem + ".ppm"))
                print(
                    f"error: gif encode failed ({msg}); wrote PPM sequence to {fallback}",
                    file=sys.stderr,
                )
                return 3
            info = probe_gif(out_gif)
            if "error" in info:
                print(f"error: ffprobe failed: {info['error']}", file=sys.stderr)
                return 4
            sha = hashlib.sha256(out_gif.read_bytes()).hexdigest()
            print(json.dumps({
                "gif": str(out_gif),
                "encoder": msg,
                "source_frames": clip.n_frames,
                "rendered_frames": len(idx),
                "downsample": step,
                "sha256": sha,
                **info,
            }, indent=2))
    return 0


def _pixel_diff(a: np.ndarray, b: np.ndarray) -> Dict:
    if a.shape != b.shape:
        return {"comparable": False, "reason": f"shape {a.shape} vs {b.shape}"}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = (d.max(axis=3) > 0).reshape(d.shape[0], -1).sum(axis=1)
    total = d.shape[1] * d.shape[2]
    return {
        "comparable": True,
        "frames": int(d.shape[0]),
        "width": int(d.shape[2]),
        "height": int(d.shape[1]),
        "identical_frames": int((changed == 0).sum()),
        "changed_pixels_total": int(changed.sum()),
        "pixels_total": int(total * d.shape[0]),
        "changed_pixel_ratio": float(changed.sum() / (total * d.shape[0])),
        "changed_pixels_worst_frame": int(changed.max()),
        "changed_ratio_worst_frame": float(changed.max() / total),
        "max_channel_delta": int(d.max()),
        "mean_abs_channel_delta": float(d.mean()),
        "per_frame_changed_pixels": [int(v) for v in changed],
    }


def load_sequence(path: Path) -> Tuple[np.ndarray, str]:
    """Load a rendered sequence: either a GIF or a directory of PPM frames.

    Comparing the PPM frames directly isolates the renderer's own pixel output
    from GIF palette quantisation, which matters when two animations differ only
    by a few anti-aliased edge pixels.
    """
    if path.is_dir():
        files = sorted(path.glob("frame_*.ppm"))
        if not files:
            raise FileNotFoundError(f"{path}: no frame_*.ppm in directory")
        return np.stack([read_ppm(f) for f in files]), f"ppm-sequence({len(files)})"
    return decode_gif(path), "gif"


def cmd_compare(args) -> int:
    pa, pb = Path(args.gif[0]), Path(args.gif[1])
    a, kind_a = load_sequence(pa)
    b, kind_b = load_sequence(pb)
    out = {
        "a": str(pa),
        "a_kind": kind_a,
        "b": str(pb),
        "b_kind": kind_b,
        "a_sha256": hashlib.sha256(pa.read_bytes()).hexdigest() if pa.is_file() else None,
        "b_sha256": hashlib.sha256(pb.read_bytes()).hexdigest() if pb.is_file() else None,
        "diff": _pixel_diff(a, b),
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_compare_npz(args) -> int:
    ca, cb = Clip(Path(args.npz[0])), Clip(Path(args.npz[1]))
    out: Dict = {"a": str(ca.path), "b": str(cb.path)}
    d = np.abs(ca.joints - cb.joints)
    out["posed_joints"] = {
        "shape": list(d.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "reference_abs_max": float(np.abs(cb.joints).max()),
        "worst_frame": int(d.reshape(d.shape[0], -1).max(axis=1).argmax()),
        "per_frame_max_abs": [float(v) for v in d.reshape(d.shape[0], -1).max(axis=1)],
    }
    if ca.frames is not None and cb.frames is not None:
        df = np.abs(ca.frames - cb.frames)
        out["frames"] = {
            "shape": list(df.shape),
            "max_abs": float(df.max()),
            "mean_abs": float(df.mean()),
            "reference_abs_max": float(np.abs(cb.frames).max()),
            "reference_abs_mean": float(np.abs(cb.frames).mean()),
        }
    out["foot_contacts"] = {
        "mismatched_frames": int((ca.contacts != cb.contacts).any(axis=1).sum()),
        "n_frames": ca.n_frames,
        "per_column_mismatch": [int(v) for v in (ca.contacts != cb.contacts).sum(axis=0)],
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cam_opts(sp):
        sp.add_argument("--width", type=int, default=520)
        sp.add_argument("--height", type=int, default=480)
        sp.add_argument("--azimuth", type=float, default=72.0, help="camera azimuth, degrees")
        sp.add_argument("--elevation", type=float, default=14.0, help="camera elevation, degrees")
        sp.add_argument("--margin-top", type=float, default=78.0,
                        help="pixels reserved at the top for the HUD strip")
        sp.add_argument("--margin-bottom", type=float, default=68.0,
                        help="pixels reserved at the bottom for the HUD strip")
        sp.add_argument("--margin-side", type=float, default=18.0,
                        help="pixels reserved at the left/right edge")

    c = sub.add_parser("camera", help="fit one camera over every clip and save it")
    c.add_argument("--npz", action="append", required=True)
    c.add_argument("--out", required=True)
    add_cam_opts(c)
    c.set_defaults(func=cmd_camera)

    r = sub.add_parser("render", help="npz -> animated GIF (or PPM sequence)")
    r.add_argument("--npz", required=True)
    r.add_argument("--gif")
    r.add_argument("--png-dir")
    r.add_argument("--camera", help="camera JSON from the `camera` subcommand")
    r.add_argument("--label", default="", help="provenance line, e.g. 'GOLDEN (reference run)'")
    r.add_argument("--model", default="")
    r.add_argument("--prompt", default="")
    r.add_argument("--git-sha", default="")
    r.add_argument(
        "--note", action="append", default=[],
        help="extra HUD line under the banner (repeatable)",
    )
    r.add_argument(
        "--offsets", default="",
        help="comma-separated autoregressive window offsets when the npz omits them, "
             "e.g. 0,40,80",
    )
    r.add_argument("--fps", type=int, default=0, help="output fps (0 = the clip's own)")
    r.add_argument("--max-colors", type=int, default=256)
    r.add_argument("--no-text", action="store_true", help="render without the HUD")
    r.add_argument("--keep-last", action="store_true", default=True)
    add_cam_opts(r)
    r.set_defaults(func=cmd_render)

    d = sub.add_parser(
        "compare",
        help="pixel-compare two rendered GIFs (or two PPM frame directories)",
    )
    d.add_argument(
        "--gif", action="append", required=True,
        help="two GIF paths, or two directories holding frame_*.ppm; pass exactly twice",
    )
    d.add_argument("--json")
    d.set_defaults(func=cmd_compare)

    n = sub.add_parser("compare-npz", help="numeric compare of two contract npz files")
    n.add_argument("--npz", action="append", required=True, help="pass exactly twice")
    n.add_argument("--json")
    n.set_defaults(func=cmd_compare_npz)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("gif", "npz"):
        v = getattr(args, name, None)
        if isinstance(v, list) and len(v) != 2 and args.func in (cmd_compare, cmd_compare_npz):
            print(f"error: --{name} must be given exactly twice", file=sys.stderr)
            return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
