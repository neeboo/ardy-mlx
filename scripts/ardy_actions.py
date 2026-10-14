#!/usr/bin/env python
"""Action-clip batch generator: crouch / crawl / jump, through the MLX batch path.

Answers the gmgn ask "give the action stack *content* for crouch/crawl/jump, via
ARDY generation, through the bridge, into playable clips". This script is the
generation-and-measurement half: it drives the **existing** generator
(``scripts/ardy_generate.py --backend mlx``, checkpoint
``ARDY-Core-RP-20FPS-Horizon40``) once per action, records the clip's provenance
and derived-tensor geometry, and measures **foot skating** so "which clip is
usable as an asset" is a number rather than an opinion.

Design rules inherited from this repository
-------------------------------------------
* ``--backend mlx`` only. The placeholder backend is a random-weight stand-in;
  producing an action asset from it would be exactly the "占位动作" the ask
  forbids. A spec that cannot be generated for real is reported as **failed**,
  never substituted.
* **No postprocessing comparison**, because this checkout cannot run it. Core
  models run ``MotionCorrection`` upstream, that C++ extension is not built here,
  and ``docs/decisions/0002`` ruling 2 pins this side at "both sides skip it".
  ``scripts/ardy_generate.py`` refuses to claim otherwise and exits 2. So the
  second run of each action is *attempted with postprocessing on* and its
  **refusal is the recorded result** (see ``postprocess_attempt`` in the
  manifest). See ``artifacts/action-clips/README.md`` for what this means for
  the foot-skating question.
* Weight-derived bytes never enter git (``.gitignore``): the clips land in
  ``artifacts/action-clips/raw/`` and only the manifest, the README, the preview
  GIF and this script are tracked.

Usage
-----
    ./.venv/bin/python scripts/ardy_actions.py encode        # text vectors -> cache
    ./.venv/bin/python scripts/ardy_actions.py generate      # clips + skating table
    ./.venv/bin/python scripts/ardy_actions.py preview       # 3x2 grid -> preview.gif
    ./.venv/bin/python scripts/ardy_actions.py preview-jump  # 3x1 grid -> preview_jump.gif
    ./.venv/bin/python scripts/ardy_actions.py all           # all four, in order
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from ardy_mlx.io import PRESETS  # noqa: E402
from ardy_mlx.model.skeleton import (  # noqa: E402
    CORE_FOOT_JOINT_NAMES,
    CORE_JOINT_NAMES,
)
from ardy_mlx.sampling.local_encoder import LocalEncoder  # noqa: E402
from ardy_mlx.sampling.pipeline import compute_feature_layout  # noqa: E402

# --------------------------------------------------------------------------- #
# the action table -- third person, matching the viewer's own vocabulary
# --------------------------------------------------------------------------- #

#: The model every action is generated with, and its frame rate.
MODEL = "ARDY-Core-RP-20FPS-Horizon40"
#: Fixed seed for every clip, so a re-run reproduces the same motion.
SEED = 0
#: The batch history (the maximum), per contract S5.2 / CLI default resolution.
HISTORY_FRAMES = 160
#: Diffusion steps: the checkpoint's own ``num_base_steps``.
DIFFUSION_STEPS = 10
#: Classifier-free guidance: ``[text, constraint]``, the CLI default.
CFG_WEIGHT = (2.0, 2.0)


@dataclass(frozen=True)
class ActionSpec:
    """One action to generate: its name, its prompt, and how long it runs."""

    name: str
    prompt: str
    duration_s: float
    note: str = ""

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]


#: Order is the preview grid order (3 columns x 2 rows).
ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        "stand_idle",
        "A person stands still.",
        2.0,
        "baseline; nothing moves, so any skating here is noise floor",
    ),
    ActionSpec(
        "crouch_idle",
        "A person crouches down and stays crouched.",
        2.0,
        "the held pose; here for 'crouched, and then what'",
    ),
    ActionSpec(
        "crouch_forward",
        "A person crouches and walks forward slowly.",
        4.0,
        "the moving-while-crouched case the ask calls out explicitly",
    ),
    ActionSpec(
        "crawl_forward",
        "A person crawls forward on hands and knees.",
        4.0,
        "the hardest one; see the README's honesty section",
    ),
    ActionSpec(
        "jump",
        "A person jumps up in place and lands.",
        2.0,
        "in place: displacement should stay near zero",
    ),
    ActionSpec(
        "walk",
        "A person walks forward.",
        4.0,
        "baseline locomotion, and the alignment reference prompt",
    ),
)

#: A second, Chinese-labelled baseline the viewer's cache already carries. It is
#: generated too, so the reference vector gets exercised by this script rather
#: than only by the viewer.
EXTRA_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec("stand_idle_zh", "站立不动", 2.0, "same pose, viewer's Chinese vocab"),
)

#: The jump, as the action-asset ask states it: the **verbatim** in-air prompt plus
#: the forward variant that carries horizontal speed through the flight. They are a
#: separate tuple, not more members of ``ACTIONS``, for one reason: ``cmd_preview``
#: renders ``ACTIONS`` into a **3x2** grid, and a 3x3 grid would silently restage a
#: deliverable that is already signed off. They get their own 3x1 GIF instead
#: (``cmd_preview_jump``).
#:
#: Why a jump is a generation job at all, when crouch/crawl/walk are not: those are
#: *gaits*, i.e. styles a MotionBricks library can already play back. A jump is a
#: **one-shot action** with a takeoff, a flight phase and a landing, so it has to be
#: generated rather than re-timed from an existing walk.
#:
#: Duration is pinned at 2.0 s = 40 frames @ 20 fps, which is exactly **one ARDY
#: window** (``horizon=40``): the clip is produced by a single denoising pass, so it
#: cannot accumulate the inter-window drift a multi-window run would. It is also the
#: same duration as the existing ``jump`` entry, so the three jump prompts are
#: directly comparable.
JUMP_ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        "jump_air",
        "A person jumps up in the air and lands on the spot.",
        2.0,
        "the ask's verbatim prompt; displacement should stay near zero",
    ),
    ActionSpec(
        "jump_forward",
        "A person jumps forward and lands.",
        2.0,
        "the forward variant: horizontal speed must survive the flight phase",
    ),
)

#: Every action this script generates, in manifest order.
ALL_ACTIONS: Tuple[ActionSpec, ...] = ACTIONS + EXTRA_ACTIONS + JUMP_ACTIONS

#: What ``preview_jump.gif`` shows, in grid order: the in-place jump that predates
#: this ask, then the ask's verbatim in-air prompt, then the forward variant. Three
#: prompts, one question -- "which of these reads as a jump".
JUMP_PREVIEW_ACTIONS: Tuple[ActionSpec, ...] = (
    tuple(s for s in ACTIONS if s.name == "jump") + JUMP_ACTIONS
)

RAW_DIR = REPO_ROOT / "artifacts" / "action-clips" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "action-clips"
EMBED_CACHE = REPO_ROOT / "artifacts" / "embeddings"
PREVIEW_GIF = ARTIFACT_DIR / "preview.gif"
#: The jump-only preview: the sibling of ``preview.gif``, same renderer, 3x1.
PREVIEW_JUMP_GIF = ARTIFACT_DIR / "preview_jump.gif"
MANIFEST = ARTIFACT_DIR / "manifest.json"
SKATING_JSON = ARTIFACT_DIR / "foot-skating.json"

#: ``foot_contacts`` column order (contract S1.5).
CONTACT_NAMES: Tuple[str, ...] = ("L-heel", "L-toe", "R-heel", "R-toe")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def encoder_repo() -> Path:
    return LocalEncoder(bits=4).repo


# --------------------------------------------------------------------------- #
# 1. text vectors
# --------------------------------------------------------------------------- #


def cmd_encode(args) -> int:
    """Encode every action's prompt into the viewer's own cache layout.

    This calls the sibling ``llm2vec-mlx`` ``scripts/encode.py`` **directly**
    (one process for all prompts) rather than going through
    :meth:`LocalEncoder.encode` per prompt. Two reasons, both practical: the
    sibling supports ``--texts-file`` for a batch in one weight load, and with
    ``--out`` pointing at a directory it writes the ``<prompt-hash>.npy`` +
    ``.json`` cache pair itself -- the same convention the viewer reads, so the
    install step is a write, not a move.

    Nothing is invented: the vector bytes and every provenance field (weight
    digests, quantization plan, template regime, git sha) come from the encoder.
    """
    encoder = LocalEncoder(bits=4)
    status = encoder.probe()
    if not status.available:
        print(f"error: local encoder unavailable: {status.reason}", file=sys.stderr)
        return 2

    target = EMBED_CACHE / encoder.encoder_id()
    target.mkdir(parents=True, exist_ok=True)

    specs = list(ALL_ACTIONS)
    prompts_path = ARTIFACT_DIR / "prompts.txt"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text("\n".join(s.prompt for s in specs) + "\n", encoding="utf-8")

    missing = [s for s in specs if not (target / f"{s.prompt_hash}.npy").is_file()]
    print(f"[encode] encoder  {status.encoder_id}")
    print(f"[encode] target   {target.relative_to(REPO_ROOT)}")
    print(f"[encode] prompts  {len(specs)} total, {len(missing)} to encode")

    if missing:
        command = [
            str(status.interpreter),
            str(encoder.encode_script),
            "--texts-file", str(prompts_path),
            "--bits", "4",
            "--out", str(target),
            "--weights-dir", str(encoder.weights_dir),
        ]
        print(f"[encode] command  {' '.join(command)}")
        started = time.time()
        completed = subprocess.run(command, cwd=str(encoder.repo), check=False)
        elapsed = time.time() - started
        print(f"[encode] elapsed  {elapsed:.2f}s  exit={completed.returncode}")
        if completed.returncode != 0:
            print("error: the encoder failed; see its output above", file=sys.stderr)
            return completed.returncode
    else:
        print("[encode] all vectors already cached; nothing to do")

    records = []
    for spec in specs:
        vector = target / f"{spec.prompt_hash}.npy"
        sidecar = target / f"{spec.prompt_hash}.json"
        if not vector.is_file():
            print(f"error: {vector} was not written", file=sys.stderr)
            return 3
        records.append(
            {
                "name": spec.name,
                "prompt": spec.prompt,
                "prompt_hash": spec.prompt_hash,
                "vector_path": str(vector.relative_to(REPO_ROOT)),
                "vector_sha256": sha256_file(vector),
                "sidecar_sha256": sha256_file(sidecar) if sidecar.is_file() else None,
            }
        )
    (ARTIFACT_DIR / "text-vectors.json").write_text(
        json.dumps(
            {
                "encoder_id": status.encoder_id,
                "quantization": status.quantization,
                "encoder_repo": str(encoder.repo),
                "encode_script": str(encoder.encode_script),
                "texts_file": str(prompts_path.relative_to(REPO_ROOT)),
                "vectors": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[encode] wrote {ARTIFACT_DIR / 'text-vectors.json'}")
    return 0


# --------------------------------------------------------------------------- #
# 2. foot skating
# --------------------------------------------------------------------------- #


@dataclass
class SkatingReport:
    """Foot skating for one clip: how far each contact joint slides while planted.

    Definition (the reason the numbers are comparable between clips):

    * A joint is *planted* at frame ``t`` when its own ``foot_contacts`` channel is
      true at both ``t-1`` and ``t``. Frame 0 has no predecessor, so it cannot
      contribute -- the ``considered`` count records how many transitions did.
    * The **step** is the horizontal (x-z) distance the joint travelled between
      those two frames. Vertically there is no sliding by definition: a foot that
      lifts is not sliding, and y is excluded for the same reason
      :func:`ardy_mlx.control.stats.displacement_m` excludes it.
    * ``mean_mm_per_frame`` and ``max_mm_per_frame`` are over those transitions;
      ``total_mm`` is their sum, i.e. the distance the contact point dragged.

    A clip whose feet never plant (a jump in flight) reports zeros with
    ``considered == 0``, which is *not* the same statement as "no skating" -- the
    ``considered`` field is what separates those two readings.
    """

    per_channel: Dict[str, Dict[str, float]]
    considered: int
    possible: int
    mean_mm_per_frame: float
    max_mm_per_frame: float
    total_mm: float
    contact_frames: Dict[str, int]
    airborne_frames: int
    root_displacement_m: float
    root_path_m: float
    posture: Dict[str, float]
    # --- jump / vertical block ------------------------------------------- #
    # Skating answers "does it slide while planted". A jump needs a second
    # question: "does it actually leave the ground, and how hard does it come
    # back down". Both are recorded, because a clip that never leaves the floor
    # scores a perfect 0.00 mm/frame of skating and is still not a jump.
    jump: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "definition": (
                "horizontal travel of a foot joint between two consecutive frames in "
                "which its own foot_contacts channel is true at both; y excluded"
            ),
            "per_channel": self.per_channel,
            "considered_transitions": int(self.considered),
            "possible_transitions": int(self.possible),
            "posture": self.posture,
            "mean_mm_per_frame": round(self.mean_mm_per_frame, 4),
            "max_mm_per_frame": round(self.max_mm_per_frame, 4),
            "total_mm": round(self.total_mm, 4),
            "contact_frames_per_channel": self.contact_frames,
            "frames_with_no_contact": int(self.airborne_frames),
            "root_displacement_m": round(self.root_displacement_m, 6),
            "root_path_m": round(self.root_path_m, 6),
            "jump": self.jump,
        }


def measure_skating(npz_path: Path) -> SkatingReport:
    """Compute :class:`SkatingReport` from a contract-S7 ``.npz``."""
    with np.load(npz_path, allow_pickle=False) as data:
        joints = np.asarray(data["posed_joints"], dtype=np.float64)
        contacts = np.asarray(data["foot_contacts"]).astype(bool)
        root = np.asarray(data["root_positions"], dtype=np.float64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    if joints.ndim == 4:  # (B, T, J, 3)
        joints = joints[0]
    if contacts.ndim == 3:
        contacts = contacts[0]
    if root.ndim == 3:
        root = root[0]

    joint_index = [CORE_JOINT_NAMES.index(name) for name in CORE_FOOT_JOINT_NAMES]
    track = joints[:, joint_index, :]  # (T, 4, 3)

    per_channel: Dict[str, Dict[str, float]] = {}
    all_steps: List[float] = []
    total = 0.0
    considered = 0
    # Per channel: four contact joints, so four chances to be planted per
    # transition. Counting only ``T-1`` here made ``considered`` exceed
    # ``possible`` on the crawl clip, which is how the first version of this
    # metric was caught being wrong.
    possible = int(max(track.shape[0] - 1, 0)) * len(CONTACT_NAMES)
    contact_frames: Dict[str, int] = {}

    for column, name in enumerate(CONTACT_NAMES):
        planted = contacts[:-1, column] & contacts[1:, column]
        delta = np.linalg.norm(
            np.diff(track[:, column, :][:, [0, 2]], axis=0), axis=1
        )
        steps = delta[planted]
        contact_frames[name] = int(contacts[:, column].sum())
        per_channel[name] = {
            "planted_transitions": int(planted.sum()),
            "contact_frames": contact_frames[name],
            "mean_mm_per_frame": round(float(steps.mean() * 1000.0), 4) if steps.size else 0.0,
            "max_mm_per_frame": round(float(steps.max() * 1000.0), 4) if steps.size else 0.0,
            "total_mm": round(float(steps.sum() * 1000.0), 4),
        }
        all_steps.extend(steps.tolist())
        total += float(steps.sum())
        considered += int(planted.sum())

    steps_array = np.asarray(all_steps, dtype=np.float64)
    root_xz = root[:, [0, 2]]

    # Posture: not decoration. "A person crawls" and "a person kneels" are
    # different assets with the same skating number, so the height profile is
    # recorded next to the skating and the README is allowed to say which one
    # the model actually produced.
    hip = joints[:, CORE_JOINT_NAMES.index("Hips"), 1]
    head = joints[:, CORE_JOINT_NAMES.index("Head"), 1]
    hand = joints[:, [CORE_JOINT_NAMES.index("LeftHand"), CORE_JOINT_NAMES.index("RightHand")], 1]
    knee = joints[:, [CORE_JOINT_NAMES.index("LeftLeg"), CORE_JOINT_NAMES.index("RightLeg")], 1]
    posture = {
        "hip_y_min_m": round(float(hip.min()), 4),
        "hip_y_max_m": round(float(hip.max()), 4),
        "hip_y_mean_m": round(float(hip.mean()), 4),
        "head_y_mean_m": round(float(head.mean()), 4),
        "hand_y_min_m": round(float(hand.min()), 4),
        # Hands at or below knee height is what separates a crawl (hands carry
        # weight down at the floor) from a kneel (hands hang, knees take it).
        "hands_at_or_below_knee_frac": round(
            float((hand.min(axis=1) <= knee.mean(axis=1)).mean()), 4
        ),
        "hand_minus_knee_y_mean_m": round(float((hand.mean(axis=1) - knee.mean(axis=1)).mean()), 4),
    }
    # ---- jump / vertical ------------------------------------------------- #
    # ``hip`` is the pelvis joint's world height. It is a *proxy* for the body's
    # centre of mass, not a measured CoM, and it is labelled as such in the
    # manifest: the contract's ``posed_joints`` gives joint positions and no mass
    # distribution, so CoM would be an invention.
    airborne_mask = ~contacts.any(axis=1)
    hip_y = joints[:, CORE_JOINT_NAMES.index("Hips"), 1]
    # d/dt of height, m/s, sign = +up. ``np.gradient`` is second-order inside and
    # one-sided at the ends, so frame 0 and frame T-1 are still defined.
    hip_vel = np.gradient(hip_y) * float(fps)
    air_idx = np.flatnonzero(airborne_mask)
    takeoff_frame = int(air_idx[0]) if air_idx.size else None
    landing_frame = None
    if air_idx.size:
        after = np.flatnonzero(~airborne_mask[air_idx[-1] + 1:])
        if after.size:
            landing_frame = int(air_idx[-1] + 1 + after[0])
    touchdown_mps = 0.0
    if landing_frame is not None and landing_frame > 0:
        # The speed on the last airborne frame is the speed the feet arrive with.
        touchdown_mps = max(0.0, -float(hip_vel[landing_frame - 1]))
    descent_peak_mps = 0.0
    if air_idx.size:
        lo = takeoff_frame
        hi = landing_frame if landing_frame is not None else int(air_idx[-1]) + 1
        descent_peak_mps = max(0.0, -float(hip_vel[lo:hi + 1].min()))
    jump = {
        "definition": (
            "hip (pelvis) world height; 'airborne' = all four foot_contacts false; "
            "touchdown speed = -d(hip_y)/dt on the last airborne frame"
        ),
        "hip_joint": "Hips",
        "fps": int(fps),
        "leaves_ground": bool(air_idx.size),
        "airborne_frames": int(airborne_mask.sum()),
        "takeoff_frame": takeoff_frame,
        "landing_frame": landing_frame,
        "hip_y_start_m": round(float(hip_y[0]), 4),
        "hip_y_min_m": round(float(hip_y.min()), 4),
        "hip_y_peak_m": round(float(hip_y.max()), 4),
        "apex_above_start_m": round(float(hip_y.max() - hip_y[0]), 4),
        "apex_above_crouch_m": round(float(hip_y.max() - hip_y.min()), 4),
        "touchdown_mps": round(touchdown_mps, 4),
        "descent_peak_mps": round(descent_peak_mps, 4),
    }

    return SkatingReport(
        per_channel=per_channel,
        considered=considered,
        possible=possible,
        mean_mm_per_frame=float(steps_array.mean() * 1000.0) if steps_array.size else 0.0,
        max_mm_per_frame=float(steps_array.max() * 1000.0) if steps_array.size else 0.0,
        total_mm=total * 1000.0,
        contact_frames=contact_frames,
        airborne_frames=int((~contacts.any(axis=1)).sum()),
        root_displacement_m=float(np.linalg.norm(root_xz[-1] - root_xz[0]))
        if root_xz.shape[0] > 1
        else 0.0,
        root_path_m=float(np.linalg.norm(np.diff(root_xz, axis=0), axis=1).sum())
        if root_xz.shape[0] > 1
        else 0.0,
        posture=posture,
        jump=jump,
    )


# --------------------------------------------------------------------------- #
# 3. generate
# --------------------------------------------------------------------------- #


def generation_command(spec: ActionSpec, embedding: Path, out_dir: Path, *,
                       postprocess: bool) -> List[str]:
    """The exact ``ardy_generate.py`` invocation, as one list of argv.

    Paths are rendered **relative to the repository root**, because that is the
    working directory the command is run from: an absolute path here would make
    the manifest's "how to reproduce this clip" line machine-specific for no
    reason.

    ``postprocess=True`` omits ``--no-postprocess``. On a Core checkpoint that is
    the upstream default, and in this checkout it is a hard refusal -- which is
    the point: the two runs differ in exactly one flag, so the refusal is
    attributable.
    """

    def rel(path: Path) -> str:
        try:
            return str(Path(path).relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    argv = [
        "./.venv/bin/python",
        "scripts/ardy_generate.py",
        "--backend", "mlx",
        "--embedding", rel(embedding),
        "--model", MODEL,
        "--duration", f"{spec.duration_s}",
        "--seed", str(SEED),
        "--history-frames", str(HISTORY_FRAMES),
        "--diffusion-steps", str(DIFFUSION_STEPS),
        "--cfg-weight", str(CFG_WEIGHT[0]), str(CFG_WEIGHT[1]),
        "--prompt", spec.prompt,
        "--out", rel(out_dir),
    ]
    if not postprocess:
        argv.append("--no-postprocess")
    return argv


def run_variant(spec: ActionSpec, embedding: Path, out_dir: Path, *,
                postprocess: bool) -> dict:
    """Run one (action, postprocess-flag) pair and record what actually happened."""
    argv = generation_command(spec, embedding, out_dir, postprocess=postprocess)
    started = time.time()
    completed = subprocess.run(
        argv, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    elapsed = time.time() - started

    record = {
        "postprocess": postprocess,
        "command": " ".join(argv),
        "exit_code": int(completed.returncode),
        "elapsed_s": round(elapsed, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "produced": [],
    }
    model_dir = out_dir / MODEL
    if model_dir.is_dir():
        record["produced"] = sorted(p.name for p in model_dir.iterdir())
    return record


def _derive_record(npz_path: Path, json_path: Path) -> dict:
    """sha256 + every derived tensor's shape and dtype."""
    tensors = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for name in sorted(data.files):
            array = data[name]
            tensors[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "npz": {
            "path": str(npz_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(npz_path),
            "bytes": npz_path.stat().st_size,
        },
        "sidecar": {
            "path": str(json_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(json_path),
            "bytes": json_path.stat().st_size,
        },
        "derived_tensors": tensors,
        "sidecar_contents": sidecar,
    }


def cmd_generate(args) -> int:
    """Generate every action, both postprocess variants, measure skating."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = LocalEncoder(bits=4)
    vector_dir = EMBED_CACHE / encoder.encoder_id()

    specs = list(ALL_ACTIONS)
    layout = compute_feature_layout(
        MODEL,
        int(round(max(s.duration_s for s in specs) * PRESETS[MODEL].fps)),
        HISTORY_FRAMES,
    )[0]

    clips: List[dict] = []
    skating: Dict[str, dict] = {}
    failures: List[str] = []

    for spec in specs:
        embedding = vector_dir / f"{spec.prompt_hash}.npy"
        if not embedding.is_file():
            print(
                f"error: no text vector for {spec.name} at {embedding}; run 'encode' first",
                file=sys.stderr,
            )
            return 2

        out_dir = RAW_DIR / spec.name
        runs = []
        for postprocess in (True, False):
            variant = "postprocess-on" if postprocess else "postprocess-off"
            print(f"\n[generate] {spec.name} ({variant})")
            record = run_variant(spec, embedding, out_dir, postprocess=postprocess)
            print(f"[generate]   exit={record['exit_code']} elapsed={record['elapsed_s']}s")
            if record["exit_code"] != 0:
                tail = record["stderr"].strip().splitlines()
                print(f"[generate]   refused: {tail[0] if tail else '(no stderr)'}")
            else:
                print(f"[generate]   produced: {', '.join(record['produced'])}")
            runs.append(record)

        ok = [r for r in runs if r["exit_code"] == 0]
        if not ok:
            failures.append(spec.name)
            clips.append(
                {
                    "name": spec.name,
                    "prompt": spec.prompt,
                    "prompt_hash": spec.prompt_hash,
                    "duration_s": spec.duration_s,
                    "note": spec.note,
                    "status": "failed",
                    "runs": runs,
                }
            )
            continue

        chosen = ok[0]
        model_dir = out_dir / MODEL
        stem = f"{spec.prompt_hash}-{SEED}"
        npz_path = model_dir / f"{stem}.npz"
        json_path = model_dir / f"{stem}.json"
        if not npz_path.is_file():
            print(f"error: {npz_path} missing after a successful run", file=sys.stderr)
            return 3

        report = measure_skating(npz_path)
        skating[spec.name] = report.as_dict()
        derived = _derive_record(npz_path, json_path)

        entry = {
            "name": spec.name,
            "prompt": spec.prompt,
            "prompt_hash": spec.prompt_hash,
            "seed": SEED,
            "fps": int(derived["sidecar_contents"]["fps"]),
            "frames": int(derived["sidecar_contents"]["num_frames"]),
            "duration_s": spec.duration_s,
            "postprocess": bool(derived["sidecar_contents"]["postprocess"]),
            "note": spec.note,
            "status": "generated",
            "model": MODEL,
            "backend": "mlx",
            "history_frames": HISTORY_FRAMES,
            "diffusion_steps": DIFFUSION_STEPS,
            "cfg_weight": list(CFG_WEIGHT),
            "embedding": {
                "path": str(embedding.relative_to(REPO_ROOT)),
                "sha256": sha256_file(embedding),
                "encoder_id": encoder.encoder_id(),
            },
            "clip": derived["npz"],
            "sidecar": derived["sidecar"],
            "derived_tensors": derived["derived_tensors"],
            "generated_by": {
                "command": chosen["command"],
                "exit_code": chosen["exit_code"],
                "elapsed_s": chosen["elapsed_s"],
            },
            "postprocess_attempt": next(
                (r for r in runs if r["postprocess"]), None
            ),
            "skating": report.as_dict(),
        }
        clips.append(entry)
        print(
            f"[generate] {spec.name}: frames={entry['frames']} "
            f"root_disp={report.root_displacement_m:.3f} m "
            f"skate={report.mean_mm_per_frame:.2f} mm/frame "
            f"({report.considered} of {report.possible} transitions planted)"
        )

    manifest = {
        "schema": "ardy-action-clips/1",
        "generated_by": {
            "script": "scripts/ardy_actions.py",
            "script_sha256": script_sha256(),
            "git_sha": _git_sha(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "model": MODEL,
        "checkpoint_dir": f"checkpoints/{MODEL}",
        "fps": int(layout.fps),
        "horizon": int(layout.horizon),
        "motion_rep_dim": int(layout.motion_rep_dim),
        "nbjoints": int(layout.nbjoints),
        "seed": SEED,
        "history_frames": HISTORY_FRAMES,
        "diffusion_steps": DIFFUSION_STEPS,
        "cfg_weight": list(CFG_WEIGHT),
        "contact_channel_order": list(CONTACT_NAMES),
        "contact_joint_order": list(CORE_FOOT_JOINT_NAMES),
        "raw_dir": str(RAW_DIR.relative_to(REPO_ROOT)),
        "raw_dir_tracked": False,
        "actions": clips,
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    SKATING_JSON.write_text(json.dumps(skating, indent=2) + "\n", encoding="utf-8")
    print(f"\n[generate] wrote {MANIFEST.relative_to(REPO_ROOT)}")
    print(f"[generate] wrote {SKATING_JSON.relative_to(REPO_ROOT)}")
    if failures:
        print(f"[generate] FAILED actions: {', '.join(failures)}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# 4. preview: six clips, one grid, one GIF
# --------------------------------------------------------------------------- #


def _render_grid(args, *, specs, out_gif: Path, out_json: Path, columns: int, label: str) -> int:
    """Compose the clips named by ``specs`` into one labelled side-by-side GIF.

    Reuses ``scripts/ardy_preview.py``'s rasteriser (its font, its camera fit,
    its ground grid, its contact markers) rather than introducing a second
    renderer that could disagree with the previews already in the repository.
    Each cell gets its **own** camera fitted to that action alone, because a walk
    crosses 3 m while a jump stays put; one shared camera would render the jump
    as an indistinguishable blob.

    This is the one renderer, called twice: ``cmd_preview`` passes ``ACTIONS``
    (the 3x2 grid, ``preview.gif``) and ``cmd_preview_jump`` passes
    ``JUMP_PREVIEW_ACTIONS`` (the 3x1 grid, ``preview_jump.gif``).
    """
    import ardy_preview as preview

    if not MANIFEST.is_file():
        print("error: run 'generate' first (no manifest.json)", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    wanted = {s.name for s in specs}
    entries = [
        a for a in manifest["actions"]
        if a["status"] == "generated" and a["name"] in wanted
    ]
    # Keep the table order, not the manifest's.
    order = {s.name: i for i, s in enumerate(specs)}
    entries.sort(key=lambda a: order[a["name"]])
    if not entries:
        print(f"error: no generated clips for {label}; run 'generate' first", file=sys.stderr)
        return 2

    cell_w = int(args.cell_width)
    cell_h = int(args.cell_height)
    rows = (len(entries) + columns - 1) // columns
    grid_w = cell_w * columns
    grid_h = cell_h * rows

    clips = {}
    cams = {}
    labels = {}
    # Each cell is rendered over the **first N frames**, not the whole clip, and
    # its camera is fitted to exactly those frames.
    #
    # This is the difference between a preview that shows where the action goes
    # and one that does not. Fitting a camera to a 4.35 m walk shrinks the figure
    # to a thumbnail; fitting it to the whole clip and then rendering only the
    # first 2 s keeps the figure large but *pans the camera with the motion*, so
    # the walk and the crouch-walk look almost stationary. Rendering and fitting
    # over the same 2 s window keeps the camera world-fixed for that window, so
    # the translation is real on screen. The 2 s cells are unaffected: 40 frames
    # is their entire clip.
    fit_frames = max(int(args.fit_frames), 1)
    for entry in entries:
        npz = REPO_ROOT / entry["clip"]["path"]
        clip = preview.Clip(npz)
        clips[entry["name"]] = clip
        window = clip.joints[: min(fit_frames, clip.n_frames)]
        cams[entry["name"]] = preview.fit_camera(
            [window], cell_w, cell_h,
            azimuth_deg=72.0, elevation_deg=14.0,
            margin_top=26.0, margin_bottom=46.0, margin_side=14.0,
        )
        labels[entry["name"]] = (
            f"{entry['name']}",
            f"{entry['prompt']}",
            f"{entry['frames']}f @{entry['fps']}fps  {entry['duration_s']:.1f}s  "
            f"skate {entry['skating']['mean_mm_per_frame']:.1f} mm/f",
        )

    # The fitted window sets the loop length: every cell stays inside the range
    # its camera was fitted to, so no figure leaves the frame mid-loop.
    frames = min(min(c.n_frames for c in clips.values()), fit_frames)
    if args.frames:
        frames = min(frames, int(args.frames))

    staging = ARTIFACT_DIR / "_gif-frames"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for t in range(frames):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        for index, entry in enumerate(entries):
            name = entry["name"]
            row, column = divmod(index, columns)
            raster = preview.render_frame(clips[name], cams[name], min(t, clips[name].n_frames - 1))
            y0 = row * cell_h
            x0 = column * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = raster
            # A hairline divider so the six cells read as six cells.
            canvas[y0:y0 + cell_h, x0:x0 + 1] = (70, 80, 100)
            canvas[y0:y0 + 1, x0:x0 + cell_w] = (70, 80, 100)
            title, prompt, stat = labels[name]
            preview.draw_text(canvas, title, x0 + 8, y0 + 6, 2, (255, 255, 255))
            preview.draw_text(canvas, stat, x0 + 8, y0 + 24, 1, (150, 236, 170))
            preview.draw_text(canvas, f'"{prompt}"', x0 + 8, y0 + 36, 1, (170, 186, 210))
            contacts = int(clips[name].contacts[min(t, clips[name].n_frames - 1)].sum())
            preview.draw_text(
                canvas, f"{min(t, clips[name].n_frames - 1) + 1}/{clips[name].n_frames}"
                f"  c{contacts}/4",
                x0 + cell_w - 74, y0 + 6, 1,
                preview.COL_CONTACT if contacts == 4 else preview.COL_NO_CONTACT,
            )
        preview.write_ppm(staging / f"frame_{t:04d}.ppm", canvas)

    ok, message = preview.encode_gif(
        staging, out_gif, fps=int(args.fps), max_colors=int(args.max_colors)
    )
    if not ok:
        print(f"error: gif encode failed: {message}", file=sys.stderr)
        return 3
    shutil.rmtree(staging, ignore_errors=True)

    info = preview.probe_gif(out_gif)
    print(f"[preview] wrote {out_gif.relative_to(REPO_ROOT)}")
    print(f"[preview] {grid_w}x{grid_h}, {frames} frames @ {args.fps} fps, "
          f"{out_gif.stat().st_size} bytes")
    print(f"[preview] encoder {message}")
    print(f"[preview] ffprobe {json.dumps(info)}")

    out_json.write_text(
        json.dumps(
            {
                "path": str(out_gif.relative_to(REPO_ROOT)),
                "bytes": out_gif.stat().st_size,
                "sha256": sha256_file(out_gif),
                "width": grid_w,
                "height": grid_h,
                "frames": frames,
                "fps": int(args.fps),
                "layout": f"{columns}x{rows}",
                "label": label,
                "cells": [
                    {
                        "name": e["name"],
                        "prompt": e["prompt"],
                        "status": e["status"],
                        "prompt_hash": e["prompt_hash"],
                    }
                    for e in entries
                ],
                "renderer": f"scripts/ardy_preview.py #{sha256_file(REPO_ROOT / 'scripts' / 'ardy_preview.py')[:7]}",
                "composer": f"scripts/ardy_actions.py #{script_sha256()[:7]}",
                "ffprobe": info,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def cmd_preview(args) -> int:
    """``preview.gif``: the full 3x2 action grid."""
    return _render_grid(
        args,
        specs=ACTIONS,
        out_gif=PREVIEW_GIF,
        out_json=ARTIFACT_DIR / "preview.json",
        columns=3,
        label="the 3x2 action grid",
    )


def cmd_preview_jump(args) -> int:
    """``preview_jump.gif``: the three jump prompts, 3x1, nothing else."""
    return _render_grid(
        args,
        specs=JUMP_PREVIEW_ACTIONS,
        out_gif=PREVIEW_JUMP_GIF,
        out_json=ARTIFACT_DIR / "preview_jump.json",
        columns=3,
        label="the jump grid",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ardy_actions.py",
        description="Generate crouch/crawl/jump action clips and measure foot skating.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("encode", help="encode the action prompts into the embedding cache")
    sub.add_parser("generate", help="generate every action and measure foot skating")

    preview = sub.add_parser("preview", help="compose artifacts/action-clips/preview.gif")
    preview.add_argument("--cell-width", type=int, default=300)
    preview.add_argument("--cell-height", type=int, default=320)
    preview.add_argument(
        "--fit-frames",
        type=int,
        default=40,
        help="frames of each clip the camera is fitted to (default: %(default)s)",
    )
    preview.add_argument("--fps", type=int, default=10)
    preview.add_argument("--frames", type=int, default=0, help="0 = the shortest clip")
    preview.add_argument("--max-colors", type=int, default=128)

    pjump = sub.add_parser(
        "preview-jump", help="compose artifacts/action-clips/preview_jump.gif (3x1)"
    )
    pjump.add_argument("--cell-width", type=int, default=300)
    pjump.add_argument("--cell-height", type=int, default=320)
    pjump.add_argument("--fit-frames", type=int, default=40)
    pjump.add_argument("--fps", type=int, default=10)
    pjump.add_argument("--frames", type=int, default=0)
    pjump.add_argument("--max-colors", type=int, default=128)

    sub.add_parser("all", help="encode, then generate, then both previews")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "encode":
        return cmd_encode(args)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "preview":
        return cmd_preview(args)
    if args.command == "preview-jump":
        return cmd_preview_jump(args)
    if args.command == "all":
        for step in ("encode", "generate", "preview", "preview-jump"):
            argv_step = [step]
            if step in ("preview", "preview-jump"):
                argv_step += [
                    "--cell-width", "300", "--cell-height", "320",
                    "--fit-frames", "40", "--fps", "10",
                ]
            code = main(argv_step)
            if code != 0:
                print(f"error: step {step!r} exited {code}", file=sys.stderr)
                return code
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
