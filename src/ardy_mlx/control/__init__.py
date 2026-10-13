"""Motion controls: the layer that makes the router's parameters *do* something.

        The router reads a sentence and produces numbers. This package is where
        those numbers change the clip that is served.

Why a separate package (and not a tweak inside the sampler)
-----------------------------------------------------------
``speed`` and ``amplitude`` are **deformations applied to a finished motion**, and
one of them additionally changes **what is asked of the sampler**. Putting either
inside ``sampling/**`` would have buried a product decision inside the arithmetic
of the autoregressive loop, and putting them inside ``model/**`` would have changed
the meaning of the checkpoint. So the rule is:

* ``sampling/**`` and ``model/**`` keep their own semantics untouched;
* this package reads a feature block and returns a feature block;
* ``scripts/ardy_viewer.py`` is the only thing that decides *when* to call it.

What each parameter does is frozen in ``docs/decisions/0008-motion-controls.md``
before any of this was written:

======================  ==========================  ===============================
parameter               conditional input?          post-hoc deformation?
======================  ==========================  ===============================
``speed``               yes (``source_frames``)      yes (time-axis resample)
``amplitude``           no                          yes (joint rotation angle)
``expression``/``gaze`` no                          **no** -- placeholders, see below
======================  ==========================  ===============================

Honest boundaries
-----------------
* ``expression`` and ``gaze`` are **not wired to anything**. The rig has no face and
  no eyes. They are carried in the payload and reported as
  ``expression_placeholder: true``, and they never appear in an ``applied`` list.
* ``amplitude`` rescales the ``global_rot_data`` block only. The rendered pose comes
  from forward kinematics over that block (``model/skeleton.py``), so the picture is
  right; the ``local_joints_positions`` block keeps its original values and is
  therefore no longer strictly consistent with the rotations.
* A continuation clip's **frame 0 is the previous clip's last frame** and must stay
  bit-identical to it. ``preserve_first_frame`` exempts that one frame from the
  amplitude deformation; ``speed`` is already exact there because reading a time axis
  at index ``0 * speed`` is index ``0``. The cost of the exemption is a one-frame
  amplitude discontinuity at the seam, which is stated rather than hidden.
"""

from __future__ import annotations

from .motion import (
    CONTROL_AMPLITUDE_RANGE,
    CONTROL_MAX_SOURCE_FACTOR,
    CONTROL_SPEED_RANGE,
    ControlError,
    ControlResult,
    MotionControls,
    apply_controls,
    required_source_frames,
)
from .stats import (
    MotionStats,
    displacement_m,
    joint_rotation_deg,
    mean_speed_mps,
    motion_stats,
)

__all__ = [
    "CONTROL_AMPLITUDE_RANGE",
    "CONTROL_MAX_SOURCE_FACTOR",
    "CONTROL_SPEED_RANGE",
    "ControlError",
    "ControlResult",
    "MotionControls",
    "MotionStats",
    "apply_controls",
    "displacement_m",
    "joint_rotation_deg",
    "mean_speed_mps",
    "motion_stats",
    "required_source_frames",
]
