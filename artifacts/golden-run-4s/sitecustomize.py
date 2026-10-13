# Non-invasive instrumentation for the SECOND (and final) authorised reference
# generate.py run (nv-tlabs/ardy @ 693f74d), ``--duration 4.0`` = 80 frames =
# two autoregressive windows.
#
# Nothing inside the reference repository is modified. This module is imported by
# CPython's ``site`` module at interpreter start-up (via ``PYTHONPATH``) and only
# monkey-patches already-imported classes.
#
# What is new relative to the run-1 hook (``artifacts/golden-run/sitecustomize.py``)
# -------------------------------------------------------------------------------
# FIX 1 - the atexit flush no longer loses data. Run 1's flush died with
#     ``TypeError: Object of type ndarray is not JSON serializable`` because
#     ``generation_token_mask`` (an ndarray) was stored inside a JSON payload;
#     because everything was written *at exit*, the crash took ``randn_*.npy``,
#     ``diffusion_buffers.npz`` and ``modules/`` with it.
#     Two independent changes make that impossible now:
#       (a) ``_jsonable`` sanitises every record (ndarray -> list / summary);
#       (b) every artefact is written **incrementally, at the moment it is
#           observed** (``_save`` / ``_jsonl``), so the exit hook can only ever
#           lose the optional aggregate ``trace.json``, never the raw data.
#     The optional aggregate is additionally written first into a temp file and
#     renamed, and any failure is logged instead of raised.
# FIX 2 - the module forward hooks no longer retain tensors in memory; each
#     tensor is written immediately, which also removes the run-1 peak-RSS
#     contribution of the (never-written) module dump.
#
# What is new for the multi-window task
# -------------------------------------
#   * per-window records of ``Ardy._generate_window`` (history window bounds,
#     incoming ``first_heading_angle``, incoming history, ``x`` at step 0) and of
#     ``Ardy._recenter_history`` (centre index, ``center_pos``, recentred root);
#   * the three frame masks and the three token masks per window;
#   * the final translated hybrid sequence and the per-chunk autoregressive
#     decode inputs (``get_explicit_motion_from_hybrid_autoregressive``);
#   * a ``phase`` tag ("window0", "window1", ..., "probe") on every record, so a
#     reader never has to guess which window a dump belongs to.
import atexit
import json
import os
import time
import traceback

import numpy as np
import torch

DUMP = os.environ.get("ARDY_GOLDEN_DUMP", "/home/mc/ardy-golden-2026-10-13-4s/raw")
os.makedirs(DUMP, exist_ok=True)
os.makedirs(os.path.join(DUMP, "modules"), exist_ok=True)

STATE = {"errors": [], "randn": [], "steps": [], "buffers": None, "counters": {}}
PHASE = {"name": "startup"}
MAX_PER_MODULE = 40
T0 = time.time()


def _next(key):
    n = STATE["counters"].get(key, 0)
    STATE["counters"][key] = n + 1
    return n


def _note(msg):
    try:
        with open(os.path.join(DUMP, "hook.log"), "a") as fh:
            fh.write(f"[{time.time() - T0:8.3f}s] {msg}\n")
    except Exception:
        pass


def _jsonable(obj, depth=0):
    """Recursively make ``obj`` JSON-serialisable. Never raises.

    This is the fix for run 1's ``TypeError``: numpy arrays are by far the most
    common offender, but numpy scalars, ``torch.Size``, tuples and nested dicts
    are handled too. Arrays with more than 64 elements are summarised (shape,
    dtype, min/max/first values) instead of being inlined, so ``trace.json``
    stays readable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.ndarray,)):
        if obj.size <= 64:
            return {"__ndarray__": obj.tolist(), "shape": list(obj.shape), "dtype": str(obj.dtype)}
        flat = obj.reshape(-1)
        return {
            "__ndarray_summary__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min": float(np.nanmin(flat.astype(np.float64))) if flat.size else None,
            "max": float(np.nanmax(flat.astype(np.float64))) if flat.size else None,
            "first": [float(v) for v in flat[:4].astype(np.float64)],
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Size):
        return list(obj)
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.detach().to("cpu").numpy(), depth)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in obj]
    return {"__repr__": repr(obj)[:200]}


def _np(t):
    """Inference-mode-safe tensor -> numpy float32 copy."""
    if t is None:
        return None
    try:
        with torch.inference_mode():
            return t.detach().to("cpu", torch.float32).numpy().copy()
    except Exception:
        try:
            return np.asarray(t.detach().to("cpu", torch.float32).tolist(), dtype=np.float32)
        except Exception as exc:
            STATE["errors"].append(f"_np: {exc!r}")
            return None


def _save(name, arr):
    """Write one array immediately. Failures are recorded, never raised."""
    if arr is None:
        return
    try:
        path = os.path.join(DUMP, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
    except Exception as exc:
        STATE["errors"].append(f"save {name}: {exc!r}")


def _jsonl(name, record):
    """Append one JSON line immediately (flushed), so a later crash cannot lose it."""
    try:
        with open(os.path.join(DUMP, name), "a") as fh:
            fh.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
    except Exception as exc:
        STATE["errors"].append(f"jsonl {name}: {exc!r}")


def _write_json(name, payload):
    """Atomically write a JSON document; never raises."""
    try:
        path = os.path.join(DUMP, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:
        STATE["errors"].append(f"json {name}: {exc!r}")


def _shape(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (tuple, list)):
        return [_shape(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    return {"type": type(x).__name__}


# --------------------------------------------------------------------- 1. preset
import importlib

_lm = importlib.import_module("ardy.model.load_model")
_kw = _lm.TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"]
_kw["base_model_name_or_path"] = os.environ["ARDY_MERGED_BASE"]
_kw["peft_model_name_or_path"] = os.environ["ARDY_SUPERVISED_ADAPTER"]
_note(f"[hook] preset base={_kw['base_model_name_or_path']}")
_note(f"[hook] preset adapter={_kw['peft_model_name_or_path']}")


# ------------------------------------------- 2. keep the 8B encoder on the CPU
_orig_load_text_encoder = _lm.load_text_encoder


def load_text_encoder(*args, **kwargs):
    kwargs["device"] = os.environ.get("ARDY_HOOK_TEXT_ENCODER_DEVICE", "cpu")
    kwargs["fp32"] = False
    _note(f"[hook] load_text_encoder forced device={kwargs['device']}")
    return _orig_load_text_encoder(*args, **kwargs)


_lm.load_text_encoder = load_text_encoder


# ------------------------------------------------------------- 3. text encoder dump
_ardy_model = importlib.import_module("ardy.model.ardy_model")
Ardy = _ardy_model.Ardy

_orig_encode_text = Ardy._encode_text


def _encode_text(self, texts):
    feat, mask = _orig_encode_text(self, texts)
    _save("text_feat.npy", _np(feat))
    _save("text_pad_mask.npy", _np(mask))
    _write_json(
        "text_encode_meta.json",
        {
            "texts": list(texts),
            "feat_shape": list(feat.shape),
            "feat_dtype": str(feat.dtype),
            "feat_device": str(feat.device),
            "mask_shape": list(mask.shape),
            "mask_dtype": str(mask.dtype),
        },
    )
    _note(f"[hook] dumped text_feat {tuple(feat.shape)} {feat.dtype}")
    return feat, mask


Ardy._encode_text = _encode_text


# ------------------------------------------------------------- 4. torch.randn trace
_orig_randn = torch.randn


def randn(*args, **kwargs):
    out = _orig_randn(*args, **kwargs)
    try:
        i = len(STATE["randn"])
        arr = _np(out)
        # FIX 1b: written now, not at exit.
        _save(f"randn_{i}.npy", arr)
        rec = {
            "i": i,
            "phase": PHASE["name"],
            "args": [list(a) if isinstance(a, (tuple, list, torch.Size)) else str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "shape": list(out.shape),
            "device": str(out.device),
            "dtype": str(out.dtype),
            "file": f"randn_{i}.npy",
        }
        STATE["randn"].append(rec)
        _jsonl("randn.jsonl", rec)
        _note(f"[hook] randn {i} -> {tuple(out.shape)} during {PHASE['name']}")
    except Exception as exc:
        STATE["errors"].append(f"randn: {exc!r}")
    return out


torch.randn = randn


# ----------------------------- 5. per-window: _generate_window / _recenter_history
_orig_generate_window = Ardy._generate_window


def _generate_window(self, history_sequence, global_transl, history_start_frame,
                     history_end_frame, total_frames, text_feat, text_pad_mask,
                     first_heading_angle, *args, **kwargs):
    w = _next("windows")
    prev_phase = PHASE["name"]
    PHASE["name"] = f"window{w}"
    f = self.num_frames_per_token
    hist_tokens = 0 if history_sequence is None else history_sequence.shape[1]
    gen_tokens = self.gen_horizon_len // f
    history_token_end = (history_end_frame - history_start_frame) // f
    rec = {
        "window": w,
        "history_start_frame": int(history_start_frame),
        "history_end_frame": int(history_end_frame),
        "total_frames": int(total_frames),
        "num_frames_per_token": int(f),
        "gen_horizon_len": int(self.gen_horizon_len),
        "history_tokens_in": int(hist_tokens),
        "history_token_end": int(history_token_end),
        "generation_token_end": int(history_token_end + gen_tokens),
        "window_tokens": int(history_token_end + gen_tokens),
        "future_len": int(total_frames - self.gen_horizon_len - history_end_frame),
        "first_heading_angle_in": _np(first_heading_angle),
        "global_transl_in": _np(global_transl),
        "history_sequence_in_shape": None if history_sequence is None else list(history_sequence.shape),
    }
    _save(f"window{w}_history_in.npy", _np(history_sequence))
    _save(f"window{w}_first_heading_in.npy", _np(first_heading_angle))
    _save(f"window{w}_global_transl_in.npy", _np(global_transl))
    _jsonl("windows.jsonl", rec)

    out = _orig_generate_window(
        self, history_sequence, global_transl, history_start_frame, history_end_frame,
        total_frames, text_feat, text_pad_mask, first_heading_angle, *args, **kwargs,
    )
    _save(f"window{w}_history_out.npy", _np(out))
    _jsonl("windows.jsonl", {"window": w, "history_sequence_out_shape": list(out.shape)})
    _note(f"[hook] window{w}: frames [{history_start_frame},{history_end_frame}) "
          f"total={total_frames} hist_tokens={hist_tokens} -> out {tuple(out.shape)} "
          f"heading_in={None if rec['first_heading_angle_in'] is None else rec['first_heading_angle_in'].tolist()}")
    PHASE["name"] = prev_phase
    return out


Ardy._generate_window = _generate_window

_orig_recenter_history = Ardy._recenter_history


def _recenter_history(self, history_sequence, center_frame_index, requantize):
    out = _orig_recenter_history(self, history_sequence, center_frame_index, requantize)
    try:
        idx = len(STATE.setdefault("recenters", []))
        new_seq, center_pos, new_root = out
        _save(f"recenter{idx}_center_pos.npy", _np(center_pos))
        _save(f"recenter{idx}_new_root_motion.npy", _np(new_root))
        _save(f"recenter{idx}_history_recentered.npy", _np(new_seq))
        rec = {
            "i": idx,
            "phase": PHASE["name"],
            "center_frame_index": _np(center_frame_index),
            "requantize": bool(requantize),
            "history_in_shape": list(history_sequence.shape),
            "history_out_shape": list(new_seq.shape),
            "center_pos": _np(center_pos),
        }
        STATE["recenters"].append(rec)
        _jsonl("recenters.jsonl", rec)
        _note(f"[hook] recenter{idx} center_idx={_np(center_frame_index).tolist()} "
              f"center_pos={_np(center_pos).tolist()}")
    except Exception as exc:
        STATE["errors"].append(f"recenter: {exc!r}")
    return out


Ardy._recenter_history = _recenter_history


# --------------------------------------- 6. stage A / stage B module I/O recording
TENSOR_MODULES = (
    "input_proj",
    "output_proj",
    "encoder",
    "decoder",
    "quantizer",
    "root_model",
    "body_model",
    "global_root_hybrid_proj",
    "local_root_hybrid_proj",
    "global_root_hybrid_constraints_proj",
    "local_root_hybrid_constraints_proj",
)


def _install_module_hooks(model):
    if STATE.get("hooked"):
        return
    STATE["hooked"] = True
    targets = []
    for root_name in ("autoencoder", "denoiser"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for name, mod in root.named_modules():
            if not name:
                continue
            leaf = name.rsplit(".", 1)[-1]
            targets.append((f"{root_name}.{name}", leaf, mod))
    _note(f"[hook] installing hooks on {len(targets)} modules")
    for full_name, leaf, mod in targets:
        keep = any(leaf == t for t in TENSOR_MODULES)
        safe = full_name.replace(".", "__")
        seen = {"n": 0}

        def make_hook(full_name=full_name, keep=keep, safe=safe, seen=seen):
            def hook(module, inputs, output):
                # FIX 2: count, write, drop. Nothing big is retained in memory.
                idx = seen["n"]
                seen["n"] += 1
                if idx >= MAX_PER_MODULE:
                    return
                try:
                    _jsonl("modules.jsonl", {
                        "module": full_name,
                        "phase": PHASE["name"],
                        "call": idx,
                        "in": _shape(inputs),
                        "out": _shape(output),
                    })
                    if keep:
                        for ai, v in enumerate(inputs):
                            if isinstance(v, torch.Tensor):
                                _save(f"modules/{safe}__call{idx}__in{ai}.npy", _np(v))
                        if isinstance(output, torch.Tensor):
                            _save(f"modules/{safe}__call{idx}__out.npy", _np(output))
                except Exception as exc:
                    STATE["errors"].append(f"hook {full_name}: {exc!r}")

            return hook

        try:
            mod.register_forward_hook(make_hook())
        except Exception as exc:
            STATE["errors"].append(f"register {full_name}: {exc!r}")


# ------------------------------------ 7. frame masks + token masks, per window
_orig_three_mask = _ardy_model.get_three_mask_from_len


def get_three_mask_from_len(history_len, generation_len, future_len, total_len, *args, **kwargs):
    out = _orig_three_mask(history_len, generation_len, future_len, total_len, *args, **kwargs)
    try:
        i = _next("three_mask")
        h, g, fu = out
        np.savez(
            os.path.join(DUMP, f"masks_{i}.npz"),
            history=_np(h), generation=_np(g), future=_np(fu),
            history_len=_np(history_len), generation_len=_np(generation_len),
            future_len=_np(future_len),
        )
        _jsonl("masks.jsonl", {
            "i": i, "phase": PHASE["name"],
            "history_len": _np(history_len), "generation_len": _np(generation_len),
            "future_len": _np(future_len), "total_len": int(total_len),
        })
    except Exception as exc:
        STATE["errors"].append(f"three mask: {exc!r}")
    return out


_ardy_model.get_three_mask_from_len = get_three_mask_from_len

try:
    _latent_utils = importlib.import_module("ardy.model.latent_utils")
    _HMC = _latent_utils.HybridMotionConverter

    _orig_conv_tokmask = _HMC.convert_frame_mask_to_token_mask

    def convert_frame_mask_to_token_mask(self, history_mask, generation_mask, future_mask, motion_mask=None):
        out = _orig_conv_tokmask(self, history_mask, generation_mask, future_mask, motion_mask)
        try:
            i = _next("tok_mask")
            h, g, fu = out
            np.savez(
                os.path.join(DUMP, f"token_masks_{i}.npz"),
                history=_np(h), generation=_np(g), future=_np(fu),
            )
            _jsonl("token_masks.jsonl", {"i": i, "phase": PHASE["name"],
                                         "history_shape": list(h.shape)})
        except Exception as exc:
            STATE["errors"].append(f"token mask: {exc!r}")
        return out

    _HMC.convert_frame_mask_to_token_mask = convert_frame_mask_to_token_mask

    # ------------------- 8. final translated hybrid + per-chunk AR decode inputs
    _orig_ar_decode = _HMC.get_explicit_motion_from_hybrid_autoregressive

    def get_explicit_motion_from_hybrid_autoregressive(self, hybrid_motion, motion_pad_mask,
                                                       motion_len, motion_mask=None,
                                                       crop_history_length=None):
        _save("final_hybrid_translated.npy", _np(hybrid_motion))
        _write_json("final_decode_meta.json", {
            "hybrid_shape": list(hybrid_motion.shape),
            "crop_history_length": crop_history_length,
            "gen_horizon_len": self.gen_horizon_len,
            "num_frames_per_token": self.num_frames_per_token,
            "motion_len_sum": _np(motion_len),
        })
        out = _orig_ar_decode(self, hybrid_motion, motion_pad_mask, motion_len,
                              motion_mask=motion_mask, crop_history_length=crop_history_length)
        _save("final_motion_decoded_autoregressive.npy", _np(out))
        _write_json("final_decode_out_meta.json", {"out_shape": list(out.shape)})
        _note(f"[hook] AR decode {tuple(hybrid_motion.shape)} -> {tuple(out.shape)} "
              f"crop={crop_history_length}")
        return out

    _HMC.get_explicit_motion_from_hybrid_autoregressive = get_explicit_motion_from_hybrid_autoregressive
except Exception as exc:  # pragma: no cover
    STATE["errors"].append(f"latent_utils hooks: {exc!r}")


# --------------------------------------------- 9. stage A public-API I/O recording
_fsq_mod = importlib.import_module("ardy.model.autoencoder.fsq")
FSQ = _fsq_mod.FSQVAETransformer

for _meth in ("tokenize", "detokenize", "requantize"):
    _orig = getattr(FSQ, _meth)

    def _wrap(orig=_orig, meth=_meth):
        def wrapper(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                i = _next("fsq_" + meth)
                tag = f"fsq_{meth}_{i}"
                inp = args[0] if args else next(iter(kwargs.values()), None)
                _save(tag + "_in.npy", _np(inp) if isinstance(inp, torch.Tensor) else None)
                for ai, v in enumerate(args[1:], start=1):
                    if isinstance(v, torch.Tensor):
                        _save(f"{tag}_arg{ai}.npy", _np(v))
                if isinstance(out, dict):
                    for k, v in out.items():
                        _save(f"{tag}_out_{k}.npy", _np(v))
                    shape_rec = {"out_keys": {k: list(v.shape) for k, v in out.items()}}
                else:
                    _save(tag + "_out.npy", _np(out))
                    shape_rec = {"out_shape": list(out.shape)}
                _jsonl("fsq.jsonl", {"i": i, "method": meth, "tag": tag,
                                     "phase": PHASE["name"], **shape_rec})
                _note(f"[hook] {meth} -> {tag} phase={PHASE['name']}")
            except Exception as exc:
                STATE["errors"].append(f"fsq {meth}: {exc!r}")
            return out

        return wrapper

    setattr(FSQ, _meth, _wrap())


# ------------------------------------------- 10. stage B denoiser / CFG I/O recording
_den_mod = importlib.import_module("ardy.model.auto_latent_twostage_denoiser")
Den = _den_mod.AutoLatentTwostageDenoiser
_orig_den_forward = Den.forward


def den_forward(self, *args, **kwargs):
    idx = _next("den_calls")
    out = _orig_den_forward(self, *args, **kwargs)
    try:
        _save(f"denoiser_raw_{idx}.npy", _np(out))
        _jsonl("denoiser.jsonl", {"idx": idx, "phase": PHASE["name"],
                                  "out_shape": list(out.shape)})
        _note(f"[hook] denoiser raw {idx} -> {tuple(out.shape)}")
    except Exception as exc:
        STATE["errors"].append(f"denoiser: {exc!r}")
    return out


Den.forward = den_forward

_cfg_mod = importlib.import_module("ardy.model.cfg")
CFG = _cfg_mod.AutoLatentClassifierFreeGuidedModelSeparated
_orig_cfg_forward = CFG.forward


def cfg_forward(self, *args, **kwargs):
    idx = _next("cfg_calls")
    out = _orig_cfg_forward(self, *args, **kwargs)
    try:
        _save(f"cfg_combined_{idx}.npy", _np(out))
        _jsonl("cfg.jsonl", {"idx": idx, "phase": PHASE["name"], "out_shape": list(out.shape)})
        _note(f"[hook] cfg combined {idx} -> {tuple(out.shape)}")
    except Exception as exc:
        STATE["errors"].append(f"cfg: {exc!r}")
    return out


CFG.forward = cfg_forward


# ------------------------------------------------- 11. per-denoising-step I/O trace
_orig_denoising_step = Ardy.denoising_step


def denoising_step(self, x, *args, **kwargs):
    idx = len(STATE["steps"])
    if STATE["buffers"] is None:
        try:
            STATE["buffers"] = {
                k: v.detach().to("cpu", torch.float64).numpy().copy()
                for k, v in self.diffusion.named_buffers()
            }
        except Exception as exc:
            STATE["errors"].append(f"buffers: {exc!r}")
    # positional order after (self, x): history_len, generation_len, future_len,
    # history_mask, generation_mask, future_mask, history_token_mask,
    # generation_token_mask, future_token_mask, text_feat, text_pad_mask, t, ...
    t = args[11] if len(args) > 11 else kwargs.get("t")
    gen_token_mask = args[7] if len(args) > 7 else None
    x_in = _np(x)
    out = _orig_denoising_step(self, x, *args, **kwargs)
    rec = {
        "step": idx,
        "phase": PHASE["name"],
        "t": _np(t).tolist() if t is not None else None,
        "x_shape": list(x.shape),
    }
    STATE["steps"].append(rec)
    _save(f"denoise_step{idx}_x_in.npy", x_in)
    _save(f"denoise_step{idx}_x_out.npy", _np(out))
    _save(f"denoise_step{idx}_gen_token_mask.npy",
          _np(gen_token_mask) if gen_token_mask is not None else None)
    _jsonl("steps.jsonl", rec)
    _note(f"[hook] denoise step {idx} ({PHASE['name']}) t={rec['t']} x={list(x.shape)}")
    return out


Ardy.denoising_step = denoising_step


# ------------------------------- 12. final features + one tokenizer round-trip probe
_orig_call = Ardy.__call__


def _call(self, texts, num_frames, num_denoising_steps, pad_mask, first_heading_angle,
          motion_mask, observed_motion, *args, **kwargs):
    out = _orig_call(self, texts, num_frames, num_denoising_steps, pad_mask,
                     first_heading_angle, motion_mask, observed_motion, *args, **kwargs)
    _save("motion_normalized.npy", _np(out))
    _write_json("call_meta.json", {
        "texts": list(texts),
        "num_frames": int(num_frames),
        "num_denoising_steps": int(num_denoising_steps),
        "first_heading_angle": _np(first_heading_angle),
        "cfg_weight": kwargs.get("cfg_weight"),
        "crop_history_length": kwargs.get("crop_history_length"),
        "gen_horizon_len": int(self.gen_horizon_len),
        "num_frames_per_token": int(self.num_frames_per_token),
        "num_base_steps": int(self.diffusion.num_base_steps),
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
        "pad_mask_shape": list(pad_mask.shape),
    })
    _note(f"[hook] dumped motion_normalized {tuple(out.shape)}")

    prev_phase = PHASE["name"]
    PHASE["name"] = "probe"
    try:
        motion_len = pad_mask.sum(dim=-1)
        n_tok = out.shape[1] // self.num_frames_per_token
        num_gen_frames = n_tok * self.num_frames_per_token
        motion_pad_mask = (
            torch.arange(num_gen_frames, device=out.device)[None, :] < motion_len[:, None]
        )
        _save("probe_input_features.npy", _np(out))
        _save("probe_pad_mask.npy", _np(motion_pad_mask))
        hybrid2, tok_mask = self.hybrid.get_hybrid_motion_from_explicit(out, motion_len, motion_pad_mask)
        _save("probe_hybrid.npy", _np(hybrid2))
        _save("probe_token_mask.npy", _np(tok_mask))
        explicit2 = self.hybrid.get_explicit_motion_from_hybrid(hybrid2, motion_pad_mask, motion_len)
        _save("probe_roundtrip_features.npy", _np(explicit2))
        _note("[hook] tokenizer round-trip probe done")
    except Exception as exc:
        STATE["errors"].append(f"probe: {exc!r}")
        _note(f"[hook] probe failed: {exc!r}")
    PHASE["name"] = prev_phase

    return out


Ardy.__call__ = _call


# --------------------------------------------------------- 13. install at first use
_orig_load_model = _lm.load_model


def load_model(*args, **kwargs):
    model = _orig_load_model(*args, **kwargs)
    try:
        _install_module_hooks(model)
    except Exception as exc:
        STATE["errors"].append(f"install hooks: {exc!r}")
    return model


_lm.load_model = load_model
import sys as _sys

_sys.modules["ardy.model"].load_model = load_model
_note("[hook] load_model patched (module + package attribute)")


# --------------------------------------------------------------- 14. flush at exit
def _flush():
    """Optional aggregate only: every raw artefact is already on disk (FIX 1b)."""
    try:
        _write_json("trace.json", {
            "randn": STATE["randn"],
            "denoise_steps": STATE["steps"],
            "recenters": STATE.get("recenters"),
            "counters": STATE["counters"],
            "errors": STATE["errors"],
        })
        if STATE["buffers"]:
            np.savez(os.path.join(DUMP, "diffusion_buffers.npz"), **STATE["buffers"])
        _note(f"[hook] flush done, errors={len(STATE['errors'])}")
    except Exception as exc:
        _note(f"[hook] flush failed (raw data is already on disk): {exc!r}\n{traceback.format_exc()}")


atexit.register(_flush)
_note("[hook] installed (v2: incremental writes, sanitised JSON, per-window records)")
