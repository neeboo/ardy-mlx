# Non-invasive instrumentation for ONE reference generate.py run (nv-tlabs/ardy @ 693f74d).
#
# Nothing inside the reference repository is modified. This module is imported by
# CPython's `site` module at interpreter start-up (via PYTHONPATH) and only
# monkey-patches already-imported classes.
#
# It captures, for the single authorised golden run:
#   * the text encoder's output (`text_feat` / `text_pad_mask`) -- the deterministic
#     input the MLX side has to reproduce;
#   * the initial noise `torch.randn(...)` sampled for each autoregressive window
#     (the only stochastic input of the whole generation);
#   * the full per-denoising-step trace (x_t, denoiser outputs, DDIM update);
#   * the stage-A (tokenizer) and stage-B (denoiser) module inputs/outputs;
#   * the diffusion schedule buffers;
#   * a tokenizer encode -> decode round-trip probe on the gold clip's own
#     features (a read-only forward pass on the frozen 35 M-parameter tokenizer,
#     NOT a second generation).
import atexit
import json
import os
import traceback

import numpy as np
import torch

DUMP = os.environ.get("ARDY_GOLDEN_DUMP", "/home/mc/ardy-golden-2026-10-13/raw")
os.makedirs(DUMP, exist_ok=True)

STATE = {"errors": [], "randn": [], "steps": [], "buffers": None, "modules": {}}
MAX_PER_MODULE = 40

# Leaf module names whose *tensors* are worth keeping (all are tiny here);
# everything else only gets shapes recorded.
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


def _note(msg):
    try:
        with open(os.path.join(DUMP, "hook.log"), "a") as fh:
            fh.write(str(msg) + "\n")
    except Exception:
        pass


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
    if arr is None:
        return
    try:
        np.save(os.path.join(DUMP, name), arr)
    except Exception as exc:
        STATE["errors"].append(f"save {name}: {exc!r}")


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
# Upstream load_text_encoder() ends with `.to(device=<cuda>, dtype=bfloat16)`,
# which would put ~16 GB of weights on an 8 GB card. The local production recipe
# pins the encoder to CPU instead; we do exactly the same, non-invasively.
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
    try:
        with open(os.path.join(DUMP, "text_encode_meta.json"), "w") as fh:
            json.dump(
                {
                    "texts": list(texts),
                    "feat_shape": list(feat.shape),
                    "feat_dtype": str(feat.dtype),
                    "feat_device": str(feat.device),
                    "mask_shape": list(mask.shape),
                    "mask_dtype": str(mask.dtype),
                },
                fh,
                indent=2,
            )
    except Exception as exc:
        STATE["errors"].append(f"text meta: {exc!r}")
    _note(f"[hook] dumped text_feat {tuple(feat.shape)} {feat.dtype}")
    return feat, mask


Ardy._encode_text = _encode_text


# ------------------------------------------------------------- 4. torch.randn trace
_orig_randn = torch.randn


def randn(*args, **kwargs):
    out = _orig_randn(*args, **kwargs)
    try:
        STATE["randn"].append(
            {
                "args": [
                    list(a) if isinstance(a, (tuple, list, torch.Size)) else str(a) for a in args
                ],
                "kwargs": {k: str(v) for k, v in kwargs.items()},
                "shape": list(out.shape),
                "device": str(out.device),
                "dtype": str(out.dtype),
                "_data": _np(out),
            }
        )
    except Exception as exc:
        STATE["errors"].append(f"randn: {exc!r}")
    return out


torch.randn = randn


# --------------------------------------- 5. stage A / stage B module I/O recording
def _install_module_hooks(model):
    """Record input/output shapes (and small tensors) for tokenizer/denoiser modules."""
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
        bucket = STATE["modules"].setdefault(full_name, {"calls": []})

        def make_hook(full_name=full_name, keep=keep, bucket=bucket):
            def hook(module, inputs, output):
                if len(bucket["calls"]) >= MAX_PER_MODULE:
                    return
                try:
                    rec = {
                        "in": _shape(inputs),
                        "out": _shape(output),
                    }
                    if keep:
                        rec["_in"] = [_np(v) for v in inputs if isinstance(v, torch.Tensor)]
                        rec["_out"] = _np(output) if isinstance(output, torch.Tensor) else None
                    bucket["calls"].append(rec)
                except Exception as exc:
                    STATE["errors"].append(f"hook {full_name}: {exc!r}")

            return hook

        try:
            mod.register_forward_hook(make_hook())
        except Exception as exc:
            STATE["errors"].append(f"register {full_name}: {exc!r}")


# --------------------------------------------- 6. stage A public-API I/O recording
_fsq_mod = importlib.import_module("ardy.model.autoencoder.fsq")
FSQ = _fsq_mod.FSQVAETransformer

for _meth in ("tokenize", "detokenize", "requantize"):
    _orig = getattr(FSQ, _meth)

    def _wrap(orig=_orig, meth=_meth):
        def wrapper(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                tag = f"fsq_{meth}_{len(STATE.setdefault('fsq_' + meth, []))}"
                inp = args[0] if args else next(iter(kwargs.values()), None)
                _save(tag + "_in.npy", _np(inp) if isinstance(inp, torch.Tensor) else None)
                if isinstance(out, dict):
                    for k, v in out.items():
                        _save(f"{tag}_out_{k}.npy", _np(v))
                    STATE["fsq_" + meth].append(
                        {"tag": tag, "out_keys": {k: list(v.shape) for k, v in out.items()}}
                    )
                else:
                    _save(tag + "_out.npy", _np(out))
                    STATE["fsq_" + meth].append({"tag": tag, "out_shape": list(out.shape)})
                _note(f"[hook] {meth} -> {tag}")
            except Exception as exc:
                STATE["errors"].append(f"fsq {meth}: {exc!r}")
            return out

        return wrapper

    setattr(FSQ, _meth, _wrap())


# ------------------------------------------- 7. stage B denoiser / CFG I/O recording
_den_mod = importlib.import_module("ardy.model.auto_latent_twostage_denoiser")
Den = _den_mod.AutoLatentTwostageDenoiser
_orig_den_forward = Den.forward


def den_forward(self, *args, **kwargs):
    idx = len(STATE.setdefault("den_calls", []))
    out = _orig_den_forward(self, *args, **kwargs)
    try:
        STATE["den_calls"].append({"idx": idx, "out_shape": list(out.shape)})
        _save(f"denoiser_raw_{idx}.npy", _np(out))
        _note(f"[hook] denoiser raw {idx} -> {tuple(out.shape)}")
    except Exception as exc:
        STATE["errors"].append(f"denoiser: {exc!r}")
    return out


Den.forward = den_forward

_cfg_mod = importlib.import_module("ardy.model.cfg")
CFG = _cfg_mod.AutoLatentClassifierFreeGuidedModelSeparated
_orig_cfg_forward = CFG.forward


def cfg_forward(self, *args, **kwargs):
    idx = len(STATE.setdefault("cfg_calls", []))
    out = _orig_cfg_forward(self, *args, **kwargs)
    try:
        STATE["cfg_calls"].append({"idx": idx, "out_shape": list(out.shape)})
        _save(f"cfg_combined_{idx}.npy", _np(out))
        _note(f"[hook] cfg combined {idx} -> {tuple(out.shape)}")
    except Exception as exc:
        STATE["errors"].append(f"cfg: {exc!r}")
    return out


CFG.forward = cfg_forward


# ------------------------------------------------- 8. per-denoising-step I/O trace
_orig_denoising_step = _ardy_model.Ardy.denoising_step


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
        "t": _np(t).tolist() if t is not None else None,
        "x_shape": list(x.shape),
        "generation_token_mask": _np(gen_token_mask).astype(np.int8) if gen_token_mask is not None else None,
    }
    STATE["steps"].append(rec)
    _save(f"denoise_step{idx}_x_in.npy", x_in)
    _save(f"denoise_step{idx}_x_out.npy", _np(out))
    _note(f"[hook] denoise step {idx} t={rec['t']}")
    return out


_ardy_model.Ardy.denoising_step = denoising_step


# ------------------------------- 9. final features + one tokenizer round-trip probe
_orig_call = _ardy_model.Ardy.__call__


def _call(
    self,
    texts,
    num_frames,
    num_denoising_steps,
    pad_mask,
    first_heading_angle,
    motion_mask,
    observed_motion,
    *args,
    **kwargs,
):
    out = _orig_call(
        self,
        texts,
        num_frames,
        num_denoising_steps,
        pad_mask,
        first_heading_angle,
        motion_mask,
        observed_motion,
        *args,
        **kwargs,
    )
    _save("motion_normalized.npy", _np(out))
    try:
        with open(os.path.join(DUMP, "call_meta.json"), "w") as fh:
            json.dump(
                {
                    "texts": list(texts),
                    "num_frames": int(num_frames),
                    "num_denoising_steps": int(num_denoising_steps),
                    "first_heading_angle": _np(first_heading_angle).tolist()
                    if first_heading_angle is not None
                    else None,
                    "cfg_weight": kwargs.get("cfg_weight"),
                    "crop_history_length": kwargs.get("crop_history_length"),
                    "gen_horizon_len": int(self.gen_horizon_len),
                    "num_frames_per_token": int(self.num_frames_per_token),
                    "num_base_steps": int(self.diffusion.num_base_steps),
                    "out_shape": list(out.shape),
                    "out_dtype": str(out.dtype),
                    "pad_mask_shape": list(pad_mask.shape),
                },
                fh,
                indent=2,
            )
    except Exception as exc:
        STATE["errors"].append(f"call meta: {exc!r}")
    _note(f"[hook] dumped motion_normalized {tuple(out.shape)}")

    # ---- one read-only tokenizer encode -> decode probe on the gold clip itself
    try:
        motion_len = pad_mask.sum(dim=-1)
        n_tok = out.shape[1] // self.num_frames_per_token
        num_gen_frames = n_tok * self.num_frames_per_token
        motion_pad_mask = (
            torch.arange(num_gen_frames, device=out.device)[None, :] < motion_len[:, None]
        )
        _save("probe_input_features.npy", _np(out))
        _save("probe_pad_mask.npy", _np(motion_pad_mask))
        hybrid2, tok_mask = self.hybrid.get_hybrid_motion_from_explicit(
            out, motion_len, motion_pad_mask
        )
        _save("probe_hybrid.npy", _np(hybrid2))
        _save("probe_token_mask.npy", _np(tok_mask))
        explicit2 = self.hybrid.get_explicit_motion_from_hybrid(
            hybrid2, motion_pad_mask, motion_len
        )
        _save("probe_roundtrip_features.npy", _np(explicit2))
        _note("[hook] tokenizer round-trip probe done")
    except Exception as exc:
        STATE["errors"].append(f"probe: {exc!r}")
        _note(f"[hook] probe failed: {exc!r}")

    return out


_ardy_model.Ardy.__call__ = _call


# --------------------------------------------------------- 10. install at first use
_orig_load_model = _lm.load_model


def load_model(*args, **kwargs):
    model = _orig_load_model(*args, **kwargs)
    try:
        _install_module_hooks(model)
    except Exception as exc:
        STATE["errors"].append(f"install hooks: {exc!r}")
    return model


# `scripts/generate.py` does `from ardy.model import load_model`, which was bound
# to the *original* function when `ardy.model` was first imported. Patch both the
# submodule attribute and the package attribute so the wrapper is the one used.
_lm.load_model = load_model
import sys as _sys

_sys.modules["ardy.model"].load_model = load_model
_note("[hook] load_model patched (module + package attribute)")


# --------------------------------------------------------------- 11. flush at exit
def _flush():
    try:
        payload = {
            "randn": [{k: v for k, v in r.items() if k != "_data"} for r in STATE["randn"]],
            "denoise_steps": STATE["steps"],
            "fsq_tokenize": STATE.get("fsq_tokenize"),
            "fsq_detokenize": STATE.get("fsq_detokenize"),
            "fsq_requantize": STATE.get("fsq_requantize"),
            "denoiser_calls": STATE.get("den_calls"),
            "cfg_calls": STATE.get("cfg_calls"),
            "modules": {
                k: [{"in": c["in"], "out": c["out"]} for c in v["calls"]]
                for k, v in STATE["modules"].items()
            },
            "errors": STATE["errors"],
        }
        with open(os.path.join(DUMP, "trace.json"), "w") as fh:
            json.dump(payload, fh, indent=2)

        for i, r in enumerate(STATE["randn"]):
            _save(f"randn_{i}.npy", r["_data"])
        for r in STATE["steps"]:
            if r["generation_token_mask"] is not None:
                _save(f"denoise_step{r['step']}_gen_token_mask.npy", r["generation_token_mask"])
        if STATE["buffers"]:
            np.savez(os.path.join(DUMP, "diffusion_buffers.npz"), **STATE["buffers"])

        # module tensors, one directory per module to keep names readable
        for full_name, bucket in STATE["modules"].items():
            for ci, call in enumerate(bucket["calls"]):
                if "_in" not in call:
                    continue
                safe = full_name.replace(".", "__")
                for ai, arr in enumerate(call.get("_in") or []):
                    _save(f"modules/{safe}__call{ci}__in{ai}.npy", arr)
                if call.get("_out") is not None:
                    _save(f"modules/{safe}__call{ci}__out.npy", call["_out"])
        _note(f"[hook] flush done, errors={len(STATE['errors'])}")
    except Exception as exc:
        _note(f"[hook] flush failed: {exc!r}\n{traceback.format_exc()}")


try:
    os.makedirs(os.path.join(DUMP, "modules"), exist_ok=True)
except Exception:
    pass

atexit.register(_flush)
_note("[hook] installed")
