中文: [README.zh-CN.md](README.zh-CN.md)
# ardy-mlx

Reproducing **ARDY** (NVIDIA, SIGGRAPH 2026 — human motion generation with "autoregressive diffusion + hybrid representation") using **MLX** on Apple Silicon.
**The goal is a library**: load the weights locally, produce a full-length motion clip, and align with the reference implementation bit-exact / within tolerance.
**Not a goal**: real-time streaming (known to be expensive: short chunks, multi-step diffusion, and an 8B text encoder that has to stay resident);
how good the real-time experience is **does not affect whether this repository ships**.

Upstream: `nv-tlabs/ardy` (code Apache-2.0; **checkpoints have their own terms**, see `NOTICE` and `docs/decisions/`).

## Scope (v1 covers the batch path only)
The reference implementation has two entry points: `scripts/generate.py` (one text prompt → one whole motion clip) and `scripts/run_demo.py` (streaming interaction).
**v1 aligns only the `generate.py` path**: fixed duration, the whole clip in one shot, no interaction loop.

**Key cost-cutting decision: do not port the text encoder.** ARDY's text side is LLM2Vec + Llama-3-8B (heavy, and only validated on a CUDA machine we do not own);
v1 **consumes only precomputed text embeddings** (computed once and saved to disk on that machine); the MLX side handles only the motion model and sampling.
That gives alignment a deterministic input (same embedding + same seed) and keeps the v1 workload under control.

## Roadmap (every step needs a falsifiable acceptance check)
1. Verify and record checkpoint licensing (**stop if it does not pass**, see `docs/decisions/`)
2. Weight conversion (reference checkpoint → MLX-readable), verifying name/shape/bytes tensor by tensor
3. Sampler and autoregressive chunk loop (diffusion denoising + horizon advance)
4. **Full-clip alignment**: same embedding + same seed, compared array by array against the reference `generate.py` output (tolerance and rationale must be written into the report)
5. Measured performance (time per chunk / per clip, on both CPU and Metal)
6. (Optional) Text encoder: Core ML, or reuse the existing MLX Llama implementation

Work order: [`docs/brief-claude-sampling.md`](docs/brief-claude-sampling.md).
Interface first: the dependency goes only through `docs/motion-contract.md` (the motion contract) and the constants exported by `src/ardy_mlx/io/`; changing the contract is a contract change.

## Environment
A repo-local `.venv` (see `docs/decisions/0001-python-environment.md`); run everything with `./.venv/bin/python`.
The reference implementation and the golden outputs are on that Linux/CUDA machine (read-only for us; a single run needs human approval).

## License
This repository's code is Apache-2.0; **until the checkpoint terms are cleared, we do not convert or redistribute any weights** (see `NOTICE`).
