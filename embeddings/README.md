# Cached text embeddings

> **Superseded in part (2026-10-13).** `ardy-mlx` now reaches a **local MLX text
> encoder** through a subprocess, so the viewer can encode a prompt it has never
> seen. The authoritative description of the cache layout, the `(prompt_hash,
> encoder_id)` key and the source-precedence rule is **[`docs/VIEWER.md` §3](../docs/VIEWER.md)**.
> This file is kept as the reference-vector recipe — adding a *reference* vector
> still requires the reference machine.

The port consumes a *precomputed* `[B, L, 4096]` float32 embedding. This directory
holds the cache's **documentation**; the files themselves live under
`artifacts/embeddings/` (a weight-derived runtime cache, kept out of the source tree
and out of git).

```
artifacts/embeddings/          the runtime cache the viewer reads
  <prompt-hash>.npy            a REFERENCE vector, [L, 4096] or [B, L, 4096] float32
  <prompt-hash>.json           sidecar: {"prompt": "<the exact text>", ...}
  <encoder-id>/                a LOCALLY-encoded vector, one directory per encoder
    <prompt-hash>.npy            e.g. mlx-llm2vec-4-bit-group-32-…/
    <prompt-hash>.json
embeddings/README.md           this file
```

`<prompt-hash>` is **`prompt_hash(prompt)`** — the first **16 hex characters** of
`sha256(prompt.encode("utf-8"))`. That is the same convention `save_outputs` uses
for its output file names (contract S7 / decision 0002 ruling 4). Using the full
64-character digest here would give one prompt two names in one repository, so
this directory does not do that. (The viewer displays the full sha256 of the
*file* separately, as `embedding_sha256` — that one identifies bytes, not text.)

The cache directory can be pointed elsewhere with
`scripts/ardy_viewer.py --embeddings-dir <path>`. `.npy` / `.npz` are ignored by
`.gitignore` wherever they live; see the rules there.

## Adding a reference vector (this still needs the reference machine)

A **reference** vector — the only kind that makes a golden comparison meaningful —
has to be produced on the machine that can run the reference text encoder: LLM2Vec +
Llama-3-8B, ~16 GB of bf16 weights, which is exactly the piece this port does not
have. Do it **there** (the reference repo at `nv-tlabs/ardy` @ `693f74d`,
`TEXT_ENCODER_MODE=local`, `TEXT_ENCODER_DEVICE=cpu` on a small box — see
`artifacts/golden-run/RUN-RECORD.md` §2 for the measured configuration).

(For a **local** vector you do not need this at all: type the prompt into the viewer.
It runs `~/dev/llm2vec-mlx`'s `scripts/encode.py` in that repository's own `.venv`
and writes `artifacts/embeddings/<encoder-id>/<prompt-hash>.npy`. See
`docs/VIEWER.md` §3.)

```python
# on the reference machine, in the reference repo
import hashlib, numpy as np
from ardy.utils.load_model import load_text_encoder   # the loader the golden run used

prompt = "A person walks forward."
encoder = load_text_encoder()          # then .to(device=..., dtype=...) as that box requires
feat, mask = encoder([prompt])         # [1, L, 4096] float32, [1, L] bool

digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]   # <- NOT [:64]
np.save(f"{digest}.npy", np.asarray(feat, dtype=np.float32))
```

Then copy the `.npy` into the **top level** of `artifacts/embeddings/` — not into a
sub-directory, which is what marks a vector as locally encoded — and write the
sidecar:

```json
{
  "prompt": "A person walks forward.",
  "prompt_hash": "e9cf8141cac79f35",
  "embedding_sha256": "<shasum -a 256 e9cf8141cac79f35.npy>",
  "encoder": "reference-llm2vec-bf16",
  "reference_source": "pancat-linux-ci: LLM2Vec + Meta-Llama-3-8B-Instruct …",
  "source": "text_feat.npy from the 2.0 s golden run"
}
```

The `encoder` field is what makes the entry a *reference* vector. It is optional for
backward compatibility — a sidecar with no `encoder` field is also read as a reference
vector, because every cache file that predates the field was one — but a new file
should say so.

Then restart the viewer (`scripts/ardy_viewer.py`) — the prompt list is rescanned
on every `/api/health`, so a reload of the page is enough if the server stays up.

### What the sidecar does and does not give you

* The viewer **uses** `prompt` to label the dropdown and to resolve a request that
  passes prompt *text* instead of a hash. Text is resolved by hashing it, so the
  text path and the hash path agree by construction.
* The sidecar is **documentation**. Nothing verifies that the `.npy` bytes really
  are the encoding of that text. If the label is wrong the viewer will show a
  right-looking clip under a wrong prompt, and no test can catch it. Recompute the
  embedding on the reference machine if you need that guarantee.
* Without a sidecar, the entry still appears but is labelled
  `<unknown prompt for <hash>>`; it can still be selected **by hash**.
* A **locally** encoded vector is identified by where it lives (a sub-directory) and
  by its `encoder_id` + `quantization` fields, both of which the viewer writes. The
  directory wins if the two disagree, and a contradicting sidecar is refused.

## Why a miss is never a fallback

If a requested prompt has no cached embedding the server does **not** substitute
another prompt's embedding and does **not** fall back to a default. It either
encodes the prompt locally (`docs/VIEWER.md` §3) or fails by name:
`local_encoder_unavailable` / `local_encoder_failed` when an encoder is configured,
`no_cached_embedding_for_prompt` when it is not. This repository's rule: a wrong
conditioning signal produces plausible-looking motion, so a silent fallback is
indistinguishable from success — the one failure mode that must be impossible.

## Current contents

| prompt hash | prompt | embedding sha256 | shape |
|---|---|---|---|
| `e9cf8141cac79f35` | `A person walks forward.` | `9105d78883196369a026375be7595c2f9e5804729980f2223bf716110c322693` | `[1, 1, 4096]` |

That is the only prompt the one authorised reference run encoded, and therefore the
only prompt whose clip the viewer can compare with a *golden reference*. Locally
encoded vectors for other prompts accumulate under `<encoder-id>/` as they are used;
they are a different condition and are never compared with the golden clips.
