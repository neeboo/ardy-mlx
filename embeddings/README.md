# Cached text embeddings

`ardy-mlx` v1 **has no text encoder**. The port consumes a *precomputed*
`[B, L, 4096]` float32 embedding, so the browser viewer can only offer the prompts
that already have one of these files. This directory holds that cache's
**documentation**; the files themselves live under `artifacts/embeddings/`
(a weight-derived runtime cache, kept out of the source tree and out of git).

```
artifacts/embeddings/          the runtime cache the viewer reads
  <prompt-hash>.npy            the embedding, [L, 4096] or [B, L, 4096] float32
  <prompt-hash>.json           optional sidecar: {"prompt": "<the exact text>", ...}
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

## Adding a prompt

The embedding has to be produced on the machine that can run the reference text
encoder — LLM2Vec + Llama-3-8B, ~16 GB of bf16 weights, which is exactly the piece
this port does not have. Do it **there** (the reference repo at `nv-tlabs/ardy`
@ `693f74d`, `TEXT_ENCODER_MODE=local`, `TEXT_ENCODER_DEVICE=cpu` on a small box —
see `artifacts/golden-run/RUN-RECORD.md` §2 for the measured configuration):

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

Then copy the `.npy` in here (the bytes are the same on any machine) and write the
sidecar:

```json
{
  "prompt": "A person walks forward.",
  "prompt_hash": "e9cf8141cac79f35",
  "embedding_sha256": "<shasum -a 256 e9cf8141cac79f35.npy>",
  "source": "text_feat.npy from the 2.0 s golden run"
}
```

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

## Why a miss is an error, not a fallback

If a requested prompt has no cached embedding the server answers
`400 no_cached_embedding_for_prompt`, naming the file it looked for and listing the
hashes it does have. It never substitutes another prompt's embedding and never
falls back to a default. This repository's rule: a wrong conditioning signal
produces plausible-looking motion, so a silent fallback is indistinguishable from
success — the one failure mode that must be impossible.

## Current contents

| prompt hash | prompt | embedding sha256 | shape |
|---|---|---|---|
| `e9cf8141cac79f35` | `A person walks forward.` | `9105d78883196369a026375be7595c2f9e5804729980f2223bf716110c322693` | `[1, 1, 4096]` |

That is the only prompt the one authorised reference run encoded, and therefore
the only clip the viewer can produce a *golden* comparison for. Any other prompt
can be added by the recipe above (generate, then compare against nothing — there
is no reference run for it).
