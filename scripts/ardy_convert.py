#!/usr/bin/env python3
"""Convert ARDY reference checkpoints to MLX-native safetensors, and prove it.

What "conversion" means here, precisely: the released checkpoints are
``safetensors`` with all-``F32`` payloads and PyTorch-style state-dict keys
(``pose_net.*`` / ``denoiser.*``). MLX reads that container natively, so no
*arithmetic* conversion is required or performed -- silently retyping to bf16 or
renaming keys would only introduce a place for the two implementations to
disagree. The conversion therefore does three things, all verifiable:

1. **Inventory parity** -- every tensor name, dtype and shape is compared against
   the source header, and the tensor/parameter totals are checked against the
   numbers recorded from the reference (docs/motion-contract.md S3.0:
   204 tensors / 35,430,052 params for the tokenizer and 224 / 155,823,252 for
   the denoiser).
2. **MLX-native re-emission** -- the weights are loaded through ``mlx.core`` and
   written back out as a canonical safetensors file for the MLX side
   (``<ckpt>/mlx/<name>.mlx.safetensors``), with provenance in ``__metadata__``.
3. **Byte-exact payload verification** -- the source file bytes and the converted
   file bytes are compared *per tensor*, byte for byte, over the raw payload
   ranges taken from each header. Any single differing byte is reported with the
   tensor name and the offset, and makes the command exit non-zero.

Outputs written by default (all inside the gitignored checkpoint directory
except the two reports under artifacts/):

    <ckpt>/mlx/tokenizer.mlx.safetensors
    <ckpt>/mlx/denoiser.mlx.safetensors
    artifacts/weight-manifest.json     tensor list + totals (committable)
    artifacts/weight-verify.json       byte-exactness results (committable)

Examples
--------
    ./.venv/bin/python scripts/ardy_convert.py checkpoints/ARDY-Core-RP-20FPS-Horizon40
    ./.venv/bin/python scripts/ardy_convert.py <ckpt> --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ardy_mlx.io.safetensors import (  # noqa: E402
    DTYPE_BYTES,
    SafetensorsError,
    param_count,
    read_header,
    read_payload,
    tensors_of,
    validate,
)

#: Numbers recorded from the reference checkpoint inspection
#: (docs/motion-contract.md S3.0 -- "A side verified with scripts/ardy_inspect.py").
EXPECTED = {
    "tokenizer.safetensors": {"tensors": 204, "params": 35_430_052, "dtype": "F32"},
    "denoiser.safetensors": {"tensors": 224, "params": 155_823_252, "dtype": "F32"},
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def first_difference(a: bytes, b: bytes) -> int | None:
    """Index of the first differing byte, or None when the two are identical."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def convert_one(src: Path, out_dir: Path, *, verify_only: bool) -> Dict:
    """Convert and byte-verify a single checkpoint file. Returns a report dict."""
    import mlx.core as mx

    result: Dict = {"source": str(src), "name": src.name}
    header, header_len, file_size = read_header(src)
    tensors = tensors_of(header)
    problems = validate(header, header_len, file_size, src)

    total_params = sum(param_count(e) for e in tensors.values())
    dtypes = sorted({e["dtype"] for e in tensors.values()})
    result["source_sha256"] = sha256_file(src)
    result["source_bytes"] = file_size
    result["source_header_bytes"] = header_len
    result["source_metadata"] = header.get("__metadata__")
    result["tensor_count"] = len(tensors)
    result["total_params"] = total_params
    result["dtypes"] = dtypes
    result["header_problems"] = problems

    expected = EXPECTED.get(src.name)
    result["expected"] = expected
    if expected:
        result["matches_recorded_totals"] = (
            len(tensors) == expected["tensors"]
            and total_params == expected["params"]
            and dtypes == [expected["dtype"]]
        )
    else:
        result["matches_recorded_totals"] = None

    out_path = out_dir / f"{src.stem}.mlx.safetensors"
    result["converted_path"] = str(out_path)

    if not verify_only:
        arrays = dict(mx.load(str(src)))
        # --- inventory parity against the header we parsed ourselves -----------
        missing = sorted(set(tensors) - set(arrays))
        extra = sorted(set(arrays) - set(tensors))
        shape_mismatch: List[str] = []
        for name, entry in tensors.items():
            if name not in arrays:
                continue
            got = tuple(int(d) for d in arrays[name].shape)
            want = tuple(int(d) for d in entry["shape"])
            if got != want:
                shape_mismatch.append(f"{name}: header {want} vs mlx {got}")
        result["mlx_missing"] = missing
        result["mlx_extra"] = extra
        result["mlx_shape_mismatch"] = shape_mismatch

        mx.save_safetensors(
            str(out_path),
            arrays,
            metadata={
                "format": "pt",
                "converter": "ardy-mlx scripts/ardy_convert.py",
                "source_file": src.name,
                "source_sha256": result["source_sha256"],
                "dtype": "preserved (F32)",
            },
        )
        del arrays

    # --- byte-exact payload verification --------------------------------------
    dst_header, dst_header_len, dst_size = read_header(out_path)
    dst_tensors = tensors_of(dst_header)
    result["converted_sha256"] = sha256_file(out_path)
    result["converted_bytes"] = dst_size
    result["converted_header_bytes"] = dst_header_len
    result["converted_metadata"] = dst_header.get("__metadata__")
    result["converted_tensor_count"] = len(dst_tensors)

    only_src = sorted(set(tensors) - set(dst_tensors))
    only_dst = sorted(set(dst_tensors) - set(tensors))
    meta_mismatch: List[str] = []
    byte_mismatch: List[Dict] = []
    bytes_compared = 0
    exact = 0

    for name in sorted(set(tensors) & set(dst_tensors)):
        s, d = tensors[name], dst_tensors[name]
        if s["dtype"] != d["dtype"] or list(s["shape"]) != list(d["shape"]):
            meta_mismatch.append(
                f"{name}: {s['dtype']}{s['shape']} -> {d['dtype']}{d['shape']}"
            )
        a = read_payload(src, name, tensors)
        b = read_payload(out_path, name, dst_tensors)
        bytes_compared += len(a) + len(b)
        diff = first_difference(a, b)
        if diff is None:
            exact += 1
        else:
            byte_mismatch.append(
                {
                    "tensor": name,
                    "first_diff_offset": diff,
                    "src_len": len(a),
                    "dst_len": len(b),
                    "src_sha256": hashlib.sha256(a).hexdigest(),
                    "dst_sha256": hashlib.sha256(b).hexdigest(),
                }
            )

    result["only_in_source"] = only_src
    result["only_in_converted"] = only_dst
    result["metadata_mismatch"] = meta_mismatch
    result["byte_mismatch_count"] = len(byte_mismatch)
    result["byte_mismatch_examples"] = byte_mismatch[:10]
    result["tensors_byte_identical"] = exact
    result["tensors_compared"] = len(set(tensors) & set(dst_tensors))
    result["payload_bytes_compared"] = bytes_compared

    result["ok"] = (
        not problems
        and not only_src
        and not only_dst
        and not meta_mismatch
        and not byte_mismatch
        and exact == len(tensors)
        and result.get("matches_recorded_totals") in (True, None)
        and not result.get("mlx_shape_mismatch")
    )
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ARDY checkpoints to MLX-native safetensors with byte-exact verification."
    )
    parser.add_argument("checkpoint_dir", help="directory holding tokenizer/denoiser .safetensors")
    parser.add_argument("--out-dir", default=None, help="converted output dir (default <ckpt>/mlx)")
    parser.add_argument("--manifest", default="artifacts/weight-manifest.json")
    parser.add_argument("--report", default="artifacts/weight-verify.json")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing conversion")
    parser.add_argument("--files", nargs="*", default=None, help="subset of files to convert")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.is_dir():
        print(f"no such checkpoint dir: {ckpt}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else ckpt / "mlx"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.files or ["tokenizer.safetensors", "denoiser.safetensors"]
    reports = []
    ok = True
    for name in names:
        src = ckpt / name
        if not src.is_file():
            print(f"missing: {src}", file=sys.stderr)
            return 2
        try:
            rep = convert_one(src, out_dir, verify_only=args.verify_only)
        except (SafetensorsError, RuntimeError) as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            return 1
        reports.append(rep)
        ok = ok and rep["ok"]
        print(
            f"{name}: {rep['tensor_count']} tensors, {rep['total_params']:,} params, "
            f"dtypes={rep['dtypes']}, {rep['tensors_byte_identical']}/{rep['tensors_compared']} "
            f"byte-identical, mismatches={rep['byte_mismatch_count']}, "
            f"recorded-totals-match={rep['matches_recorded_totals']}"
        )

    manifest = {
        "converter": "scripts/ardy_convert.py",
        "checkpoint_dir": str(ckpt),
        "files": [
            {
                "name": r["name"],
                "source_sha256": r["source_sha256"],
                "converted_sha256": r["converted_sha256"],
                "tensor_count": r["tensor_count"],
                "total_params": r["total_params"],
                "dtypes": r["dtypes"],
                "tensors": [
                    {
                        "name": t,
                        "dtype": e["dtype"],
                        "shape": list(e["shape"]),
                        "params": param_count(e),
                    }
                    for t, e in sorted(tensors_of(read_header(ckpt / r["name"])[0]).items())
                ],
            }
            for r in reports
        ],
        "totals": {
            "tensors": sum(r["tensor_count"] for r in reports),
            "params": sum(r["total_params"] for r in reports),
        },
        "reference": {"repo": "nv-tlabs/ardy", "commit": "693f74d"},
        "note": (
            "Weights themselves are NOT committed (checkpoints/ is gitignored); this manifest "
            "lists names/shapes only. F32 is preserved exactly: no dtype change, no key rename."
        ),
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    Path(args.report).write_text(json.dumps({"ok": ok, "files": reports}, indent=2) + "\n")

    print(f"manifest -> {args.manifest}")
    print(f"report   -> {args.report}")
    print("RESULT: OK" if ok else "RESULT: MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
