#!/usr/bin/env python3
"""Is this machine ready? Reports Apple Silicon, Python version and MLX."""
import platform, sys

def main() -> int:
    ok = True
    machine, system = platform.machine(), platform.system()
    print(f"platform      {system} {platform.mac_ver()[0]} ({machine})")
    if system != "Darwin" or machine != "arm64":
        print("  ! MLX targets Apple Silicon (Darwin/arm64)"); ok = False
    print(f"python        {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ! need Python 3.11+"); ok = False
    try:
        import mlx.core as mx, numpy as np
        print(f"mlx           {getattr(mx, '__version__', 'present')}")
        a = mx.array(np.arange(6, dtype=np.float32).reshape(2, 3))
        print(f"mlx smoke     sum={float(mx.sum(a)):.1f} (expected 15.0)")
        print(f"device        {mx.default_device()}")
    except Exception as exc:
        print(f"mlx           MISSING ({exc.__class__.__name__}: {exc})")
        print("  -> pip install mlx"); ok = False
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
