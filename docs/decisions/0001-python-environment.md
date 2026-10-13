# 0001 · Python 环境（沿用 motionbricks-mlx 的约定）

**决定**：仓库内 `.venv`，Homebrew `python@3.13` 创建；两个 agent 与人类**只用** `./.venv/bin/python`。

```sh
cd ~/dev/ardy-mlx
python3.13 -m venv .venv
./.venv/bin/python -m pip install -U pip mlx numpy pytest
./.venv/bin/python scripts/check_env.py     # 期望最后一行 READY
```

**理由**：与 `motionbricks-mlx` 一致，避免两个 agent 各建一套环境。
**注意**：MLX 只负责**本机推理**；参考实现与金标准产物在那台 Linux/CUDA 机器上，
**不在这台机器上跑**（那台机器我们只读，且一次运行需人类批准）。
