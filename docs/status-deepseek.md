# 状态 · DeepSeek（A：权重转换 / 模型前向 / 数值对齐）

- **当前轮次**：A 侧第 1 轮（只做不需要跑模型的三件事）。起于 HEAD `8b272b1`。
- **已完成**
  1. **checkpoint 许可核查（硬门）**：**建议通过**。4 个 checkpoint 均为 **NVIDIA Open Model License Agreement**
     （模型卡明示 `This model is ready for commercial or non-commercial use.`；协议授权 `create derivative works of / distribute / sell`）。
     全文见 `docs/decisions/0002-checkpoint-terms.md`（**提案，待人类批准**）。
  2. **契约六项 `?` 全部补齐**（每项带 `文件:行`）：见 `docs/motion-contract.md`
     第 1–6 节；磁盘布局已**定稿 v1**（第 7 节）。
  3. **接口常量落地**：`src/ardy_mlx/io/__init__.py`（`FeatureLayout` + `PRESETS` + 自校验 `assert`）。
  4. **只读权重探针**：`scripts/ardy_inspect.py`（纯 stdlib 解析 safetensors header，不 import torch/mlx、不写文件）。
  5. **测试**：`tests/test_io_layout.py`、`tests/test_io_inspect.py` —— `26 passed`。
  6. **环境**：`.venv` 已建（Python 3.13.12 / mlx 0.32.3 / numpy 2.5.4 / pytest 9.1.1），
     `./.venv/bin/python scripts/check_env.py` → **READY**。
- **正在动的文件**：本文件。
- **下一步（等人类批准后才动）**
  1. 批准 `docs/decisions/0002` → 更新 `NOTICE:12-14` 的措辞（**当前不动**，它在禁改清单里）。
  2. 批准后：按 `docs/motion-contract.md` §3 的形状逐张量核对，写 `scripts/ardy_convert.py`
     （tokenizer 键前缀 `pose_net.`、denoiser 键前缀 `denoiser.`，均为 `F32`，共 428 张量）。
  3. 再等人类**单独批准**在那台 Linux 机器上跑一次 `generate.py` 取金标准输出（golden）。
- **阻塞 / 需要人类裁定**
  - **B1**：`docs/decisions/0002` 提案待批（硬门虽建议通过，但按纪律不自行放行）。
  - **B2**：金标准输出是否**含 MotionCorrection 后处理**。参考默认对 Core **开启**后处理，
    而该 C++ 扩展在本机未编译（`postprocess.py:307-314` 缺扩展直接 `raise`）。
    A 建议第一版**两侧都 `--no-postprocess`** 以求同口径，需人类确认。
  - **B3**：MotionBricks/Unity 的坐标与旋转转换**待定**（上游无任何导出器；`README.md:263` 只是相关工作总结链接）。
  - **B4**：模型卡写 326 M 参数，实测两个权重文件合计 **191,253,304**（结构自洽但口径不同，未向上游确认）。
- **纪律留痕**：未跑推理、未跑 `generate.py`、未转换/未下载任何权重、未 push。
  那台 Linux 机器全程只读；过程中曾误建 2 个 `/tmp` 临时文件，**已即刻删除并复核不存在**。
  未改动 `NOTICE` / `LICENSE` / `README*` / `pyproject.toml` / `.gitignore`，未触碰 B 侧文件。
