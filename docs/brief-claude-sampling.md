# 交给 Claude Code 的工单（B 侧：采样与自回归循环）

**工作目录**：`~/dev/ardy-mlx`。先读 `README.md`、`NOTICE`（只走源码路线、checkpoint 单独核）、
`docs/motion-contract.md`（接口，未定稿前不要动代码）、`docs/decisions/0001-python-environment.md`（用 `./.venv/bin/python`）。

## 你只写这些文件（与 A 零重叠）
- `src/ardy_mlx/sampling/**`（新建）
- `scripts/ardy_generate.py`
- `tests/test_sampling_*.py`、`tests/test_generate_*.py`
- `docs/status-claude.md`

**绝对不要碰**：`src/ardy_mlx/model/**`、`src/ardy_mlx/io/**`、`scripts/ardy_{convert,align,inspect}.py`、
`tests/test_model_*.py`、`tests/test_io_*.py`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml`、
`docs/motion-contract.md`、`docs/decisions/**`。

## 任务
1. **等契约**：`docs/motion-contract.md` 里那六项由 A 补齐后，按它实现，不许自行改字段名/维度/顺序；
   不够用就在 `docs/status-claude.md` 写"需要新增字段＋原因"，交人类裁定。
2. **采样器**：按契约里的调度（步数、噪声表、guidance 与否）实现扩散去噪；输入 = 文本 embedding（预计算文件）＋ seed ＋ 时长，
   输出 = 整段动作（**第一版只做批处理路径**，不做流式）。
3. **自回归分块循环**：按 horizon 推进，把上一块的末尾作为下一块的上下文；上下文窗口怎么裁**按契约**，不要自创。
4. **CLI**：`scripts/ardy_generate.py`，参数与参考 `generate.py` 对齐（prompt embedding 文件、`--duration`、`--seed`、`--model`、`--out` 必填），
   输出与契约同形状。
5. **测试**：形状、mask、一次前向、一次极小规模采样循环（用合成权重或 A 的小样本权重）、以及"改坏就红"的负对照。

## 判据
- 不许新造正确性判据：对齐以 A 的 `scripts/ardy_align.py` 为准（对参考金标准产物逐数组比对）。
- 每个结论带证据：真实输出、数字、失败就贴原始报错；看不清就写"未验证"。
- 不做长训练；不 push；不装系统级依赖；不改别人的文件。

## 汇报
做了什么、跑了什么验证、数字、下一步、阻塞。
