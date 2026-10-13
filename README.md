# ardy-mlx

在 Apple Silicon 上用 **MLX** 复现 **ARDY**（NVIDIA，SIGGRAPH 2026，"自回归扩散 ＋ 混合表示"的人体动作生成）。
**目标是一个库**：能在本机加载权重、跑出整段动作、并与参考实现逐位/逐容差对齐。
**不作为目标**：实时流式体验（已知成本高：块短、扩散多步、还要常驻 8B 文本编码器），
实时效果好坏**不影响本仓库是否交付**。

上游：`nv-tlabs/ardy`（代码 Apache-2.0；**checkpoint 各自条款**，见 `NOTICE` 与 `docs/decisions/`）。

## 范围（第一版只做批处理路径）
参考实现有两条入口：`scripts/generate.py`（一句文本 → 一整段动作）与 `scripts/run_demo.py`（流式交互）。
**第一版只对齐 `generate.py` 那条**：固定时长、一次性出整段、没有交互循环。

**关键降本决定：不搬文本编码器。** ARDY 的文本侧是 LLM2Vec ＋ Llama-3-8B（重且只在我们不拥有的 CUDA 机上验证过）；
第一版**只吃预先算好的文本 embedding**（在那台机器上算一次存盘），MLX 侧只负责动作模型与采样。
这样对齐有确定输入（同一 embedding ＋ 同一 seed），也把第一版的工作量压到可控。

## 路标（每步都要有可失败的验收）
1. 核 checkpoint 许可并记录（**不通过则停止**，见 `docs/decisions/`）
2. 权重转换（参考 checkpoint → MLX 可读），逐张量核对名字/形状/字节
3. 采样器与自回归分块循环（扩散去噪 + horizon 推进）
4. **整段对齐**：同一 embedding ＋ 同一 seed，与参考 `generate.py` 的输出逐数组比对（容差与理由要写进报告）
5. 性能实测（每块/每段耗时，CPU 与 Metal 两端）
6. （可选）文本编码器：Core ML 或复用 MLX 已有的 Llama 实现

## 分工（一份文件同一时间只有一个写手）
| 归属 | 谁 | 文件范围 |
|---|---|---|
| A. 权重转换 / 模型前向 / 数值对齐 | **DeepSeek** | `src/ardy_mlx/model/**`、`src/ardy_mlx/io/**`、`scripts/ardy_{convert,align,inspect}.py`、`tests/test_model_*.py`、`tests/test_io_*.py`、`README.md`、`docs/status-deepseek.md` |
| B. 采样器 / 自回归循环 / 生成 CLI | **Claude Code** | `src/ardy_mlx/sampling/**`、`scripts/ardy_generate.py`、`tests/test_sampling_*.py`、`tests/test_generate_*.py`、`docs/status-claude.md` |
| C. 冻结区 | **人类** | `LICENSE`、`NOTICE`、`pyproject.toml`、`.gitignore`、`docs/decisions/**`（agent 只写提案）、`docs/motion-contract.md` |

工单：[`docs/brief-claude-sampling.md`](docs/brief-claude-sampling.md)。
接口优先：两边只通过 `docs/motion-contract.md`（数据契约）与 `src/ardy_mlx/io/` 导出的常量互相依赖；A 改契约＝契约变更。

## 环境
仓库内 `.venv`（见 `docs/decisions/0001-python-environment.md`），两个 agent 与人类统一用 `./.venv/bin/python`。
参考实现与金标准产物在那台 Linux/CUDA 机器上（我们只读、一次运行需人类批准）。

## 许可证
本仓库代码 Apache-2.0；**未核清 checkpoint 条款之前，不转换、不再分发任何权重**（见 `NOTICE`）。
