# 0002（已批准 2026-10-13） · checkpoint 许可核查（**提案** · 待人类批准）

- 状态：**提案（proposal）**，尚未生效；本文件不改写任何上游条款，只记录证据与建议。
- 核查人：A 侧（DeepSeek）
- 核查日期：2026-10-13
- 核查对象：上游 `nv-tlabs/ardy` @ `693f74d`（Apache-2.0）的 4 个 checkpoint
- 核查方式：只读（`ls/cat/grep/sed -n/head/tail/git show`）；未安装、未修改、未运行任何推理

## 0. 硬门结论

> **建议：通过硬门。** 4 个 checkpoint 的许可均为 **NVIDIA Open Model License Agreement**，
> 明示"commercially usable / free to create and distribute Derivative Models"，
> 并把 `create derivative works of`、`distribute`、`sell` 纳入授权范围。
>
> 本次核查中**未转换、未复制、未再分发任何权重**。
> 本提案需人类批准后才可进入权重转换（README 路线图第 2 步）。

## 1. 证据 A · 上游 README 的 checkpoint 许可列

`ardy/README.md:65`（表头，含 `License` 列）：

```
| Model | Skeleton | Training Data | FPS | Horizon | Release Date | Hugging Face | License |
```

`ardy/README.md:67-70`（4 行，逐行均以 NVIDIA Open Model 作为 License）：

```
| **ARDY-Core-RP-20FPS-Horizon40** | Core | [Bones Rigplay 1](...) | 20 | 40 | July 10, 2026 | [Link](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-Core-RP-20FPS-Horizon8**  | Core | [Bones Rigplay 1](...) | 20 |  8 | July 10, 2026 | [Link](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon8)  | [NVIDIA Open Model](...) |
| **ARDY-G1-RP-25FPS-Horizon52**   | Unitree G1 | [Bones Rigplay 1](...) | 25 | 52 | July 10, 2026 | [Link](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) | [NVIDIA Open Model](...) |
| **ARDY-G1-RP-25FPS-Horizon8**    | Unitree G1 | [Bones Rigplay 1](...) | 25 |  8 | July 10, 2026 | [Link](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon8)  | [NVIDIA Open Model](...) |
```

`ardy/README.md:287`（明确代码与权重分别授权）：

```
This codebase is licensed under [Apache-2.0](LICENSE). Note that model checkpoints and data are licensed separately as indicated on the HuggingFace download pages.
```

## 2. 证据 B · checkpoint 自带的 LICENSE 全文

本地唯一的 checkpoint 目录里**随权重附带了 LICENSE 与本模型 README**：

```
~/services/ardy-engine/checkpoints/ARDY-Core-RP-20FPS-Horizon40/
├── LICENSE                 (162 行, NVIDIA Open Model License Agreement)
├── README.md               (= HuggingFace model card)
├── config.yaml
├── denoiser.safetensors    (623,324,336 B)
├── tokenizer.safetensors   (141,746,112 B)
└── stats/{motion,post_quantization,pre_quantization}/
```

`LICENSE:1,3`：

```
Reference: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

NVIDIA Open Model License Agreement
```

`LICENSE:13-17`（**授权基调整段引用**）：

```
of this Agreement, NVIDIA confirms that:
- Models are commercially usable.
- You are free to create and distribute Derivative Models.
- NVIDIA does not claim ownership to any outputs generated using the Models or
  Derivative Models.
```

`LICENSE:61-67`（授权范围，**决定"能否本地推理/微调/再分发/商用"**）：

```
2.2. License Grant. The rights granted herein are explicitly conditioned on Your
full compliance with the terms of this Agreement. Subject to the terms and
conditions of this Agreement, NVIDIA hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, revocable (as stated in
Section 2.1) license to publicly perform, publicly display, reproduce, use,
create derivative works of, make, have made, sell, offer for sale, distribute
(through multiple tiers of distribution) and import the Model.
```

条件与限制（必须一并遵守）：

| 条款 | 行 | 原文要点 |
|---|---|---|
| 2.1 条件 | `LICENSE:44-58` | 绕过/削弱 Guardrail（安全护栏、加密、鉴权等）→ 权利**自动终止**；对 NVIDIA 提版权/专利诉讼 → 授权终止 |
| 2.2 可撤销性 | `LICENSE:63` | 授权是 `revocable (as stated in Section 2.1)` |
| 2.3 AI 伦理 | `LICENSE:68-70` | 使用须符合 NVIDIA Trustworthy AI 条款 |
| 2.4 输出归属 | `LICENSE:71-78` | `NVIDIA claims no ownership rights in outputs. You are responsible for outputs and their subsequent uses.` |
| 3.1 再分发条件 | `LICENSE:83-86` | 分发 Model 须给接收方**一份本协议副本**，并在 `Notice` 文本文件中包含归属声明 `"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"` |
| 3.2 Cosmos 专属 | `LICENSE:87-92` | `"Built on NVIDIA Cosmos"` 标注义务**仅针对 NVIDIA Cosmos Model** |
| 3.3 可加自己的条款 | `LICENSE:93-98` | 可为自己的修改/Derivative Model 另附条款，但 Model 本身的使用仍须合规 |
| 5. 商标 | `LICENSE:113-115` | 未授予 NVIDIA 商标使用权，仅"描述来源"的合理使用 |
| 11. 贸易合规 | `LICENSE:156-161` | 须遵守美国出口管制/制裁法规 |
| 版本 | `LICENSE:162` | `Version Release Date: October 24, 2025` |

关于 3.2 是否适用于 ARDY：`LICENSE:36` 定义 `1.4. "NVIDIA Cosmos Model" means a multimodal Model shared under this Agreement.`
在整仓 grep `cosmos`（忽略大小写）**只命中 LICENSE 自身**，上游从未把 ARDY 标注为 Cosmos Model。
**建议判定：3.2 不适用**；但这属于**法律解释**，标注为残留未知项（见 §8）。

## 3. 证据 C · checkpoint 自带 README（= HuggingFace model card）

`README.md:1-5`（YAML frontmatter，许可字段）：

```yaml
license: other
license_name: nvidia-open-model-agreement
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement
```

`README.md:22`（**最直接的一句**）：

```
This model is ready for commercial or non-commercial use.
```

`README.md:25-26`：

```
### License/Terms of Use:
Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
```

`README.md:151` 与 `README.md:158`（模型卡的 Licensing / Use Case Restrictions 两行）同样只指向该协议。

## 4. 证据 D · 4 个 HuggingFace 页面复核（已取到）

已通过 `https://huggingface.co/nvidia/<name>/raw/main/README.md` 取回 4 张模型卡（HTTP 200），
逐张校验 `license_name` 与第 22 行，**四者完全一致**：

| HuggingFace repo | HTTP | frontmatter `license_name` | 第 22 行 |
|---|---|---|---|
| `nvidia/ARDY-Core-RP-20FPS-Horizon40` | 200 | `nvidia-open-model-agreement` | `This model is ready for commercial or non-commercial use.` |
| `nvidia/ARDY-Core-RP-20FPS-Horizon8` | 200 | `nvidia-open-model-agreement` | 同上 |
| `nvidia/ARDY-G1-RP-25FPS-Horizon52` | 200 | `nvidia-open-model-agreement` | 同上 |
| `nvidia/ARDY-G1-RP-25FPS-Horizon8` | 200 | `nvidia-open-model-agreement` | 同上 |

> 未取到的内容：4 个 repo 的**文件清单/权重文件级**校验（`?recursive=true` API）本次未抓；
> NVIDIA Open Model Agreement 的**在线最新版**未与随附文本逐字 diff（见 §8）。

## 5. 证据 E · 代码许可与第三方组件

- 上游代码：`ardy/LICENSE:1` = `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` + Apache-2.0 全文；`README.md:287` 明示代码 Apache-2.0。
- `ardy/ATTRIBUTIONS.MD:1` = `LLM2Vec MIT License …`（MIT）
- `ardy/ATTRIBUTIONS.MD` 中段 = `Unitree mujoco BSD 3-Clause License …`（BSD-3-Clause）
- `MotionCorrection/` 是 C++ pybind11 扩展（`setup.py:99-102`，扩展名 `motion_correction._motion_correction`），随 Apache-2.0 代码库分发。

## 6. 逐 checkpoint 判定表

"允许范围"一列**只引用上述原文**；没有原文支撑的一律写"未知"。

| # | checkpoint | 本地是否存在 | 来源 | 许可名称 | 本地推理 | 微调 / Derivative Model | 再分发 | 商用 |
|---|---|---|---|---|---|---|---|---|
| 1 | `ARDY-Core-RP-20FPS-Horizon40` | ✅ **在**（本机唯一） | 上游 README:67 + `checkpoints/…/LICENSE` + HF | NVIDIA Open Model License Agreement（v2025-10-24） | **允许**（`use`） | **允许**（`create derivative works of`） | **允许，有条件**（`distribute`；须附协议副本 + Notice 归属声明，`3.1`） | **允许**（`sell, offer for sale`；`Models are commercially usable`） |
| 2 | `ARDY-Core-RP-20FPS-Horizon8` | ❌ 不在 | 上游 README:68 + HF | 同上 | 同上 | 同上 | 同上 | 同上 |
| 3 | `ARDY-G1-RP-25FPS-Horizon52` | ❌ 不在 | 上游 README:69 + HF | 同上 | 同上 | 同上 | 同上 | 同上 |
| 4 | `ARDY-G1-RP-25FPS-Horizon8` | ❌ 不在 | 上游 README:70 + HF | 同上 | 同上 | 同上 | 同上 | 同上 |

> 注意：**#1 是本机唯一存在的 checkpoint**（`checkpoints/ARDY-Core-RP-20FPS-Horizon40/`）。
> #2–#4 的条款依据是上游 README 表 + 各自的 HF 模型卡；它们的**随附 LICENSE 文件本次无法读取**（未下载），
> 因此"本地随附文本与 HF 一致"这一点对 #2–#4 属于**未核实**（但四张模型卡本身已逐张取到，一致）。

## 7. 建议结论（允许做什么 / 禁止做什么）

**建议允许（在本仓库内）**

1. **本地推理**：下载/使用权重在本机（Apple Silicon / MLX）跑推理。
2. **权重格式转换**（即 README 路线图第 2 步）：把 `tokenizer.safetensors` / `denoiser.safetensors` 转成 MLX 可读格式 —— 这构成 `Derivative Model`（`1.1` 定义），**被 `2.2` 明确授权**。
3. **微调**：`create derivative works of` 覆盖。
4. **商业使用**：`LICENSE:13` + 模型卡 `README.md:22`。
5. **再分发**（含转换后的权重），但必须同时满足：
   - 随附**本协议副本**（`3.1`）；
   - 在 `Notice` 文本文件中写入 `"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"`（`3.1`）；
   - 若改动过，可另附自己的条款（`3.3`）。

**建议禁止 / 必须避免**

1. **不要绕过或削弱任何 Guardrail**（安全护栏、加密、鉴权、DRM 等）——`2.1` 规定权利**自动终止**。
2. **不要对 NVIDIA 提版权/专利诉讼**——`2.1` 规定授权即刻终止。
3. **不要用 NVIDIA 商标/商品名**做背书（`5`），除非描述来源的合理使用。
4. **不要忽略出口管制与制裁合规**（`11`）。
5. **不要假定许可是不可撤销的**：`2.2` 写明 `revocable`。
6. **本仓库当前 `NOTICE:12-14` 的"未清条款前不转换/不再分发"约束**：若本提案获批，该句应更新为"已核查，见 `docs/decisions/0002`"。**本次不修改 `NOTICE`**（在禁改清单内），仅作为提案建议。

## 8. 残留未知（明确写"未知"，不猜）

| 项 | 状态 | 原因 |
|---|---|---|
| #2–#4 checkpoint 的**随附 LICENSE 文件**原文 | **未知** | 本机未下载这 3 个 checkpoint；仅能读到 HF 模型卡与上游表格 |
| NVIDIA Open Model Agreement **在线最新版** vs 随附文本 | **未知** | 未逐字 diff；`LICENSE:47-49` 允许 NVIDIA 更新协议并要求你遵守更新版或停止使用 |
| `3.2` Cosmos 标注义务是否适用于 ARDY | **未核实（建议不适用）** | 上游从未称 ARDY 为 Cosmos Model；但这是法律解释，留待人类/法务裁定 |
| **训练数据** Bones Rigplay 1 的许可 | **未知** | 未读取 bones.studio 的数据条款；不影响"使用权重"，但影响"再分发权重"的风险评估 |
| **文本编码器**链路的许可 | **未知（已定位证据位置）** | 见下 |
| 输出的商用责任 | 已明确 | `2.4`：NVIDIA 不主张输出所有权，但**责任在使用方** |

**文本编码器（v1 不移植，但 v1 消费其预计算 embedding）** —— 本次只定位、未审：

- `~/services/ardy-engine/source-models/Meta-Llama-3-8B-Instruct/LICENSE:1` = `META LLAMA 3 COMMUNITY LICENSE AGREEMENT`（`LICENSE:2` = `Meta Llama 3 Version Release Date: April 18, 2024`），并附 `USE_POLICY.md`（`# Meta Llama 3 Acceptable Use Policy`）。
- `~/services/ardy-engine/source-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp{,-supervised}/` 为 LLM2Vec 适配器（MIT，见 `ATTRIBUTIONS.MD:1`）。
- 合并产物 `~/services/ardy-engine/llm2vec-base-merged/model.safetensors`（15.0 GB）**目录内没有 LICENSE/model card**。

→ 建议：v1 只消费**预计算 embedding 文件**这一决定同时规避了该链路；但若将来分发 embedding 或移植文本编码器，
**必须单独做一次许可核查**（Llama 3 Community License 有月活与归属等额外条款）。

## 9. 本次未做的事（留痕）

- **未转换任何权重**（没有读取/写出 safetensors 张量数据）。
- **未下载**任何 checkpoint（#2–#4 仍未落盘）。
- **未运行** `generate.py` 或任何推理；金标准输出仍需人类单独批准。
- **未修改**那台 Linux 机器上的任何文件（只读；过程中曾误建 2 个 `/tmp` 临时文件，已即刻删除并复核不存在）。
- **未修改** `NOTICE` / `LICENSE` / `README*` / `pyproject.toml` / `.gitignore`。

---

## Lead 批准（2026-10-13）

**硬门通过**，并同时裁定本轮提出的另外三件事：

1. **许可**：批准本提案。四个 checkpoint 同为 **NVIDIA Open Model License（2025-10-24）**：
   允许本地推理、**格式转换（构成 Derivative Model）**、微调、商用；**再分发须满足条件**（随附协议副本 ＋
   在 Notice 写明归属声明）。**风险记在案**：协议 `revocable`、绕过/削弱 Guardrail 会**自动终止**、
   不得用 NVIDIA 商标、出口管制条款；**Cosmos 标注义务**按提案判断为不适用（属法律解释，保留为残留未知）。
   **金标准输出不在本仓库分发**（它由 NVIDIA 权重派生），与"不再分发权重"的既有立场一致。
2. **金标准是否含后处理**：**批准"两侧都 `--no-postprocess`"**。理由：Core 默认跑 `MotionCorrection`，
   本机缺该扩展会直接 `raise`；两侧都不跑才能同口径比对。**报告里必须写明**这条取舍及其后果
   （保留 foot skating、与"Core 默认含后处理"的官方产物存在系统性差异）。
3. **坐标转换（MotionBricks/Unity）**：**v1 不做**。第一版一律保留上游约定（**右手系、Y 轴向上、地面 XZ、世界坐标、米、6D 旋转**），
   与产品侧的转换留到接线时再定；`docs/motion-contract.md` 里的建议表**标注为未验证**，B 侧不得据此写死。
4. **`<prompt-hash>` 口径**：批准 **prompt UTF-8 原文**（稳定、可复算），并**冻结进契约**；改口径必须走契约变更流程。

**附带授权**（用户已开工并要求"英文优先、不写分工"）：本裁定允许把 `NOTICE` 里"未清条款前不转换/不再分发"那句
更新为"已核查，见 `docs/decisions/0002`"。`
