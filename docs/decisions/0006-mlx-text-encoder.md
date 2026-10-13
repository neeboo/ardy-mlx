# 0006 · MLX 文本编码器（ARDY 条件向量）：配方、预注册判据与许可

- 状态：**已冻结（判据先于实现写死；本文不含任何实现）**
- 日期：2026-10-13
- 依据：模型卡 `HF: voxta/Llama-3-8B-LLM2Vec-ARDY-INT8` 的调查；参考源码
  `nv-tlabs/ardy` @ `693f74d`（`ardy/model/llm2vec/**`）；本机 MLX 0.32.3（Apple Silicon）
- 关系：沿用 `0003`（容差与金标准口径）、`0004`（契约裁定）、`0005`（多窗口基线）；
  **不改**其中任何一条

---

## 0. 目标（一句话）

在本机（Apple Silicon / MLX）把**一句 prompt** 编码成 ARDY 需要的 **4096 维池化向量**
（`[B, L=1, 4096]`，与金标准 `text_feat.npy` 同形），作为**额外一条路**。

**两条路并存，互不替代：**

| 路 | 是什么 | 地位 |
|---|---|---|
| **参考编码器** | CI 机（`pancat-linux-ci`）上的 bf16 组装：`Meta-Llama-3-8B-Instruct` + MNTP LoRA + supervised LoRA，transformers/peft，CPU bf16，`batch_size=1` | **对齐基准**。金标准的条件信号由它产生，`artifacts/golden-run*/raw/text_feat.npy` 即其输出 |
| **MLX 编码器**（本次新增） | 同一组权重，适配器先合并进 bf16 底座、再转 MLX 并量化（4-bit / 8-bit），本机推理 | **本机可用性**。它让"任意 prompt → 向量"在本机成立；**它不产生金标准**，也不替换参考缓存 |

判据里的每一个数字都比对**参考编码器**的向量；MLX 向量永远不被称为"金标准向量"。

## 1. 配方（逐条来自模型卡与参考源码，附来源）

1. **底座**：`meta-llama/Meta-Llama-3-8B-Instruct`（bf16）。
2. **双向注意力**：去掉因果掩码。来源：`ardy/model/llm2vec/models/bidirectional_llama.py`
   的 `ModifiedLlamaAttention.__init__` 把 `self.is_causal = False`；注意力只带 **padding 掩码**，
   没有 causal mask。**不要**用 `bos/eos` 之外的任何"近似"。
3. **MNTP 适配器**：
   - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`（先合并进底座）
   - `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`（在 MNTP 之上再合并）
   CI 机上已有同一份（`~/services/ardy-engine/source-models/`，各 167,829,552 B LoRA），
   CI 还预置了 MNTP 已合并的 bf16 底座 `~/services/ardy-engine/llm2vec-base-merged`
   （290 张量、7,504,924,672 参数、全 BF16，键无 `model.` 前缀、不含 `lm_head`）。
4. **池化**：对 **prompt token 做均值池化，跳过 instruction**。来源：`llm2vec.py`
   `get_pooling()` + `_skip_instruction()`：`attention_mask` 被 `embed_mask` 覆盖后
   `mean(last_hidden_states[-length:])`，其中 `length` = **只对 prompt 文本**（`add_special_tokens=False`）
   tokenize 出来的 token 数。`LLM2VecEncoder.__call__` 再把 `(N,4096)` 变成 `(N,1,4096)`。
   参考包装器**强制 `batch_size=1`**（注释明说 batch size 会改变结果），本实现默认逐句编码。
5. **必须把 `config._name_or_path` 设成 `meta-llama/Meta-Llama-3-8B-Instruct`**。
   来源：`llm2vec.py` 的 `prepare_for_tokenization()` 用 `self.model.config._name_or_path`
   选择 Llama-3 对话模板（`<|start_header_id|>user<|end_header_id|>\n\n … <|eot_id|>`）。
   transformers 5 **不再序列化** `_name_or_path`，所以合并后要显式写回；**不设则向量与参考对不上**。
   本实现从 `config.json` 读该字段，**读不到就报错**，绝不静默退回别的模板。
   （CI 上的 `llm2vec-base-merged/config.json` 已带该字段，值正确。）
6. **适配器先合并进 bf16 底座，再量化**。**先量化后合并会严重累积误差**（LoRA 的增量是在
   全精度权重上定义的，量化后的底座再叠加适配器无法复原）。
   本实现：读 bf16 → 提升到 fp32 做 `W + (alpha/r)·B@A` → 落回 bf16 → 再转 MLX 并量化。
7. **真实输入串**：`LLM2Vec.encode` 把 `["", prompt]` 变成
   `"!@#$%^&*()" + prompt`（`_convert_to_str("", text)`），tokenize 时 `add_special_tokens=True`
   （前置 BOS `128000`），`padding_side="left"`，`max_length=512`。
   本实现按同一构造做，并用 CI 上导出的逐 prompt `input_ids` 逐 token 校验分词器。

## 2. 预注册验收判据（**先写死，之后不许改**）

任何一条不满足都按"未通过"如实上报；**不得**事后移动阈值、换 prompt 集合或换对照。

### 2.1 prompt 集合（冻结，16 条，含金标准用的那条）

```
 1 A person walks forward.                 9  A person does a cartwheel.
 2 A person runs forward.                 10  A person crawls on the floor.
 3 A person jumps in place.               11  A person dances energetically.
 4 A person turns around.                 12  A person climbs a ladder.
 5 A person sits down on a chair.         13  A person punches the air twice.
 6 A person waves with their right hand.  14  A person lies down and rolls over.
 7 A person kicks a ball with their left foot.  15 A person jumps over a small obstacle.
 8 A person walks backward slowly.        16  A person stretches their arms above their head.
```

参考向量在 CI 机上以**仅编码器**的方式产生（§3）。第 1 条同时有金标准 `text_feat.npy`，
两者必须互相印证（作为额外的一致性检查，不替换 §2.2）。

### 2.2 保真度（对参考编码器向量）

- 在 §2.1 的 16 条 prompt 上，逐条算 `cosine(MLX(bits), reference)`。
- **8-bit：16 条全部 ≥ 0.997**（模型卡实测 0.9976 / 0.9980）
- **4-bit：16 条全部 ≥ 0.985**（模型卡 NF4 实测 0.9841 / 0.9857；本实现是 MLX 仿射量化，
  不是 NF4——**阈值按任务冻结为 0.985，不因量化格式不同而放宽**）
- **最近邻自证**：对每条 prompt 的 MLX 向量，在 16 条参考向量里余弦最近的那个**必须是它自己**；
  16/16 命中才算通过。（若两条 prompt 的参考向量彼此极近，允许把这一对记为"歧义对"并如实列出，
  但至少要有一条判定为"最近邻非自身即失败"的硬断言。）

### 2.3 可证伪性（负对照必须掉出判据）

三条负对照，均在 **8-bit** 模型上测（与正例同口径）：

| # | 对照 | 期望 |
|---|---|---|
| N-A | **保留因果掩码**（`is_causal=True`，其他一切不变） | 16 条 cosine 全部 **< 0.985** |
| N-B | **不跳过 instruction**（池化用完整 `attention_mask`，含 BOS/模板 token） | 全部 **< 0.985** |
| N-C | **池化取错 token**（不取 prompt 尾部 `length` 个、而取整段均值或最后一个 token） | 全部 **< 0.985** |

硬判据：三条**全部**满足上表。任何一条落在 `[0.985, 0.997)` 记为**弱对照**并如实上报；
落在 `≥0.997` 视为该判据**不可证伪**，本轮判据判为未通过。

### 2.4 动作层（用 MLX 向量真的跑一次生成）

1. 用 MLX 编码器产出的 `A person walks forward.` 向量，走**出货路径**
   （`scripts/ardy_generate.py --backend mlx`）生成 2.0 s 片段：形状 `(40,330)`、
   时长 2.0 s（= 2 个 horizon 的整数倍）、**所有数值有限**（无 NaN/Inf）；再用查看器
   的同一读取路径确认**可播放**。
2. **必须明确记录它与"参考编码器向量生成的动作"不同**，并给出
   `max_abs(motion_normalized_MLX向量 − 金标准)`。模型卡原话：扩散对条件混沌
   → 不同但同样有效。**要求该差值 > 1e-3**（即"确实不同"；若 ≤1e-3 说明向量等价到了
   与参考难以区分的程度，需重新检查编码器）。
3. **禁止**把这条动作描述成"与金标准一致"。措辞固定为
   "**不同条件向量下的另一次有效生成**"。

### 2.5 性能与内存

- 编码一句的 **p50 / p90**（同机、≥10 次、warm、8-bit 与 4-bit 各一组）
- **冷启动**（进程启动 → 第一个向量）
- **峰值内存**：4-bit **目标 ≤ 5 GB**（超出即如实报告超额）
- 记录测量时的 `loadavg`（本机常被其它负载占用，必须如实标注）

### 2.6 证据强度降级条款（预先约定）

若 CI 机的"仅编码器"运行**无法完成**（不可达 / 缺权重 / 授权不允许），则退化为：
**只用手上那 1 条参考向量（`A person walks forward.`，金标准 `text_feat.npy`）+ 模型卡数字**，
并在 `docs/status-deepseek.md` **显式标注证据强度下降**（§2.2 的 16 条表退化为 1 条、
最近邻自证退化为 1 条、§2.3 的负对照仍可做）。**不得**把弱化后的证据说成满足 §2.2 全文。

## 3. 参考向量的取得（纪律）

- 在 `pancat-linux-ci` 上做**仅编码器**的运行：CPU bf16、`batch_size=1`、
  **不占 GPU、不跑动作生成、不跑参考仓库的 `generate.py`**。
- 一次性跑完 16 条 prompt（模型只加载一次），导出每条 `input_ids` 与向量，并记录
  耗时 / 峰值 RSS / 退出码 / **在 CI 上新建了哪些路径**。
- **不装、不改、不 sudo**；只在自己新建的 `~/ardy-encoder-only-*` 目录下写；
  跑完立即退出进程（无残留 `pgrep` 命中）；**不打印任何凭据**。
- 向量与 CI 记录**一律不入库**（`*.npy` 已在 `.gitignore`；参考向量放
  `artifacts/reference-embeddings/`，与 `artifacts/embeddings/` 一样属权重派生数据）。

## 4. 许可（与仓库许可不同）

- 底座 `meta-llama/Meta-Llama-3-8B-Instruct`：**Llama 3 Community License**
- MNTP / supervised 适配器：**MIT**（McGill-NLP）
- 本仓库代码：Apache-2.0
- ⇒ 本组件是**独立组件**：在 `NOTICE` **只加一段许可说明**（不改既有立场、不改"不重新分发
  NVIDIA 权重/金标准输出"的条款），**权重与参考向量不入库**。

## 5. 明确不做

1. **不改** `/api/plan` 的对齐语义（`x_T` 注入、`no_cached_embedding_for_prompt` 仍是错误、
   时长仍须是 horizon 整数倍）。
2. **不替换**参考缓存的地位：查看器**优先使用参考向量**；MLX 向量只在"没有参考向量"时兜底，
   且必须在 HUD/sidecar/接口里**标明来源**。
3. **不做**训练 / 微调 / 蒸馏；不改底座数值（只做 LoRA 合并与量化）。
4. 不改 `src/ardy_mlx/model/**` 的既有行为、不改 `ArdyBatch` 语义。

## 6. 查看器接线判据（可区分两种来源）

- MLX 编码器的缓存写在参考缓存的**子目录**里（`artifacts/embeddings/mlx/`）：同名 prompt
  的两个来源**不可能互相覆盖**，且两条路可按文件名区分而无需读内容。
- sidecar 必须带 `encoder` 字段：参考向量 `"reference-llm2vec-bf16"`，MLX 向量
  `"mlx-llm2vec"`，并带 `bits` / 权重指纹 / 合并与量化参数。
  **缺 `encoder` 字段的历史文件按"参考"解释**（现存 `e9cf8141cac79f35.npy` 因此保持原意）。
- `/api/health` 的每个 embedding 条目带 `encoder`（含 `bits`），并有
  `preferred_encoder` 标记；`/api/plan` 的响应记录**本次实际用了哪一条**。
- 页面 HUD **显示来源**；两种来源**同时存在时必须选参考**，这条要有测试。

---

## 冻结声明

以上 §2 的阈值、prompt 集合、负对照与动作层口径，**在实现开始前冻结**。
实现完成后只允许**追加**测量结果与解释，不允许修改本节数字。
