# 状态 · DeepSeek（A：权重转换 / 模型前向 / 数值对齐）

- **当前轮次**：A 侧第 2 轮（金标准 + 权重转换 + 前向 + 三级对齐 + 性能）。起于 HEAD `a13b6d9`。
- **本轮授权**：用户明确授权在那台机器上**跑一次（且仅一次）**参考推理取金标准；除这一次外仍按只读对待。

## 0. 一句话结论

**三级全部对齐，全部通过 1e-3 容差**——tokenizer 往返 ≤ **7.0e-06**、
denoiser 单步（全部 10 步）≤ **5.4e-05**、整段 40 帧 `frames (T,330)` ≤ **3.9e-05**；
denoiser 权重 **224/224 张量全部被前向用到**（零未使用）；权重转换 **428/428 张量逐字节一致、0 处不一致**。

## 1. 金标准（一次运行，40.87 s，exit 0）

完整记录见 [`artifacts/golden-run/RUN-RECORD.md`](../artifacts/golden-run/RUN-RECORD.md)。要点：

- 参考实现 `nv-tlabs/ardy` @ `693f74d`；checkpoint `ARDY-Core-RP-20FPS-Horizon40`（盘上唯一）。
- **实际用的最小配置**：prompt `A person walks forward.`、`--duration 2.0`（40 帧 @20fps = **正好一个 horizon**）、
  `--num_samples 1`、`--seed 0`、`--no-postprocess`、`--diffusion_steps` 默认（=10）、`--history_frames` 默认（=160）。
- **偏离记录（唯一一处）**：文本编码器放 **CPU bf16**（未降精度，只改放置）。
  依据：本机 RAM 14 GB / VRAM 8 GB，8B 编码器 bf16 ≈ 16 GB；这正是该机既有生产配方
  （`run_ardy.sh` 的 `TEXT_ENCODER_DEVICE=cpu` + 专用 `swap-gmgn-ardy.img`）。
  另外 `load_text_encoder()` 结尾会把编码器 `.to(cuda)`，必须显式挡住；
  用本地 `llm2vec-base-merged` + 本地 supervised adapter，并设 `HF_HUB_OFFLINE=1` 防止悄悄下载 15 GB。
- **资源**：墙钟 **40.87 s**、峰值 RSS **13.34 GB**、峰值 VRAM **2,308 MiB**（其中既有服务占 1,097 MiB，本轮净增 ≈1.2 GB）、
  swap ≈2.2 GB、10 步去噪 ≈1.2 s。
- **`--no-postprocess` 的后果（须写明）**：按 `0002` 裁定 2 两侧都跳过后处理，因此金标准**保留了 foot skating**，
  与"Core 官方默认产物"存在系统性差异；本仓 B 侧无需移植 `MotionCorrection` C++ 扩展。
  两侧同口径，对齐本身不受影响。
- **仪器化**：用 `PYTHONPATH` 注入 `sitecustomize.py`（见 `artifacts/golden-run/sitecustomize.py`），
  **未修改参考仓库任何文件**（跑完 `git status` 干净）；除金标准外多跑了一次
  tokenizer 的 encode→decode 只读探针（35 M 参数，非第二次生成）。
- **同时 dump 了文本 embedding**（契约要求的确定输入）：`raw/text_feat.npy`，`(1,1,4096)` float32。
  并且 dump 了参考自己的初始噪声 `x_T`（`raw/denoise_step0_x_in.npy`）——见第 5 节。

## 2. 权重转换（A 侧核心）——**逐张量 + 逐字节，全部一致**

`scripts/ardy_convert.py`（新增）。**转换语义**：发布权重是 all-`F32` safetensors，
MLX 原生可读，因此**不做任何数值改写**（不改 dtype、不改键名——任何改写只会制造两侧分歧点）；
真正做的是三件可证伪的事：清单核对、MLX 原生重导出、逐字节比对。

| 文件 | 张量 | 参数量 | dtype | 逐字节一致 | 不一致 | 对照 A1 记录 |
|---|---|---|---|---|---|---|
| `tokenizer.safetensors` | **204** | **35,430,052** | 全 `F32` | **204 / 204** | **0** | ✔ 一致 |
| `denoiser.safetensors` | **224** | **155,823,252** | 全 `F32` | **224 / 224** | **0** | ✔ 一致 |
| 合计 | 428 | 191,253,304 | — | 428 / 428 | 0 | ✔ |

- 比对方式：源文件与转换后文件**各自按 header 的 `data_offsets` 取出原始载荷字节**，逐张量逐字节比较；
  任何一处不同都会打印张量名与首个差异字节偏移并使命令非零退出。负载字节合计比对 **765,013,216 B**，差异 **0**。
- 产物：`<ckpt>/mlx/{tokenizer,denoiser}.mlx.safetensors`（`checkpoints/` 已 gitignore，**未入库**）；
  入库的只有名称/形状清单 `artifacts/weight-manifest.json` 与报告 `artifacts/weight-verify.json`。
- 源文件 sha256：tokenizer `58a887e2…`、denoiser `1019d0bf…`（见报告）。
- 解析路径**唯一化**：`scripts/ardy_inspect.py` 已改为复用 `src/ardy_mlx/io/safetensors.py`，
  两者不可能再对同一个 header 有不同理解（原 26 个测试仍全绿）。
- **仍未解释**：模型卡写 326 M，实测 191.3 M（A1 记为残留未知，本轮未变）。

## 3. 前向实现结构与张量名对齐

新增 `src/ardy_mlx/model/`：`nn.py`（原语）、`motion_rep.py`、`fsq.py`（Stage A）、
`backbone.py`、`denoiser.py`（Stage B）、`diffusion.py`、`ardy.py`（整段装配）。
非显然常量全部带参考 `文件:行`。

- **Stage A `FSQVAETransformer`**：encoder `patchify(4帧)→input_proj(1300→512)→8×因果层→output_proj(512→128)`;
  FSQ `bound` 走 `tanh(z+atanh(0.5/half_l))*half_l-0.5` 再 `round/32`（**不是**朴素 `round(z*32)/32`）;
  `post_quantization` 归一化。decoder `input_proj(128→512) → cat(local_root 16) → Linear(528→512)+ReLU →
  正弦 PE → 8×因果层 → output_proj(512→1316) → 拆 root(4)/body(325)`。
- **Stage B `AutoLatentTwostageDenoiser`**：hybrid `[root 20 | latent 128]`；
  stage 1 用 `global_root_hybrid_constraints_proj`（**即使无约束也走这条**，约束张量喂零）、
  stage 2 用 `local_root_hybrid_constraints_proj`；`future_constraints_proj` 只在 `future_token_mask` 为真处生效。
  条件注入是 **prefix 拼接**：`[text(1) | timestep(1) | first_heading_angle(1)]` + 3 个 learned prefix 位置嵌入；
  **backbone 无因果掩码**（只传 `src_key_padding_mask`）。
- **CFG**：`separated` 三路 batch（真文本 / 真约束 / 全零），`uncond + w_t(text-uncond) + w_c(uncond→cstr)`；
  注意 `timesteps` 也必须一起三倍化（曾漏掉，已由 Stage B 对齐抓出）。
- **Diffusion**：余弦 β 表硬编码；`space_timesteps` 的 `use_timesteps` 与 `map_tensor` **不能混用**；
  `calc_diffusion_vars` 每步重算；DDIM `eta=0` 确定性。
- **未使用张量**：denoiser **0 个未使用**（224/224 全覆盖）；
  tokenizer 有且仅有 2 个未使用 = `pose_net.decoder.target_cond_blocks.0.{weight,bias}`，
  原因：batch 路径不传 `target_cond`（`fsq.py:253-259`）。已在测试中断言这一精确集合。

### 3.1 本轮抓到并修掉的三个"静默"陷阱（值得记进契约）

1. **`global_root_to_local_root(normalized=True)` 的输出是"再归一化"的**
   （`base.py:145-146`：`local_root_stats.normalize(...)`）。
   即交给 Stage A decoder 与 Stage B body 级的 local root 条件在**归一化空间**，不是米 / 米每秒。
   误按物理量传会让 decoder 输出错 **0.47**（而正确的残差是 6e-06）——这是本轮最大的一个坑。
2. **tokenizer 的 `requantize` 方法与 `detokenize` 内联的重取整不是一回事**：
   前者是 `unnormalize → round(clamp·32)/32 → normalize`（带 stats），后者只有中间那一步。
3. **`x.at`/`[..., 0]` 索引语义**：MLX 的 `a[..., 0]` 取最后一个轴（`[B,T,1,3]→[B,T,1]`），
   而参考的 `[..., 0, [0,2]]` 是先取第 3 轴再取 x/z；写错会静默拿到错的通道。

## 4. 对齐结果（`scripts/ardy_align.py`，可复跑）

同一个 embedding、同一个 `seed=0`、同一份参考 `x_T`。容差 **1e-3**，并附**负对照**证明容差可证伪。

### 4.1 容差理由（必须写进报告）

金标准是 **CUDA float32**（cuBLAS/cuDNN 的归约顺序与 MLX 不同），本侧是 **Metal/CPU float32**。
两侧都不精确，所以正确的问题不是"是否逐位相等"，而是"残差是否只是 float32 求和重排的量级，
还是真的错了"。实测残差 1e-7 ~ 5.4e-05，信号量级 O(1)，相差 4~8 个数量级；
而**结构性错误**会产生 O(1e-1)。负对照（`x_T += 0.25`）实测 max_abs = **2.56**（比容差大 2560×），
证明 1e-3 确实能红。

### 4.2 第 1 级 · tokenizer encode→decode 往返

| 比较 | max_abs | 参考 \|max\| |
|---|---|---|
| `tokenize` → latent `(1,10,128)` | **2.384e-07** | 2.696 |
| `detokenize` → root `(1,40,4)` | **5.841e-06** | 0.689 |
| `detokenize` → body `(1,40,325)` | **7.018e-06** | 2.122 |
| 生成路径 `detokenize_0` root / body | **4.947e-06** / **6.676e-06** | 0.688 / 2.115 |
| `requantize`（stats 包装） | **2.384e-07** | 2.696 |

latent 是**离散**量（1/32 网格），实测 2.4e-07 ⇒ 量化层级**逐个命中**。

### 4.3 第 2 级 · denoiser 单步（全部 10 步都测，不只看最大）

| 量 | 10 步最差 max_abs |
|---|---|
| 三路原始输出 `(3,10,148)` | **2.760e-05** |
| CFG 合成后 `(1,10,148)` | **5.388e-05** |
| DDIM 更新 `x_{t-1}` | **1.234e-05** |

测试里对**每一步**单独断言，避免单个坏时间步被最大值掩盖。

### 4.4 第 3 级 · 整段（40 帧 = 完整 2.0 s 片段）

| 比较 | max_abs | 参考 \|max\| |
|---|---|---|
| `motion_normalized (1,40,330)` | **3.931e-05** | 2.115 |
| `frames (T,330)`（反归一化） | **1.869e-05** | 1.679 |
| root 块 vs 参考 `.npz` 的 `root_positions` | **5.141e-06** | 0.960 |
| heading vs 参考 `.npz` 的 `global_root_heading` | **2.991e-06** | 1.001 |

**到哪一级**：三级**全部做到**。2.0 s = 40 帧 = `gen_horizon_len`，所以这一段整片**只有 1 个自回归窗口**，
`ardy.py` 里的窗口循环按通用多窗口写，但**只有单窗口这一支有金标准**——多窗口（>2.0 s）尚未验证，属明确残留未知。
B 侧已交付 `src/ardy_mlx/sampling/`（`ArdySampler` + `DenoiseRequest`/`DenoiseResult` 的 step-callable 契约），
把本轮的窗口逻辑包成 `step_fn` 即可推广到多窗口；本轮**未改 B 的任何文件**。

> 关于 RNG：金标准里 dump 了参考自己的 `x_T`，因此对齐**不依赖**复刻 CUDA 的 Philox RNG。
> `ArdyBatch.generate` 要求显式注入噪声（`noise_fn`），把"移植 CUDA RNG"留给后续单独处理。

## 5. 性能（M1 Max 32 GB，macOS 26.5.2，mlx 0.32.3，float32）

当时 load：**`os.getloadavg()` ≈ (6.9, 8.9, 9.1)**（有其他 agent 在跑，非空闲机），故为**偏保守**的读数。
每项 warm-up 后重复；MLX 是惰性求值，计时区间内已 `mx.eval` 强制求值。

| 路径 | CPU p50 | CPU p90 | Metal p50 | Metal p90 | 倍数 |
|---|---|---|---|---|---|
| tokenizer encode（40 帧） | 7.90 ms | 8.39 ms | **3.87 ms** | 6.83 ms | 2.0× |
| tokenizer decode（40 帧） | 8.05 ms | 9.96 ms | **5.04 ms** | 5.98 ms | 1.6× |
| denoiser 单步（3 路 CFG batch） | 60.99 ms | 68.60 ms | **12.45 ms** | 13.48 ms | 4.9× |
| **整段 40 帧 / 10 步** | 601.53 ms | 605.04 ms | **108.21 ms** | 110.11 ms | 5.6× |
| 冷启动（加载 428 张量） | 0.130 s | — | 0.115 s | — | — |
| 峰值内存 | 1,883 MB | — | 1,921 MB | — | — |

整段 108 ms ≈ 10 × 12.45 ms（去噪） + 2×约 5 ms（编解码）⇒ 与单步数据自洽。
注意：本表只覆盖**运动模型**；v1 明确不移植文本编码器（8B），它在这台机器上是 CPU/bf16、
单条短 prompt 秒级，金标准的 40.87 s 里绝大部分是它。

## 6. 测试

新增 30 个测试，全部通过：

- `tests/test_io_safetensors.py`（9）：容器读/写、逐字节载荷、损坏检测。
- `tests/test_model_contract.py`（13）：**不需要 checkpoint**，用合成 stats 跑，钉住上面三个陷阱
  （local root 的归一化方向、`re_round` ≠ `requantize`、FSQ 的 tanh/offset 公式与不对称值域 [-1, 31/32]、
  `map_tensor` 与循环下标之别、PE 表、掩码语义、末帧速度复制）。
- `tests/test_model_alignment.py`（8）：三级金标准对齐 + 负对照 + 张量使用率断言；
  权重/金标准缺失时 **skip 并说明原因**，不会静默变绿。

跑 `./.venv/bin/python -m pytest tests/test_io_*.py tests/test_model_*.py -q` → **56 passed**。

## 7. 未能确认 / 失败的部分

1. **钩子的 atexit flush 崩了**：`TypeError: Object of type ndarray is not JSON serializable`
   （`generation_token_mask` 是 ndarray 却放进了 JSON）。因为只授权一次，**没有重跑**。
   丢失了 `randn_*.npy`、`diffusion_buffers.npz`、`modules/*.npy`。
   `x_T` 已由 `denoise_step0_x_in.npy` 精确恢复（该窗口无历史，`x == x_T`），三级对齐不受影响；
   丢的只有"模块内部张量"这一层（若将来需要逐层深挖，重跑一次即可，代价 41 s）。
   另注意 `raw/trace.json` 是**预检残留的空文件**，不是本次运行产物。
2. **只跑了一次**：授权写明"一次（且只一次）"，因此没有跑 `--duration 4.0`（2 个窗口）。
   多窗口路径（history cropping / 多次 recenter / `first_heading_angle` 更新）**已实现但无金标准验证**。
3. **未复刻 CUDA RNG**：靠注入参考 `x_T` 绕开；`seed` 语义对齐仍需单独工作。
4. **模型卡 326 M vs 实测 191.3 M** 仍未向上游确认（A1 的残留未知）。
5. 本机跑全量 `pytest` 时 B 侧 7 个测试失败（`test_sampling_loop.py` / `test_sampling_window.py`），
   属 B 侧并行进行中的工作，**A 未触碰其文件**；本侧 56 个测试全绿。

## 8. 纪律留痕

- 那台机器：**只跑了一次** `generate.py`；未安装、未 sudo、未改系统/配置、未占 GPU 做别的事；
  未打印任何凭据；跑完**无进程残留**（GPU 回到 1,097 MiB 基线）；参考仓库 `git status` 干净。
  只新建了 `~/ardy-golden-2026-10-13/`（hook + raw + 日志 + 金标准 npz），可随时删除。
  残留：swap 使用量从 2,035 MB → ≈2,248 MB（加载 8B 的页缓存/交换残留，未改任何配置）。
- 本机：**未 push**；未做长训练；未使用任何合成输入（CGEvent/AX/osascript）。
- 未触碰禁改文件：`src/ardy_mlx/sampling/**`、`scripts/ardy_generate.py`、
  `tests/test_sampling_*.py`、`tests/test_generate_*.py`、`docs/brief-claude-sampling.md`、
  `README*`、`NOTICE`、`LICENSE`、`pyproject.toml`、`.gitignore`。
- `docs/motion-contract.md` 只**补"实测确认"注记**，未改任何字段/维度/顺序。
