# 状态 · DeepSeek（A：权重转换 / 模型前向 / 数值对齐）

- **当前轮次**：A 侧第 3 轮（第二次参考运行 + **多窗口整段对齐** + `make_step_fn` 接缝）。起于 HEAD `7dde062`。
- **本轮授权**：`docs/decisions/0003-alignment-baseline.md` 裁定 4，用户明确"直接搞完，不要等我"——
  **第二次（也是最后一次）参考运行**已用掉：`--duration 4.0`（80 帧 = 2 个自回归窗口），其余参数与第 1 次完全一致。
  除这一次外那台机器仍按只读对待。

## 0. 一句话结论

**第 3 级对齐从单窗口推广到多窗口整段，全部通过 `0003` 冻结的 1e-3**：
两窗口逐窗口 `x`（= 历史 + 注入噪声）**逐位 / 2.2e-06** 命中，重定心位移、平移累积、
`first_heading_angle` 同步更新各 **1.5e-06 / 1.6e-07**，20 个去噪步全部逐步断言
（raw **8.5e-05**、CFG **9.1e-05**、DDIM **1.7e-05**），整段 `frames (1,80,330)` **3.5e-05**、
分段 ≤3.5e-05、root/heading 对参考 npz **4.7e-06 / 1.9e-06**。
过程中**抓到并修掉两个只有多窗口才会暴露的移植错误**（`lengths` 语义、`future_token_mask` 规则），
并用 **11 条负对照**证明每条多窗口路径的断言都能红；其中 2 条被证明在本金标准上**必然空转**，
已如实记为限制而不是掩盖。A 侧测试 **81 passed + 1 xfailed**（新增 26 个）。

## 1. 第二次参考运行（一次，38.26 s，exit 0）

完整记录见 [`artifacts/golden-run-4s/RUN-RECORD.md`](../artifacts/golden-run-4s/RUN-RECORD.md)。

- 参数与第 1 次**逐字相同**，只把 `--duration 2.0` 改成 **`4.0`**：同 prompt `A person walks forward.`、
  `--num_samples 1`、`--seed 0`、`--no-postprocess`、`--diffusion_steps` 默认（10）、
  `--history_frames` 默认（160）、文本编码器 **CPU bf16**、`HF_HUB_OFFLINE=1`、
  `PYTHONPATH` 注入 `sitecustomize.py`。**未改参考仓库任何文件**（跑完 `git status` 干净，前后都是 0 行）。
- 参考 `nv-tlabs/ardy` @ `693f74d`；checkpoint `ARDY-Core-RP-20FPS-Horizon40`（盘上唯一）。
- **资源**：墙钟 **38.26 s**、user/sys 70.59/43.68 s（298 % CPU）、**峰值 RSS 13.26 GB**、
  **峰值 VRAM 2,308 MiB**（其中既有服务占 1,097 MiB ⇒ 本轮净增 ≈1.2 GB）、
  swap 2,241 → 峰值 3,393 → 跑完 2,255 MB、主缺页 101,657、exit 0。
  与第 1 次几乎同价——瓶颈是 8B 文本编码器，多出来的只有第二个 10 步循环（运动模型本体每窗 ≈108 ms）。
- 新目录 `~/ardy-golden-2026-10-13-4s/`（**未覆盖第 1 次的目录**）。跑完：无进程残留、
  GPU 回到 1,097 MiB 基线、预检 scratch 目录已删（已确认不存在）。

### 1.1 钩子 atexit flush 崩溃的修复点（第 1 轮的 `TypeError`）

第 1 轮的 `atexit` flush 死在 `TypeError: Object of type ndarray is not JSON serializable`
（`generation_token_mask` 是 ndarray 却进了 JSON）。因为**所有东西都在退出时写**，
崩一次就带走了 `randn_*.npy`、`diffusion_buffers.npz`、`modules/`——而运行机会已经用掉。
v2 钩子用**两条互不依赖**的改动让这类失败不可能再造成数据丢失：

1. **flush 不再是承重结构**：`_jsonable()` 递归消毒每条记录（ndarray ≤64 元素内联，
   否则记 shape/dtype/min/max 摘要；numpy 标量、`torch.Size`、tensor、嵌套 dict/list 都覆盖）；
   同时**每个产物在观测到的当下就落盘**（`_save` 写 npy、`_jsonl` 每条事件一行并 flush、
   `_write_json` 走临时文件 + `os.replace`）。退出钩子现在只写可选的聚合 `trace.json`，
   失败只记日志不抛异常。**本次实跑 `trace.json` 的 `errors: []`，所有 dump 齐全。**
2. **模块 forward hook 不再在内存里攒张量**：每个 `modules/*.npy` 在 hook 里写完即丢
   （顺带消掉了第 1 轮那份从未落盘的 RSS 占用）。

**落盘前做过真实预检**：用 v2 钩子在 scratch 目录里塞进"第 1 轮那种把 ndarray 放进 JSON 记录"的形状，
让退出钩子真的跑一遍 → `flush done, errors=0`，然后删除 scratch 目录。预检通过后才启动唯一一次运行。

### 1.2 本轮新增的仪器化（这是"逐窗口对齐"能存在的前提）

| 产物 | 内容 |
|---|---|
| `randn_{w}.npy` | 参考自己的 `x_T`（每个窗口一次）——**第 1 轮丢失的就是它** |
| `window{w}_x0.npy` | 该窗口起始的 `x`（= `denoise_step{10w}_x_in.npy`），即"历史 + 新噪声" |
| `window{w}_history_{in,out}.npy` | 进出该窗口的累积 hybrid 序列 |
| `window{w}_first_heading_in.npy` | **喂进**窗口 w 的 `first_heading_angle`；w≥1 时它就是 w−1 刷新后的值 |
| `window{w}_global_transl_in.npy` | 进入该窗口时已累积的平移 |
| `recenter{w}_{center_pos,new_root_motion,history_recentered}.npy` | 在 `Ardy._recenter_history` 里抓的每步重定心 |
| `masks_{w}.npz` / `token_masks_{w}.npz` | 每窗口的 3 个帧级 mask 与 3 个 **token** 级 mask |
| `final_hybrid_translated.npy` | 送去解码的平移后 hybrid 序列 |
| `*.jsonl`（windows/recenters/steps/fsq/denoiser/cfg/randn/masks/token_masks） | 每条事件一行、带 window phase 标签 |
| `modules/*.npy`（506 个） | 逐模块输入输出——**第 1 轮全丢的那一层** |

## 2. 多窗口对齐结果（`scripts/ardy_align.py --levels L4`，可复跑）

同一个 embedding、同一份参考注入噪声（每个窗口各自一份 `x_T`）。容差 **1e-3**。完整 JSON 见
[`artifacts/align-report.json`](../artifacts/align-report.json)（`levels.L4`）。

### 2.1 逐窗口整段（不是只看最终整体误差）

| 量 | w0（`[0,0)`，20 token，`future_len=40`） | w1（`[0,40)`，20 token，`future_len=0`） |
|---|---|---|
| 几何（`history_start/end_frame`、token 数、`future_len`） | **与参考完全相同** | **与参考完全相同** |
| token mask（history/generation/future，逐元素） | **全相等**（`future` 20 个全 0） | **全相等** |
| 窗口起始 `x`（历史＋噪声） | **0.0（逐位）**，参考 \|max\| 3.32 | **2.198e-06**，参考 \|max\| 3.53 |
| `first_heading_angle` 传入 | 0.0（逐位） | **1.625e-07**（参考 \|max\| 2.61e-03） |
| `first_heading_angle` 刷新后（喂给下一个窗口） | **1.625e-07**（参考 \|max\| 2.61e-03） | — |
| `global_transl` 传入 | 0.0（逐位） | **1.505e-06**（参考 \|max\| 0.1314） |
| `global_transl` 累积后 | **1.505e-06**（参考 \|max\| 0.1314） | — |
| `center_pos`（末帧重定心位移） | **1.505e-06**（参考 \|max\| 0.1314） | **1.580e-06**（参考 \|max\| 0.1000） |
| 重定心后的 hybrid 序列 | **2.198e-06**（参考 \|max\| 2.696） | **2.667e-06**（参考 \|max\| 2.861） |
| `center_frame_index` | 39（= 参考） | 79（= 参考） |

### 2.2 20 个去噪步逐步断言（每步单列，不让坏步藏进最大值）

两窗口 `raw`/`cfg`/`ddim` 的最差分别是 **8.476e-05 / 9.114e-05 / 1.660e-05**；
逐步明细在 `align-report.json` 的 `levels.L4.comparisons.per_step`（20 行）。
每行还断言了**参考自己的不变式**：窗口 ≥1 的 `x_out` 里历史 token 与 `x_in` **逐位相同**
（DDIM 只更新 generation token，`ardy_model.py:302-313`）——20 步全部 `True`。

### 2.3 整段与分块

| 比较 | max_abs | 参考 \|max\| |
|---|---|---|
| `motion_normalized (1,80,330)` | **7.620e-05** | 2.115 |
| `frames (1,80,330)` | **3.543e-05** | 1.682 |
| root 块 vs 参考 npz `root_positions` | **4.709e-06** | 0.960 |
| heading vs 参考 npz `global_root_heading` | **1.866e-06** | 1.001 |
| 平移后 hybrid（`final_hybrid_translated`） | **3.234e-06** | 2.861 |
| **chunk 0**：`frames` / root / heading | **3.543e-05** / **1.818e-06** / **1.554e-06** | 1.679 / 0.960 / 1.001 |
| **chunk 1**：`frames` / root / heading | **3.054e-05** / **4.709e-06** / **1.866e-06** | 1.682 / 0.960 / 1.001 |

单窗口三级（L1/L2/L3）在本轮改动后**复跑无回归**，数字与第 2 轮记录逐位相同：
L1 ≤7.0e-06、L2 ≤5.4e-05、L3 `frames` 1.869e-05。

## 3. 多窗口四条路径：断言清单与证据

| 路径 | 在哪断言 | 证据（本轮实测） |
|---|---|---|
| **① history cropping** | `per_window.geometry_matches_reference` + `token_*_mask_equal`；纯算术单测 `autoregressive_chunk_slices`；负对照 `crop=4` | 几何与参考相同；6 个 token mask 逐元素相等；`crop=4`（真截断）使 `frames` 错 **1.002** |
| **② 每步"末帧重定心 ＋ 平移累积"** | `per_window.center_frame_index` / `center_pos` / `recentered_hybrid` / `global_transl_{in,after}`；负对照 `center_index_offset=±4/±8` | 中心 39→79 与参考一致；`center_pos` ≤1.58e-06；w1 的 `global_transl` = w0 的 `center_pos`（1.505e-06）；错一格 → `frames` 错 **1.013** |
| **③ 裁剪后 `first_heading_angle` 同步更新** | `per_window.first_heading_angle_in/after`；负对照 `refresh_first_heading=False` | w0 传入 0.0、w1 传入 **0.0026115**（非零 ⇒ 不是空转），残差 1.625e-07；关掉刷新 → `frames` 错 **3.265e-03**（容差的 3.3 倍） |
| **④ 分块自回归解码** | `final`/`chunks` 对金标准；`decode_chunked_vs_whole`；负对照 `autoregressive_decode=False` | 分块解码 `frames` 3.543e-05；**整段解码 vs 分块解码差 9.496e-02（归一化空间，容差的 95 倍）** |

> **诚实限制（写进结论而不是省略）**：`crop_history_length=160` 在 80 帧上**从未真正截断**
> （`history_start_frame = max(0, history_end_frame − 160) = 0` 两个窗口都成立）。所以路径 ① 的金标准
> 只覆盖"裁剪分支被走到 + 算术正确 + mask 正确"，**不覆盖真正的截断**。因此"把 crop 长度改一帧"
> 作为负对照在本金标准上**必然空转**（实测 `crop=164` 与 `crop=160` 输出逐位相同），
> 真正的可证伪性由 `crop=4` 提供。这条已同步写进 `docs/motion-contract.md` §10 的 N13。

## 4. 负对照（容差可证伪，也能验伪——11 条）

`scripts/ardy_align.py --controls`（信号 = 端到端 `frames (1,80,330)` 的 max_abs；容差 1e-3）：

| 对照 | frames max_abs | 检出 | 预期 |
|---|---|---|---|
| clean baseline（与移植默认行为完全一致） | 3.543e-05 | 否 | 否 ✔ |
| `x_T += 0.25`（错的初始噪声） | **1.793** | 是 | 是 ✔ |
| 重定心中心早一个 token（−4 帧） | **1.013** | 是 | 是 ✔ |
| 重定心中心早两个 token（−8 帧） | **1.012** | 是 | 是 ✔ |
| 重定心中心晚一个 token（+4 帧；会夹回参考中心） | 3.543e-05 | 否 | 否 ✔ |
| `first_heading_angle` 不刷新（用陈旧值） | **3.265e-03** | 是 | 是 ✔ |
| `future_token_mask` 由 `future_mask` 推导 | **1.342** | 是 | 是 ✔ |
| `crop_history_length = 4`（真截断到 1 个历史 token） | **1.002** | 是 | 是 ✔ |
| `crop_history_length = 164`（多一个 patch；80 帧不截断） | 3.543e-05 | 否 | 否 ✔ |
| `crop_history_length = None`（完全不裁剪） | **4.423e-02** | 是 | 是 ✔ |
| 用整段解码替代分块解码 | **4.428e-02** | 是 | 是 ✔ |

两条"未检出"都是**语义上不可表达**的缺陷，不是容差太松：
① 参考中心 `history_end_frame + generation_len − 1` 本来就是序列最后一帧，往后再挪只能夹回；
② 80 帧永远碰不到 160 帧的裁剪预算（见 §3 的限制）。两者都以 `expect_detected` 显式登记，
`matches_expectation` 全为真——即"该红的都红、该绿的不许红"。

## 5. 本轮抓到并修掉的两个多窗口移植错误

两个都**只在多窗口才会现形**，都是"静默"型（不报错、不留 NaN，只是数值错 O(1)）。

### BUG-1 · `global_root_to_local_root` 忽略了 `lengths`（影响最大）

`compute_vel_xyz` / `compute_vel_angle`（`motion_rep/tools.py:58-110`）先补零，再
`v[lengths−1] = v[clamp(lengths−2,0)]`。denoiser 传的是
**`lengths = history_len + generation_len`**（`auto_latent_twostage_denoiser.py:219`），
而张量长度是 `history_len + generation_len + future_len`。单窗口时两者相等（40 vs 40），
所以第 2 轮全绿也发现不了；**4.0 s 的 w0 是 40 vs 80**（后 40 帧是被 mask 掉的 future 区）。

- 症状：w0 的 Stage-2 local root 条件全错 → 整段 `frames` 错 **6.75**（容差的 6750 倍）。
- 定位方式：用钩子 dump 的 `modules/denoiser__model_{root,body}_model__call0__*.npy` 逐级二分——
  `root_model` 输出残差 8.3e-07（说明 Stage 1 完全正确），而 `body_model` 输入残差 **11.39**
  ⇒ 定位到两者之间的 `global_root_to_local_root`。
- 修复：`compute_vel_xyz(..., lengths=None)` / 新增 `_repeat_diff_at_last(...)`；
  `_pad_last(x, lengths)`；`global_root_to_local_root(..., lengths=...)` 真正转发；
  denoiser 传 `lengths`；解码路径按 pad-mask 求 `lengths`。
- 回归测试：`tests/test_model_contract.py::test_velocity_repeats_at_lengths_minus_one_not_at_the_sequence_end`
  与 `::test_local_root_only_changes_at_lengths_minus_one`（都断言"只有 `lengths−1` 那一帧变"）。

### BUG-2 · `future_token_mask` 在无约束时必须是全 False

`latent_utils.py:207`：`motion_mask is None` 时 `future_token_mask = zeros_like(history_token_mask)`，
**不是** `future_mask` 降采样到 token。80 帧 w0 的 `future_mask` 有 40 帧为真，
而 `future_token_mask` 是 20 个 0。误用会改变 backbone 的 `src_key_padding_mask`。

- 症状：`frames` 错 **1.342**。
- 修复：新增 `convert_frame_mask_to_token_mask(...)` 把这条规则显式钉住（原实现直接
  `future_mask.reshape(...).all(-1)`）。
- 回归测试：`tests/test_model_multiwin.py::test_future_token_mask_is_zero_when_there_are_no_constraints`
  （同时断言帧级 future mask 确实是 40，证明"两者本来就不一样"）。

### 顺带修掉的**测量**错误（不是移植错误，但会误判）

- 多窗口 w1 的 `x_out` 里**历史 token 原样保留**，把整个张量拿去做 DDIM 比较会在历史段产生
  0.37 的假差异。现在只比较 generation 段，并**单独断言**历史段逐位不变。
- 逐窗口的 `x0` 比较在负对照改变窗口几何时必须按 shape 兜底（否则报 broadcast 错，
  把"检出差异"变成"测试崩了"）；per-step 的 mask 改为从**参考自己的几何**推导，
  让去噪器/mask 算术与窗口规划解耦，各自独立断言。

## 6. 第 2 轮成果（保留，复跑无回归）

- **权重转换：428/428 张量逐字节一致**（tokenizer 204 + denoiser 224；比对载荷 765,013,216 B，差异 0）；
  denoiser **224/224 张量全部被前向用到**；tokenizer 恰好 2 个 `target_cond_blocks` 未用（batch 路径不传 `target_cond`）。
- **单窗口三级对齐**：L1 tokenizer 往返 ≤7.0e-06、L2 denoiser 10 步 ≤5.4e-05、L3 `frames (1,40,330)` ≤3.9e-05。
- **性能**（M1 Max，本轮复测）：tokenizer encode 4.55 ms、decode 7.08 ms、denoiser 单步 10.98 ms、
  整段 40 帧/10 步 **108.4 ms**（p50，Metal）。冷启动 0.115 s，峰值 ≈2.8 GB。
- 三层契约陷阱（归一化 local root / `requantize` ≠ 内联重取整 / FSQ 非朴素取整）见
  `docs/motion-contract.md` §9 的 N1–N9。

## 7. 给 B 的接缝：`make_step_fn`

`src/ardy_mlx/model/ardy.py` 导出 **`make_step_fn(batch, text_feat, text_mask, *, num_steps=10,
cfg_weight=(2.0,2.0), crop_history_length=160, reuse_latent=True)`**，返回
`Callable[[DenoiseRequest], DenoiseResult]`——正是 `ardy_mlx.sampling.ArdySampler` 需要的
step callable。**只读 B 的文件确认签名，未改动 B 的任何文件**（`src/ardy_mlx/sampling/**` 与
`scripts/ardy_generate.py`、`tests/test_{sampling,generate}_*.py` 本轮零改动）。

字段映射（逐项来自 B 的 `autoregressive.py`，不复制其语义）：

| `DenoiseRequest` 字段 | 适配器怎么用 |
|---|---|
| `plan.history_len` | 干净历史帧数（必须是 patch 的整数倍，否则报错） |
| `plan.generation_len` | 必须等于 `batch.gen_horizon_len`，否则报错 |
| `history_frames`（归一化 **330** 维显式帧） | 转成 148 维 hybrid 窗口（root 块展平 + tokenizer latent） |
| `generation_noise`（`[B,nt,148]`） | 直接用作用窗噪声 `x_T`（token 数不符则报错） |
| `first_heading_angle` / `text_embeddings` / `text_mask` / `cfg_weight` / `num_steps` | 转发；后四个缺省时回落到构造参数 |
| 返回 | `DenoiseResult(generated_frames=[B, horizon, 330])`，**只含新生成的这一段**，世界坐标 |

两个设计点：

1. **`reuse_latent` 一槽缓存**。B 的循环把**显式帧**交回来，如果每步都重新 tokenize，
   就会把 denoiser 自己的（requantize 后的）latent 悄悄换成"解码再编码"的结果，链条因此漂移。
   适配器缓存自己产出的 hybrid，当传入历史是缓存帧的后缀时直接复用对应 latent
   （测试断言 `reuse_latent=False` 会给出不同结果，证明缓存不是装饰）。
2. **窗口几何是调用方的选择**。B 的 `plan_window` 把窗口右对齐到它持有的上下文上
   （默认 = `default_history_frames` = 160 帧零历史），所以 **B 的 step 0 是
   `history_len=160`、50 个 token，而参考批处理路径的 window 0 是 `history_len=0`、20 个 token**。
   适配器**按 request 说的做**（这就是契约），但这条差异已被
   `tests/test_model_stepfn.py::test_step_fn_window_geometry_is_Bs_choice_not_the_reference_batch_path`
   钉住，并作为提案写进 `docs/decisions/0005-multi-window-alignment.md` §4 交给 B 决策——
   A 侧不改 B 的文件。
   **实测（scratch 脚本里把 `mx.random.normal` 临时换成参考自己的 `randn_{w}.npy`，
   然后跑 B 真正的 `ArdySampler.run()`；未改 B 任何文件）**：B 的链条对金标准
   `frames (1,80,330)` 的 max_abs = **5.466**（金标准 |max| = 1.682）。
   另测到 `num_frames` 对 B 是"种子 + 生成"：`num_frames=80, history_frames=160` 生成 **0 帧**，
   要 80 帧输出得传 `num_frames=240`；该配置下两步的 `history_len` 都是 160（参考是 0 → 40）。
   该差异以 `xfail(strict=False)` 登记（现为 XFAIL，B 改好后会 XPASS 提醒更新提案）。
   **影响范围**：B 的 `0004` 裁定 5 已定「以 A 的 `ArdyBatch` 为标准、`ardy_generate.py` 的 MLX 后端
   默认走 A 的路径」，所以**发布路径不受此差异影响**（它本轮 80 帧对齐 3.543e-05）；
   受影响的只有「B 的独立循环能否拿金标准当端到端判据」。

## 8. 测试

A 名下 **82 个测试：81 passed + 1 xfailed**（本轮新增 26 个）。那 1 个 xfail 是**有意登记**的
已知接口差异（B 的窗口几何），`strict=False`：B 改好后会 XPASS 提醒更新提案，不会误报成回归。

| 文件 | 数量 | 说明 |
|---|---|---|
| `tests/test_io_inspect.py` / `test_io_layout.py` / `test_io_safetensors.py` | 13 / 13 / 8 | 容器与布局（第 1/2 轮） |
| `tests/test_model_contract.py` | **17** | 合成 stats，不需要 checkpoint；**新增 `lengths` 语义 2 个**（BUG-1 的回归） |
| `tests/test_model_alignment.py` | 7 | 单窗口三级金标准 + 负对照 + 张量使用率 |
| `tests/test_model_multiwin.py` | **15（新）** | 多窗口逐窗口/逐步断言、四条路径、5 个纯算术测试、负对照表 |
| `tests/test_model_stepfn.py` | **9（新；8 passed + 1 xfail）** | `make_step_fn` 契约/校验/确定性/缓存，用 B 自己的 `ArdySampler` 端到端跑通整条链，以及登记 B 窗口几何差异的 xfail |

跑法：

```sh
./.venv/bin/python -m pytest tests/test_io_*.py tests/test_model_*.py -q     # A 侧：81 passed, 1 xfailed
./.venv/bin/python -m pytest -q                                              # 全量：180 passed, 1 xfailed
```

**非我文件的失败**：本轮**没有**。全量 `pytest` 现在是 **180 passed + 1 xfailed（那个 xfail 是我自己登记的）/ 0 failed**
（B 侧 99 个：`test_generate_cli` 12、`test_generate_pipeline` 16、`test_sampling_cfg` 8、
`test_sampling_loop` 23、`test_sampling_schedule` 16、`test_sampling_window` 24）。
第 2 轮记录的 B 侧 7 个失败在本轮开始前已由 B 自己修掉；A 未触碰其文件。

## 9. 仍未确认 / 失败的部分

1. **真正的历史截断没有金标准**（80 帧 < 160 帧预算）：只有纯算术单测 + `crop=4` 负对照。
   若要覆盖，需要一次 >200 帧（`--duration ≥ 11.0`）的参考运行——**本次授权不含**。
2. **带约束路径完全没跑**（两次运行都没有 `--constraints`）：`motion_mask`/`observed_motion`
   非零、`future_token_mask` 非零那一支只有源码级复刻，没有数字验证。
3. **seed / CUDA RNG 语义**未对齐（靠注入参考 `x_T` 绕开，同 `0003`）。
4. 模型卡 326 M vs 实测 191.3 M 仍未向上游确认（A1 的残留未知）。
5. 训练数据 Bones Rigplay 1 的许可、文本编码器链路的许可（`0003` 的残留未知）。
6. B 的 `plan_window` 窗口几何与参考批处理路径不同（§7 第 2 点）——A 已钉住并提案，**决定权在 B**。

## 10. 纪律留痕

- **那台机器**：第二次运行**只跑了一次**，未重试、未补跑；未安装、未 sudo、未改任何系统/配置；
  未打印任何凭据；跑完**无进程残留**（GPU 回到 1,097 MiB 基线）；参考仓库 `git status` 前后均干净。
  只新建 `~/ardy-golden-2026-10-13-4s/`（第 1 轮的目录未动）；预检 scratch 目录已删除并确认不存在。
  残留：swap 2,241 MB → 2,255 MB（加载 8B 的页缓存/交换残留，未改任何配置）。
- **本机**：未 push；未做长训练；未使用任何合成输入（CGEvent/AX/osascript）。
- **未触碰禁改文件**：`src/ardy_mlx/sampling/**`、`scripts/ardy_generate.py`、
  `tests/test_sampling_*.py`、`tests/test_generate_*.py`、`docs/brief-claude-sampling.md`、
  `README*`、`NOTICE`、`LICENSE`、`pyproject.toml`、根 `.gitignore`。
- **权重派生数据不入库**：`artifacts/golden-run-4s/.gitignore` 堵 `raw/` + `out/`（与第 1 轮同规则，
  不动根 `.gitignore`）；`git ls-files` 审计见 §11。
- `docs/motion-contract.md` 只**补"实测确认"注记**（新增 §10 与 N10–N13），
  未改任何字段/维度/顺序。

## 11. 入库审计

```
$ git ls-files | grep -E '\.safetensors$|golden-run[^/]*/(raw|out)/'   # 期望：零命中
```
结果：**零命中**。上面那条命令同时覆盖 `checkpoints/**/*.safetensors` 与
`artifacts/golden-run{,-4s}/(raw|out)/**`。入库的只有名字/形状清单
（`artifacts/weight-manifest.json`、`weight-verify.json`）、对齐报告
（`artifacts/align-report.json`）、运行记录与仪器化（`RUN-RECORD.md`、`run.sh`、
`sitecustomize.py`、`run.log`、`gpu_samples.txt`、`mem_samples.txt`）。
