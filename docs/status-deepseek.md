# 状态 · DeepSeek（A：权重转换 / 模型前向 / 数值对齐）

- **当前轮次**：A 侧第 4 轮（收尾：`posed_joints` 骨架 FK + 最终端到端验收 + 文档收口）。起于 HEAD `4f03545`。
  第 3 轮的记录保留在 §1–§11；**A 侧最终状态见 §12**。
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

## 8. 测试（第 3 轮计数；第 4 轮见 §12.3）

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

---

# 12. 第 4 轮（收尾）· `posed_joints` 骨架 FK + 最终端到端验收

起于 HEAD `4f03545`。本轮**没有再跑参考实现**（`0005` 已裁定不买第三次运行），
所有数字都来自已有的两次金标准 + 本机 MLX。

## 12.1 `posed_joints` 的骨架 FK（`0004` 裁定 4 落地）

**问题**：`posed_joints` 之前是占位（`local_joints_positions` + `root_pos`），与金标准差 **0.132**。
**根因**：契约要的是 `inverse()` 的世界关节位置（`ardy_motionrep.py:259-264`），
它需要骨架的**父子关系 + 静息姿态**，而这两样都**不在 checkpoint 里**——在参考的源码里。

**做法**（新文件 `src/ardy_mlx/model/skeleton.py`，纯 A 侧）：

1. 从参考源码逐字转录 `CoreSkeleton27`：关节名/父子关系（`skeleton/definitions.py:348-381`，27 关节，根 `Hips`）
   与静息姿态 `neutral_joints`（`ardy/assets/skeletons/cskel27/joints.p`，27×3，写成 Python 字面量，
   **不落任何 `.npy`**——入库审计对它零容忍）。
2. 逐行移植 `skeleton/kinematics.py:87-171` 的 `batch_rigid_transform`
   （`rel_j = neutral_j − neutral_parent(j)`、`T_j = [[R_j, rel_j],[0,0,0,1]]`、`G_j = G_parent @ T_j`，
   根取 `G_root = T_root`；`posed_joints = G[..., :3, 3] + root_positions`，
   `global_rot_mats = G[..., :3, :3]`；支持任意前导 batch 维）。
3. 移植 `geometry.cont6d_to_matrix` 与 `transforms.global_rots_to_local_rots`，
   于是 `frames (T,330) → posed_joints (T,27,3)` 这条链与参考 `inverse()` 完全一致。

**对齐数字（对 `artifacts/golden-run{,-4s}/out/*.npz` 的 `posed_joints`）**

| 比较 | 2.0 s / 40 帧 | 4.0 s / 80 帧 |
|---|---|---|
| FK(金标准自己的 `local_rot_mats`+`root_positions`) | **2.4e-07** | **1.2e-07** |
| 从**金标准特征向量**走的完整链 | **3.0e-07** | **4.8e-07** |
| **出货路径**（我们自己的 `frames`）→ FK | **5.178e-06** | **4.813e-06** |
| 同链的 `global_rot_mats` vs npz | 1.0e-06 | 8.9e-07 |
| **占位实现（补前）** | **0.132** | **0.236** |
| 负对照：父子关系拍平 | **6.508e-01** | **6.546e-01** |

- 占位是 **0.132**（正是 `0004` 裁定 4 记录的数）；补后 5.18e-06 / 4.81e-06，
  与 `frames` 自身残差（1.87e-05 / 3.54e-05）同量级，而 `posed_joints` 的量级是 1.68。
- 负对照（错误的父子关系）→ 0.65，**是残差的 12.5 万倍**：这条比较确实能红。
- 三项都进了 `scripts/ardy_align.py`（L3/L4 的 `posed_joints_vs_npz`、
  `posed_joints_placeholder_vs_npz`、`posed_joints_flat_hierarchy_vs_npz`），
  后两项登记为**诊断项**（`DIAGNOSTIC_KEYS`，按设计必须"大"，不参与通过/失败判定）。
- 单测：`tests/test_model_skeleton.py`（15 个，无需 checkpoint）；金标准对齐测试在缺数据时
  **显式 skip 并说明原因**。

**仍未接上（B 侧，我按要求没有动 B 的文件）**：`scripts/ardy_generate.py` 的
`_common_outputs` 仍用 `synthesize_posed_joints(frames, layout)`，sidecar 仍写
`extra.posed_joints_source = "local_joints_positions (root-local) + root_pos; real FK not implemented"`。
接法（一行）与文案已写进 `docs/motion-contract.md` §11 的 N14；**接上之前，CLI 落盘的 `posed_joints`
不是契约值**（2.0 s 差 0.132 / 4.0 s 差 0.236）。

## 12.2 最终端到端验收（结论钉死）

### (1) CLI 多窗口复现金标准（4.0 s / 80 帧 / 2 窗口，A 的 `ArdyBatch`）

以 `0004` 裁定 5 的标准路径跑 B 自己的 `scripts/ardy_generate.py --backend mlx`
（未改其任何文件），并把参考自己的 `x_T`（`randn_0/1.npy`，每窗口一份）注入：

| 比较 | max_abs | 参考 \|max\| |
|---|---|---|
| **`frames (1,80,330)`** | **3.543e-05** | 1.682 |
| `motion_normalized` | 7.619e-05 | 2.115 |
| root 块 vs npz `root_positions` | 4.709e-06 | 0.960 |
| heading vs npz | 1.866e-06 | 1.001 |
| **`posed_joints`（FK）vs npz** | **4.813e-06** | 1.682 |
| npz 里实际落盘的 `posed_joints`（仍是 B 的占位） | 2.363e-01 | 1.682 |

与 `0005` 的 3.543e-05 **逐位同值**（同一条路径、同一份注入噪声）。

### (2) 单窗口回归（2.0 s / 40 帧 / 1 窗口）

| 比较 | max_abs | 参考 \|max\| |
|---|---|---|
| **`frames (1,40,330)`** | **1.869e-05** | 1.679 |
| `motion_normalized` | 3.931e-05 | 2.115 |
| root / heading vs npz | 5.141e-06 / 2.991e-06 | 0.960 / 1.001 |
| `posed_joints`（FK）vs npz | 5.178e-06 | 1.679 |

与第 2/3 轮记录**逐位同值**，无回归。

### (3) 两条验收都固化进了 `scripts/ardy_align.py --cli-acceptance`

复跑命令（一条命令给出上面两张表）：

```sh
./.venv/bin/python scripts/ardy_align.py --levels L1 L2 L3 L4 --cli-acceptance \
    --controls --negative-control --repeats 20 --json artifacts/align-report.json
```

**过程中查实两处 B 侧接线问题**（A 只报告、不改 B 的文件；细节见契约 §11 的 N16）：

1. CLI 的 `--history-frames` 默认是 `None`，落到模型是 `crop_history_length=None`，
   会同时关掉裁剪、`first_heading_angle` 刷新与分块解码 → 与金标准 **4.42e-02**（容差的 44 倍）。
   **必须显式 `--history-frames 160`**（= 参考 batch 默认）才能复现 1.869e-05 / 3.543e-05。
   建议 B 把默认值改成 `PRESETS[...].default_history_frames`。
2. `--noise-file` 在 ≥2 窗口时**注入不进去且完全静默**：判据
   `injected.shape[1:] == shape[1:]` 忽略了 batch 轴（`(W,1,10,148)` vs `(1,10,148)`），
   于是每个窗口都回落到 MLX RNG（实测整段错 **4.42**，无警告）。
   单窗口文件是好的。本轮多窗口注入用"进程内替换 `mx.random.normal`"完成（与第 3 轮同一手法）。

### (4) 全量测试

```
$ ./.venv/bin/python -m pytest -q
199 passed, 1 xfailed in 36.13s
```

- 第 3 轮是 **180 passed + 1 xfailed**；本轮 **+19**（15 个骨架单测/对齐 + 4 个 CLI 端到端验收），
  **没有任何测试被删或改弱**；两个既有测试因新增诊断项各加了三行（诊断项必须"大"，其余仍 ≤ 1e-3）。
- 那 1 个 xfail 仍是**有意登记**的 B 侧窗口几何差异（`strict=False`，B 改好后会 XPASS 提醒），保留。

### (5) 性能复测（本机 M1 Max / 10 核，**整机被其它 agent 占用**）

整段 80 帧 / 20 步 = 2 窗口 × 10 步 DDIM，注入参考 `x_T`；10 遍记录（`--bench-only`），
**机器全程被其它 agent 占用，loadavg 13.5–20.9**，所以下面是**上界**而不是安静机器上的能力值：

| 设备 | 80 帧/20 步 p50（记录里最低的一遍） | 同一路径全部 10 遍 p50 区间 | denoiser 单步(batch3) | 冷启动 | 峰值内存 |
|---|---|---|---|---|---|
| **Metal** | **527.1 ms**（p90 1045.3 ms，loadavg 16.95） | 527–977 ms | 10.9–27.0 ms | 0.12–0.38 s | **1010 MB** |
| **CPU** | **1124.7 ms**（p90 1144.9 ms，loadavg 18.37） | 1125–3312 ms | 63.6–152.0 ms | 0.17–0.62 s | **947 MB** |

- `benchmarks` 是每设备 p50 最低的那一遍，`benchmark_passes` 是全部 20 条记录（含每遍 loadavg）。
- 若在对齐同一个进程里连跑 L1–L4 再测，峰值内存会记到 2.75/2.82 GB——那是对齐级别留下的分配，
  不代表出货路径；上表的 1010/947 MB 来自只加载权重 ＋ 跑一段 80 帧的干净进程。
- 新增的 FK 本身很便宜：80 帧 p50 **0.97–2.80 ms**（CPU/Metal），可以忽略。

## 12.3 本轮改动清单（全部 A 侧）

| 文件 | 改动 |
|---|---|
| `src/ardy_mlx/model/skeleton.py` | **新增**：CoreSkeleton27 定义 + FK + 6D/全局→局部 + `posed_joints_from_features` + 占位对照 |
| `src/ardy_mlx/model/__init__.py` | 导出 `Skeleton` / `core_skeleton` / `posed_joints_from_features` |
| `scripts/ardy_align.py` | L3/L4 增加 `posed_joints` 三类比较 + 诊断项机制；`--cli-acceptance`；benchmark 增加 80 帧/20 步与 FK；`benchmark_passes` |
| `tests/test_model_skeleton.py` | **新增** 15 个（结构/FK/旋转转换/金标准对齐/占位对照） |
| `tests/test_model_cli_acceptance.py` | **新增** 4 个（CLI 端到端 2 窗口/1 窗口/落盘现状） |
| `tests/test_model_alignment.py`、`tests/test_model_multiwin.py` | 各自新增 1 个 + 3 个断言（诊断项必须大） |
| `artifacts/align-report.json` | 重跑：L1–L4 + CLI 验收 + 11 条负对照 + 20 条性能记录 |
| `README.md`、`README.zh-CN.md` | 由用户授权改写为"已完成" |
| `docs/motion-contract.md` | 只加注记 §11（N14–N16），未改任何字段/维度/顺序 |

**未触碰**（禁改清单）：`NOTICE`、`LICENSE`、`pyproject.toml`、根 `.gitignore`、
`src/ardy_mlx/sampling/**`、`scripts/ardy_generate.py`、`tests/test_{sampling,generate}_*.py`、
`docs/brief-claude-sampling.md`。

## 12.4 A 侧最终状态：已完成的定义

1. **权重转换**：`tokenizer` 204 + `denoiser` 224 = **428/428 张量逐字节一致**，
   比对载荷 **765,013,216 B，差异 0**；denoiser 224/224 张量全部被前向用到
   （`artifacts/weight-verify.json`、`weight-manifest.json`）。
2. **三级对齐（单窗口）**：L1 tokenizer 往返 ≤7.0e-06、L2 denoiser 10 步 ≤5.4e-05、
   L3 整段 `frames (1,40,330)` **1.869e-05**。
3. **多窗口对齐**：窗口几何/token mask 与参考逐元素相同，重定心/平移/heading 刷新 ≤2.7e-06，
   20 步逐步断言（raw 8.48e-05 / CFG 9.11e-05 / DDIM 1.66e-05），整段 **3.543e-05**。
4. **`posed_joints` 真 FK**：出货路径 **5.18e-06 / 4.81e-06**（占位 0.132 / 0.236），
   负对照 0.65；A 侧 API 与测试齐备（**B 侧 sidecar 未接**）。
5. **CLI 端到端**：`--backend mlx`（A 的 `ArdyBatch`）+ `--history-frames 160` + 注入参考 `x_T`
   → 2.0 s **1.869e-05**、4.0 s **3.543e-05**，可一键复跑（`--cli-acceptance`）。
6. **`make_step_fn` 接缝**：B 的 `ArdySampler` 可直接调用；窗口几何差异已钉进 xfail + 提案。
7. **测试**：**199 passed + 1 xfailed**（xfail = B 侧窗口几何差异，`strict=False`）。
8. **性能**：见 §12.2(5)；本机共享，数值是上界。

## 12.5 仍未验证 / 未接（保留为残留未知）

1. **B 的 CLI/sidecar 未接 FK**：落盘 `posed_joints` 仍是占位；B 改一行即可（契约 N14）。
2. **CLI 两个默认值/接线问题**（`--history-frames` 默认 `None`；`--noise-file` 多窗口静默失效），
   决定权在 B（契约 N16）。
3. **`--constraints` 路径**（`motion_mask`/`observed_motion` 非零、`future_token_mask` 非零）
   两次参考运行都没跑，只有源码级复刻。
4. **真正的历史截断**（>200 帧）没有金标准；覆盖来自纯算术单测 + `crop=4` 负对照（N13）。
5. **seed / CUDA RNG 语义**未对齐：靠注入参考 `x_T` 绕开；MLX 侧按窗口 `mx.random.split`。
6. **B 的独立循环（`ArdyAutoregressive`）不能用金标准判**：窗口框架不同（160 帧零种子 + 40 生成 = 50 token，
   参考是 0 历史 + 40 生成 + 40 掩码未来 = 20 token），注入参考噪声后 **5.466**（金标准 |max| 1.682）。
   出货路径不受影响（走 A 的 `ArdyBatch`）；已按 `0005` 的 lead 批准写成 xfail，**保留**。
7. 模型卡 326 M vs 实测 191.3 M；训练数据 Bones Rigplay 1 与文本编码器链路的许可。
8. §2.4 MotionBricks/Unity 转换：仍待定（`0002` 裁定 v1 不做）。
9. 8B 文本编码器未移植：v1 只吃预计算 embedding（预计算的许可/可复现性仍是开放项）。

## 12.6 纪律留痕（本轮）

- **没有再跑参考实现**（`0005` 已裁定不买第三次运行）；本机之外未碰任何机器。
- 未 push；未做长训练；未安装任何依赖；未使用任何合成输入（CGEvent/AX/osascript）。
- `docs/motion-contract.md` 只**追加** §11 注记（N14–N16），未改字段/维度/顺序。
- 权重派生数据**一律不入库**：本轮的 CLI 产物落在 `tempfile.mkdtemp()` 里、读完即删；
  契约里的 `neutral_joints` 以 Python 字面量入库（不是 `.npy`）。审计见 §12.7。
- 未触碰禁改清单（见 §12.3 末）。

## 12.7 入库审计（本轮结束）

```
$ git ls-files | grep -E '\.(safetensors|npz|npy)$|golden-run[^/]*/(raw|out)/'
（零命中）
```

结果：**零命中**。

## 12.8 可视化预览（回答"效果怎么样，能预览吗"）

**新增交付：[`docs/PREVIEW.md`](PREVIEW.md) ＋ `artifacts/preview/`（5 支骨架 GIF，合计 ≈ 2.0 MB）
＋ [`scripts/ardy_preview.py`](../scripts/ardy_preview.py)（纯 numpy 光栅化 + `subprocess` 调 `ffmpeg`）。**

要点（细节与复现命令见 `PREVIEW.md`）：

- 用 npz 里的 **世界坐标 `posed_joints`** 画 27 关节骨架，叠加 XZ 地面网格、地面影子、
  以及 `foot_contacts` 的接触标记（真 = 实心绿盘 + 连到脚关节的细线，假 = 琥珀空心环）；
  HUD 标出 prompt / 模型 / 帧数 / fps / 金标准还是我们 / 噪声来源 / 生成 git sha / 渲染器指纹。
- **保真度证据（同源，注入参考 `x_T`）**：`posed_joints` max_abs **5.18e-06**（2.0 s）、
  **4.81e-06**（4.0 s），参考量级 |max| **1.68**；`foot_contacts` **逐位相同**（0/40、0/80）。
  这两个数与 `artifacts/align-report.json → cli_acceptance.posed_joints_vs_npz` **逐位相同**，
  说明本次复现就是当时那次验收运行。330 维 `frames` 沿用已登记的 **1.87e-05 / 3.54e-05**。
- **渲染成图的差异**：同源两支 GIF 在 **raster 层**只差 **0.0726 %** 的像素、
  **最大通道差 2/255** → 视觉上无法区分。256 色 GIF 那一层是 0.478 %、最大 71，
  差额全部来自调色板量化把 1–2 灰阶的边缘差放大成另一个调色板索引（两个数字要一起读）。
- **诚实的边界**：① 骨架预览，不是角色/蒙皮渲染；② 两边都 `--no-postprocess`，
  所以有 foot skating（脚会打滑），接触帧之间滑动是预期行为；③ 主交付的
  `ardy-mlx-2s.gif` / `ardy-mlx-4s.gif` 用的是**我们自己的噪声**，所以是"同风格不同样本"，
  要逐位同源得看额外的 `ardy-mlx-2s-xT.gif`；④ 8B 文本编码器未移植，本次用参考 dump 的
  预计算 `text_feat.npy`；⑤ 参考只跑过两次，多窗口只有 4.0 s 这一支。
- **顺手证实的两件事**：`--seed 0` 在 MLX 上可复现（连跑两次 `frames` 逐位相同）；
  4.0 s 的自回归拼接处**不是**逐帧位移的最大步（边界/最大 = 0.950，且注入版与金标准三项统计完全一致），
  即续接平滑、无抽搐。
- 一个采样事实：参考 seed 0 几乎原地走（2 s 位移 0.131 m），我们自己采样的两支分别走
  1.096 m / 3.819 m。**不是实现差异**——注入同一 `x_T` 后复现到 5e-06。
- 纪律：未 push；未装依赖（`.venv` 未变，无 matplotlib/mujoco/smplx/torch 新增）；
  未跑参考实现；未用合成输入；只动了 `scripts/ardy_preview.py`、`docs/PREVIEW.md`、
  `docs/status-deepseek.md`、`artifacts/preview/**`。GIF 与相机 JSON 入库，
  `golden-run*/{raw,out}` 与任何 `.npz/.npy` **一律不入库**（§12.7 的审计仍为零命中）。

## 13. 文本编码器已迁出：本仓库只留指针（2026-10-13）

**文本编码器不再属于本仓库。** 它已独立成 `~/dev/llm2vec-mlx`（仓库名 `llm2vec-mlx`），
本仓库与它之间**只有向量文件这一条接口**：

- **接口**：`(1, 1, 4096)` **float32** 池化向量（与 `artifacts/golden-run/raw/text_feat.npy` 同形）
  ＋ 一个同名 `.json` sidecar，sidecar 至少带 `encoder`（参考向量
  `"reference-llm2vec-bf16"`、MLX 向量 `"mlx-llm2vec"`）与 `bits`；**缺 `encoder` 字段读作参考向量**。
- **本仓库的立场不变**：查看器仍**优先使用参考向量**；MLX 向量只在没有参考向量时兜底。
  本节只是指针，不改任何既有语义（`0004` 契约、`/api/plan` 对齐、时长整数倍等一律不动）。
- **原来那份判据文档 `docs/decisions/0006-mlx-text-encoder.md` 已删除**，内容整体搬到
  `llm2vec-mlx` 的 `docs/decisions/0002-mlx-text-encoder.md`（编号按新仓库，阈值一字未改）。
  随之迁走的还有：文本编码器包（原 `src/ardy_mlx/text/**`，未跟踪、从未提交）、
  参考向量（原 `artifacts/reference-embeddings/*.npy`，权重派生数据，不入库）、
  以及"仅编码器"参考运行的记录（原 `artifacts/reference-encoder-run/**`）。
- 本仓库因此**回到干净状态**：无 `src/ardy_mlx/text/**`、无 `artifacts/reference-*`。

## 14. 第 6 轮（2026-10-13）· 查看器接上本机编码：自由文本 → 本机编码 → 本机生成

**本节只加注记，不改既有契约与数值。** §13 说"查看器优先参考向量"，本轮把它连同
"本机编码"一起落进代码：查看器现在可以输入**任意 prompt**，缺向量就在本机编码。

### 14.1 接口与缓存

- **跨仓调用走文件接口，不改 `llm2vec-mlx` 一个字节**：`subprocess` 跑它 `.venv` 里的
  `scripts/encode.py --text … --bits 4 --out <tmp>/<prompt-hash>.npy --quiet`，读回
  `.npy` ＋ `.json`。**没有**跨仓 import、没有拷贝它任何代码、没有改它任何文件。
  只试对方 `.venv` 自己的解释器；本仓库的 `sys.executable` **不是**候选（那是另一套依赖，
  会让"编码器缺失"变成"编码器跑在错的 python 上"）。
- **缓存键 `(prompt_hash, encoder_id)`**：参考向量在 `artifacts/embeddings/<prompt-hash>.npy`，
  本机向量在 `artifacts/embeddings/<encoder-id>/<prompt-hash>.npy`，**一个 prompt 可同时持有两者**。
- **量化全名入库**：`encoder_id = mlx-llm2vec-4-bit-group-32-last-2-layers-embed-tokens-at-8-bit`，
  侧车同时写 `quantization = "4-bit group 32, last-2 layers + embed_tokens at 8-bit"`。
  名字从 artifact 自己的 manifest 读，不硬编码；纯写 "4-bit" 会与**过不了线**的那个 artifact
  （纯 4-bit group 32，最小 cos 0.9828 < 0.985）同名，所以必须写全。
- **优先级**：参考向量 → 当前档本机向量 → 现场编码 → 否则**具名报错**。
  绝不退到别的 prompt、也绝不用别的编码器的向量（`encoder_variant_not_cached` 只针对显式指定编码器的查询）。

### 14.2 现场验证（HTTP，不点界面、不合成输入）

服务 `nohup ./.venv/bin/python scripts/ardy_viewer.py --port 8850`（旧 PID 56914 已先停掉，
新 PID 86755）。全新 prompt `A person walks forward and waves.`：

| 量 | 值 |
|---|---|
| `timing.encode_ms`（编码子进程，含冷启动） | **1367 ms**（首次）；20 条全新 prompt 的 p50 **974 ms** / p90 993 ms |
| `timing.generate_ms`（采样） | 221 ms（首次）；p50 129 ms |
| `timing.total_ms`（服务端整体） | **1598 ms**（首次）；p50 1114 ms |
| 向量来源 | `local` / `source_kind: local-mlx` |
| 量化档 | 4-bit group 32, last-2 layers + embed_tokens at 8-bit |
| 同一 prompt 再来一次 | `encode_ms: null`、`cache_hit: true`、**总 111 ms**、frames 逐位相同 |

**旧 prompt `A person walks forward.` 仍然优先用参考向量，且逐位未变**：

- `source: reference` / `encoder_id: reference-llm2vec-bf16` / `encode_ms: null`；
- 与"完全停用本机编码器（`--no-local-encoder`）的那台服务"对比，
  `posed_joints` / `root_positions` / `foot_contacts` **三个数组逐位相同**；
- 对金标准 npz 的 `posed_joints` `max_abs = 5.1781535e-06`，与 §7/`align-report.json`
  记录的 **5.178e-06** 一致（不是新的、更好看的数字）。

### 14.3 负向证据（都是具名报错，原文）

```
POST /api/plan {"prompt": "<2010-token prompt>"}
502 local_encoder_failed
message: the local encoder exited 3 for prompt hash a532f2b3a994c000
detail : encoder exit code: 3
         encoder refused: prompt needs 2010 tokens, over the reference's max_length 512;
         truncation would change the pooled vector, so the encoder refuses to guess

POST /api/plan {"prompt": "A person bows politely and stands up."}   # encoder 不可用
503 local_encoder_unavailable
message: the local text encoder is not usable: missing scripts/encode.py (…);
         an interpreter (tried: …/.venv/bin/python, …/.venv/bin/python3);
         the 4-bit weight manifest (…); run llm2vec-mlx's scripts/build_weights.py first
```

`encoder_id` 混用（把 8-bit 向量塞进 4-bit 的键、或反过来）在测试里是**具名拒绝**
或"当作 miss 重新编码"，绝不静默复用：
`tests/test_viewer_local_encoder.py::{test_negative_mixing_two_encoders_vectors_is_refused,
test_negative_a_cached_vector_is_never_returned_for_a_different_encoder_key,
test_negative_a_mislabelled_file_in_an_encoders_directory_is_refused}`。

### 14.4 性能与内存（loadavg 8.41 / 9.15 / 10.07 → 8.96 / 9.23 / 10.08，10 核）

| 量 | 值 |
|---|---|
| 编码（整只子进程，HTTP 20 条全新 prompt） | p50 **974 ms**、p90 993 ms |
| 采样 2.0 s（热缓存） | p50 **108 ms**（忙时 129 ms） |
| 采样 4.0 s（热缓存） | p50 **257 ms** |
| 查看器进程峰值 RSS | **1030 MiB**（编码循环中），空闲 938 MiB，起始 334 MiB |
| 编码子进程峰值 RSS | 4-bit **5190 MiB**；8-bit **7862 MiB** |

编码器与查看器是**两个进程**：5.2 GB 不落在 ARDY checkpoint 的地址空间里，
编码结束查看器回到 ~0.9 GB。这也是它必须是子进程的原因之一（另两个：两仓解耦、耗时可见）。

### 14.5 测试与纪律

- 本仓库 **332 passed, 1 xfailed**（原 303 + 1，只增不减）。新增
  `tests/test_viewer_local_encoder.py`（27 条，其中 2 条要真编码器、缺席时**带原因 skip**）。
- `llm2vec-mlx` **一个字未改**（`git status --short` 为空，HEAD 仍 `7708d08`）。
- 未 push；权重/向量不入库（§11 的审计仍零命中，见本轮汇报）。

---

## 15. 第 7 轮（2026-10-13）· 前端的 400：一个没解析出 prompt 的请求

**本轮只动前端与文档**（`viewer/app.js`、`tests/viewer_js_guard.test.mjs`、
`tests/test_viewer_http.py` 新增一条、`docs/VIEWER.md`、本文件）。`src/**`、
`scripts/ardy_viewer.py` 与既有测试断言**一个字未改**；`~/dev/llm2vec-mlx` 未触碰。

### 15.1 现象与成因

浏览器控制台：

```
app.js:275  POST http://127.0.0.1:8850/api/plan 400 (Bad Request)
  getJson @ app.js:275
  generateClip @ app.js:463
  （匿名） @ app.js:725
```

服务端具名 `no_prompt_given`。引擎与编码器都正常：带 prompt、`duration=8`、
`noise=random` → **200**（160 帧 / 4 窗口 / 752.7 ms）。**错在前端**：

- 服务端 `plan` 路由读 `params.get("prompt")`，`_embedding_for_request` 对 `None`/`""`
  具名拒绝（`scripts/ardy_viewer.py`）；
- **`JSON.stringify` 会丢掉 `undefined`**：prompt 没解析出来时，body 不是"空 prompt"，而是
  **根本没有 `prompt` 键** —— `{"seed":0,"duration":2,"noise":"golden"}`。失败的请求**看起来
  不小，只是少了一个键**；
- 触发点**没有共用一道守卫**：唯一的检查在 `generateClip` 内部，各触发点各自读控件；第 725 行
  `Reload` 的处理函数**没有 await `loadHealth()`**（那正是填下拉框的往返）就先调
  `generateClip()`，于是在下拉框还没就绪时读了控件，接着把**服务端的错误名**当红框弹出来 ——
  页面侧的"还没准备好"被写成了 `no_prompt_given`。

**能下与不能下的结论**：`bc00b26` 的源码在空 prompt 时 `requestedPrompt()` 已返回 `null`，
所以"空文本框 ＋ 空下拉"那条路是**弹红框、不发请求**；真正会丢字段的，是从**未解析**的 prompt
直接拼 body 的路径 —— 这次消掉的就是这一类（body 生成器拒绝未解析 prompt，**且**每个触发点都先
过守卫）。仓库历史里没有留下那一瞬的中间态提交，**无法**从本树复现当时的网络 400，如实记为残留
未知（§15.4）。

### 15.2 改了什么

- `viewer/app.js`
  - 新增**纯函数区**（`// --- pure-prompt-guard: start … end`）：`resolveRequest()` /
    `shouldPlan()` / `planBody()` / `referenceDecision()` —— 不碰 DOM、不 fetch、不读 `state`；
  - `requestedPrompt()` 退化为"把控件值塞进 `resolveRequest`"的适配器；
  - **所有**触发点（`Generate` / `Enter` / `Go` / `Reload` / boot）统一走 `planIfReady()`：
    解析不出 prompt 就**不发请求**、`clearError()`、状态栏说一句人话
    （`Nothing to generate yet (<trigger>). Type a prompt, or pick a cached embedding from the list.`），
    **不弹红框**；
  - `generateClip(resolved)` 保留 `shouldPlan()` 兜底，body 一律由 `planBody()` 生成；
  - `Reload` 先 `await loadHealth()` 再判定，消掉竞态；
  - `/api/reference` 先按 `/api/health.golden_references` 判定：该时长没有金标准就**不请求**、
    toggle 自动取消勾选、面板写明（消掉 404 噪音）。`golden_reference_prompt_mismatch` 是**真
    拒绝**，仍按名上报；
  - 选中项 / 时长 / 步进三处原本各写一遍的"作废已载入参考"合并为 `refreshGoldenForControls()`。
- `tests/viewer_js_guard.test.mjs`（新，Node，12 条；纯函数区从**出货文件**里现读现跑）。
- `tests/test_viewer_http.py` 新增 1 条（`subprocess` 调 node；无 node 时带原因 skip）。
- `docs/VIEWER.md`：§4 控件说明、§6 边界 4 ＋**新增边界 13**（前端不做浏览器测试）、错误表两行、
  §8.1（node 单测）与 §8.2（本次 400 成因）。

### 15.3 验证（原始输出）

```
$ node --check viewer/app.js
（无输出，exit 0）

$ node tests/viewer_js_guard.test.mjs
✔ empty text box + empty dropdown: nothing resolves, so nothing is planned
✔ a dropdown that is not ready yet is not a prompt either
✔ whitespace is not a prompt: it falls through to the dropdown
✔ text in the box is used verbatim (trimmed), and wins over the selection
✔ a selected cached embedding is planned by its prompt_hash
✔ negative control: bypassing the guard produces a body with no prompt key
✔ the reference is only requested for a duration /api/health says has a golden run
✔ a configured golden run whose npz is missing is also known in advance
✔ reference decisions with no runs, no duration, or malformed runs never fetch
✔ the guard region is pure: it runs with no document, no fetch, no state
✔ wiring: every trigger reaches the guard, and the body is built by planBody()
✔ negative control: the wiring checks fail on a guard-free copy of the page
ℹ tests 12   ℹ pass 12   ℹ fail 0

$ ./.venv/bin/python -m pytest -q
333 passed, 1 xfailed in 35.23s        （原 332 + 1 xfailed，只增不减）

$ curl -s -X POST http://127.0.0.1:8850/api/plan -H 'Content-Type: application/json' \
      -d '{"seed":0,"duration":2,"noise":"golden"}'
HTTP 400  {"error": "no_prompt_given", ...}

$ curl -s -X POST ... -d '{"prompt":"A person walks forward and waves.","seed":0,"duration":8,"noise":"random"}'
HTTP 200  frames 160 | windows 4 | duration_s 8.0 | generate_ms 752.7 | source local-mlx
```

**负对照是跑过的**：在 `/tmp` 的一份**拷贝**上（仓库文件未动，sha256 前后一致）分别注入四种绕过 ——
① 去掉 `generateClip` 的 `shouldPlan`；② `Reload` 不 await `loadHealth`；③ `shouldPlan` 恒真；
④ `loadReference` 先请求后判定 —— `node tests/viewer_js_guard.test.mjs` 分别以
`fail 2 / fail 2 / fail 4 / fail 2` 退出 1。守卫被绕过时测试**确实会失败**。

### 15.4 残留未知

- **浏览器运行时仍需人眼确认**（新增边界 13）：静态检查 ＋ Node 逻辑单测证明不了"点击确实走到
  这段逻辑"，也证明不了渲染。全绿而页面坏掉，是这套测试**不声称**覆盖的情形。
- 那次网络 400 的确切中间态提交无法从本树复现（§15.1 末）。
- `golden_reference_prompt_mismatch`（404）仍在：页面只知道**哪些时长**有金标准，不知道金标准的
  prompt 文本，所以用**本地**向量去比对 2.0 s 金标准时仍会被具名拒绝 —— 这是真拒绝，不消。
- 选中下拉项**不会**自动重新生成（只是作废并重载参考）。若期望"选完即生成"，那是行为变更，
  需另行确认。

---

## 16. 第 8 轮（2026-10-13）· System 1 落地：代码掌流程，Jev 只提供判断

**先回答上一轮的状态**：第 7 轮（空 prompt 守卫 ＋ 404 噪音）的前端修复**已经落地**，
并在本轮开始后由父 agent 提交为 `0238624 viewer: one guard for every trigger, and no
/api/reference without a golden run`（17:31:58，含 `docs/VIEWER.md`、`docs/status-deepseek.md`、
`tests/viewer_js_guard.test.mjs`、`viewer/app.js`、`tests/test_viewer_http.py`）。
我**接着它继续**，一个字未丢；本轮全部改动都在它之上。

本轮新增/改动：`src/ardy_mlx/router/**`（新）、`scripts/ardy_viewer.py`（加路由接线，既有行为未动）、
`viewer/{app.js,index.html,style.css}`（气泡）、`tests/test_router_jev.py`（新）、
`tests/test_viewer_router.py`（新）、`tests/test_viewer_http.py`（加 5 条）、
`tests/viewer_js_guard.test.mjs`（12 → 20 条）、`pyproject.toml`（加 `router` extra）、
`docs/VIEWER.md`（§9 等）、`docs/motion-contract.md`（§14 只加注记）、本文件。
**未碰**：`src/ardy_mlx/{model,sampling,io}/**` 的既有行为、`scripts/ardy_{generate,align}.py`、
`tests/test_model_*.py`、`tests/test_io_*.py`、`NOTICE`、`LICENSE`、根 `.gitignore`、
`~/dev/llm2vec-mlx`。

### 16.1 依赖：装了什么，为什么

`typesafe-sdk==0.7.4` 装进**本仓库 `.venv`**（纯 Python）。它带入 `httpx2`、`pydantic`(+core)、
`tenacity`、`truststore`、`anyio`、`h11`、`idna`、`typing-extensions`、`annotated-types`、
`typing-inspection`。`pyproject.toml` 里它是**可选 extra** `router`，不是硬依赖：

```
./.venv/bin/python -m pip install -e '.[router]'
```

不加这个 extra，查看器照样起，`/api/health.router.state` 具名报 `router_unavailable`。
所有路由测试都用**假 client**，因此核心测试既不需要这个包，也不需要网络。

**接口细节以已装 SDK 的类型定义为准**（`typesafe_sdk/_core/question_types.py`、
`response_types.py`、`_schemas/models.py`、`constants.py`）。文档站点在我这里是**通的**
（`https://docs.typesafe.ai/llms.txt` 返回 200），所以我也读了 live docs 的
`sdk/python/*`、`primitives/*`、`confidence`、`patterns/confidence-routing`，没有臆造字段。

### 16.2 结构

- **`src/ardy_mlx/router/jev.py`**：状态构造、问句集、答案归一化、映射与阈值（纯函数）、
  判定缓存、线程与 deadline、七种具名降级、按键解析。**key 只有 `resolve_api_key()` 一个入口**。
- **问句集**（一次请求，8 条，各问一件事，闭集都留出口）：
  `is_motion`(Noul) / `known`(Noul) /
  `route`(Choice: `library`/`compose`/`generate`/`non_motion`) /
  `tempo`(Score: `slow and relaxed`/`normal pace`/`hurried`) /
  `intensity`(Score: `subtle`/`moderate`/`vigorous`) /
  `emotion`(Choice: 平静/高兴/烦躁/低落) / `interrupt`(Noul) /
  `library_pick`(Choice: `L1…Ln` ＋ `none`) ——
  最后一条是**投机问句**，与其余同批发出：没有它 `route=library` 只说了"走库"却没说**哪一条**。
- **`state` 用命名字段**：`utterance`、`context.current_action`、`context.known_intents`、
  `context.library_entries`（**为空数组也如实传**）。
- **映射表**（`ROUTE_ACTIONS`，全函数）：
  `library→load_library`（**换成库里那条的 prompt_hash 做条件**）、
  `compose→compose_knobs`、`generate→generate`、
  `non_motion→hold`（**不出片、保持当前动作**）。
  `speed = 1 + (tempo-1)*0.35`；`amplitude = 0.6 + intensity/2*0.8`；
  `emotion → expression/gaze`（**占位**，`expression_placeholder: true`）；`interrupt → replaces_current`。
- **阈值**（三个不同的数，因为三种代价不同）：`route_confidence=0.5`、
  `route_top_probability=0.5` → 走**保守分支**（取最近路由、参数只许更慢更小、
  `conservative=true`、气泡加 `我拿不准，先用最接近的：`）；`is_motion=0.5` → **绝不开新动作**；
  `known=0.5` → 不改动作，只让气泡承认在即兴。
- **异步不阻塞**：`submit()` 起线程立刻返回 → 请求去解析向量（未命中约 1 s 子进程）→
  再 `resolve(deadline)` 取判定。两者**真的重叠**（`test_viewer_router.py` 用会合事件证明：
  判定线程要等编码器发信号才结束；串行时必定超时）。超时 → `router_timeout`，
  退化为纯向量路由，气泡明写"这次没做语义解析"；**跑过期的调用仍会写入缓存**，下次同句直接命中。
- **缓存键**：`(prompt_hash, router_id, known_intents_rev)`；`router_id = "<model>+typesafe-sdk==<ver>"`，
  `known_intents_rev` 是 `(known_intents, library_entries)` 的哈希 —— 库一变，判定自动失效。
- **留痕**：`/api/plan` 与 `/api/health` 都带 `router{state, router_id, route, probabilities,
  latency_ms, cache_hit, degraded, detail, deadline_ms, model, sdk_version, judgment, known_intents_rev}`，
  plan 另带 `decision{route, action, starts_new_action, conservative, source, reason, message,
  params, library_entry}`。
- **开关与隐私**：`--no-router`、`--router-deadline-ms`（默认 2500）。
  `router.privacy` 三态明写：on（发到托管 API）/ off（没 key，不发）/ 关（本地）。

### 16.3 真实验收（`jev-latest` + `typesafe-sdk==0.7.4`，deadline 8000 ms）

四句各跑 2 次，**每次调用前把库还原**（用仓库缓存的**临时副本**，绝不写 `artifacts/`）：

| 句子 | 期望 | run 1 | run 2 | tempo / intensity | emotion |
|---|---|---|---|---|---|
| `往前走` | library | **library 0.96** | **library 0.96** | 1.00 normal / 0.87 moderate | 平静 0.99 |
| `慢慢走过去` | compose | **compose 0.99** | **compose 0.99** | 0.00 slow / 0.13 subtle | 平静 1.00 |
| `跳过去挥挥手` | generate | compose 0.71 | compose 0.70 | 1.14 normal / 0.95 moderate | 高兴 0.96 |
| `今天好累` | non_motion | **non_motion 0.99** | **non_motion 0.99** | 0.07 slow / 0.08 subtle | 低落 0.99 |
| `A person walks forward.` | library | **library** | **library** | 1.00 normal / 0.89 moderate | 平静 1.00 |

**最后一行是跑真服务时抓到的一个真 bug**：`is_motion` 第一版问的是"说话人是不是在**要求**动作"，
而本查看器的全部 prompt 词表都是**第三人称描述**（`A person walks forward.`），于是**每一条缓存条目
都被读成非请求 → hold → 不出片**。这是驱动运行中的服务（`POST /api/plan`，缓存哈希
`e9cf8141cac79f35`）发现的，没有任何测试抓到。问句已改成"这句话是不是**要求或命名**一个动作"，
false 一侧改说"说话人自己的状态"。五行两句全部一致。

5 条里 4 条与期望一致。`跳过去挥挥手` **稳定读成 `compose`**：
路由定义把 `generate` 留给"库里没有**且**组合不出来"的动作，而本库**已经有**
`A person jumps over a puddle.` 与 `A person waves with both hands.` —— "跳过去＋挥挥手"
正好是这两条的**组合**。这是**可辩护的读法**，不是 bug；如实记录，不调参掩盖。

动作与气泡原文：

```
往前走        action=load_library   embedding='A person walks forward.'
              照库里那条来：A person walks forward.（Jev 读出这句最接近库里已有的一条）。
慢慢走过去     action=compose_knobs speed=0.65 amplitude=0.656
              用现有动作组合：速度 0.65×、幅度 0.66×（Jev 读出 slow and relaxed / subtle）。
跳过去挥挥手    action=compose_knobs speed=1.07 amplitude=0.99 expression=smile
              用现有动作组合：速度 1.07×、幅度 0.99×（Jev 读出 normal pace / moderate）。
今天好累      action=hold starts_new_action=False kind=hold
              这句不是动作要求（Jev 读 non_motion 的概率 0.99），保持当前动作。
A person walks forward.  action=load_library  embedding='A person walks forward.'
              照库里那条来：A person walks forward.（Jev 读出这句最接近库里已有的一条）。
```

延迟（同进程 10 次 warm）：**min 284.7 / p50 440.6 / max 1646.7 ms**（更早的 8 次那轮在机器较闲时是 min 281.6 / p50 379.3 / max 560.0 ms；两轮之差是这台共享机器的负载，不是路由）。
判定缓存：call 1 `ok`（miss）→ call 2 `ok`（`known_intents_rev` 变了：编码器把句子加进了库）
→ call 3 `cached`，`latency_ms 0.0`。

### 16.4 负对照（四条，原文）

跑法：`./.venv/bin/python -m pytest -v <每条>` —— 18 条全 PASS。

- **① `non_motion` 绝不启动新动作**：`test_negative_control_non_motion_never_starts_a_new_action`
  （概率 0.5/0.6/0.9/0.99 四档，且 `is_motion=0.99`、`interrupt=1.0` 也说不动它）、
  `test_viewer_router.py::test_negative_control_non_motion_returns_a_hold_and_draws_nothing`
  （`runner.calls == []`：**模型一次都没被调用**，`kind == "hold"`，`last_action` 不变）。
- **② 超时绝不假装"已解析"**：`test_negative_control_a_degraded_verdict_never_claims_a_semantic_parse`
  （六种降级逐一参数化，且**故意把一份完美 judgment 与降级 state 一起传进去**）、
  `test_a_timeout_names_itself_and_the_deadline_it_missed`、
  `test_viewer_router.py::test_negative_control_a_degraded_verdict_still_plans_but_says_it_did_not_parse`
  （四种具名降级，`route` 必须是 `None`、`judgment` 必须是 `None`、气泡必须含"没做语义解析"）。
- **③ 低置信必须走保守分支并明示**：
  `test_negative_control_a_low_confidence_reading_takes_the_conservative_branch_and_says_so`
  （`conservative is True`、消息以 `我拿不准，先用最接近的：` 开头、`speed ≤ 1.0`、`amplitude ≤ 1.0`）。
- **④ key 出现在前端或接口里必须能被测出来**：
  `test_negative_control_the_key_is_in_no_output_the_router_can_produce`（递归扫所有字符串）、
  `test_negative_control_the_key_is_in_no_api_response`（`/api/health` 与 `/api/plan` 全字段）、
  `test_negative_control_the_key_is_in_no_file_the_browser_can_fetch`（**把 sentinel 写进环境变量后
  扫 `viewer/**` 每一个文件**，并在同一条测试里把哨兵写进一个临时文件证明这个检查**会红**）。

**这些负对照本身也会红**（在仓库文件上注入后立即按 sha256 逐字节还原）：

```
1 non_motion gate removed          -> exit 1  1 failed
2 degraded branch removed          -> exit 1  6 failed
3 low-confidence branch removed    -> exit 1  1 failed
4 key written into the page        -> exit 1  1 failed
all four files restored byte-for-byte
   src/ardy_mlx/router/jev.py  sha256=a61afa1f20bc3074…
   viewer/app.js               sha256=10a4cf099c3f8c7e…
```

### 16.5 测试计数

`./.venv/bin/python -m pytest -q` → **411 passed, 1 xfailed**（上一轮 333 + 1 xfailed；只增不减）。
其中本轮新增 57（`test_router_jev.py`）＋ 16（`test_viewer_router.py`）＋ 5（`test_viewer_http.py`）
＝ 78，另 `node tests/viewer_js_guard.test.mjs` 从 12 条加到 **20 条**（仍由 pytest 里的桥接条驱动）。

### 16.6 残留未知

- **本轮把一次真实调用写进了仓库缓存，已清理并如实记录**：第一次验收脚本用了仓库真实的
  `artifacts/embeddings`，`FakeLocalEncoder` 因此把 4 条假向量（`git_sha` 全零）装了进去
  （`047a2ea98fe79ec2`、`14375452c554aeea`、`540b358400fad375`、`ab60cf41271af9d3`）。
  已按 `git_sha` 全零精确识别并**逐文件删除**，复扫为 0；随后所有验收都在**临时副本**上跑，
  每次调用前还原。`artifacts/embeddings/*` 本来就在 `.gitignore` 里，未进版本库。
- **参数未接入采样器**（新增边界 15）：`speed` / `amplitude` / `expression` / `gaze` 只是携带，
  本轮出片仍只由文本向量决定。
- **判定正确性未验证**（新增边界 14）：四句里有一句与预期不符（见 §16.3），且这是**一个模型一次
  回答**，不是准确率。低置信、对抗输入、CJK 的稳定性都未测。
- **隐私是真实变化**（新增边界 14）：开启路由时用户那句话离开本机，本地测试无法观测这件事。
- **浏览器运行时仍需人眼确认**（上一轮边界 13 仍然成立）：气泡的 DOM、颜色与点击路径没有任何
  本仓库的测试声称覆盖。
