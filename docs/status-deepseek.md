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
