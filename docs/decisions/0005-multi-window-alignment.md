# 0005 · 多窗口对齐基线（**提案** · 待人类批准）

- 状态：**提案（proposal）**。本文件不改任何上游条款，只记录证据与建议。
- 提出人：A 侧（DeepSeek）
- 日期：2026-10-13
- 依据：`0003` 裁定 4 授权的**第二次参考运行**已按纪律用完一次（`--duration 4.0`，80 帧 = 2 个自回归窗口）
- 证据：`artifacts/golden-run-4s/RUN-RECORD.md`、`artifacts/align-report.json`（`levels.L4`）、
  `docs/status-deepseek.md` §2–§5

## 0. 结论摘要

> **多窗口整段对齐通过 `0003` 冻结的 1e-3。**
> 逐窗口 `x`（历史＋注入噪声）残差 **0.0 / 2.198e-06**；重定心位移 ≤ **1.580e-06**、
> 平移累积 ≤ **1.505e-06**、`first_heading_angle` 刷新 **1.625e-07**；
> 20 个去噪步逐步 raw/CFG/DDIM 最差 **8.476e-05 / 9.114e-05 / 1.660e-05**；
> 整段 `frames (1,80,330)` **3.543e-05**，分段 **3.543e-05 / 3.054e-05**，
> root/heading 对参考 npz **4.709e-06 / 1.866e-06**。
>
> 过程中抓到并修掉 **2 个只在多窗口现形的静默移植错误**（见 §2），
> 并用 **11 条负对照**（§3）证明每条多窗口路径的断言都能红。

## 1. 建议的裁定

1. **多窗口基线口径冻结**，与 `0003` 的单窗口基线并列：
   同一个 prompt、`--seed 0`、`--no-postprocess`、`--duration 4.0`（80 帧、2 窗口）、
   文本编码器 CPU bf16；容差沿用 **1e-3**；每次比较都必须附负对照。
2. **把 N10 / N11 / N12 提升为契约的强约束**（`docs/motion-contract.md` §10）：
   `lengths = history_len + generation_len` 的速度复制落点；无约束时 `future_token_mask` 全 False；
   `crop_history_length is not None` 时必须走分块解码。三条都是 O(1e-1)~O(1e1) 级静默错误。
3. **承认一条覆盖缺口**（不是"已对齐"，是"未覆盖"）：`crop_history_length = 160` 在 80 帧上
   **从不真正截断**，所以"真截断"没有金标准。**建议不要为此再花第三次参考运行**；
   如果将来确实需要，`--duration ≥ 11.0`（>200 帧）才是能触发截断的最小成本，
   但那需要新的授权。
4. **`make_step_fn` 接缝归 A 维护**（`src/ardy_mlx/model/ardy.py`），B 只调用、不修改 A 的文件；
   A 只读 B 的 `autoregressive.py` 确认签名。B 的文件本轮零改动。
5. **窗口几何的差异交 B 决策**（§4）：B 的 `plan_window` 把每个窗口右对齐到它持有的上下文，
   默认是 160 帧零历史，因此 B 的 step 0 是 `history_len=160`，而参考批处理路径的 window 0 是
   `history_len=0`。A 不改 B 的文件，只把差异钉进测试并在此提案。

## 2. 两个只在多窗口现形的移植错误（已修）

两个都是**静默**的：不报错、不产生 NaN，只是数值错 O(1)。

### 2.1 `global_root_to_local_root` 忽略 `lengths`（最严重）

- 参考：`compute_vel_xyz` / `compute_vel_angle`（`motion_rep/tools.py:58-110`）先补零，
  再 `v[lengths−1] = v[clamp(lengths−2, 0)]`；denoiser 传
  `lengths = history_len + generation_len`（`auto_latent_twostage_denoiser.py:219`），
  而张量长度是 `history_len + generation_len + future_len`。
- 单窗口 40 帧时 `lengths == T`，所以第 2 轮三级全绿**也发现不了**；
  4.0 s 的 window 0 是 **40 vs 80**（后 40 帧是 mask 掉的 future 区）。
- **实测后果**：Stage-2 的 local root 条件全错 ⇒ 整段 `frames` 错 **6.75**（容差的 6750 倍）。
- 定位：用参考 dump 的 `modules/denoiser__model_{root,body}_model__call0__*.npy` 逐级二分——
  `root_model` 输出残差 8.3e-07，`body_model` **输入**残差 11.39 ⇒ 两者之间的
  `global_root_to_local_root` 是唯一嫌疑。
- 修复 + 回归：见 `docs/status-deepseek.md` §5；两个新单测钉住"只有 `lengths−1` 那一帧变"。

### 2.2 无约束时 `future_token_mask` 必须是全 False

- 参考：`latent_utils.py:207`，`motion_mask is None` ⇒ `zeros_like(history_token_mask)`，
  **不是** `future_mask` 降采样。80 帧 window 0 的 `future_mask` 有 **40 帧**为真，
  而 `future_token_mask` 是 **20 个 0**——两者本来就不是一回事。
- **实测后果**：`frames` 错 **1.342**（改的是 backbone 的 `src_key_padding_mask`）。
- 修复：把这条规则提取成 `convert_frame_mask_to_token_mask(...)` 并加单测。

## 3. 负对照（每条路径都可证伪）

信号 = 端到端 `frames (1,80,330)` 的 max_abs；容差 1e-3。完整表见
`docs/status-deepseek.md` §4。

| 路径 | 能让它红的对照 | 实测 |
|---|---|---|
| history cropping | `crop_history_length = 4`（真截断） | **1.002** |
| 每步末帧重定心 + 平移累积 | 中心早一个 token（−4 帧） | **1.013** |
| `first_heading_angle` 同步更新 | 关掉刷新 | **3.265e-03** |
| 分块自回归解码 | 换整段解码 | **4.428e-02** |
| （总闸）错的初始噪声 | `x_T += 0.25` | **1.793** |
| 基线与"不许红" | clean / `crop=164` / 中心晚一个 token | 3.543e-05（= 干净值） |

`crop=164` 与"中心晚一个 token"这两条**在本金标准上必然空转**，原因分别是
"80 帧碰不到 160 帧预算"与"参考中心本来就是序列最后一帧"。两条都以 `expect_detected=False`
显式登记并断言"不许红"——**空转的负对照会给出假的安全感，所以必须写下来**。

## 4. 交给 B 决策的一条接口差异（A 不擅自决定）

> **前置**：B 的 `0004-contract-rulings.md` 裁定 5 已经定了"**哪条循环是标准**"——
> **A 的 `ArdyBatch`** 是通过金标准验证的那条，`ardy_generate.py` 的 MLX 后端默认走它；
> B 的 `ArdyAutoregressive` 保留为独立实现/占位/测试路径，**出现分歧时以 A 的行为准，
> 并把分歧写成提案而不是改 B 的文件**。本节就是那条分歧的提案。
> 也就是说：**发布路径不受影响**（它走 A 的 `ArdyBatch`，本轮 80 帧对齐 3.543e-05）；
> 受影响的只有 B 的独立循环作为端到端判据时的可用性。

B 的 `plan_window(held_frames=...)` 把窗口右对齐到当前上下文，`held_frames` 默认是
`init_history_len = min(default_history_frames, history_frames)` = **160 帧**（`history_frames=None` 时）。于是：

| | B 的 step 0 | 参考批处理路径的 window 0 |
|---|---|---|
| `history_len` | **160 帧**（40 个 token，全零种子） | **0 帧** |
| `generation_len` | 40 | 40 |
| 窗口 token 数 | 50 | 20（10 generation + 10 masked future） |
| **同一份参考 `x_T` 下与金标准的差异** | **max_abs 5.466**（金标准 \|max\| = 1.682） | — |

上表最后一行的测法（**未改 B 的任何文件**）：在 scratch 脚本里把
`mx.random.normal` 临时替换成"依次返回参考自己的 `randn_{w}.npy`"，然后调用 B 真正的
`ArdySampler.run()`；对比对象是金标准 `motion_normalized` 反归一化后的 `frames (1,80,330)`。
`translated()` 与 `frames` 给同一个数（5.466）。

另外测到两条**属于 B 循环语义**的定量事实（A 只报告，不评判）：

- `num_frames` 对 B 的循环是"种子 + 生成"：`num_frames=80, history_frames=160` 会生成
  **0 步 / 0 帧**（`plan_autoregressive_steps` 以 `init_history_len=160` 为起点）；
  要拿到 80 帧输出需要 `num_frames = 160 + 80 = 240`。
  （注意这与 B 的 `0004` 裁定 1"`num_frames` = 生成帧数、v1 路径 `init_history_len = 0`"是
  **两条不同的口径**：裁定 1 描述的是希望 v1 采用的语义，而 `ArdySampler` 当前按
  `init_history_len = min(160, history_frames)` 起算。）
- 该配置下每一步的 `StepDiagnostics.history_len` 都是 **160**（两次窗口都是 50 token），
  而参考是 **0 → 40**。

**A 侧验算过的最小改法**（不要求 B 接受，只是把选项写具体）：把**种子上下文留空**
（`init_history_len = 0`，即 `initial_history` 长度为 0）**同时把 `crop_history_length` 留在 160**，
B 自己的 `plan_window` 就会给出参考的规划序列：

| step | `held_frames`（= 当前 context 长度） | `plan_window` 结果 | 参考 |
|---|---|---|---|
| 0 | 0 | `history_len=0`, `generation_len=40` | `history_end_frame=0` ⇒ `history_token_end=0` ✔ |
| 1 | 40（= 上一段轨迹；`take = 160-40` 从空种子里只取到 0 帧） | `history_len=40`, `generation_len=40` | `history_end_frame=40` ⇒ `history_token_end=10` ✔ |

即：**分歧的根因只是"零种子被当成 160 帧真实历史"**，不是 `plan_window` 的算术。
两条路都可行，请 B 选一个（或提出第三个）：

- **(a) 保持现状**：把 160 帧零种子当作"真实 init history"，
  等价于参考的 `init_history_sequence = 160 帧零`——语义自洽，但**与 `generate.py` 批处理路径不等价**，
  因此 B 的独立循环不能用单窗口/多窗口金标准作为端到端判据（发布路径仍走 A 的 `ArdyBatch`，不受影响）。
- **(b) 让第一个窗口不带历史**：`held_frames` 在 `trajectory is None` 时取 0
  （或让 `initial_history` 不参与第一个窗口），使其与参考的 `init_history_sequence=None` 分支一致。
  这样 B 的链条就能用金标准端到端验证。

这一差异已在 `tests/test_model_stepfn.py` 里以
`xfail(strict=False)` 钉住（现在 XFAIL；哪天 B 的链条变得与金标准等价就会 XPASS 提醒更新本节）。
无论选哪个，A 的 `make_step_fn` 都无需改动。

## 5. 残留未知（继续挂着）

1. **真正的历史截断**没有金标准（80 帧 < 160 帧预算）；现有覆盖是纯算术单测 + `crop=4` 负对照。
2. **带约束路径**（`motion_mask`/`observed_motion` 非零、`future_token_mask` 非零）两次运行都没跑，
   只有源码级复刻。
3. seed / CUDA RNG 语义对齐（沿用 `0003`）。
4. 模型卡 326 M vs 实测 191.3 M。
5. 训练数据 Bones Rigplay 1 与文本编码器链路的许可。
6. B 的窗口几何差异（§4），决定权在 B。
