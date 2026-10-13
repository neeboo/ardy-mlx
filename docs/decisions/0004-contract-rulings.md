# 0004 · 契约裁定：num_frames / 时长→步数 / 短片段 / posed_joints / 哪条循环是"标准"

**背景**：B 侧交付了采样链（cosine 调度、DDIM eta=0、separated 三路 CFG、自回归窗口、`ardy_generate.py`），
155 个测试全绿（含 **12 个负对照**），并且**用 A 的转换权重通过 CLI 复现了金标准**：
整行 `(40,330)` **max_abs 1.87e-05**（root 5.14e-06、heading 2.99e-06、local_joints 1.27e-06、
global_rot 2.50e-06、velocities 1.87e-05、foot_contacts 2.44e-06），有测试锁定。
它把 5 个契约问题**挂起来等裁定、没有自创**，本决定逐条裁定。

## 裁定

1. **`num_frames` 语义 = "生成帧数"**（不含初始上下文）。冻结：`steps = ceil((num_frames − init_history_len)/horizon)`，
   在 v1 的路径上 `init_history_len = 0`；本地若以 160 帧上下文起手，则 `clip = seed + steps × horizon`。
   两种算法在 12 s 上同为 240 帧，但**语义以"生成帧数"为准**，sidecar 必须同时记录
   `num_frames`（生成）与 `seed_context_frames`。
2. **`--duration → 步数` 的算术冻结**：沿用模型侧那套 `ceil((round(duration·fps) − init_history_len)/horizon)`；
   任何 CLI 不得自造第二套换算。
3. **片段短于默认 history 时：保持显式报错，不许静默缩短**。（与全程的"失败要具名"一致。）
4. **`posed_joints` 是占位**：它需要**骨架 FK**，属 A 侧（`model/**` / `motion_rep`）。在补上之前，
   sidecar 必须写 `extra.posed_joints_source = "placeholder"`，并且**文档里要写明它与金标准差 0.132**，
   禁止把占位数据当结果发布。补 FK 后必须对金标准重新对齐并更新此条。
5. **哪条循环是"标准"**：**A 的 `ArdyBatch`** 是通过金标准验证的那条（`0003` 的三级对齐即基于它）；
   B 的 `ArdyAutoregressive` 保留作为**独立实现与占位/测试路径**，但：
   - `ardy_generate.py` 的 **MLX 后端默认走 A 的路径**；
   - 两条循环的**帧数与窗口划分必须一致**（已有测试），出现分歧时**以 A 的行为准**，并把分歧写成提案而不是改 B 的文件。

## 观察项（记录在案，不改契约）

* **6D 旋转存的是矩阵的前两"行"**：行 2.2e-06、列 2.6e-02 —— 契约是对的，**不要"修"**。
* `foot_contacts` 在特征里是**连续回归值**；sidecar 的二值化版本与金标准 bool 数组**完全一致**（2 s 片段 141 个 true）。
* `--diffusion-steps 1` 会得到长度 2 的链（`space_timesteps(10,1)==[0,9]`），**忠实于参考**。
* **不要**用 npz 的 `posed_joints`＋`global_rot_mats` 去重建片段（那是 `inverse()` 的信息损失，4e-3~2e-1，
  不是 CLI 缺陷）；**一律以 `motion_normalized.npy` 为对齐基准**（6D 走 Gram–Schmidt 重建到 2.2e-06）。
* RNG：MLX 无 `manual_seed`，按窗口 `mx.random.split`；**不追求与 CUDA Philox 逐位一致**（构造上未验证），
  对齐靠注入参考 `x_T`。

## 下一步（A 侧）

补 `posed_joints` 的骨架 FK（与 `motion_rep` 同一套骨架定义），然后重跑多窗口对齐与金标准比对；
`0003` 已授权的第二次参考运行（`--duration 4.0`）由 A 侧 A3 轮执行。
