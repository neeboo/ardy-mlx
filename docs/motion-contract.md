# 数据契约（v1 · A 侧已补齐六项，带文件:行）

**本文件是 A/B 之间唯一的接口。** 所有数值均由 A 从参考实现读出，每项附 `文件:行`。
**不许猜**；填错会让 B 白干。若实现中发现契约不够用，**不要自行改字段名/维度/顺序**，
在 `docs/status-claude.md` 写"需要新增字段＋原因"交人类裁定。

- 参考实现：`nv-tlabs/ardy` @ `693f74d`，只读镜像在 `pancat-linux-ci:~/services/ardy-engine/ardy`
- 本文中裸文件名（如 `ardy_model.py:569`）均指该仓库根下的相对路径
- 权重：`pancat-linux-ci:~/services/ardy-engine/checkpoints/ARDY-Core-RP-20FPS-Horizon40/{tokenizer,denoiser}.safetensors`
- 补齐日期：2026-10-13（A 侧）

## 0. 已确认（上游 README / checkpoint config / HF 模型卡）

| 项 | 值 | 证据 |
|---|---|---|
| checkpoint | `ARDY-Core-RP-20FPS-Horizon40`（本机唯一存在）、`…Horizon8`、`ARDY-G1-RP-25FPS-Horizon52`、`…Horizon8` | `README.md:67-70` |
| 骨架 | **Core** = `CoreSkeleton27`（27 joints，SMPL-X 系）；**G1** = `G1Skeleton34`（34 joints） | `skeleton/definitions.py:338`、`:286`；`registry.py:33-40` |
| 帧率 | **20 fps**（Core）/ **25 fps**（G1） | `checkpoints/…/config.yaml:13,45` |
| horizon（块长） | **40**（Core H40）/ 8 / 52 / 8；即 `gen_horizon_len` | `config.yaml:4` |
| `num_base_steps` | **10** | `config.yaml:2` |
| `cfg_type` | **`separated`** | `config.yaml:3` |
| 训练数据 | Bones Rigplay 1（不影响接口，仅记录） | `README.md:67` |

## 1. ✅ 输出表示的精确字段与单位

### 1.1 特征向量 `frames (T, D)` 的构成

`D` 由 6 个 block **按固定顺序**拼接而成（顺序 = `size_dict` 的键序）：

| # | block 名 | 形状（每帧） | 单位 | 含义 |
|---|---|---|---|---|
| 1 | `root_pos` | `[3]` | **米** | 根（pelvis）**世界**位置 `(x, y, z)` |
| 2 | `global_root_heading` | `[2]` | — | 根朝向 `(cos θ, sin θ)` |
| 3 | `local_joints_positions` | `[J-1, 3]` | **米** | **非根**关节位置，在**根局部**坐标、并已对齐到地面 |
| 4 | `global_rot_data` | `[J, 6]` | — | **全局**关节旋转，**6D 连续表示** |
| 5 | `velocities` | `[J, 3]` | 米/秒 | 全局关节速度 |
| 6 | `foot_contacts` | `[4]` | 0/1 | 脚接触标签 |

证据：`ardy/motion_rep/reps/ardy_motionrep.py:57-64`（`size_dict` 定义）、`:169-181`（`einops.pack` 的拼接顺序，与上表一致）

```python
# ardy_motionrep.py:57-64
self.size_dict = {
    "root_pos": torch.Size([3]),
    "global_root_heading": torch.Size([2]),
    "local_joints_positions": torch.Size([nbjoints - 1, 3]),  # removed the pelvis joint
    "global_rot_data": torch.Size([nbjoints, 6]),
    "velocities": torch.Size([nbjoints, 3]),
    "foot_contacts": torch.Size([4]),
}
```

### 1.2 root block / body block 的切分

```
root block  = block1 + block2                      # 前 5 维
body block  = block3 + block4 + block5 + block6
```

证据：`ardy/motion_rep/reps/base.py:47-54`

```python
# base.py:47-54
self.root_slice = slice(0, self.slice_dict[self.last_root_feature].stop)   # last_root_feature == "global_root_heading"
self.body_slice = slice(self.root_slice.stop, self.motion_rep_dim)
self.body_dim = self.body_slice.stop - self.body_slice.start
self.global_root_dim = self.motion_root_dim = self.root_slice.stop
self.local_root_dim = sum(val.numel() for val in self.local_root_size_dict.values())
self.local_motion_rep_dim = self.local_root_dim + self.body_dim
self.global_motion_rep_dim = self.motion_rep_dim
```

`last_root_feature = "global_root_heading"`（`ardy_motionrep.py:65`）⇒ **root block = 前 5 维**。

每个 block 的切片 = `np.cumsum([0] + 各 block 元素数)`（`base.py:43-44`）。

### 1.3 具体数字（Core J=27）

| 项 | 值 | 推导 |
|---|---|---|
| `body_dim` | **325** | (27-1)×3 + 27×6 + 27×3 + 4 = 78+162+81+4 |
| `local_root_dim` | **4** | `local_root_rot_vel[1] + local_root_vel[2] + global_root_y[1]`（`ardy_motionrep.py:66-70`） |
| **`motion_rep_dim` = D** | **330** | 5 + 325 |
| `local_motion_rep_dim` | **329** | 4 + 325 |

**D = 330 有三重独立验证**（数值级，不只靠求和）：

1. `tokenizer` 编码器输入投影 `pose_net.encoder.input_proj.weight` 形状 `[512, 1300]`；
   `transformer.py:87` = `Linear(input_dim * num_frames_per_token, latent_dim)`，`num_frames_per_token=4`
   ⇒ `input_dim = 1300/4 = 325 = body_dim` ✔
2. `tokenizer` 解码器输出投影 `pose_net.decoder.output_proj.weight` 形状 `[1316, 512]`；
   `transformer.py:229` = `Linear(latent_dim, output_dim * num_frames_per_token)`
   ⇒ `output_dim = 1316/4 = 329 = local_motion_rep_dim` ✔
3. `denoiser.global_root_hybrid_constraints_proj.weight` 形状 `[1024, 2768]`；
   `auto_latent_twostage_denoiser.py:87-90` = `global_root_hybrid_token_dim + (motion_rep_dim + body_dim) * num_frames_per_token`
   ⇒ `2768 = 148 + (D + 325)*4` ⇒ `D + 325 = 655` ⇒ **D = 330** ✔

同源校验（stats 文件）：`stats/motion/mean.npy` 形状 `(334,)` = `global_root_dim(5) + local_root_dim(4) + body_dim(325)`
（`base.py:69-72` 的 `self.stats` 切片索引正是 `0..5` 与 `9..333`，跳过中间的 4 个 local-root 槽位）✔

**G1（J=34）**：`body_dim = 33×3+34×6+34×3+4 = 99+204+102+4 = 409`；**D = 414**；`local_motion_rep_dim = 413`。

### 1.4 各类表示的精确定义

| 量 | 定义 | 证据 |
|---|---|---|
| 旋转 | **6D 连续旋转 = 旋转矩阵的前两"行"**：`cont6d = concat(matrix[...,0], matrix[...,1])` | `geometry.py:16-18` |
| 6D→矩阵 | Gram–Schmidt：`x=normalize(r0)`、`z=normalize(x × r1)`、`y=z × x`，再按**列**拼 `[x,y,z]`（右手基） | `geometry.py:21-37` |
| 朝向 θ | `θ = atan2(dz, -dx)`，其中 `d = 右髋 - 左髋`（世界坐标） | `motion_rep/tools.py:124-126` |
| 接触（4 通道） | **`(左 heel, 左 toe, 右 heel, 右 toe)`** | `motion_rep/feet.py:23-25,52` |
| 接触判据 | `‖v_foot‖ < 0.15` **且** `y_foot < 0.10`，两者同时满足记 1 | `feet.py:36-50`；阈值由调用方给定 `ardy_motionrep.py:166` |
| 速度 | `v[t] = fps * (p[t+1] - p[t])`，末帧复制前一帧 | `tools.py:58-71` |
| 单位 | **米**（BVH 解析时 `cm -> m`） | `skeleton/bvh.py:560`、`:573`；`data_processing/skeleton.py:398,406` |
| `local_joints_positions` | FK 得到、去掉根关节、再加回根的 y 做"对齐地面" | `ardy_motionrep.py:160-163` |

接触通道顺序的依据：`CoreSkeleton27` 的 `right_foot_joint_names = ["RightFoot", "RightToeBase"]`、
`left_foot_joint_names = ["LeftFoot", "LeftToeBase"]`（`skeleton/definitions.py:342-343`），
而 `feet.py:52` 是 `torch.cat((feet_l, feet_r), axis=-1)` ⇒ **左在前、右在后**。

### 1.5 解码 `inverse()` 的输出字典（落盘用）

| key | 形状 | 说明 |
|---|---|---|
| `local_rot_mats` | `(B,T,J,3,3)` | 局部旋转矩阵 |
| `global_rot_mats` | `(B,T,J,3,3)` | 全局旋转矩阵 |
| `posed_joints` | `(B,T,J,3)` | 全局关节位置 |
| `root_positions` | `(B,T,3)` | 根世界位置 |
| `smooth_root_pos` | `(B,T,3)` | 此表示下等于 `root_positions` |
| `foot_contacts` | `(B,T,4)` **bool** | 由特征 `> 0.5` 得到 |
| `global_root_heading` | `(B,T,2)` | `(cos,sin)` |

证据：`ardy_motionrep.py:271-279`。参考实现落盘的 `.npz` 另加 `fps` 与 `text` 两个标量（`scripts/generate.py:159-164`）。

### 1.6 归一化（**必须区分**）

- 模型内部/权重张量走的是**归一化**特征：`base.py:232-238` 用 `stats/motion/{mean,std}.npy`（形状 `(334,)`）做 `normalize/unnormalize`。
- 契约里的 `frames (T,D)` 指**反归一化后的物理量**（见 §7）。B 侧采样器内部用的归一化张量**不是**契约的一部分。

## 2. ✅ 坐标系与手性、是否世界坐标、与 MotionBricks/Unity 的转换

### 2.1 坐标系

- **右手系，Y 轴向上**；地面 = **XZ 平面**。
  - Y 是高度：`feet.py:37` `positions[:, :, fid_l, 1]`；`base.py:134` `global_root_y = root_pos[..., 1]`。
  - 平面位移只动 `x` 与 `z`：`ardy_motionrep.py:233-234`（`[0]`/`[2]`）、`base.py:204`（`root_pos[:, 0, [0, 2]]`）、`base.py:126`（`center_pos[:, 1] = 0`）。
  - 右手：`geometry.py:8-13` 与 `motion_rep/tools.py:168-171` 的 Y 旋转矩阵均为 `[[c,0,s],[0,1,0],[-s,0,c]]`；
    `cont6d_to_matrix` 构造 `z = x × r1`、`y = z × x`（`geometry.py:26-30`）也是右手基。
- 旋转作用约定为**列向量** `x' = R x`（`tools.py:184-186` 注释 `(since we use the convention x' = R x)`）。

### 2.2 是否已是世界坐标：**是**

`ArdyMotionRep.__call__` 直接接收**世界系** `root_positions` 与局部旋转，经 FK 得到全局量（`ardy_motionrep.py:151-155`）；
`root_pos`、`global_rot_data`、`velocities` 都在世界系。**没有** per-frame 的局部骨架平移。

### 2.3 但生成结果被**规范化**到固定起始帧

- 朝向：`first_heading_angle = torch.zeros(num_samples, device=device)  # facing +Z`（`scripts/generate.py:240`）
  ⇒ 第 0 帧朝向 **+Z**；`θ=0` 表示面向 +Z。
- 位置：`global_transl` 初值为 0（`ardy_model.py:616-619`），末尾由 `translate_normalized_root_motion(root_motion, global_transl, …)` 加回累积平移（`ardy_model.py:679`）。
  `translate_normalized_root_motion` 断言平移的 y 分量为 0（`ardy_model.py:29-38`）⇒ **只在水平面平移**，高度不偏移。

⇒ 契约口径：**世界坐标，锚定在"起点在地面原点附近、起始朝向 +Z"**。B 若需要别的起始帧位姿，在契约外自行刚体变换。

### 2.4 与 MotionBricks / Unity 的转换：**上游没有任何转换代码（新定义，尚未与 B 对齐）**

事实：在参考实现里 grep `unity|motionbricks|\.fbx|gltf|glb|usd|usdc`，**唯一命中是 README:263 的"相关工作总结"链接**
（`README.md:263`），不存在导出器。BVH 只作为**输入**解析（`scripts/interactive_demo/motion_io.py:77-81`），不是输出格式。

因此这一层约定由**我们**定义，**本条为待评审项，B 不要据此写死**。A 的建议（**未在上游验证**，属推断）：

| 目标 | 上游（ARDY） | Unity | 需要的变换 |
|---|---|---|---|
| 手性 | 右手 | **左手** | 取反一个轴（推荐取反 **x**，保持 +Z 为前） |
| 上轴 | +Y | +Y | 无需变换 |
| 前向（θ=0） | +Z | +Z | 无需变换 |
| 旋转表示 | 6D 连续（前两行） | 四元数 | 6D→矩阵→四元数；**取反轴后须做手性修正**（转置/共轭），不能直接取反四元数分量 |
| 单位 | 米 | 米 | 无 |
| 根平移 | `root_pos` 世界 | Transform.position | x 取反 |

> 结论：MotionBricks/Unity 的转换**待定**，需要人类或 B 侧给出目标约定后再写进契约（附本项为"待定稿"）。

## 3. ✅ 两阶段结构：tokenizer / denoiser 的输入输出张量名与形状

### 3.0 权重的实测清单（A 侧用 `scripts/ardy_inspect.py` 核过）

| 文件 | 字节 | header | tensors | 参数量 | dtype | data_offsets |
|---|---|---|---|---|---|---|
| `tokenizer.safetensors` | 141,746,112 | 25,896 | **204** | **35,430,052** | 全 `F32` | 连续 ✔ |
| `denoiser.safetensors` | 623,324,336 | 31,320 | **224** | **155,823,252** | 全 `F32` | 连续 ✔ |
| 合计 | 765,070,448 | — | 428 | **191,253,304** | — | — |

两文件 `__metadata__` 均为 `{"format": "pt"}`。校验方式：`scripts/ardy_inspect.py <file>`（只读，不载入 tensor 数据）。

> ⚠️ **观察到的数字不一致（不阻塞，但需留意）**：模型卡 `README.md:45` 写
> `**Number of model parameters:** 326 M`，而两个已发布权重文件实测合计 **191.3 M**。
> 两个文件的张量集合与 `config.yaml` 的结构**自洽**（见下表的逐层核算），
> 因此 A 的判断是"**文件完整、模型卡数字口径不同或未更新**"，而非权重缺失。
> 该差异尚未向上游确认 → 记为残留未知。

### 3.1 总览

```
motion 特征 [B,T,330]
   │
   │  Stage A: FSQVAETransformer  (tokenizer.safetensors, 141,746,112 B, 204 tensors, 键前缀 "pose_net.")
   │    encoder 只吃 body(325)；FSQ 量化成 128 维码 → 每 4 帧 1 个 token
   ▼
body latent [B, T/4, 128]
   │
   │  Stage B: AutoLatentTwostageDenoiser (denoiser.safetensors, 623,324,336 B, 224 tensors, 键前缀 "denoiser.")
   │    hybrid token = [ global root 20 | body latent 128 ] = 148
   │    两级：root_model 先预测 global root，body_model 再据 local root 预测 body latent
   ▼
hybrid token [B, T/4, 148]
   │
   │  Stage A 的 decoder 反量化 + external_cond(local root) → 还原 [B,T,329]，再拼回 root → [B,T,330]
   ▼
motion 特征 [B,T,330]
```

`num_frames_per_token = 4`（`config.yaml:25`）——即 **4 帧 = 1 个 token**（"patch size"）。

### 3.2 Stage A · tokenizer（`FSQVAETransformer`）

**`feature_mode: [body, pose, body, root]` 的语义**（`config.yaml:17-21` → `fsq.py:66-72`）：

```python
# fsq.py:66-72
assert len(feature_mode) == 4 and np.all([f in self.ALLOWED_FEATURE_MODE for f in feature_mode])
(
    self.encoder_input_feature_mode,        # body
    self.decoder_output_feature_mode,       # pose
    self.decoder_target_cond_feature_mode,  # body
    self.decoder_external_cond_feature_mode,# root
) = feature_mode
```

三个维度的取值（`fsq.py:74-83`，注释 `always use the local root`）：

```python
dim_dict = {
    "pose": motion_rep.local_motion_rep_dim,  # local root + body = 329
    "body": motion_rep.body_dim,              # 325
    "root": motion_rep.local_root_dim,        # 4
}
```

| 部件 | 输入张量名 / 形状 | 输出张量名 / 形状 |
|---|---|---|
| `encoder` | `x_in` = **body** `[B, T, 325]`（`fsq.py:197`，经 `extract_body` `fsq.py:171-172`） | `x_encoder` `[B, T/4, 128]`（`fsq.py:198`） |
| patchify | `rearrange(x, "b (t f) d -> b t (f d)", f=4)` ⇒ `[B,T/4,1300]`；`input_proj = Linear(325*4, 512)` | `transformer.py:117`、`:87` |
| 8 层 transformer | 宽度 `latent_dim=512`，`num_heads=4`，`ff=1024`，**causal** | `config.yaml:26-35`；`fsq.py:86-99` |
| `output_proj` | `Linear(512, 128)` | `transformer.py:104`；权重 `[128,512]` |
| `quantizer` (FSQ) | `FSQ(levels=[64]*128, return_indices=False)` ⇒ **128 个通道，每通道 64 级** | `fsq.py:117-122` |
| post-quant 归一化 | `stats/post_quantization/{mean,std}.npy`，形状 `(128,)` | `fsq.py:206-211` |
| `decoder` | `token_embeddings` `[B,T/4,128]` + `external_cond` = **local root** `[B,T,4]`（+ `target_cond` = **body** `[B,T,325]`） | `x_decoder` `[B,T,1316]`（`fsq.py:239-243`） |
| decoder 输出投影 | `output_proj = Linear(512, 329*4=1316)` | `transformer.py:229`；权重 `[1316,512]` |
| 拆分 | `output["root"] = x_decoder[..., :4]`、`output["body"] = x_decoder[..., 4:]` | `fsq.py:246-248` |

关于 `num_fsq_levels: 128` 与 `fsq_level_list: 64` 的关系（`config.yaml:22-23`）：
`fsq_level_list` 是**整数**时被展开成 `[64] * num_fsq_levels`（`fsq.py:117-118`），
且 `latent_embedding_dim = num_fsq_levels`（`fsq.py:85`，**覆盖**了构造参数传入的 128）⇒ 两者都是 128，一致。

> `encoder.input_proj` 的权重形状 `[512,1300]` 与 `decoder.output_proj` 的 `[1316,512]`
> 是上面形状表的**权重级核对**依据（见 §1.3）。

### 3.3 Stage B · denoiser（`AutoLatentTwostageDenoiser`）

**token 维度**：

```python
# ardy_model.py:476-487
shape = (batch_size, num_generation_tokens, nframe_root_dim + latent_embedding_dim)
x_t = torch.randn(shape, device=device)
```

```
dim_token = nframe_root_dim + latent_embedding_dim = 20 + 128 = 148
```

- `latent_embedding_dim = 128`（`config.yaml:24`；与 `stats/post_quantization/mean.npy` 的 `(128,)` 一致）
- `nframe_root_dim = 20`（`config.yaml:41`）= `motion_root_dim(5) × num_frames_per_token(4)`；核对见 `latent_utils.py:177-182` 与权重 `global_root_hybrid_proj` = `[1024,148]`

| 部件 | 输入 | 输出 |
|---|---|---|
| 整体 `forward` | `x` `[B, T/4, 148]`（hybrid：前 20 = 全局 root 扁平，后 128 = body latent） | 与 `x` **同形状** `[B, T/4, 148]`（`auto_latent_twostage_denoiser.py:184-186`、`:391-399`） |
| 拆分 | `root_motion` → `[B,T,5]`；`latent_body_motion` → `[B,T/4,128]` | `latent_utils.py:67-76` |
| **Stage 1** `root_model` | `root_stage_input` `[B,T/4,1024]` | `global_root_motion_pred` `[B,T/4,20]` → reshape `[B,T,5]`（`:271-294`） |
| **Stage 2** `body_model` | `body_stage_input`（`x_new = cat[local_root_16 \| latent_body_128] = 144` 投影到 1024） | `latent_body_motion_pred` `[B,T/4,128]`（`:367-389`） |
| 输出拼接 | `cat([global_root_motion_pred(20), latent_body_motion_pred(128)], -1)` | `[B,T/4,148]`（`:391-399`） |
| 每级层数/宽度 | 8 层，`latent_dim=1024`，`num_heads=8`，`ff=2048` | `config.yaml:53-56` |

**Stage 1→2 的衔接**：先把预测的 global root 转成 **local root** 作为 body 级的条件
（`:300-318`：`global_root_to_local_root` → `rearrange` → `cat([local_root, latent_body_motion])`）；
权重核对 `local_root_hybrid_proj` = `[1024,144]`（`16+128`）。

**预测目标 = x_start（干净样本）**，不是 ε 或 velocity：
`denoising_step` 里该输出被命名为 `token_seq_pred_clean`（`ardy_model.py:280`），
DDIM 反推 ε 的公式是 `eps = (sqrt_recip_alphas_cumprod * x_t - pred_xstart) / sqrt_recipm1_alphas_cumprod`（`diffusion.py:128-130`）。

**条件注入方式 = prefix 拼接（不是 cross-attention）**：

- `embed_text = Linear(4096 → 1024)`（权重 `[1024,4096]`），文本被投影后作为**前缀 token** 拼到序列前（`model/backbone.py:321`、`:339`）
- `embed_timestep`：`timesteps [B]` → `[B,1,1024]`（`backbone.py:459-468`）
- `linear_first_heading_angle = Linear(2 → 1024)`（权重 `[1024,2]`），把 `(cos,sin)` 投影后加入前缀（`backbone.py:342-361`）
- `learned_prefix_embedding.embedding.weight` 形状 `[3,1024]` ⇒ **3 个可学习前缀 token**
  （对应 `positional_encoding_mode: learned_prefix_zero_at_first_generation`，`config.yaml:62`）

**文本特征维度 = 4096**，即 `llm_shape: [1, 4096]`（`config.yaml:49-51`）——Llama-3-8B 的 hidden size；`use_text_mask: false`（`config.yaml:52`）。
⇒ 对应 README 的"v1 只消费预计算 embedding"决定：**契约的文本输入是 `[B, L, 4096]` float32 + `[B, L]` bool mask**。

## 4. ✅ 采样调度（步数、噪声表、是否有 CFG、默认 `num_steps`）

### 4.1 噪声表：**cosine β schedule**（硬编码，不来自 checkpoint）

```python
# model/diffusion.py:12-26
def get_beta_schedule(num_diffusion_timesteps, max_beta=0.999):
    """Get cosine beta schedule."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float)
```

- `alphas_cumprod_base = cumprod(1 - betas_base)`（`diffusion.py:38`）
- **只有 tokenizer 与 denoiser 一套调度**：`Ardy` 只建一个 `Diffusion`（`ardy_model.py:67`）；tokenizer 是冻结的、不参与扩散（`fsq.py:131-135`）。

### 4.2 采样器：**DDIM，eta = 0（确定性）**

```python
# model/diffusion.py:113-133
class DDIMSampler(nn.Module):
    """Deterministic DDIM sampler (eta = 0)."""
    def __call__(self, x_t, pred_xstart, t):
        eps = (self.diffusion.sqrt_recip_alphas_cumprod[t, None, None] * x_t - pred_xstart) \
              / self.diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t, None, None]
        x = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps
        return x
```

### 4.3 步数

| 量 | 值 | 证据 |
|---|---|---|
| `num_base_steps`（调度长度） | **10** | `config.yaml:2`；`Diffusion(num_base_steps)` `diffusion.py:32-41` |
| CLI `--diffusion_steps` 默认 | `None` → 回落到 `num_base_steps` = **10** | `scripts/generate.py:56-60,196-197` |
| 允许范围 | `1 <= steps <= num_base_steps`，否则抛错 | `generate.py:198-202` |
| 子采样 | `space_timesteps(n)`：`use_timesteps = round(arange(num_base_steps) * (num_base_steps-1)/max(1,n-1))`，并 clamp | `diffusion.py:50-58` |
| 循环方向 | `indices = list(range(num_denoising_steps))[::-1]`（t 从大到小） | `ardy_model.py:622` |
| 每步刷新调度 | `space_timesteps` + `calc_diffusion_vars` **每步都做** | `ardy_model.py:263-264`、`:627-628` |

> ⇒ **第一版默认 `num_steps = 10`，DDIM，确定性，无随机性**（除初始噪声与 token 级采样）。

### 4.4 Classifier-Free Guidance：**有，`cfg_type = "separated"`（三路）**

`config.yaml:3` = `cfg_type: separated` → `AutoLatentClassifierFreeGuidedModelSeparated`（`cfg.py:12-19`）。

三路 batch（`B → 3B`，`cfg.py:74-77` 注释）：

```
Pass 0: text-only       (真文本, 零约束)
Pass 1: constraint-only (零文本, 真约束)
Pass 2: unconditional   (零文本, 零约束)
```

合成公式（`cfg.py:148-149`）：

```python
out_text, out_cstr, out_uncond = torch.chunk(out_3, 3, dim=0)
return out_uncond + cfg_weight_text * (out_text - out_uncond) + cfg_weight_cstr * (out_cstr - out_uncond)
```

| 量 | 值 | 证据 |
|---|---|---|
| CLI 默认 | `--cfg_weight` = `[2.0, 2.0]`（= `[text_weight, constraint_weight]`） | `generate.py:90-96` |
| 单值时 | 视为纯 text 权重，constraint 权重记 0.0 | `generate.py:176-181`；`ardy_model.py:272-277` |
| `Ardy.__call__` 形参默认 | `cfg_weight: Optional[float] = 2.0` | `ardy_model.py:541` |

### 4.5 初始噪声与种子

- 每个自回归窗口重新采纯噪声：`x_t = torch.randn(shape, device=device)`（`ardy_model.py:481`），`shape = (B, num_generation_tokens, 148)`
- history token 直接来自 `history_sequence`（干净），generation token 用 `x_t`（`ardy_model.py:488-490`）
- 种子：`if args.seed is not None: seed_everything(args.seed)`（`generate.py:232-233`），在**建输入之前**调用
- ⇒ B 的采样器必须用同一 RNG 语义才能与金标准逐位对齐（**待 A 后续提供 golden 时确认 RNG 细节**）

## 5. ✅ 自回归推进时上下文窗口怎么裁

### 5.1 训练窗口上限 = **10 秒**

```python
# scripts/generate.py:106-115
def _default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """Longest history that, together with the generation horizon, fits the model's trained window.

    ARDY is trained on windows of at most 10 s. Without history cropping the attention window grows
    with the output, and generations longer than the trained window degrade into jitter — so each
    autoregressive step must see at most this many history frames (same budget as the interactive
    demo).
    """
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token
```

### 5.2 默认值（Core / 20 fps / horizon 40）

| 量 | 计算 | 值 |
|---|---|---|
| patch（token 大小） | `num_frames_per_token` | **4 帧** = 0.2 s |
| `max_window_len` | `(10*20 // 4) * 4` = `(200//4)*4` | **200 帧** (10 s) |
| **默认 history（batch 路径）** | `((200 - 40)//4)*4` | **160 帧** (8 s) |
| `gen_horizon_len` | `config.yaml:4` | **40 帧** (2 s) |
| 每步总窗口 | `history + horizon` | **200 帧** |

> ⚠️ **batch 路径与 GUI 的默认值相反**，不要混用：
> GUI 滑块 `History Crop Length` 的 default 是 **min**（= patch size，`README.md:194`），
> 而 `scripts/generate.py` 的 `--history_frames` default 是 **max**（= 160 帧，`generate.py:209-210`，`README.md:252`）。

`--history_frames` 约束：必须是 `patch` 的正整数倍，否则抛错（`generate.py:211-212`）。

### 5.3 推进与裁剪的算术

```python
# ardy_model.py:569-570
num_autoregressive_steps = int(np.ceil((num_frames - init_history_len) / gen_horizon_len))
num_gen_frames = num_autoregressive_steps * gen_horizon_len + init_history_len

# ardy_model.py:630-634
for auto_step in range(num_autoregressive_steps):
    history_end_frame = auto_step * gen_horizon_len + init_history_len
    history_start_frame = (
        0 if crop_history_length is None else max(0, history_end_frame - crop_history_length)
    )
```

- **保留的历史帧数 = `crop_history_length`（默认 160）**，用滑动窗口右对齐到 `history_end_frame`
- 窗口内的 frame 级 mask 分三段（history / generation / future）：
  `get_three_mask_from_len`（`ardy_model.py:19-26`），在 `_generate_window` 中构造（`:439-445`）
- token 级 mask 由 frame 级 mask 取 `.all(dim=-1)` 得到（`latent_utils.py:192-220`）
- **只有 generation token 会被去噪并覆盖**；history token 保持干净（`denoising_step:302-313`：只对 `generation_token_mask` 位置做 DDIM 更新，其余 `x.clone()` 原样保留）

### 5.4 每个 auto step 之后的重定心（避免坐标漂移）

```python
# ardy_model.py:655-663
#  recenter the history sequence as specified
# transl_center_mode == "last_history":
center_frame_index = (history_end_frame + generation_len - 1).clamp(min=0)  # [B,]
history_sequence, center_pos, new_root_motion = self._recenter_history(history_sequence, center_frame_index, requantize=True)
global_transl = global_transl + center_pos
```

- 以**刚生成块的最后一帧**为原点，把 root 的 x/z 平移回原点（y 不动，`recenter_root_motion` `ardy_motionrep.py:112-134`）
- 平移量**累积**进 `global_transl`，最后统一加回（`ardy_model.py:679`）⇒ 输出仍是世界坐标
- 同时重新量化 body latent（`requantize=True`，`ardy_model.py:356-359`）

### 5.5 裁剪时同步更新起始朝向

```python
# ardy_model.py:665-674
if crop_history_length is not None:
    heading_angles = self.motion_rep.get_root_heading_angle(
        self.motion_rep.global_root_stats.unnormalize(new_root_motion),
    )
    full_history_len = new_root_motion.shape[1]
    cropped_history_start_frame = max(0, full_history_len - crop_history_length)
    first_heading_angle = heading_angles[:, cropped_history_start_frame]
```

⇒ **`first_heading_angle` 不是全程固定为 0**：裁剪生效后它被更新为"裁剪窗口首帧"的朝向。

### 5.6 horizon 与 fps 的关系

`gen_horizon_len` 是**帧数**（Core H40 = 40 帧 @20fps = **2.0 s**；G1 H52 = 52 帧 @25fps = **2.08 s**）。
"10 秒窗口"是**帧数预算**：`10 * fps`，再向下取整到 patch 的整数倍（`generate.py:114`）。
⇒ 20 fps 下 200 帧，25 fps 下 `(250//4)*4 = 248` 帧。

### 5.7 解码阶段也要按同样的窗口切

`get_explicit_motion_from_hybrid_autoregressive`（`latent_utils.py:103-144`）在 detokenize 时
按 `num_generation_tokens` 分块，并用 `crop_history_length // num_frames_per_token` 取历史 token 起点
（`:118-125`）——**B 若自己实现解码，必须复刻这个分块方式**，否则与参考实现不等价。

## 6. ✅ 是否必须跑 `MotionCorrection` 扩展、它改哪些自由度

### 6.1 先澄清：**不存在 "MotionCompression"**

在参考实现里 grep `MotionCompression|motion_compression` → **零命中**。
只有 `MotionCorrection/`（C++ pybind11 扩展，`setup.py:99-102`，模块名 `motion_correction._motion_correction`），
源码含 `AnimProcessing/InverseKinematics.cpp` 与 `AnimProcessing/TrajectoryCorrector.cpp`。

### 6.2 批处理路径（`scripts/generate.py`）**默认会跑，但可关**

```python
# generate.py:265-275
# G1: postprocessing is disabled (does not work well for this model).
use_postprocess = "g1" not in resolved_model.lower() and not args.no_postprocess
if use_postprocess:
    corrected = post_process_motion(
        output["local_rot_mats"],
        output["root_positions"],
        output["foot_contacts"],
        model.skeleton,
        constraint_lst=constraint_lst or None,
    )
    output.update(corrected)
```

| 情形 | 是否跑 |
|---|---|
| **Core**（core / soma）+ 默认 | **跑** |
| Core + `--no-postprocess` | 跳过 |
| **G1** | **永远跳过**（`"g1" not in ...`） |

`--no-postprocess` 的说明：`Don't apply motion post-processing to reduce foot skating (ignored for G1)`（`generate.py:79-83`）。

### 6.3 它改的是哪些自由度

签名（`postprocess.py:184-192`）：

```python
def post_process_motion(
    local_rot_mats: torch.Tensor,   # (B,T,J,3,3)
    root_positions: torch.Tensor,   # (B,T,3)
    contacts: torch.Tensor,         # (B,T,4)
    skeleton: SkeletonBase,
    constraint_lst: Optional[List] = None,
    contact_threshold: float = 0.5,
    root_margin: float = 0.04,
) -> Dict[str, torch.Tensor]:
```

实际改动（`postprocess.py:277-344`）：

1. 把 `local_rot_mats` 转**四元数**（`:280`），把 `root_positions` 拷成 `hip_translations_corrected`（`:279`）
2. 调 `motion_postprocess.correct_motion(...)`，**原地**修改这两个量（`:315-327`）；
   该 Python 包装最后写回 `hipTranslations` 与 `jointRotations`（`MotionCorrection/python/motion_correction/motion_postprocess.py:97-101`）
3. `local_rot_mats_corrected = quaternion_to_matrix(rotations_corrected)`（`:329`）
4. 再用 **FK** 重算 `global_rot_mats` 与 `posed_joints`（`:331-337`）
5. 返回 `{local_rot_mats, root_positions, posed_joints, global_rot_mats}`（`:339-344`）

⇒ **修改的自由度 = ① 根（髋）平移 `root_positions`（含 y？以 `root_margin` 为准）② 全部局部关节旋转 `local_rot_mats`**；
**不修改** `foot_contacts`（只作为输入标签）。算法性质：脚接触驱动的 **trajectory correction + IK**（由 C++ 侧实现，本次未读 C++ 细节）。

参数默认：`contact_threshold = 0.5`、`root_margin = 0.04`（`:190-191`）。

### 6.4 第一版能否跳过？——**建议跳过，但必须固定口径**

**依赖代价**：`postprocess.py:307-314` 是**硬依赖**，缺扩展直接抛错：

```python
try:
    from motion_correction import motion_postprocess
except ImportError as e:
    raise RuntimeError(
        "Motion correction is required for this postprocessing path but the "
        "motion_correction package is not installed. Install with -e ."
    ) from e
```

⇒ 若本机不编译这个 C++ 扩展，跑 Core 时必须显式传 `--no-postprocess`。

**跳过的后果（必须写进对齐报告）**：

1. 输出会保留 **foot skating（脚底打滑）**，视觉质量下降（`generate.py:82` 的措辞即 "to reduce foot skating"）。
2. **与参考金标准产生系统性数值差异** —— 金标准默认是**跑了**后处理的（Core）。
   ⇒ `scripts/ardy_align.py` 必须**同口径**比较：要么两侧都 `--no-postprocess`，要么必须复刻后处理。
   **建议第一版：两侧都跳过后处理**，并在对齐报告中写明；B 侧无需重写 C++ 扩展。
3. 后处理还会**修正约束跟随**（`README.md:177` "improve constraint following"），所以带 constraint 的场景差异更大。

## 7. 磁盘布局（**定稿 v1**）

```
out/<model>/<prompt-hash>-<seed>.npz      # 数据
out/<model>/<prompt-hash>-<seed>.json     # 元数据 sidecar
```

`<model>` = checkpoint 全名（如 `ARDY-Core-RP-20FPS-Horizon40`）；
`<prompt-hash>` = prompt 字符串 UTF-8 的 sha256 前 16 个十六进制字符；`<seed>` = 十进制整数（未指定 seed 时用 `noseed`）。

**`.npz` 成员（定稿）**

| 成员 | dtype | 形状 | 说明 |
|---|---|---|---|
| `frames` | `float32` | `(T, D)` | 契约特征向量，**已反归一化**，世界坐标，单位米。`D = 330`（Core）/ `414`（G1）；block 顺序见 §1.1 |
| `posed_joints` | `float32` | `(T, J, 3)` | 世界关节位置（`inverse()` 直接给出，免下游重算 FK）；`J = 27` / `34` |
| `root_positions` | `float32` | `(T, 3)` | 根世界位置（与 `frames[:, 0:3]` 一致，冗余但便于消费） |
| `foot_contacts` | `bool` | `(T, 4)` | 顺序 `(左heel, 左toe, 右heel, 右toe)` |
| `offsets` | `int64` | `(K+1,)` | 每个自回归块的**累计帧边界**，`offsets[0]=0`、`offsets[K]=T`；用于诊断长时程漂移 |
| `fps` | `int32` | 标量 | `20` / `25` |
| `text` | `str`（`<U…`） | 标量 | 原始 prompt |

**`.json` sidecar（定稿）**

```json
{
  "model": "ARDY-Core-RP-20FPS-Horizon40",
  "skeleton": "CoreSkeleton27",
  "fps": 20,
  "horizon": 40,
  "num_frames_per_token": 4,
  "feature_dim": 330,
  "num_frames": 100,
  "duration_s": 5.0,
  "seed": 0,
  "num_steps": 10,
  "cfg_weight": [2.0, 2.0],
  "postprocess": false,
  "text_embedding_sha256": "<sha256 of the precomputed embedding file>",
  "git_sha": "<ardy-mlx HEAD>",
  "reference": {"repo": "nv-tlabs/ardy", "commit": "693f74d"},
  "created_utc": "2026-10-13T00:00:00Z"
}
```

**定稿理由与取舍**

- 存**反归一化 + 世界坐标**的 `frames`：这是唯一能让下游在不持有 stats 文件的情况下解释数据的口径；归一化张量不落盘。
- `frames` 是"特征向量"，**需要 skeleton + §1 的 block 约定才能解释**。若下游只需要可播放的骨架姿态，
  用 `posed_joints`（同时落盘，避免下游自己实现 FK）。此项若评审认为冗余，可在 v1.1 移除，但**移除前需人类批准**。
- `offsets` 取代了原草案里含糊的"offsets"，明确为**自回归块边界**。
- `meta.json` 从"npz 内成员"改为**独立 sidecar**：NPZ 不适合放嵌套结构，且 sidecar 便于 `git`/人工 diff。
- **待定稿的一处**：`<prompt-hash>` 的哈希口径（UTF-8 原文 vs 规范化后文本）。本稿取**UTF-8 原文**，
  若将来做 prompt 规范化，需在此处明确并**重新定稿**（原因：hash 变了会导致既有产物路径失效）。

## 8. 与 B 的接口小结（可直接照此写测试）

```python
# 常量（A 侧导出，见 src/ardy_mlx/io/__init__.py）
FEATURE_DIM_CORE = 330      # CoreSkeleton27, 20 fps
FEATURE_DIM_G1   = 414      # G1Skeleton34, 25 fps
FRAMES_PER_TOKEN = 4
LATENT_DIM       = 128      # FSQ channels (levels=64)
HYBRID_TOKEN_DIM = 148      # 20 (root) + 128 (latent body)
ROOT_BLOCK_DIM   = 5
BODY_DIM_CORE    = 325
NUM_BASE_STEPS   = 10
CFG_TYPE         = "separated"
DEFAULT_CFG_WEIGHT = (2.0, 2.0)
TEXT_FEAT_DIM    = 4096
MAX_WINDOW_SECONDS = 10
```

| 步骤 | 输入 | 输出 |
|---|---|---|
| 文本 | `[B,L,4096]` + `[B,L]` mask | prefix（内部） |
| 采样循环 | `x_t [B,T/4,148]`（`randn`） | 每个 auto step 更新 generation token |
| detokenize | hybrid `[B,T/4,148]` + local root cond `[B,T,4]` | 特征 `[B,T,330]` |
| 落盘 | `inverse()` 的 dict | `out/<model>/<hash>-<seed>.npz` + `.json` |

**未解决（B 不要自行决定，需要时写进 `docs/status-claude.md` 请求裁定）**

1. MotionBricks/Unity 的坐标与旋转转换（§2.4）——**待定**。
2. `<prompt-hash>` 的哈希口径（§7）——本稿取 UTF-8 原文。
3. 金标准是否含后处理（§6.4）——**建议第一版两侧都跳过**，等人类确认。

## 9. 实测确认（A 侧第 2 轮 · 2026-10-13，**只加注记，未改任何字段/维度/顺序**）

本节是**新增注记**：契约本身的字段、维度、顺序一律未动。证据来自那台机器上**唯一一次**授权参考运行
（`nv-tlabs/ardy` @ `693f74d`，`ARDY-Core-RP-20FPS-Horizon40`，prompt `A person walks forward.`，
`--duration 2.0` → 40 帧 @20fps = 1 个 horizon，`--seed 0`，`--no-postprocess`，其余默认）。
运行记录见 `artifacts/golden-run/RUN-RECORD.md`。

**已确认（数字级）**

1. **§1.3 的 D = 330 与 block 尺寸**：整段 `frames (1,40,330)` 已在 MLX 侧逐元素复现，残差 1.9e-05；
   `root` 块（前 5 维）与参考 `.npz` 的 `root_positions`/`global_root_heading` 残差 5.1e-06 / 3.0e-06。
2. **§1.6 的 stats 切片**：`stats/motion/{mean,std}.npy` 是 334 长，模型内部特征 330 长，
   切片确为 `0..4` + `9..333`（跳过 4 个 local-root 槽位）。
   **实测：`global_root_stats.unnormalize(frames_norm[:, :, :5])` 与参考 `.npz` 的
   `[root_positions | global_root_heading]` 逐位相等（max_abs = 0.0）**。
3. **§3.3 的 hybrid token = 148**、`nframe_root_dim = 20`、`latent_embedding_dim = 128`：
   dump 到的张量形状与之一致（`(1,10,148)`），且 denoiser **224 张量全部被前向用到**（零未使用）。
4. **§3.2 的 tokenizer 形状表**：encoder `[512,1300]`/`[128,512]`、decoder `[512,128]`/`[512,528]`/`[1316,512]` 全部核对通过；
   Stage A 往返（encode→decode）残差 ≤ 7.0e-06。
5. **§4.x 的调度与采样**：余弦 β 表、`space_timesteps`、`calc_diffusion_vars`、DDIM `eta=0`
   已按 `diffusion.py` 复刻，**全部 10 个去噪步**逐步对齐（原始输出 ≤ 2.8e-05，CFG 合成 ≤ 5.4e-05，DDIM ≤ 1.2e-05）。

**新增注记（契约里没写、但 B 必须知道，否则会静默错）**

| # | 注记 | 证据 | 误用后果（实测） |
|---|---|---|---|
| N1 | `global_root_to_local_root(..., normalized=True)` 的**返回值是归一化后的** local root：先 `global_root_stats.unnormalize` 输入，算完 4 维物理量后再 `local_root_stats.normalize` 输出 | `reps/base.py:145-146` | 误当物理量传给 Stage A decoder：输出错 **0.47**（正确残差 6e-06） |
| N2 | tokenizer 的 `requantize()` **方法**（带 stats：`unnormalize → round(clamp·32)/32 → normalize`）与 `detokenize()` **内联**的那一次重取整（只有中间一步）**不是同一件事** | `fsq.py:236-238` vs `:257-278` | 混淆后 Stage A 往返错 **1.696** |
| N3 | FSQ 量化不是朴素 `round(z*32)/32`：先 `tanh(z + atanh(offset/half_l))·half_l − offset`，其中 `half_l=(64−1)·(1+1e-3)/2`、`offset=0.5`；值域**不对称**，实测 `[−1.0, 31/32]` | `finite_scalar_quantization.py:124-131` | 直接量化会让离散层级大范围错位 |
| N4 | backbone 的 prefix 是 `[text(1) \| timestep(1) \| first_heading_angle(1)]`，且 **denoiser backbone 没有因果掩码**（只传 `src_key_padding_mask`）；`timestep` 是**查正弦 PE 表的第 t 行**（不是把 t 连续嵌入） | `backbone.py:300-396`、`:439-461` | 加因果掩码：单步输出错 **1.28**（正确 5.4e-05） |
| N5 | `pos_encoding_token_index` 是**相对第一个 generation token** 的（可为负）；`history_len // num_frames_per_token` 是原点 | `auto_latent_twostage_denoiser.py:190-193` | 多窗口时位置编码整体偏移 |
| N6 | 无约束生成时**约束投影仍然参与计算**（`observed_motion`/`motion_mask` 喂全零，`future_token_mask` 全 False），不要走"跳过"的捷径 | `auto_latent_twostage_denoiser.py:200-260` | `global_root_hybrid_proj` 与 constraints 投影不可互换 |
| N7 | CFG 三路 batch 时 **`timesteps` 也必须一起三倍化**（以及 `history/generation/future` 的 len、mask、`first_heading_angle`） | `cfg.py:74-116` | 直接 shape 报错（已抓到） |
| N8 | 金标准里 dump 了参考自己的初始噪声 `x_T`（`(1,10,148)`）——该窗口无历史故 `x == x_T`。对齐应**注入**它，不要试图复刻 CUDA 的 Philox RNG | `ardy_model.py:481` | 复刻 RNG 是另一件事，会掩盖真正的数值差异 |
| N9 | `global_root_to_local_root` 的末帧速度 = **前一帧复制**（先补零再 scatter），不是 0 | `motion_rep/tools.py:58-105` | 末帧速度静默为 0 |

**仍未验证（保留为残留未知，B 不要据此写死）**

- **多窗口**（`--duration > 2.0`，即 >1 个 horizon）的 history cropping / 多次 recenter /
  `first_heading_angle` 更新路径：已按 `ardy_model.py:620-680` 实现，但**本轮只有单窗口金标准**。
  → **第 3 轮已补**，见 §10。
- 带 `--constraints` 的路径（`motion_mask`/`observed_motion` 非零）本轮**完全没跑**。
- §2.4 MotionBricks/Unity 转换：仍待定（`0002` 裁定 v1 不做）。

## 10. 实测确认（A 侧第 3 轮 · 2026-10-13，**只加注记，未改任何字段/维度/顺序**）

本节同样是**纯注记**：契约的字段、维度、顺序一律未动。证据来自 `0003` 授权的**第二次、也是最后一次**
参考运行（同 prompt、`--seed 0`、`--no-postprocess`、文本编码器 CPU bf16，**只把 `--duration` 改成 4.0**
→ 80 帧 @20fps = **2 个 horizon**）。运行记录见 `artifacts/golden-run-4s/RUN-RECORD.md`；
逐窗口对齐数字见 `docs/status-deepseek.md`。容差仍为 `0003` 冻结的 **1e-3**，并附 11 条负对照。

**§5.x 的多窗口算术：现在有金标准支持（数字级）**

| 契约小节 | 实测确认 | 残差 |
|---|---|---|
| §5.3 窗口规划 | w0 `history_start_frame`/`history_end_frame` = `0`/`0`，`total_tokens` = 20；w1 = `0`/`40`，`total_tokens` = 20 | 几何**完全相同** |
| §5.3 token mask | 两窗口的 `history/generation/future` token mask 与参考 dump **逐元素相等** | 相等 |
| §5.4 每步末帧重定心 | `center_frame_index` = `39`（w0）、`79`（w1） | 相等 |
| §5.4 平移累积 | w1 收到的 `global_transl` = `[-0.04001, 0, 0.13138]`（= w0 的 `center_pos`） | 1.5e-06 |
| §5.5 `first_heading_angle` 同步更新 | w0 传入 `0.0`，w1 传入 **`0.0026115`**（来自重定心后的 w0 root motion 第 0 帧） | 1.6e-07 |
| §5.7 解码分块 | 分块方案 `[(0,0,10), (0,10,20)]`（crop=160 时两块的 `history_token_start` 都为 0） | 见 N12 |

**新增注记（契约里没写、但 B 必须知道，否则会静默错）**

| # | 注记 | 证据 | 误用后果（实测） |
|---|---|---|---|
| N10 | **速度"复制末帧"的落点是 `lengths − 1`，不是张量的最后一帧。** `compute_vel_xyz`/`compute_vel_angle` 收一个 `lengths`（`tools.py:58-105`）：先补零，再 `v[lengths−1] = v[clamp(lengths−2,0)]`。denoiser 传的是 **`lengths = history_len + generation_len`**（`auto_latent_twostage_denoiser.py:219`），而张量长度是 `history_len + generation_len + future_len`。单窗口时二者相等（40 vs 40）所以看不出来；**4.0 s 的 w0 是 40 vs 80** | `tools.py:66-79`、`:100-110`、`auto_latent_twostage_denoiser.py:219` | 忽略 `lengths`：local root 条件全错，整段 `frames` 错 **6.75**（容差的 6750 倍） |
| N11 | 无约束路径下 **`future_token_mask` 恒为全 False**（`latent_utils.py:207`：`torch.zeros_like(history_token_mask)`），与 `future_mask`（帧级）**不是**同一件事。80 帧 w0 的 `future_mask` 有 40 帧为真，而 `future_token_mask` 是 20 个 0 | `latent_utils.py:199-220` | 用 `future_mask` 推 token mask：attention key padding 改变，`frames` 错 **1.34** |
| N12 | `crop_history_length is not None` 时**必须走分块解码** `get_explicit_motion_from_hybrid_autoregressive`（`ardy_model.py:684-691`），不能用整段 `get_explicit_motion_from_hybrid`。两者差的不只是浮点归约顺序：每一块只拿到自己那段（crop 后）的历史，`lengths` 也跟着变，local root 条件因此不同 | `latent_utils.py:103-140` | 整段解码 vs 分块解码差 **4.4e-02**（容差的 44 倍） |
| N13 | `history_frames`（裁剪预算）在 80 帧时**根本没触发截断**：`history_start_frame = max(0, history_end_frame − 160)` 在两窗口下都是 0。所以"裁剪长度差一帧"这类负对照在 `--duration 4.0` 上是**空转**（实测 `crop=164` 与 `crop=160` 输出逐位相同）；要证伪裁剪必须把 crop 压到会真正截断的值（实测 `crop=4`：`frames` 错 1.00） | `ardy_model.py:630-634` | 用空转的 crop 负对照会得出"裁剪已对齐"的假结论 |

**仍未验证（保留为残留未知）**

- 带 `--constraints` 的路径（`motion_mask`/`observed_motion` 非零），含 `future_token_mask` 非零时
  的 `future_constraints_proj`：两次运行都**没有约束**，N11 只覆盖了全 False 那一支。
- 真正发生历史截断的多窗口（>200 帧，或显式小 `crop_history_length`）：本轮金标准只有 80 帧，
  截断分支**没有**金标准，只有纯算术单测与 `crop=4` 负对照。
- seed / CUDA RNG 语义对齐（沿用 `0003` 的残留未知）。
- §2.4 MotionBricks/Unity 转换：仍待定。


## 11. 实测确认（A 侧第 4 轮 · 2026-10-13，**只加注记，未改任何字段/维度/顺序**）

本轮补上 §1.5/§7 里 `posed_joints` 的**真骨架 FK**（`0004` 裁定 4 的收尾），并做最终端到端验收。
契约的字段、维度、顺序、单位一律未动；下面每条都是注记，可直接照此消费。

### N14 · `posed_joints` 现在是真 FK，骨架定义来自**参考源码**（不是 checkpoint）

- **骨架 = `CoreSkeleton27`**：关节名与父子关系取自 `ardy/skeleton/definitions.py:348-381`
  （27 关节，根 = `Hips`）；静息姿态 `neutral_joints` 取自 `ardy/assets/skeletons/cskel27/joints.p`；
  FK 本体是 `ardy/skeleton/kinematics.py:87-171`（`batch_rigid_transform`：
  `rel_j = neutral_j − neutral_parent(j)`、`T_j = [[R_j, rel_j],[0,0,0,1]]`、`G_j = G_parent @ T_j`、
  `posed_joints = G[..., :3, 3] + root_positions`）。**checkpoint 里没有骨架信息**，所以这份定义只能
  从 Apache-2.0 的源码抄；A 侧落在 `src/ardy_mlx/model/skeleton.py`。
- **特征向量 → `posed_joints` 的链**与参考 `inverse()` 完全一致（`ardy_motionrep.py:259-264`）：

  ```
  frames[..., 0:3]                      -> root_positions
  frames[..., 83:245] (6D, 27×6)        -> cont6d_to_matrix -> global_rot_mats
  global_rots_to_local_rots(global)     -> local_rot_mats
  fk(local_rot_mats, root_positions)    -> posed_joints (T,27,3)
  ```

  `local_joints_positions`（§1.1 的第 3 块）**不参与**这条链——它是 FK 的*输出*再加工，不是输入。
- **实测（本金标准）**：

  | 比较 | 2.0 s / 40 帧 | 4.0 s / 80 帧 |
  |---|---|---|
  | FK(金标准自己的 `local_rot_mats`+`root_positions`) vs npz `posed_joints` | **2.4e-07** | **1.2e-07** |
  | 同一 FK 的 `global_rot_mats` vs npz | 1.0e-06 | 8.9e-07 |
  | 从**金标准特征向量**走的链（上行整条） | **3.0e-07** | **4.8e-07** |
  | **出货路径**（我们自己的 `frames`）→ FK | **5.18e-06** | **4.81e-06**（分段 2.06e-06 / 4.81e-06） |
  | 占位实现（`local_joints_positions` + `root_pos`）→ 差 | **0.132** | **0.236** |

  即：补 FK 前是 `0004` 裁定 4 记录的那个 **0.132**，补后与 `frames` 自身残差（1.87e-05 / 3.54e-05）
  同量级；`posed_joints` 的 `|max|` 是 1.68，所以这是 5 个数量级的收紧。
- **负对照**（`0003` 裁定 1 要求每条新比较都能红）：把父子关系拍平成"全部挂在根上"
  → `posed_joints` 错 **6.51e-01**（2.0 s）/ **6.55e-01**（4.0 s），是容差的 651/655 倍。
  该对照在 `scripts/ardy_align.py` 里登记为诊断项 `posed_joints_flat_hierarchy_vs_npz`。
- **仍未接上（B 侧，A 不动其文件）**：`scripts/ardy_generate.py` 落盘的 `posed_joints` 仍是
  `synthesize_posed_joints` 的占位，sidecar 仍写
  `extra.posed_joints_source = "local_joints_positions (root-local) + root_pos; real FK not implemented"`。
  B 只需把 `_common_outputs` 里的 `synthesize_posed_joints(frames, layout)` 换成
  `ardy_mlx.model.skeleton.posed_joints_from_features(frames)`，并把那行文案改成
  `"skeleton FK (CoreSkeleton27) on global_rot_data + root_pos"`。
  **在这行接上之前，CLI 落盘的 `posed_joints` 不是契约值**（2.0 s 差 0.132 / 4.0 s 差 0.236），
  而 A 侧 API 已可用并有测试锁定。

### N15 · 6D 的 `matrix[..., 0]` 取的是**列**，不是"行"（§1.4 的措辞是标签，代码行才是行为）

`geometry.py:16-18` 的 `cont6d = concat(matrix[..., 0], matrix[..., 1])` 里，`matrix[..., 0]` 索引
**最后一个轴**，所以 6D 存的是旋转矩阵**前两列**；`cont6d_to_matrix` 也按列拼回（`x=normalize(c0)`、
`z=x×c1`、`y=z×x`，再 `stack([x,y,z], -1)`）。实测（2.0 s 金标准）：

| 读法 | 与 npz `global_rot_mats` 的差 |
|---|---|
| 把同一段 6D 当**列**读（= 代码的行为） | **0.026** |
| 当**行**读 | **2.01** |

0.026 不是错误：模型回归出来的 6D 并不严格正交（`|r0|−1` 最大 0.017、`r0·r1` 最大 0.027），
Gram–Schmidt 正是为修它；把 `cont6d_to_matrix` 的输出再转回 6D 稳定在 **2.4e-07**（幂等）。
`0004` 的"契约是对的，不要修"依然成立——**代码行是权威**，本注记只把"行/列"这个标签说清楚。

### N16 · 复现金标准的最小 CLI 调用（以及两处 B 侧接线现状）

```
./.venv/bin/python scripts/ardy_generate.py --backend mlx \
    --embedding <text_feat.npy> --prompt "A person walks forward." \
    --duration {2.0|4.0} --seed 0 --no-postprocess --history-frames 160
```

- **`--history-frames 160` 必须显式给**：CLI 自己的默认是 `None`，落到模型就是
  `crop_history_length=None`，会同时关掉**裁剪**、`first_heading_angle` 刷新（§5.5）与
  **分块解码**（§5.7）。实测该默认与金标准差 **4.42e-02**（容差的 44 倍，正是负对照
  `crop_history_length=None` 的数）；显式传 `160`（= 参考 batch 默认，§5.2）后
  2.0 s **1.869e-05**、4.0 s **3.543e-05**。
- **`--noise-file` 在 ≥2 个窗口时注入不进去**：它的判据是
  `injected.shape[1:] == shape[1:]`，而 `injected` 还带 batch 轴（`(W,1,10,148)` vs `(1,10,148)`），
  于是多窗口文件永远不匹配，**每个窗口都静默回落到 MLX RNG**（实测整段错 4.42，但没有任何报错/警告）。
  单窗口文件（`(1,10,148)`）是好的：2.0 s 用它可以复现 1.869e-05。
  多窗口要注入参考 `x_T`，本轮用的是"在进程内替换 `mx.random.normal`"（不改 B 的文件，同第 3 轮手法）；
  该验收已固化成 `scripts/ardy_align.py --cli-acceptance`，可一键复跑。
- 两条都在 B 的文件里，A 只报告不改；`--history-frames` 的默认值建议改成
  `layout.default_history_frames`（A 侧已把该值导出为 `PRESETS[...].default_history_frames`）。

**仍未验证（保留为残留未知）**

- 带 `--constraints` 的路径（`motion_mask`/`observed_motion` 非零）：两次参考运行都没有约束。
- 真正发生历史截断的多窗口（>200 帧）：没有金标准（沿用 §10 的 N13 结论）。
- seed / CUDA RNG 语义：未对齐（靠注入参考 `x_T` 绕开）。
- §2.4 MotionBricks/Unity 转换：仍待定。
- `posed_joints` 的 CLI/sidecar 接线（N14 末条）：A 侧 API 与测试完成，**B 侧未接**。

## 12. 注记（A 侧第 5 轮 · 2026-10-13）：文本编码器已迁出本仓库

**本节只加注记，契约的字段、维度、顺序、单位一律未动**；`text_feat` 这一路接口的定义也没变。

- **文本编码器已独立成 `~/dev/llm2vec-mlx`**，不再属于本仓库；判据文档随之搬到
  `llm2vec-mlx/docs/decisions/0002-mlx-text-encoder.md`（本仓库原 `0006` 已删除）。
- **接口（不变）**：条件向量是 `(1, 1, 4096)` **float32** —— 与
  `artifacts/golden-run/raw/text_feat.npy` 逐维同形同序；新仓库另附同名 `.json` sidecar
  （`encoder` / `bits` / 权重与适配器指纹 / 池化与模板口径 / git sha）。
- **消费方口径（不变）**：查看器**优先参考向量**（`encoder: "reference-llm2vec-bf16"`），
  MLX 向量（`encoder: "mlx-llm2vec"`）只在没有参考向量时兜底；**sidecar 缺 `encoder` 字段按参考解释**。
- **不许混用**：两种 `encoder` 的向量不可互换 —— 编码器不同 → 条件不同 → 动作不同（扩散对条件混沌）。
  这不影响 §1 的任何一维、任何单位，也不影响 `/api/plan` 的 `x_T` 注入与时长整数倍约束。

## 13. 注记（A 侧第 6 轮 · 2026-10-13）：查看器接上本地编码器，**契约未动**

**本节只加注记**：字段、维度、顺序、单位、`(1, 1, 4096)` 的形状与 float32 类型一律未动；
`/api/plan` 的 `x_T` 注入与"时长必须是 horizon 整数倍"两条约束也未动。

§12 写的"消费方口径"（参考向量优先、不许混用）本轮**落进了代码**，并补上了实现细节：

- **缓存键是 `(prompt_hash, encoder_id)`**，不再是单独的 `prompt_hash`。侧车多写两个字段：
  `encoder_id`（如 `mlx-llm2vec-4-bit-group-32-last-2-layers-embed-tokens-at-8-bit`）与
  `quantization`（全名，如 `4-bit group 32, last-2 layers + embed_tokens at 8-bit`）。
  纯写 "4-bit" 会与**过不了线的那个 artifact**同名（纯 4-bit group 32 最小 cos 0.9828 < 0.985），
  所以全名是必须的，不是装饰。
- **落盘布局**：参考向量在 `artifacts/embeddings/<prompt-hash>.npy`（历史约定不变），
  本机向量在 `artifacts/embeddings/<encoder-id>/<prompt-hash>.npy`。
  一个 prompt 因此可以同时持有参考向量与多个精度的本机向量，互不覆盖。
- **优先级不变**：有参考向量就用参考向量（HUD 明示 `reference bf16 (golden)`）；
  否则用**当前配置的那一档**本机向量；都没有才现场编码，编码失败**具名报错**
  （`local_encoder_unavailable` / `local_encoder_failed`），**绝不**退到别的 prompt 或别的编码器的向量。
- **本机向量不能与金标准对齐**：4-bit 对 16 条参考向量最小 cos 0.9875765（8-bit 0.9997453），
  但扩散对条件混沌 —— 两个相近但不同的文本向量给出的是**不同但同样有效**的两段动作，
  **不是**"与金标准一致"。所以本机向量那一路**不参与**金标准对照，这一点写进了 HUD 与文档。
- **编码器不可用不是错误状态，是具名状态**：`/api/health` 的 `encoder` 块给出
  `ready` / `unavailable`（带 `reason` 与逐候选检查）/ `disabled` 三态，探测**不加载权重**。
