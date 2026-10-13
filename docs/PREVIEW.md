# 预览：金标准 vs 我们的 MLX（骨架 GIF）

> 回答"效果怎么样，能预览吗"。
> 本文所有数字都是**实测**的，不是估计；机器可读版本见
> [`artifacts/preview/preview-manifest.json`](../artifacts/preview/preview-manifest.json)。
> 生成侧的 git sha：**`b4918ba`**（两处 `golden-run*` 就是在这个 HEAD 上做的）。
> 渲染器：[`scripts/ardy_preview.py`](../scripts/ardy_preview.py)（内容指纹
> `ardy_preview.py #3a889e3`，即该文件 sha256 的前 7 位；它也印在每支 GIF 的 HUD 上）。

---

## 1. 五支 GIF 分别是什么

| 文件 | 是什么 | 噪声来源 | 尺寸 | 帧数 | fps | 时长 | 字节 | sha256（前 16） |
|---|---|---|---|---|---|---|---|---|
| `ardy-golden-2s.gif` | **金标准**（参考实现 nv-tlabs/ardy @ `693f74d`）2.0 s | 参考自己的 `x_T`（seed 0） | 520×480 | 40 | 20 | 2.00 s | 307,789 | `d48d8daceffcabd3` |
| `ardy-mlx-2s.gif` | **我们自己**的 MLX 输出 2.0 s | **我们自己的 MLX RNG** | 520×480 | 40 | 20 | 2.00 s | 402,918 | `f2334a59c070e405` |
| `ardy-mlx-2s-xT.gif` | **我们自己**的 MLX 输出，注入参考 `x_T` | **参考 `x_T`**（金标准的同源孪生） | 520×480 | 40 | 20 | 2.00 s | 307,903 | `6f9d98a88581dfa9` |
| `ardy-golden-4s.gif` | **金标准** 4.0 s（2 个自回归窗口） | 参考自己的 `x_T` | 520×480 | 80 | 20 | 4.00 s | 558,052 | `ea431942bea2e126` |
| `ardy-mlx-4s.gif` | **我们自己**的 MLX 输出 4.0 s（2 个窗口） | **我们自己的 MLX RNG** | 520×480 | 80 | 20 | 4.00 s | 510,064 | `7d420773a9de5c44` |

合计 **2,086,726 B ≈ 2.0 MB**（预算 5 MB 以内；分辨率 520×480、256 色、20 fps 未压缩）。

**"真的在动"的核实**：`ffprobe -count_frames` 实测帧数与时长如上（全部 >1 帧、时长非零）；
再把每支 GIF 解码回 RGB 逐帧算 sha256，**每帧的 sha256 都不同**
（2 s 三支 40/40 唯一，4 s 两支 80/80 唯一），所以不存在"重复帧堆出来的假动画"。
逐帧 sha256 全部落在 `preview-manifest.json` 的 `decoded_frame_sha256` 字段里。

配套入库的还有三份相机参数（很小，纯几何，不含权重）：
`camera-2s.json`（2 s 三支**共享**）、`camera-4s-golden.json`、`camera-4s-mlx.json`。

### 怎么渲的

* **几何**：直接用 npz 里的 **`posed_joints`（世界坐标，Y 轴向上，米，地面 = XZ 平面）**
  画 27 关节棍状骨架。没有重算 FK——CLI 已经把契约 S7 的 `posed_joints` 落盘了。
* **投影**：单一固定针孔相机 + 透视投影；每支 GIF 的相机由 `ardy_preview.py camera`
  按"该 GIF 要覆盖的关节在整段时长内都不出画"拟合一次（`azimuth 72° / elevation 12°`，
  即从侧前方看，前进方向横过画面）。相机**不跟随**角色，所以画面里的位移是真实位移。
  - **只有 2 s 的三支共享同一个相机**（`camera-2s.json`），因此第 3 节的像素比较只对 2 s 成立。
  - 4 s 两支相机各自拟合，因为我们的 own-noise 4 s 走了 3.82 m 而金标准只走 0.24 m（见 §3.4），
    共用一个相机就会把金标准缩成一小团，失去可看性。
* **画面元素**：XZ 地面网格 + 世界坐标轴（红 = X，蓝 = Z）、骨架投影到地面的**暗色影子**、
  按深度排序的骨骼（`painter's algorithm`）、按肢体分色（躯干灰白 / 右臂橙 / 左臂蓝 /
  右腿粉 / 左腿绿）。
* **脚部接触标记**：`foot_contacts` 为**真** → 地面上实心绿圆盘（直径 10 px + 外环）并画一条
  细绿线连到脚关节（"这只脚踩在这里"）；为**假** → 琥珀色空心环。列序按契约 S1.5
  =（左 heel, 左 toe, 右 heel, 右 toe）。截图里能看到 4/4 全绿与 2/4 带琥珀环两种状态。
* **HUD**：prompt、模型名、帧数、fps、时长、关节数、自回归窗口数、
  **是金标准还是我们的 MLX**、噪声来源、生成 git sha、渲染器指纹；右下角是
  当前帧号与当帧接触数；底部还有一条**时间轴**，1 个窗口的 clip 是实心进度条，
  多窗口时在窗口边界画黄色刻度（4 s 那支的第 40/80 帧处可见）。
* **技术栈**：**纯 numpy** 手写光栅化（距离场抗锯齿）+ `subprocess` 调系统 `ffmpeg`
  做 `palettegen/paletteuse` 编码。**没有** matplotlib / mujoco / smplx / torch，
  `.venv` 里没装任何新依赖。这台机器的 ffmpeg 没有 `libfreetype`（无 `drawtext` 滤镜），
  所以文字是**自己写的 5×7 点阵字库**，也因此 PNG/PPM 帧序列回落路径同样自带文字。

---

## 2. 一句话结论

**同源（注入参考 `x_T`）时，我们和金标准在数值上等价、在画面上几乎逐像素相同。**

* 世界关节 `posed_joints`：2.0 s → **5.18e-06**，4.0 s → **4.81e-06**（参考量级 |max| = **1.68**）。
* 330 维特征 `frames`（`artifacts/align-report.json → cli_acceptance`，同一个注入路径）：
  2.0 s → **1.87e-05**，4.0 s → **3.54e-05**（参考量级 |max| = **1.68**）。这是 5 个数量级的收紧。
* 渲染成图之后：**最大通道差 = 2/255**（金标准 vs `ardy-mlx-2s-xT.gif`）。
  也就是说两支 GIF 在**视觉上无法区分**。

一句话的证据强度：**渲染差异的最大值只有 2 个灰阶**，而画面里任何一条骨骼的宽度是 8 px。

---

## 3. 像素差异（实测）

**测量方法**：为了让数字只反映"动作/绘制"而不是"HUD 上写着 GOLDEN 还是 OURS"，
比较用的是**去掉文字**的渲染（`--no-text`）、**同一个 2 s 相机**、同尺寸同帧数。
"raster" 比的是渲染器自己吐出的 PPM；"gif" 比的是用户真正打开的 256 色 GIF。

### 3.1 金标准 vs 我们（注入参考 `x_T`，同源）

| 层级 | 不同像素数 / 总像素 | 比例 | 最大通道差 | 平均通道差 | 逐帧最大不同像素 |
|---|---|---|---|---|---|
| raster（PPM） | 7,250 / 9,984,000 | **0.0726 %** | **2 / 255** | 0.00026 | 56 |
| GIF（256 色） | 47,715 / 9,984,000 | 0.4779 % | 71 / 255 | 0.0061 | 1,266 |

**结论**：渲染层面两者差 **0.073 %**，且最大只差 **2 个灰阶**——差异全部落在骨骼边缘的
抗锯齿像元上，肉眼不可见。GIF 那一行的数字更大（0.478 %、最大差 71），
**不是运动变差了**，而是 GIF 只有 256 色 + Bayer 抖动：边缘上 1–2 个灰阶的差别会被量化成
**另一个调色板索引**，于是"几乎相同"被放大成"不同像素"。这两个数字请一起读。

### 3.2 金标准 vs 我们（**我们自己的噪声**，非同源）

| 层级 | 不同像素数 / 总像素 | 比例 | 最大通道差 | 平均通道差 |
|---|---|---|---|---|
| raster（PPM） | 600,273 / 9,984,000 | **6.0124 %** | 234 / 255 | 5.3011 |
| GIF（256 色） | 8,571,235 / 9,984,000 | 85.85 % | 234 / 255 | 6.0931 |

这一对比**本来就该不一样**：两支是不同的噪声采样，是两个不同的动作样本，
不是同一个动作的两次实现。raster 只差 6 % 而不是 100 %，是因为画面里
天空渐变 + 地面网格是逐像素相同的，骨架像元只占一帧的 ~6 %。

### 3.3 数值差异汇总

| 对比 | `posed_joints` max_abs（世界坐标，米） | 参考 |max| | `foot_contacts` 不一致帧 |
|---|---|---|---|
| 金标准 2 s ↔ 我们（注入 `x_T`） | **5.1782e-06** | 1.6792 | **0 / 40** |
| 金标准 4 s ↔ 我们（注入 `x_T`） | **4.8131e-06** | 1.6818 | **0 / 80** |
| 金标准 2 s ↔ 我们（自己噪声） | 1.2426 | 1.6792 | 23 / 40 |
| 金标准 4 s ↔ 我们（自己噪声） | 3.9111 | 1.6818 | 60 / 80 |

注意：我们这次注入 `x_T` 跑出来的 `posed_joints` 误差
（5.1781535148620605e-06 / 4.813075065612793e-06）与
`artifacts/align-report.json` 里登记的 `posed_joints_vs_npz`
（`5.1781535148620605e-06` / `4.813075065612793e-06`）**逐位相同**，
说明本次复现与当时那次验收是同一次运行。

### 3.4 自回归续接处是否平滑（不是"看着还行"，是量出来的）

逐帧"平均关节位移"这个序列里，窗口边界（第 39 → 40 帧那一步）**从来不是最大步**：

| clip | 边界步长 (m) | 中位步长 (m) | 最大步长 (m) | 边界/最大 |
|---|---|---|---|---|
| 金标准 4 s | 0.01576 | 0.00395 | 0.01660 | 0.950 |
| 我们 4 s（注入 `x_T`） | **0.01576** | **0.00395** | **0.01660** | 0.950 |
| 我们 4 s（自己噪声） | 0.07658 | 0.07340 | 0.09718 | 0.788 |

金标准与注入版的三项统计**完全一致**；三支里边界都不是最大步 → 拼接处没有抽搐。

### 3.5 一个值得知道的采样事实：走的距离差很多

| clip | 根节点世界位移（整段，米） |
|---|---|
| 金标准 2 s / 我们注入 `x_T` 2 s | 0.131 |
| 金标准 4 s / 我们注入 `x_T` 4 s | 0.236 |
| 我们 **自己噪声** 2 s | **1.096** |
| 我们 **自己噪声** 4 s | **3.819** |

参考 seed 0 这个样本几乎是**原地走**（0.065 m/s），我们自己采样的两个样本分别是
0.55 m/s 与 0.95 m/s（正常步速）。**这不是实现差异**——注入同一个 `x_T` 后我们复现到
5e-06、`foot_contacts` 逐位相同。这只是"同一个条件分布里的两个不同样本"，
正好说明为什么 §3.1 那种同源对比才是保真度证据。

---

## 4. 诚实的边界（五条 + 两条本文特有的）

1. **这是骨架预览，不是角色/蒙皮渲染。** 只有 27 个关节连成的棍状图，
   没有网格、没有蒙皮、没有光照材质、没有布料。它能证明"关节轨迹对不对"，
   不能证明"看上去像不像人"。
2. **金标准和我们两边都关闭了后处理**（`--no-postprocess`；契约 S6.4 / 决定 0002 裁定 2
   把 v1 钉在"两边都跳过 MotionCorrection"）。所以**会有 foot skating——脚会轻微打滑**，
   根轨迹也没有被修正。GIF 里看到脚在接触帧之间滑动是**预期行为**，不是 bug。
3. **`ardy-mlx-2s.gif` / `ardy-mlx-4s.gif` 用的是我们自己的随机噪声**，
   不是注入参考 `x_T`。所以**动作不同、但风格与指令相同**：同一 prompt、同一模型、
   同一 seed、同一条代码路径，只是换了噪声采样。
   想要"逐位同源"的对比，用 `--noise-file` 注入 → 那就是
   **`ardy-mlx-2s-xT.gif`**（与金标准 max 通道差 2/255）。
4. **没有移植 8B 文本编码器。** v1 只吃**预计算 embedding**，本次用的是参考实现
   自己 dump 的 `text_feat.npy`（`(1,1,4096)`，来自 LLM2Vec-Meta-Llama-3-8B-Instruct）。
   所以两支 GIF 的"文本条件"是**同一个**向量；预计算 embedding 的许可与可复现性
   仍是未决项。（顺带：`/tmp/ardy-emb/golden_only.npy` 那个 `(1,8,4096)` 文件是**全零**的
   占位 padding，不是有效 embedding；真正有效的是 `text_feat.npy`。）
5. **参考实现只跑过两次**（2.0 s 与 4.0 s 各一次，都是 seed 0、`--no-postprocess`）。
   所以**多窗口只有 4.0 s 这一支**，没有更长的参考、也没有第二个种子的参考，
   "8 个窗口以上 / 真正发生历史截断（>200 帧）"仍然没有金标准。
6. **GIF 里的文字会让像素比较失去意义**，所以 §3 的比较用的是无文字渲染；
   真正的 `ardy-*.gif` 与 `ardy-mlx-*.gif` 之间 HUD 上必然不同（标签就不同），
   不要拿那两支 GIF 直接做像素 diff。
7. **金标准的 npz 没有 `offsets` 成员**，直接渲染会把 4.0 s 错标成"1 window"
   （实际是 2 个窗口）。渲染时用 `--offsets 0,40,80` 显式给出；`ardy_preview.py`
   在遇到"无 offsets 且 >40 帧"时会打警告，避免这个错默认静默存在。

---

## 5. 复现命令

### 5.1 生成我们那支（`ardy-mlx-2s.gif` 的输入）

```bash
cd ~/dev/ardy-mlx

# 我们自己的 MLX 输出，2.0 s，自己采样噪声（= ardy-mlx-2s.gif 的来源）
./.venv/bin/python scripts/ardy_generate.py --backend mlx \
    --embedding artifacts/golden-run/raw/text_feat.npy \
    --prompt "A person walks forward." \
    --duration 2.0 --seed 0 --no-postprocess --history-frames 160 \
    --out out/preview/own2s
```

* **实测耗时：端到端 `real 0.99 s`**（CLI 自报 `elapsed 0.77s`；含权重加载）。
  同批的其它三支：注入 `x_T` 2 s → `0.47s`；自己噪声 4 s → `0.90s`；注入 4 s → `1.49s`。
* **`--history-frames 160` 必须显式给**：这是参考 batch 路径的默认（契约 S5.2），
  不传会退化成 `crop_history_length=None`，实测离金标准 **4.42e-02**（容差的 44 倍，契约 S11 N16）。
* **`--seed 0` 在 MLX 上是可复现的**（seed 被转成 `mx.random.key`）：连跑两次
  `frames` 逐位相同（已实测 `np.array_equal == True`）。
* 输出：`out/preview/own2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz`
  （`out/` 在 `.gitignore` 里，**不入库**）。

### 5.2 生成"同源孪生"那支（`ardy-mlx-2s-xT.gif` 的输入）

```bash
./.venv/bin/python scripts/ardy_generate.py --backend mlx \
    --embedding artifacts/golden-run/raw/text_feat.npy \
    --prompt "A person walks forward." \
    --duration 2.0 --seed 0 --no-postprocess --history-frames 160 \
    --noise-file /tmp/ardy-emb/x_t.npy \
    --out out/preview/xt2s
```

* 单窗口时 `--noise-file` 的形状是 `(1,10,148)`（= batch × 10 token × 148 维）。
  4.0 s 是 2 个窗口，必须传**堆叠**数组 `(2,1,10,148)`（由参考的
  `randn_0.npy` / `randn_1.npy` `np.stack` 而来）——旧的判据只比 `shape[1:]`，
  堆叠文件永远匹配不上，会**静默**回落到 MLX RNG（契约 S11 N16 记录的就是这个坑）。

### 5.3 渲染

```bash
cd ~/dev/ardy-mlx

# (1) 先拟合相机：2 s 三支共享一个，保证三者可以逐像素比较
.venv/bin/python scripts/ardy_preview.py camera \
    --npz artifacts/golden-run/out/golden_core_h40_2s_seed0.npz \
    --npz out/preview/own2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz \
    --npz out/preview/xt2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz \
    --width 520 --height 480 --azimuth 72 --elevation 12 \
    --out artifacts/preview/camera-2s.json

# (2) 渲染我们那支
.venv/bin/python scripts/ardy_preview.py render \
    --npz out/preview/own2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz \
    --gif artifacts/preview/ardy-mlx-2s.gif \
    --camera artifacts/preview/camera-2s.json \
    --label "OURS - ardy-mlx (MLX backend)" \
    --model "model ARDY-Core-RP-20FPS-Horizon40" \
    --prompt "A person walks forward." --git-sha b4918ba \
    --note "noise: our own MLX RNG (NOT the reference x_T)"

# 金标准那支（同一相机、同一几何管线，只换输入与标签）
.venv/bin/python scripts/ardy_preview.py render \
    --npz artifacts/golden-run/out/golden_core_h40_2s_seed0.npz \
    --gif artifacts/preview/ardy-golden-2s.gif \
    --camera artifacts/preview/camera-2s.json \
    --label "GOLDEN - reference implementation" \
    --model "model ARDY-Core-RP-20FPS-Horizon40" \
    --prompt "A person walks forward." --git-sha b4918ba \
    --note "noise: the reference x_T (seed 0)"

# 4 s 金标准：注意 --offsets —— 金标准 npz 没有 offsets 成员
.venv/bin/python scripts/ardy_preview.py render \
    --npz artifacts/golden-run-4s/out/golden_core_h40_4s_seed0.npz \
    --gif artifacts/preview/ardy-golden-4s.gif \
    --camera artifacts/preview/camera-4s-golden.json \
    --label "GOLDEN - reference implementation" \
    --model "model ARDY-Core-RP-20FPS-Horizon40" \
    --prompt "A person walks forward." --git-sha b4918ba \
    --note "noise: the reference x_T (seed 0)" --offsets 0,40,80
```

### 5.4 复现数值与像素差异

```bash
# 数值：世界关节 / 330 维特征 / 脚部接触
.venv/bin/python scripts/ardy_preview.py compare-npz \
    --npz artifacts/golden-run/out/golden_core_h40_2s_seed0.npz \
    --npz out/preview/xt2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz

# 像素：先出无文字渲染，再逐帧同尺寸比较（raster 层，避开 GIF 调色板量化）
for spec in "golden artifacts/golden-run/out/golden_core_h40_2s_seed0.npz" \
            "xt out/preview/xt2s/ARDY-Core-RP-20FPS-Horizon40/e9cf8141cac79f35-0.npz"; do
  set -- $spec
  .venv/bin/python scripts/ardy_preview.py render --npz "$2" \
      --png-dir /tmp/raster-$1 --camera artifacts/preview/camera-2s.json --no-text
done
.venv/bin/python scripts/ardy_preview.py compare \
    --gif /tmp/raster-golden --gif /tmp/raster-xt
# → changed 7250 / 9984000 = 0.0726 %, max_channel_delta 2

# 也可以直接比 GIF（数字会因 256 色量化而偏大，见 §3.1）
.venv/bin/python scripts/ardy_preview.py compare \
    --gif artifacts/preview/ardy-golden-2s.gif --gif artifacts/preview/ardy-mlx-2s-xT.gif
```

---

## 6. 关于入库的说明（权重派生数据）

* **入库**：`artifacts/preview/*.gif`（我们自己渲染的图像，不含任何权重）、
  `artifacts/preview/camera-*.json`、`artifacts/preview/preview-manifest.json`、
  `scripts/ardy_preview.py`、本文件。GIF 合计 ≈ 2.0 MB。
* **不入库**：`artifacts/golden-run/{raw,out}/`、`artifacts/golden-run-4s/{raw,out}/`、
  以及任何 `.npz` / `.npy` / `.safetensors`。它们只用于本地对齐与本次渲染，
  `NOTICE` 明确禁止再分发。渲染用的 `out/preview/**` 也被根 `.gitignore` 的 `out/` 覆盖。
* 审计命令（必须零命中）：

  ```bash
  git ls-files | grep -E '\.(safetensors|npz|npy)$|golden-run[^/]*/(raw|out)/'
  ```
