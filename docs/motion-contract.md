# 数据契约（v0 草案 · 待从参考实现补齐）

**本文件是 A/B 之间唯一的接口。** 下面标 `?` 的地方必须由 A 从参考实现里读出真实值后填上，
**不许猜**；填错会让 B 白干。

## 已确认（来自上游 README 与本地 checkpoint 名）
| 项 | 值 |
|---|---|
| checkpoint | `ARDY-Core-RP-20FPS-Horizon40`（本地已存在）、`Horizon8`、`ARDY-G1-RP-25FPS-Horizon52/8` |
| 骨架 | **Core**（SMPL-X 系）与 **Unitree G1** |
| 帧率 | **20 fps**（Core）/ **25 fps**（G1） |
| horizon（块长） | **8 / 40 / 52 帧** |
| 训练数据 | Bones Rigplay 1（不影响接口，仅记录） |

## 待 A 补齐（不许猜）
| 问题 | 为什么重要 |
|---|---|
| 输出表示的精确字段与单位（旋转表示？根平移？接触？） | 决定 `clip` 的 `frames (T,D)` 与 `offsets` |
| 坐标系与手性、以及是否已是世界坐标 | 决定与 MotionBricks/Unity 之间怎么转 |
| 两阶段结构（tokenizer / denoiser）各自的输入输出张量名与形状 | 决定权重转换与模块划分 |
| 采样调度（步数、噪声表、是否 classifier-free guidance） | 决定 B 的采样器实现 |
| 自回归推进时上下文窗口怎么裁（保留多少历史帧） | 决定长时程是否漂移 |
| 是否必须跑 `MotionCorrection` 扩展、以及它改的是哪些自由度 | 决定第一版要不要重写它 |

## 磁盘布局（待定稿）
`out/<model>/<prompt-hash>-<seed>.npz`，含 `frames (T,D)`、`offsets`、`meta.json`（模型名、fps、horizon、embedding 指纹、git sha）。
