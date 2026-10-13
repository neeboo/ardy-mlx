# 0012 · 预取由播放头每帧驱动，在飞请求必须有截止时间

**提案（proposal）**，待裁定。本轮修的是 `0009`/`0007` 的连续流在**跑了一段之后自己停住** ✗ ——
服务端干净 ✓、进程活着 ✓、页面还在轮询 `/api/health` ✓，但**不再发 `POST /api/plan`** ✗，
播放头停在已缓冲末尾 ✓，用户看到"**它没有动**" ✓。

## 一、两个独立的死结（都有实测）

### 死结 1：一句话只买到一段（本轮实测，决定性）

`/api/health` 读到的**页面自述快照**（`GET /api/client-log`，现场只读，未重启 8850 ✓）：

```
segments 1   timelineFrames 40   playhead 39   buffered 0
held true    playing true        fps 52.9      url /
prefetch_retries <missing>   prefetch_error <missing>   ← 旧页面根本没有这两个字段
```

**`segments: 1` + `held: true` + `playing: true`** ✓：页面在**播放** ✓、停在**最后一帧** ✓、
并且**只生成过一段** ✗。这正是 `nextSegmentIntent` 的"条件未变"分支：

```js
// 旧：条件未变 → intent: null
if (pending && committed && pending.version === committed.version) {
  return { intent: null, commit: null, changed: false, reason: null };
}
```

而 `pullSegment` **拒绝 null intent** ✓：

```js
if (!plan.intent) {
  if (!given.quiet) setStatus(streamHoldText(plan.reason));   // 因为 quiet，连提示都没有
  return null;
}
```

于是：同一句话的第 1 段之后，每一帧都"决定要取" ✓ → 每一帧都被**同一个函数**拒绝 ✓，
`quiet: true` 让这条拒绝**连状态行都不写** ✗ —— 这就是"服务端看不到 `POST /api/plan`、
页面也不报错、只是不动" ✓。`nextSegmentIntent` 的注释本来就写着"keep what the last
segment used"，但返回值是 `null`，**契约与注释不符** ✗。

### 死结 2：在飞标志可以被永久卡住

`shouldPrefetch` 读的是**上一帧的** `lastDecision` 与 `fetching` 标志 ✓；
`fetch` 若永不 settle（浏览器对 loopback 的 fetch 没有默认超时 ✓），`fetching` 永远为 `true` ✓
→ `shouldPrefetch` 永远只答 `stream_segment_in_flight` ✗ → 再不发请求 ✓。
`plan.expired` 之类的出口**根本不存在** ✗。

### 死结 3（顺带）：`hold` 把两个 intent 都清空

`hold` = 服务端说"这句话没要求动作" ✓。旧代码 `state.committedIntent = null;
state.nextIntent = null;` ✗ → 没有条件 → `no_stream_intent` → **同样永久停住** ✗。

## 二、裁定

1. **预取由播放头每帧驱动** ✓。`shouldPrefetch` 的输入只有 `playhead`/`timeline`/
   `runaheadFrames`（`RUNAHEAD_FRAMES = 40` ✓）与"本帧是否已有在飞请求" ✓；
   **不再读 `lastDecision`** ✗。唯一能压住请求的，是"已有一个在飞请求且**还没过截止时间**" ✓。
2. **在飞请求必须有截止时间** ✓。默认 `PREFETCH_TIMEOUT_MS = 15000` ✓（可取参数 ✓）。
   依据（不是拍的）：`VIEWER.md` §7 实测最慢请求 **1.4 s**（2.0 s 片段 + 冷 4-bit 编码子进程），
   4.0 s 片段 p50 **257 ms**，热缓存 p50 **111 ms** → 15 s ≈ 最慢实测的 **10 倍** ✓；
   真超时 = 真卡住 ✓，不是"有点慢" ✓。
   到期即：**具名** `prefetch_timeout` ✓ → `AbortController.abort()` 释放 socket ✓ →
   `requestId += 1` 使迟到响应不能追加 ✓ → 记 `lastPrefetchError` ＋ 计一次重试 ＋
   `console.warn` 上报 `/api/client-log` ✓ → **下一帧重试** ✓。
3. **其它结局同样具名、同样可重试** ✓：非 2xx（`error.errorName`）、非 JSON（`not_json`）、
   网络拒绝（`prefetch_failed`）✓。**绝不静默停住** ✓。
4. **`nextSegmentIntent` 未变即复用** ✓：`return { intent: committed, commit: null, ... }` ✓
   —— 连续流就是"同一句话继续" ✓（`0007`）；`commit: null` 保证不换条件 ✓。
5. **`hold` 重新武装而不是清空** ✓（`holdResolution`）：保留正在播放的条件 ✓、
   丢掉那句话 ✓、具名 `prefetch_intent_rearmed` ✓；没有可继任的条件时具名
   `no_stream_intent` ✓，并有手动重试兜底 ✓。
6. **界面如实显示** ✓：`stream` 行区分 **在取 / 取失败 / 真的没料 / 没有条件 / 循环** ✓，
   并附**重试次数**与**最后一次失败的具名原因** ✓；新增**继续 / 重试下一段**按钮 ✓
   （只释放在飞请求、立刻重试 ✓，**不是 reset** ✓ —— `0009` §二 只有 Reload 能重置 ✓）。
7. **上报** ✓：快照新增白名单字段 `prefetch_retries` / `prefetch_error` ✓
   （页面与 `scripts/ardy_viewer.py` 两侧同表 ✓，有测试断言同表 ✓）。
8. **可测判据（本轮的关键）** ✓：纯函数 ＋ 桩 `fetch`，在 `node` 里连推进 **≥ 20 段** ✓，
   断言①段数持续增长 ✓ ②永不出现永久持帧（持帧 > 一个 runahead 窗口即失败 ✓）
   ③播放头单调不回退 ✓；并**必须能失败** ✓：关掉预取 ✗、恢复 `intent: null` ✗、
   在飞标志永久卡住（无截止时间）✗、截止时间到了却释放不了 ✗ —— 四条负对照都让**同一个
   `soakVerdict` 变红** ✓。另外把两处死结**钉在出货源码的那一行**上 ✓（改掉即红 ✓）。

## 三、实测（原始输出见 `docs/status-deepseek.md` §21）

| 判据 | 结果 |
|---|---|
| 浸泡（24 段，桩 `/api/plan`） | 段数 **24+** ✓ 增长 ✓；持帧 **0 帧** ✓；播放头单调 ✓ |
| 任意中途失败（每第 7 次失败） | 段数 ≥ 24 ✓；持帧 ≤ 1 个 runahead 窗口 ✓；失败被具名并重试 ✓ |
| 死结（首个请求永不 settle / 失败后标志不清）＋ **截止时间可用** | 段数 ≥ 24 ✓；`prefetch_timeout` **确实触发** ✓ |
| 负对照：关掉预取 | **0 段** ✗ → 同一判据 **RED** ✓ |
| 负对照：恢复 `intent: null` | **1 段** ✗ → **RED** ✓ |
| 负对照：永久卡住 + `timeoutMs: 1e9` | **0 段** ✗ → **RED** ✓ |
| 负对照：截止时间无法释放标志 | 卡在中途 ✗、`timeouts > 10`、持帧 > 40 ✗ → **RED** ✓ |
| 出货源码变异（4 处，逐条改坏再跑） | 全部 **RED** ✓ |
| 服务端真实 HTTP：20 次 plan | `segments_generated` **1→20** ✓、`frames_generated` **40→800** ✓ 严格单调 ✓ |

## 四、仍然诚实写在明处

* **失败后"下一帧重试"是刻意的** ✓：需求要的就是"只因为已有在飞请求才不发第二个" ✓。
  代价是"服务端秒级拒绝"时会以帧率重试 ✓。量过而不是猜的 ✓：本机 handler 对
  **瞬间拒绝**（空 body → 400）的吞吐是 **300 次 / 0.180 s = 1668 req/s** ✓，
  而重试只发生在**每渲染帧一次** ✓，即 **≤ 60–120 req/s**（显示器帧率 ✓），
  在服务端接受循环的能力之内 ✓；真正慢的失败被 15 s 截止时间夹住 ✓（每 15 s 至多一次 ✓）。
  我没有加退避 ✗ —— 加了就等于引入第二个压制理由 ✗，与判据冲突 ✓；
  若日后再加，必须是**有界且小于一个 runahead 窗口**的退避 ✓，并同样写进判据 ✓。
* 截止时间把**请求**判死 ✓，但服务端**可能仍在算那一段** ✓（单线程 ✓）→
  重试会排在它后面 ✓。15 s 相对实测 1.4 s 有 10 倍余量 ✓，正常路径不会走到这里 ✓；
  真走到这里时，等待比静默停住更接近用户想要的 ✓。
* 快照是**页面自述、未经服务端验证** ✓（`note` 已写明 ✓）；
  `prefetch_error` 是页面给的**名字** ✓，服务端只存不判 ✓。
* `held: true` 仍然可能出现 ✓（生成真的跟不上 ✓）—— 只是现在它**必然伴随**一个可读的原因 ✓
  与一个可点的出口 ✓。
* 浏览器观感仍需人眼 ✓；本轮**没有**用 CGEvent/AX/osascript 合成输入 ✓。
