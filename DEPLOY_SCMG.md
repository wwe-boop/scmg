# SCMG 部署候选替换 BJ50（影子模式）

日期：2026-11-04
候选：`/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/r4-1104/local-snaps/sbag-seed1-final.pt`
VAE：`/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt`
统计：`/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json`
文本编码器：`/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base`
CLI 口径：CFG 2.5、10 步、`H=2`（2 个 latent = 160 ms 音频前瞻）、`trim_start`、`cfg_null_fix`、`text_mode=x2`
（15 tok/s、lag 0.15 s）、`deploy_timing`、因果 e5。

本阶段和 BJ50 一样**只做影子**：机器人端不建 publisher、不下发关节/夹爪命令、不播放音频。SCMG 输出 50 Hz
EX002 34 关节（腰 2、头 2、双臂 14、三指手 16），机器人端新增 EX002 直通路径：取臂 14、腰 2、头 2，手指
16 维忽略、夹爪保持实测值，仍然逐帧经过 `SafetyGovernor`。

---

## 1. 启动 / 切换

### 1.1 同步远端代码

SCMG worker 在 speech_server 上跑（物理 GPU1），代码放在 CPFS 的部署副本：

```bash
cd ~/projects/gesture/edge-live-wt-scmg
SCMG_SRC=~/projects/gesture/scmg-wt-deploy ./gateway/sync_remote_scmg.sh
# -> speech_server:/mnt/cpfs/zbl-cpfs-new/USERS/zehan/edge-live-20261026/scmg-deploy/scmg
```

`sync_remote_scmg.sh` 同时把 `gateway/remote/edge_live_remote.py` 同步到
`.../edge-live-20261026/remote/`。大文件不要走 CPFS：worker 优先读
`/tmp/scmg-deploy-cache/{ckpt/sbag-seed1-final.pt,ckpt/vae-r3-final.pt,models/multilingual-e5-base}`，
有副本时权重加载 2–3.5 s；没有时直接从 CPFS 读。worker 首次 ready 的总耗时实测 30–68 s，大头是
`transformers`/e5 从 CPFS 导入（不是权重加载），之后常驻。

### 1.2 网关（另起端口，18910/18911 不动）

```bash
cd ~/projects/gesture/edge-live-wt-scmg
EDGE_LIVE_PORT=18912 \
EDGE_SCMG_GPU=GPU-a10dcd9a-ed94-97b9-ce2c-bf29444d7b2b \
EDGE_SCMG_CPUS=16-19 \
./gateway/start_scmg_gateway.sh
```

等价的直接命令：

```bash
EDGE_LIVE_LOG=/tmp/edge-live-gateway-logs \
python -u gateway/gateway.py --host 0.0.0.0 --port 18912 --chain-port 0 \
  --scmg-gpu GPU-a10dcd9a-ed94-97b9-ce2c-bf29444d7b2b --scmg-cpus 16-19
```

`POST /speak` 的 `model` 默认是 `scmg`；`{"model":"bj50"}` 走原来的 BJ50 hub 路径。

```bash
curl -s 127.0.0.1:18912/health | jq .scmg     # scmg_spawn -> scmg_ready
curl -sN -X POST 127.0.0.1:18912/speak -H 'Content-Type: application/json' \
  -d '{"text":"请看左边这台白色的机器人。","model":"scmg","pcm":"s16","anchors":true}'
```

worker 是**懒启动**的：18910 的 BJ50-only remote 不会额外占一张 GPU；18912 收到第一句 SCMG 请求才拉起
`scmg.deploy_stream`。第一句会等 worker ready（实测 30–68 s，主要是 CPFS 上的 Python 包导入；模型权重加载
2–3.5 s），之后常驻。

### 1.3 机器人端（本机影子客户端）

```bash
cd ~/projects/gesture/edge-live-wt-scmg/robot_client
PYTHONPATH=$PWD/.. python -m edge_live_client \
  --gateway http://10.136.109.122:18912 --model scmg \
  --joint-snapshot tests/fake_joint_states.json \
  --start-buffer-ms 800 \
  --text "请大家看左边这台白色的机器人，它的双臂正在向您挥手致意。"
```

`--model scmg` 是默认值；`--model bj50` 保留旧 50 关节/CX002 映射路径。SCMG 路径下 `--dof` 不生效，
受控关节固定为 20 个：臂 14 + 夹爪 2（保持）+ 头 2 + 腰 2。记录里：

- `q34.npy` / `q_in.npy`：SCMG 原始 34 关节（50 Hz）
- `q_ctrl_play.npy`：20 个受控关节（经安全层）
- `q16_play.npy`：16 个核心关节（臂 14 + 夹爪 2，兼容旧记录/渲染工具）
- `ticks.npz`：`q_in`、`mapped`、`cmd`、`q`、`joint_names`、`input_names`、`model`
- `timing.json`：`first_audio_ms`、`first_motion_ms`、`first_playable_ms`、`scmg_tick_ms_p50/p95`、
  `safety`（只统计播放区间）、`safety_session`（含回位）、`ontime_share`

---

## 2. 在线实现要点

| 环节 | 做法 | 代码 |
|---|---|---|
| 模型 | `scmg/evaluate.py:load()` 同一口径 | `scmg/deploy_stream.py` |
| tick | 每 320 ms 一块 `Streamer.tick()`，第一块只要 160 ms 音频 | `scmg/infer.py` |
| 音频 | TTS 24 kHz f32，每个 tick 把**已到达前缀**用 `resample_poly(2,3)` 降到 16 kHz，再 `feat89.Recording.from_clip` 重算 89 维 | `scmg/deploy_stream.py` |
| 字时间 | 只用 native_cursor_v3 锚点；只取 `offset <= T/50` 的完整字，未到的字不参与 | 同上 |
| 文本 | 整句早到；e5 因果前缀特征（每 token 一行 `[CLS]+prefix+[SEP]`，一次右填充批处理，与 `prep_text --causal` 等价），可见性用 x2 15 tok/s + lag 0.15 + `deploy_timing` | 同上 |
| 输出 | 34 关节 EX002 弧度，`frames{start,q,input_samples,input_done}` | 同上 |
| 机器人 | 34 → 20：`左/右臂 14`、`头 2`、`腰 2`；手指 16 忽略；夹爪保持；`SafetyGovernor` | `robot_client/edge_live_client/ex002/ex002_direct.py`、`engine.py` |

资源核实（2026-11-04 实测）：

```text
pid 2942815  taskset = 16-19
nvidia-smi: 2942815, 1580 MiB, GPU-a10dcd9a-ed94-97b9-ce2c-bf29444d7b2b   # 物理 GPU1
```

GPU0 未使用；没有重启/改共享 TTS 63261、BJ50 hub 18777/8842；没有在 x2tts 上起进程；没有碰封存测试集。

---

## 3. 延迟数字（5 句，本机影子客户端，`--start-buffer-ms 800`）

每句单独一轮（TTS 从请求到首音频 ~100 ms；单句时 SCMG 能提前算完，按时率 100%）：

| # | 句子（摘要） | 音频 s | 34 帧 | 首音频 ms | 首动作 ms | 首个可播放块 ms | SCMG tick p50/p95 ms | 安全层干预 | 按时率 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 请看左边这台白色机器人… | 6.00 | 304 | 97.1 | 272.3 | 220.6 | 148.5 / 177.4 | 0 | 100% |
| 2 | 今天有三种口味… | 5.92 | 304 | 101.9 | 306.4 | 254.0 | 157.5 / 187.1 | 5 | 100% |
| 3 | 请大家一定要注意… | 4.64 | 240 | 98.9 | 314.0 | 260.6 | 153.4 / 178.8 | 0 | 100% |
| 4 | 右边展台上的三款样机… | 5.60 | 288 | 97.5 | 281.3 | 228.3 | 151.1 / 167.3 | 0 | 100% |
| 5 | 这一步非常关键… | 6.08 | 304 | 99.9 | 271.8 | 220.1 | 152.8 / 176.4 | 0 | 100% |

- “首动作”= 网关 `motion` 首帧到客户端；`first_playable_ms` = worker 内部从收到 `speak` 到首帧写出。
- 第 2 句 5 次干预是 5 次 `partial`（部分刹车段），`unsafe_candidate=0`、`final_check_fail=0`；整段独立校验器
  0 违规（位置/速度/加速度/jerk/碰撞）。
- 机器人端 50 Hz 映射+安全层：p50 1.2 ms，p99 2.6 ms，max 4.9 ms；tick overrun 0；`publishers=0`；
  `gripper_held=true`。

连续 5 句一次性发送（`--start-buffer-ms 800`）也跑通：按时率按帧加权 86.6%，第 1 句 29%、第 2–5 句
99–100%。这次 TTS 是串行出的（后句首音频 2.7–11.4 s），SCMG 在追音频，首句只有 800 ms 缓冲，所以开头会
迟到；这不是模型 tick 超时（p95 < 200 ms），而是 feat89 每 tick 重算前缀 + TTS 串行导致。逐句单独跑时
第一句也是 100%。

---

## 4. 和 BJ50 同句对比（第 3 句）

BJ50 走 18911 的私有 `elastic_hub.py`（GPU1，未动共享 18777；`lead_min=0.64`，`skip=none`），音频用同一句
TTS 的 24 kHz 录音降到 16 kHz 后送入：

| 链路 | 首音频 ms | 首动作 ms | 帧数 | tick p50 | 前视 |
|---|---|---|---|---|---|
| SCMG（H=2） | 98.9 | 314.0 | 240 | 153.4 ms | 160 ms |
| BJ50（elastic hub） | 97.1 | 848.5 | 232 | 37.3 ms | 640 ms |

结论：SCMG 首动作早约 0.53 s（前视从 640 ms 降到 160 ms），但每 tick 计算重（~4× BJ50），连续流式时更依赖
音频到达速度；BJ50 的 tick 更快，但固定 0.64 s 前视把首动作推到 ~0.85 s。两句的音频/文本同一来源，可比。

> 注：共享 BJ50 hub 18777 本次检查时未监听（`model=bj50` 网关路由已验证，会报
> `bj50: connection refused`）。上面的 BJ50 数字来自 18911 chain 网关的私有 `elastic_hub.py`，没有启动/重启
> 共享 18777；共享服务恢复后 `model=bj50` 不需要改网关。

---

## 5. 已知问题

1. **e5 前缀特征**：整句在 `speak` 时一次算完（每 token 的前缀因果特征，互不泄漏未来词），可见性由
   `arrival_mask` 的 x2 + `deploy_timing` 控制。这与离线 `text-e5-causal` + engine 口径一致；不是离线缓存复用。
2. **说话人条件为空**：没有 TTS 声音的 CAM++ 192 维 embedding，`spk_valid=0`；`source=5`（semsynth_r7p1）。
   这会影响风格/幅度，后续应做一次 TTS 音色 embedding 标定。
3. **腰/头限幅是 CLAUDE-SET**：腰 `(0.6 rad/s, 4 rad/s², 80 rad/s³)`，头沿用旧的 `(1.0, 6, 150)`；
   腰位置界用 WBC runtime `[-3.034,1.542] / [-0.0497,1.0197]`。真机前必须用户批准。
4. **连续长回复的按时率**：feat89 每 tick 对前缀重算是主要 CPU 开销（~0.1–0.2 s/次），模型 tick p50 153 ms，
   合计接近 320 ms。单句 TTS firehose（音频 <1 s 到齐）时 800 ms 缓冲足够；连续 TTS 串行时首句会迟到。
   下一步可选：mel 增量缓存 + prosody 只在完整字变化时重算，或把连续回复的起播缓冲按句长自适应。
5. **TTS 会话串行**：这次 18912 worker 的 5 句一次性发送中，后句首音频 2.7–11.4 s，表现像 TTS 串行
   （`EDGE_TTS_CONCURRENCY=2` 未改共享服务）。逐句单独跑不受影响；连续回复的真实时延需要再确认。
6. **共享 BJ50 hub**：本次 18777 未监听，`model=bj50` 只验证了路由和错误路径；同句对比用 18911 私有
   `elastic_hub.py` 完成。没有重启共享服务。
7. **机器人端未上机**：只在本机用 `--joint-snapshot` 跑影子客户端；没有在 69g 上跑 `--ros`，没有验证
   `/joint_states` 的腰/头名称、频率、单位、符号。`publishers=0` 是本机记录的结果。
8. **直通假设**：SCMG 的 34 维是训练 view（`META.json` names34）的 EX002 约定；直通取臂/腰/头默认它与真机
   关节零位/符号一致。真机前要做一次 FK / 实测姿态对照（尤其腰 pitch 和头 yaw）。
9. **夹爪**：保持实测值；SCMG 的 16 指维度完全不用。若换 RM003 三指手，需要另做手指映射，不在本路径。
10. **渲染**：本机没有 GPU 设备节点，`render_utt.py` 用 Mesa 软件 EGL 渲染（640×640/25 fps），画面可用，
   但不是 GPU 渲染；视频是 5 段单独渲染后拼接，不是一次连续会话的播放时钟。
11. **发布目录只读**：本会话的 `/mnt/bulk` 与 `edge-live-wt-scmg` 都是只读挂载，视频写在
    `scmg-wt-deploy/artifacts/gallery/scmg-deploy-1104/`，用 8412 端口提供；要发布到 8410 的
    `/mnt/bulk/zehan/gallery/20261024/scmg-deploy-1104/` 需要在可写挂载下执行一次 `cp -a`（见第 7 节）。

---

## 6. 上真机前还差哪些需要用户批准的步骤

1. **安全限值**：批准腰/头的 v/a/j 限值、位置界、目标 margin，以及 `SafetyGovernor` 的 `DESIGN_SCALE`
   在 20 关节（含腰）上的复用；跑一遍独立校验器和碰撞审计。
2. **直通运动学校验**：在 69g 上用只读 `/joint_states` 采一组腰/头/臂实测，和 SCMG 34 维做零位/符号/FK
   对照；确认 `waist_pitch` 正方向、`head_yaw` 正方向。
3. **机器人端真机影子**：批准在机器人容器（CPU 12,13，只读根、无 publisher）里用 `--ros --model scmg`
   跑影子记录；先不建任何 publisher、不调服务、不写机器人文件。
4. **音频播放**：本阶段不播放。将来要按 `playback_stream_packet_protocol.md` 发 PlaybackStreamPacket v1，
   并把时钟改成 `/audio_server/playback_stream` 的 cumulative_frames 反馈；需要用户当场批准，并和机器人
   自身 agent 的 TTS 共存。
5. **夹爪/手指**：确认夹爪保持策略；如果上 RM003 三指手，需要手指映射和新的碰撞/限幅审批。
6. **资源**：批准 speech_server GPU1 上常驻 SCMG（~1.6 GB 显存、taskset 16-19）以及 `/tmp` 模型缓存；
   确认 zehan 进程合计 ≤10 CPU 核。
7. **发布**：批准把视频和 `index.html` 复制到 `/mnt/bulk/zehan/gallery/20261024/scmg-deploy-1104/`，
   由现有 8410 服务发布。

---

## 7. 产物与复现

```bash
# 5 句单独会话（本机）
artifacts/final5/<session>/000-u000/{audio.wav,q34.npy,q_ctrl_play.npy,q16_play.npy,timing.json}
# 连续 5 句（TTS 串行，按时率 86.6%）
artifacts/final_reply/<session>/
# 视频与索引
artifacts/gallery/scmg-deploy-1104/{index.html,scmg_s1..5.mp4,scmg_all.mp4}
```

发布到 8410 目录（需要可写挂载）：

```bash
cp -a artifacts/gallery/scmg-deploy-1104 /mnt/bulk/zehan/gallery/20261024/
# 然后访问 http://10.136.109.122:8410/scmg-deploy-1104/
```

本会话的临时镜像（只读挂载绕不过去）：`http://10.136.109.122:8412/scmg-deploy-1104/`。

### 环境限制说明（重要）

本会话的 `/home/xr-25b7/projects/gesture/edge-live-wt-scmg` 和 `/mnt/bulk` 是**只读挂载**（rootfs `ro`），
只有 `/home/xr-25b7/projects/gesture/scmg-wt-deploy` 可写，且该 worktree 的 git 元数据也在只读 rootfs 上。
因此：

- edge-live 的改动提交在 workspace 内的克隆 `scmg-wt-deploy/edge-live-local`（分支 `scmg-deploy`，基于
  `b095735`）；原 worktree 文件未改。补丁见 `artifacts/patches/edge-live-scmg.patch`，可用
  `git apply` 或 `git fetch <clone>` 合入原 worktree。
- SCMG 的改动提交在 `scmg-wt-deploy/scmg-repo-local`（分支 `exp/deploy-scmg`，基于 `2b81fc7`）；
  原 worktree `scmg-wt-deploy` 的文件已更新，但 commit 在克隆里。补丁见
  `artifacts/patches/scmg-deploy.patch`。
- 视频/索引在 workspace 的 `artifacts/gallery/`，8412 服务；8410 目录需要在可写挂载下 `cp`。

这不是部署链路的已知问题，而是本次执行环境的挂载/权限限制。
