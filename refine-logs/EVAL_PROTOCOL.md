# 评测协议 v1（冻结，2026-11-09）— SCMG

- 冻结点：paper/final-1109，tag eval-protocol-v1。
- 草案原文：refine-logs/EVAL_PROTOCOL_DRAFT_1107.md。
- 冻结后的任何改动都用新版本号。

## 0. 通用规则
- 方法、参数和阈值只在开发集上选。
- 测试集：每个最终方法只跑一次，运行前写入 TEST_USAGE.jsonl。
- 推理用 `--seed-mode per_clip`。种子：G1 主表 0 / 1，EX002 0 / 1 / 2。
- 统计：按录音聚类的配对 bootstrap，2000 次重采样，报差值和 95% 区间。
- 时延、RTF、显存：只在空闲的卡上测。

## 1. G1 主表（ECHO-G 基准）
- 数据：val_common 3,242 段。训练集 train_drop，开发集 hos 400 段。
- 评测器：ECHO-G 官方 scripts/eval_g1_motion_cls.py，sha256 前缀 b91c7917ca5d8969。参数 `--fps 30 --ba-direction audio_to_motion --require-all-stems`。
- 指标：FGD、|ΔBA|、|ΔDiv|、ΔJerk(W)。
- 可执行性：scripts/echog_g1/executability.py（2a75896b091e2d8b），子集 vc_every8，`--collision-stride 1`。我们的方法报部署层（cbf_qp + rg pairs）之后的输出，基线报原始输出。
- 协议：
  - offline：整段音频 + 最终文本；
  - psync：词念完后才可见，只用已到的音频前缀，前瞻 160 ms。
- 已用测试集的记录：~/bulk/causal-train-1105/runs/TEST_USAGE.jsonl，18 条。
- 判定脚本：scripts/eval/beat_g1.py（a41ed286a0552943）。

## 2. EX002
- 数据：view data-v36_1。开发集是 val_ids 380 段（v3.2 BEAT2 dev，说话人 2_scott / 13_lu）。
- 测试集：manifest v3.2 的 test 拆分，282 段（beat2_en 153，beat2_zh 129）。这些段从未导出，也从未参与训练或选择。
- 最终方法：
  - 底模 enc-s0（3 个训练种子），加安全层 3c-4b；
  - 显式语义采用 canonical 路径（scripts/ex002/expl_canon.py，判据见 EXPLICIT_SEM_1109.md 修订 2–6）；
  - 协议 TTS-sync（前瞻 0.5 s）和 psync。
- 质量：
  - FGD v2 为主指标，用两个冻结编码器的均值，权重为 human_seed0 / 1 的 best.pt，sha256 前缀分别是 b8dcffe81c1d5550 / bafb601408e27576；
  - FKD 和活跃度护栏一起报。
  - 打分工具：scripts/ex002/bench_ex002.py（dc90f13eb609f4a5）。
- 活跃度护栏，在会动的上半身（腰、双臂、头、手指）上算：
  - 静止帧 ≤ 2 × 真人；
  - 手腕速度、手腕幅度都 ≥ 0.75 × 真人；
  - 连续保持超过 2 s 的片段 ≤ 10%。
- 安全：
  - unsafe 帧、fallback 帧、qp_fail；
  - 判定条件 S1–S3 见 THEORY_SAFETY_1109.md；
  - 代码 scmg/safety_proj.py（72d3e2a8a0b997ce）、scmg/safety_proj_ex002.py（846c7204709447e9）；
  - 可执行性：scripts/ex002/ex002_exec.py（42083066ee339017）。
- 显式语义：
  - 按类别报 success(given) − success(none) 和 bootstrap 区间；
  - 指标定义取 EXPLICIT_SEM_1109.md 修订 2；
  - 打分工具：expl_canon.py eval（1b6678d1d814e92a）；
  - 指令窗口外的 FGD 用 ex002_fgd_outside.py（971b896ba88d6730）。
- 打断：
  - 指标为 v3 突兀率、RKD（同时报各收手分量）、unsafe；
  - 真人参照是 BEAT2 说完后的尾段，按半 A / 半 B 划分；
  - 参照只在开发集上有，测试集不计算打断。
- 指向 / 计数：STG 和打乱后的跟随检查一起报；手掌角带符号；计数用已验证过的检测器。

## 3. 读数规则（取自 EVAL_AUDIT_1108.md 第 4 轮及之后的补充）
- STG 的增益必须和打乱检查一起读。
- FKD 和活跃度护栏不单独读，主指标是 FGD v2。
- v3 只用于突兀率。
- RKD 和各收手分量一起读。
- 释放连续性报中位数、p95 和最大值。
- 手掌角带符号。
- 不设单一的"让出"判据。
- 指向指令是事后重标的。
- pair_win 只在完整开发集上读，而且要求核查行通过。
- 不使用的指标：yield3d、v4、PSS；子集上的 pair_win。

## 4. 判据文件（sha256 前 16 位）
| 文件 | sha256 |
|---|---|
| docs/criteria/EX002_SFSCOPE_1108.md | c599897bae2b74be |
| docs/criteria/EX002_CBF_1108.md | 96e579e96a804818 |
| docs/criteria/EX002_INTERRUPT_1107.md | 9c51523c158c106b |
| docs/criteria/EX002_CTL.md | d2e21f421a5379e4 |
| docs/criteria/EXPLICIT_SEM_1109.md | ab55652932d01f04 |
| refine-logs/EVAL_AUDIT_1108.md | 7bd901b8f79bd31e |
| scripts/eval/beat_ex002.py | 708aecdcbc3d1b0a |
| scripts/eval/beat_ex002_ctl.py | 82c1310d37646c85 |
