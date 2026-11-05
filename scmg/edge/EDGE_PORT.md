# SCMG 端侧改造（Thor）计划与进度

分支：scmg `exp/edge-scmg`（基于 `deploy-chain-v0.5.0`），edge-live `edge-scmg`（基于 `robot-client/v0.5.0`）。
目标：SCMG（landing-l1-1104/sibag-coll-s1，sha256 cb51022d）在 EX002 Thor 上推理，去掉网关 + speech_server
链路；协议、弹性延迟、安全层、exec_server、新旧切换保持不变（新增 `--motion-backend scmg-local`）。
路线与 BJ50 端侧一致：服务器导出 ONNX → Thor 上构建 TensorRT（10.13.3.9）→ 控制逻辑与特征前端改 numpy → 逐级对齐。
Thor 上没有 PyTorch（69g 实查）。

## E0 测速（2026-11-05，本机 5090，fp32 eager，`python -m scmg.edge.profile_tick`）

| 模块 | 每步调用 | 每步 ms |
|---|---|---|
| flow_velocity | 20（10 步 × CFG 2 路） | 35.2 |
| plan | 2 | 5.6 |
| audio_tokens | 1 | 1.3 |
| vae.decode | 1 | 0.6 |
| 其余（text_tokens、local_audio、global_cond、掩码） | — | < 0.5 |
| 整步 p50 / p95 | — | 44.4 / 47.8 |

- 参数：模型 116.7M（planner 52.5M、flow 39.4M、audio 22.2M），VAE 4.4M；d=512、heads 8、planner 8 层、flow 6 层、z=16。
- flow_velocity 每次约 1.8 ms，矩阵很小，主要是 kernel 启动开销 → TensorRT + CUDA graph、CFG 两路合成 batch 2 是主要收益。
- feat89 完整特征（含韵律，`Recording.from_clip`）x86 单核：前缀 3.2 s 1.9 ms、24 s 12.1 ms；首次调用 327 ms 为 numba JIT。
- speech_server 上同一 tick 约 160 ms：共享 A800 + taskset 4 核 + nice 10 的 Python 启动开销，不是模型算力。

预算：每 320 ms 一步。Thor 估计（未实测）：TensorRT fp16 整步 20–60 ms；feat89 2 核 ARM 约 3–5 倍于 x86。

## 步骤

| 步 | 内容 | 验收 |
|---|---|---|
| E1 | 固定形状导出 ONNX：audio_tokens（窗口）、plan（cond/uncond batch 2）、flow_velocity（batch 2）、vae.decode/encode、e5（L≤64，前缀因果批） | onnxruntime 与 torch 逐模块最大误差 < 1e-3（fp32） |
| E2 | numpy 版 Streamer：窗口、掩码、x2 到达、Euler 10 步、CFG 合成、潜变量提交、首块修正 | 与 torch Streamer 在真机录音上逐帧 q34 最大误差 < 1e-2 rad |
| E3 | 机器人端特征前端：feat89 去 numba/librosa 依赖或预编译；复用 BJ50 numpy 前端可复用部分 | 与 feat89 逐维误差、2 核 ARM 耗时 |
| E4 | Thor 上 TensorRT fp16 引擎（trtexec / builder）+ CUDA graph；edge-live `--motion-backend scmg-local`（子进程，同 bj50_link） | Thor 上 tick p95 < 120 ms；回放 8 段 q34 与云端对齐 |
| E5 | 镜像 `gesturelab-edge-live-scmg-local`，切换脚本加 `edge`；影子 → 发布 | 真机影子覆盖率、首动作时延 |

注意：端侧不消除句首约 0.24 s 空档（x2tour 首包后 0.02 s 即开播，见 robot-client v0.5.0 实测），只省网络往返与共享 GPU 排队。
