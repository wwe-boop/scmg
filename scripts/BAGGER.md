# BAgger 1103

实现依据：[BAgger §3、附录 B](https://arxiv.org/html/2512.12080v1)。本实现把视频的像素空间换成 34 维关节空间：真实前缀 → Streamer 自由滚动 → VAE 解码 → 逐帧反转 → VAE 重编码。每轮采集使用上一轮 final.pt，微调始终从同一初始检查点开始。

## 数据与开关

`scripts/bagger_collect.py --ckpt ... --out ... --cond null|reverse` 默认 2000 条、1 块真实前缀 + 23 块生成，10 步 Euler、CFG 从 checkpoint `cfg_w` 取值，缺省采用现有 lane 的 2.0。明确启用 trim_start、cfg_null_fix 和 VoxCPM window_lm_text。支持 --dry-run、--shard/--shards、按索引断点续采；种子由 round 和 index 决定。

只从 View(split='train') 采集，使用原类别/帧长采样权重并排除 holdout，拒绝 glitch 窗口。每条保存前向与反向关节（rad）、原始与条件处理后的音频特征（已归一化）、重编码 latent（已归一化）、原始文本/时间等条件。保留真实 seed 的原始关节，不以 VAE 重建代替。反转后的监督标签不包含原文本、speaker、reference style。

每个采集目录为 COLLECTION.json + NPZ/逐条 JSON + MANIFEST.json，逐条 SHA256 和原始 item ID 可追溯；仅全部完成且 hash 验证通过才写 MANIFEST。View 的 `bagger_dir` 和 WindowDataset 的 `bagger_share` 接入它；多个采集目录以冒号连接，累积池按条均匀采样。集合是实验用纠正池，未通过 COMMON_RULES §15 全部正式数据门禁，不写回 canonical view，不声称可直接执行或正式入库。

`--bagger-share 0` 完全走原迭代器、RNG、模型与损失路径，不读取增量目录。启用后新来源 bagger_reversed 的 id=7，旧模型加载后扩展零行，不消耗随机数；新模型 checkpoint 显式记录 n_source=8。Config 默认 7 保证旧初始化逐位不变。N_SOURCE 表示所有可用来源数 8。

## 监督范围

完整 24 块轨迹 `loss_from_chunk=16`（零基）；前 16 块提供漂移上下文，后 8 块监督。为保持 ft_lane 的 256 帧输入，训练采尾部 16 块，窗口内阈值变为 8，含全部 8 块目标及 8 块漂移历史。若改为 --frames 384，则读取完整的 16+8 块。训练在实际裁剪后的关节窗口重新 VAE 编码；完整重编码 latent 保留作审计，不直接反转 latent、不跨裁剪边界复用编码。

BAgger 样本的 planner 历史不加 sigma_tf，local flow 上下文不加 ctx_noise，也不做 self-forcing 或随机丢上下文；prefix target tokens 标为 known。普通样本沿用 ft_lane 原始配置（StageCfg 默认 self_force_p=0.5，其他默认不变）。监督掩码覆盖 flow、planner aux、KL、连续性、节奏、帧语义/intent、FSQ；空文本样本不做 word/slot 监督。state simulation 和 TTS hidden conditioning 不支持，与任务的 base-stage/audio-feature 实验无关。

null 条件隐藏音频并进入 audio_null 分支，文本/说话人/参考风格为空；reverse 条件逐帧反转归一化音频特征，文本/说话人/参考风格同样为空。后者仅是任务规定的条件消融，不把倒放音频解释为正常语音。

## 运行与评测

远端在 taskset / nice 10 的 tmux 会话内执行 `bash scripts/bagger_rounds.sh GPU null|reverse`。默认 ROUNDS=3、COUNT=2000、STEPS=4000、SHARE=.33、batch=32、lr=1e-4、warmup=200、cfg `{"vox_prefsq_norm":true}`，0 个 loader worker，OMP/MKL 2 线程。一次只用一个 GPU，两个 lane 最多两个 GPU。支持上述环境变量调节，--dry-run 只预检和打印参数。bagger_env.sh 记录环境、因果 E5 路径；不安装依赖、不启动 HTTP 服务。

训练完成检查 DONE.json 步数，按本任务明确要求删除自己的 resume.pt；保留 final.pt/best.pt。用 final.pt 跑原始 rollout_probe.py 的默认和 ROLLOUT_TEXT_MODE=x2，20 dev、seed 0，输出 free/oracle 的 still/speed/copy。评测脚本保持未改，因而默认评测使用其原有 trim_start=False/cfg_null_fix=False；采集按任务要求启用这两项。历史 ft-ctl 的 .15/.71/.80 只作已知参考，不冒充本轮重新跑出的对照。

`timing.jsonl` 逐阶段记录命令、开始时间、耗时、退出码；RESULTS.json 汇总存在的结果及通过线。`scripts/test_bagger.py --orig BASE --out OUT` 用独立子进程对比原版 loss/grad/RNG，并验证混合样本掩码、空条件、source 扩展和采样分布。bagger_smoke.sh 使用实际检查点验证采集 → 混合训练 → 重载 → 下一轮反向采集。

## 与论文的差异

本任务是关节/语音条件模型，不是视频/自然语言提示词模型；论文通过改写文字描述倒放，本实验将文本置空，并比较音频置空/反转。窗口长度、初始模型、batch、学习率、每轮步数按任务而非论文设定。默认训练裁剪保留纠正尾段，不观察完整 24 块前缀。保留真值起始块原关节，最后一块仍含 VAE 重新编码误差。是否有效必须以完成的 rollout 结果判定，代码和数值测试通过不等于达到运动指标。
