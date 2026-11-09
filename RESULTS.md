# RESULTS

- 代码：code/ 来自 paper/final-1109 e716721（git archive）。
- 评测协议：docs/EVAL_PROTOCOL.md v1（tag eval-protocol-v1，332c7ea）。

## 1. G1 消融：和 czmws 的配对 bootstrap
- 差值 = FGD(SCMG) − FGD(消融)。
- 349 条录音，2000 次重采样，随机种子 20261106。
- 文件：results/g1_bootstrap/BOOT_ABL_G1_1109.{json,md}。
- 脚本：zehan-server 上的 /workspace/zehan/boot1108/boot_resume.py。

| 消融 | psync 差值 [95% CI] | offline 差值 [95% CI] |
|---|---|---|
| ncws | −0.103 [−0.129, −0.078] | −0.111 [−0.144, −0.080] |
| 训练时去文本 nt | +0.003 [−0.029, +0.032] | +0.102 [+0.044, +0.159] |
| 训练时去音频 na | −0.151 [−0.265, −0.044] | −0.196 [−0.257, −0.137] |
| 推理时去文本 | −0.036 [−0.062, −0.010] | +0.060 [+0.015, +0.106] |
| 推理时去音频 | −0.147 [−0.261, −0.035] | −0.146 [−0.201, −0.092] |

SCMG 本身的 FGD：psync 1.946，offline 1.961。

## 2. EX002 测试集（manifest v3.2 test，说话人 23_hailing，282 段，17 段录音）
测试集使用登记：results/ex002_test/TEST_USAGE_entries.jsonl。

运行：
- DLC dlc1l5pzijpf5x9v。
- release：/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/20261109-101014-ex002-test-1109（a937df0）。
- 脚本：code/scripts/ex002/ex002_test_1109.sh。
- 输出目录：/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-test-1109/run。

安全层 3c-4b 和 BENCH_EVAL 在本机运行：
- 环境：解释器 ~/envs/part2-exec-20261029，PYTHONPATH 为 scmg-wt-paper:~/bulk/pydeps/safeproj。
- 开发集 ci 0 / psync s0 复跑结果和原文件逐位一致。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=<scmg-wt-paper>:~/bulk/pydeps/safeproj \
  python scripts/ex002_cbf_dev.py run --src <run>/enc --out <out>/proj4b --proto psync --seed <s> --layer proj_l2 --B 13 --recursive --max-iter 8
SCMG_EXPL_VIEW=<ex002-test-1109>/view python scripts/ex002/bench_ex002.py eval --out BENCH_EVAL.json <label>=<dir> ...
```

### 2.1 显式语义：success(given) − success(none)
- 文件：results/ex002_test/EVAL_{tts,psync}.json。
- 3 个种子合并。

| 类别 | n（TTS-sync） | TTS-sync：none → given | 差值 [95% CI] | psync：none → given | 差值 [95% CI] |
|---|---|---|---|---|---|
| SELF | 174 | 0.000 → 0.937 | 0.937 [0.901, 0.970] | 0.000 → 0.925 | 0.925 [0.889, 0.962] |
| YOU | 57 | 0.000 → 0.982 | 0.982 [0.952, 1.000] | 0.000 → 1.000 | 1.000 [1.000, 1.000] |
| POINT | 75 | 0.013 → 0.893 | 0.880 [0.784, 0.951] | 0.013 → 0.947 | 0.933 [0.871, 0.986] |
| NUMBER | 6 | 0.000 → 1.000 | 1.000 [1.000, 1.000] | 0.167 → 1.000 | 0.833 [0.667, 1.000] |
| BIG | 21 | 0.000 → 0.619 | 0.619 [0.444, 0.833] | 0.000 → 0.714 | 0.714 [0.556, 1.000] |
| SMALL | 18 | 0.000 → 0.889 | 0.889 [0.667, 1.000] | 0.000 → 0.833 | 0.833 [0.500, 1.000] |

### 2.2 质量和活跃度（BENCH_EVAL.json）
真人参考（测试集）：still 0.006，wrist_speed 0.343，wrist_amp 0.094，held_share 0.0。

| 行 | 种子 0 / 1 / 2 的 FGD v2 | 种子 0 / 1 / 2 的 FKD | 种子 0 / 1 / 2 的 still | 种子 0 / 1 / 2 的 wrist_speed |
|---|---|---|---|---|
| enc-s0（none） | 281.73 / 284.82 / 290.66 | 3.501 / 3.880 / 3.397 | 0.099 / 0.128 / 0.079 | 0.242 / 0.216 / 0.227 |
| enc-s0 + 3c-4b | 282.16 / 285.07 / 290.61 | 3.478 / 3.879 / 3.385 | 0.099 / 0.128 / 0.079 | 0.243 / 0.217 / 0.232 |
| 显式语义 TTS-sync | 278.00 / 282.91 / 287.24 | 3.229 / 3.573 / 3.153 | 0.077 / 0.104 / 0.068 | 0.290 / 0.266 / 0.275 |
| 显式语义 psync | 271.59 / 275.54 / 283.14 | 3.349 / 3.698 / 3.251 | 0.083 / 0.109 / 0.071 | 0.292 / 0.269 / 0.276 |
| 显式语义 TTS-sync + 修订 6（种子 0） | 277.64 | 3.229 | 0.077 | 0.290 |

参照行（results/ex002_test/sanity/）：
| 比较 | FGD v2 | FKD |
|---|---|---|
| 开发集真人（17 段录音）对测试集真人 | 430.22 | 3.353 |
| 开发集真人 17–33 对开发集真人 0–16 | 52.18 | 1.189 |

### 2.3 安全（check.log，enc-s0 + 3c-4b，psync，每个种子 54,992 帧）
| 种子 | wbc 违规帧 | 原始 wbc 违规帧 | 位置 / 速度违规帧 | fallback 帧 | 每块耗时 p50 / p95 / p99 / 最大（ms） |
|---|---|---|---|---|---|
| 0 | 0 | 2347 | 0 / 0 | 0 | 50.6 / 81.6 / 158.9 / 347.0 |
| 1 | 0 | 2582 | 0 / 0 | 0 | 51.1 / 79.6 / 140.3 / 366.1 |
| 2 | 0 | 2325 | 0 / 0 | 0 | 51.3 / 80.3 / 160.0 / 530.7 |

### 2.4 可执行性（EXEC.json，显式语义各行，经过部署安全层之后）
- exec_after_cbf：6 行全部为 1.0。
- raw_violation_any：0.33–0.35。

### 2.5 指令窗口外的 FGD v2（FGDOUT_tts_s0.json，种子 0）
| 窗口半径 | none | 显式语义 TTS-sync |
|---|---|---|
| R0 | 281.74 | 278.01 |
| R1（1.0 s） | 284.42 | 284.43 |

## 3. 开发集判定输出
results/verdicts/beat_{g1,ex002,ex002_ctl}.{txt,json}，在 f3fa4e7 上运行，0 MISMATCH。
