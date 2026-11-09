# SCMG 论文材料

来源：代码仓库 main f3fa4e7（分支 paper/final-1109），整理于 2026-11-09。

## 目录
| 路径 | 内容 |
|---|---|
| code/ | main f3fa4e7 的代码，由 git archive 导出 |
| results/raw/ | 判定脚本读取的结果文件和显式语义 dev 全量结果文件，原样复制，保留原来的相对路径 |
| results/FILES.txt | results/raw 里每个文件的原始绝对路径 |
| results/verdicts/ | 三个判定脚本在 f3fa4e7 上的输出（txt 和 json） |
| docs/FINAL_RECIPE.md | 最终实现的组成和配置 |
| docs/THEORY_SAFETY_1109.md | 安全投影和打断安全提交的定理与证明 |
| docs/EVAL_AUDIT_1108.md | 评测审计记录和读数规则 |
| docs/NOVELTY_CHECK_1109.md | 相关工作检索记录 |
| docs/criteria/ | 各实验的预注册判据和结果记录 |
| docs/NEGATIVE_RESULTS_1107.md | 负结果表，取自 notes/negatives-1107 3c7d1f9 |

## 复现判定输出
判定脚本从原始路径读取结果文件。CPFS 要已挂载，~/bulk 要存在。

```bash
cd code
python scripts/eval/beat_g1.py --json /tmp/beat_g1.json
python scripts/eval/beat_ex002.py --json /tmp/beat_ex002.json
python scripts/eval/beat_ex002_ctl.py --json /tmp/beat_ex002_ctl.json
```

## 显式语义 dev 全量结果
- results/raw/control-token-1105/ex002/expl-full-1109/：EVAL_tts.json、EVAL_psync.json、BENCH_EVAL.json、EXEC.json、FGDOUT_tts_s0.json，由 code/scripts/ex002/ex002_expl_full.sh 生成。
- 子目录 a6/：NUMBER 修订 6 的结果，由 code/scripts/ex002/ex002_expl_a6.sh 生成。

## 未包含
- G1 与 czmws 的配对 bootstrap（BOOT_ABL.json）：还在跑。
- EX002 测试集结果：还没跑。
