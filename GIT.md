# SCMG 代码版本管理（2026-11-03 起）

仓库：`~/projects/gesture/scmg`（本机，主仓库）。历史：15 个 DLC 训练包（tag `pkg-r1` … `pkg-r7h`）、评测快照（`rel-*`）、
Codex 侧分支包（`codex-loss-r8`、`codex-loss-r8-seed0`、`codex-vox-tslm`）按时间导入；未导入的旧 releases/ 快照仍在 CPFS 原处。

## 规则
1. 改代码前看 `git status`；改完先自测，再 `git commit`（信息写方法名和验证结果，如 "multi-round self-forcing; test_semaux ALL_OK"）。
2. **DLC 训练包只用 `scripts/make_package.sh <名字> [phase]` 生成**：要求工作区干净，`git archive HEAD` 导出，包内写 `GIT_COMMIT`，
   自动生成哈希校验的 `-submit/command.sh` 并打 tag `pkg-<名字>`。台账记录作业 ID + tag + commit。
3. **评测 / 自有卡验证用的代码快照只用 `scripts/make_release.sh [label]`**（tag `rel-<时间>[-label]`）。不再手工 cp 主仓库到 CPFS。
4. 一个实验一条分支（`exp/<方法名>`），验收通过再合回 main；DLC 包可从分支出，tag 记录确切版本。
5. Codex 在 `codex/<任务>` 分支工作，不直接改 main；Claude 审阅后合并。
6. 不进 git：检查点、npy/npz、日志（见 .gitignore）。产物路径写进提交信息或台账。
7. 查某个作业用的代码：`git show pkg-r7h:scmg/losses.py`；对比两个版本：`git diff pkg-r7g pkg-r7h -- scmg/`。
