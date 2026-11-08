# EX002 self-forcing scope ablation + causal match on encoder-only self-forcing (pre-registered 2026-11-08, before any run)

## Background (bisect 1108, dlc6860az51fcdvg / dlcrlj66jp1f0xr6)
The lively EX002 base sibag-s2 (and sibag-s0 / s1) was trained by release 2a81db6. The only training-path change between it and the
current code is dcd50fb (11-04 17:52), the fix of an autocast weight-cache bug: the no-grad self-forcing generation pass cast the
planner / Local Flow Linear weights to bf16 inside the outer autocast with the cache on, and the gradient pass reused those copies
(no grad_fn). So on a self-forced batch (self_force_p 0.5) the planner and the Local Flow got no gradient, but the gradient still
flowed through their activations to the condition encoders. Measured (sibag recipe, seed 0, 2000 steps, offline s0, 13-point
moving-upper-body guard, GT still 0.0166 / wrist 0.411 m/s / amplitude 0.120 m):
  5d8f154 (pre-fix)     still 0.025  wrist 0.315  amp 0.094  PASS   train flow 0.6411  gn 0.662  dev_flow 0.6699 / 0.6697
  dcd50fb (fix, = HEAD) still 0.062  wrist 0.231  amp 0.078  FAIL   train flow 0.5718  gn 0.865  dev_flow 0.6758 / 0.6825
  current, sf0          still 0.145  wrist 0.176  amp 0.065  FAIL   train flow 0.4314  gn 0.686  dev_flow 0.6706 / 0.6710
So the old behaviour is NOT "self-forcing off"; it is a distinct scheme: self-forced batches train only the condition encoders.

## Method switch (exp/duplex, default off)
StageCfg.self_force_scope: 'all' (default, = current code) | 'encoders' (every nn.Linear executed by the no-grad generation pass
gets grad None on that batch; train.py, single process, --accum 1). scripts/tests/test_selfforce_scope.py ALL_OK (GPU, synthetic):
default bit-identical to the pre-change code (self_force_p 1 and 0); 'encoders' == the pre-fix code bit-for-bit (same no-grad
parameter set 325 / 327 with intent_aux, loss and every other gradient identical); non-self-forced batches unchanged.

## Gate 0 (reproduction, before any comparison is read)
enc-s0 (current code + scope encoders, sibag recipe, seed 0) must reproduce the 5d8f154 training curve to 4 decimals: mean flow and
gn over steps >= 500 (0.6411 / 0.662) and dev_flow at 1000 / 2000 (0.6699 / 0.6697). If not: stop, report, no tuning.

## Ablation (supplementary table; dev, 43 recordings)
sibag recipe (r4 base, cfg adsc + intent_aux, sc p_stream 0.67 / p_async 1 / p_async_seg 0 / w_int 0.3 / sem_w 2.0, self_force_steps
4, rounds 1, cfg / mix / noise 0, BAgger 0.33, semsynth 0.6, batch 64, 2000 steps, lr 1e-4), training seeds 0 and 1:
  sf0 (self_force_p 0) | all (self_force_p 0.5, scope all) | encoders (self_force_p 0.5, scope encoders)
seed-0 models: sf0 = bisect cur-sf0, all = lively-1108 sib-s0 (identical training to dcd50fb / HEAD), encoders = enc-s0.
bench_ex002 offline + psync, per_clip seeds 0-2, cfg 2.0; reported: FGD v2, FKD, the 13-point guard (still, wrist speed, wrist
amplitude, held clips, stutter), means over inference seeds, per training seed and the 2-seed mean. No gate: a reported finding.

## Causal match on encoder-only self-forcing (quick lane, then selection)
czm-enc-s0 = enc-s0 + causal_front / causal_spk / causal_text (pre-train-slim2m, --causal-spk-atol 1e-3 --causal-no-spk-check).
Matched control = enc-s0. Selection (frozen rule, as EX002_SWEEP_1107.md): czm-enc-s0 becomes the EX002 base iff vs enc-s0
(a) psync FGD v2 seed range entirely below, (b) offline FGD v2 mean <= 1.05 x, (c) psync FKD mean <= 1.10 x, (d) skipped steps
<= 15 %, (e) the guards (1)-(5) in offline AND psync (means over seeds 0-2). If czm-enc-s0 fails but enc-s0 passes (e), enc-s0
(current code, no causal match) is the base: the clean-code successor of sibag-s2. If neither passes, sibag-s2 stays interim.

## Addition 2026-11-08 06:5x (coordinator check; written while enc-s0 trains, before any Gate 0 number is read)
World size of the original runs: sibag-s0 / s1 / s2 (mainline-seeds-1104b, dlc14zvll7y68jfe) were single-process runs, one GPU per
lane (5 lanes on 3 GPUs, batch 64 per process, no torchrun; log.txt has no rank / world lines), and the single-GPU retrain at
2a81db6 (bisect) reproduced sibag-s0 to 4 decimals (flow 0.6411 / gn 0.662 / dev_flow 0.6699 / 0.6697 vs 0.6699 / 0.6696-0.6697).
So the single-process 'encoders' emulation is the same recipe as the original; DDP support is not needed for this base and
'encoders' refuses world > 1 (the None-grad / unused-parameter semantics under DDP were never exercised by the bug runs).

## RESULT 2026-11-08 08:0x (dlc1w474ln7rftyd SFSCOPE_DONE; ~/bulk/causal-train-1105/ex002_sfscope_table.py; all-s0 guard rows from
## lively_floor_ex002.py per seed, the lane predates the always-on guard)
Release check: REF_TRAIN_CHECK OK on 20261108-0643-ex002-sfscope before any lane. Gate 0 PASS: enc-s0 flow 0.6411 / gn 0.6619 /
dev_flow 0.6699 / 0.6696 vs 5d8f154 0.6411 / 0.6619 / 0.6699 / 0.6697 (original sibag-s0 0.6411 / 0.6620 / 0.6699 / 0.6697); enc-s1
0.6361 / 0.6656 / 0.6702 / 0.6698 vs original sibag-s1 0.6361 / 0.6657 / 0.6702 / 0.6699. No skipped steps in any lane.
| scope | seed | protocol | FGD v2 (seeds 0-2) | FKD | still | wrist m/s | amp m | held | stutter | guards |
|---|---|---|---|---|---|---|---|---|---|---|
| sf0 | 0 | offline | 181.0 (164.3-197.8) | 3.01 | 0.148 | 0.177 | 0.067 | 0.023 | 1.032 | FAIL still,wrist,amp |
| sf0 | 0 | psync | 222.7 (197.2-242.5) | 3.54 | 0.181 | 0.151 | 0.060 | 0.031 | 1.023 | FAIL still,wrist,amp |
| sf0 | 1 | offline | 162.9 (145.7-181.7) | 2.76 | 0.133 | 0.192 | 0.070 | 0.000 | 1.032 | FAIL still,wrist,amp |
| sf0 | 1 | psync | 201.8 (192.7-211.0) | 3.27 | 0.175 | 0.161 | 0.068 | 0.023 | 1.022 | FAIL still,wrist,amp |
| all | 0 | offline | 125.1 (117.7-138.6) | 2.03 | 0.064 | 0.234 | 0.080 | 0.008 | 1.063 | FAIL still,wrist,amp |
| all | 0 | psync | 155.4 (128.3-171.0) | 2.28 | 0.097 | 0.204 | 0.079 | 0.016 | 1.067 | FAIL still,wrist,amp |
| all | 1 | offline | 114.4 (101.3-127.3) | 1.73 | 0.038 | 0.263 | 0.083 | 0.000 | 1.066 | FAIL still,wrist,amp |
| all | 1 | psync | 131.4 (111.9-142.4) | 1.78 | 0.047 | 0.238 | 0.081 | 0.008 | 1.063 | FAIL still,wrist,amp |
| encoders | 0 | offline | 86.3 (84.1-87.7) | 1.51 | 0.024 | 0.316 | 0.095 | 0.000 | 1.056 | PASS |
| encoders | 0 | psync | 109.0 (107.0-112.0) | 1.54 | 0.029 | 0.323 | 0.095 | 0.008 | 1.053 | PASS |
| encoders | 1 | offline | 86.7 (81.5-89.5) | 1.49 | 0.024 | 0.353 | 0.096 | 0.008 | 1.067 | PASS |
| encoders | 1 | psync | 102.6 (98.2-106.0) | 1.49 | 0.023 | 0.376 | 0.098 | 0.008 | 1.061 | PASS |
| czm+encoders | 0 | offline | 85.3 (82.6-87.4) | 1.64 | 0.050 | 0.277 | 0.092 | 0.000 | 1.064 | FAIL still,wrist |
| czm+encoders | 0 | psync | 126.3 (115.0-135.1) | 1.87 | 0.049 | 0.267 | 0.086 | 0.008 | 1.062 | FAIL still,wrist,amp |
2-seed mean sf0       offline FGD v2 172.0  still 0.141  wrist 0.185
2-seed mean sf0       psync   FGD v2 212.3  still 0.178  wrist 0.156
2-seed mean all       offline FGD v2 119.7  still 0.051  wrist 0.248
2-seed mean all       psync   FGD v2 143.4  still 0.072  wrist 0.221
2-seed mean encoders  offline FGD v2 86.5  still 0.024  wrist 0.334
2-seed mean encoders  psync   FGD v2 105.8  still 0.026  wrist 0.350
SELECTION czm-enc-s0 vs enc-s0: (a) psync FGD range below False (115.0-135.1 vs 107.0-112.0) (b) offline <= 1.05x True (c) psync FKD <= 1.10x False (e) guards False; enc-s0 guards True
BASE -> enc-s0 ((d) skipped steps: see train.jsonl)
Reading: encoder-only self-forcing is the best of the three on every column (2-seed FGD v2 86.5 / 105.8 vs full self-forcing
119.7 / 143.4 vs sf0 172.0 / 212.3, offline / psync) and the only regime that passes the liveliness guards; sf0 is the worst.
For reference, the original sibag-s0 / s1 / s2 (bench-only, lively-1108 diag): FGD v2 offline 85.5 / 86.4 / 91.2 (mean 87.7, SD 3.1),
psync 107.8 / 102.5 / 96.2 (mean 102.2, SD 5.8) -- enc-s0 / enc-s1 match their seed twins (86.3 / 109.0, 86.7 / 102.6).
Selection (frozen rule): czm-enc-s0 fails (a) (psync 115.0-135.1 vs 107.0-112.0), (c) (FKD 1.87 vs 1.54) and (e) (still / wrist,
and amplitude in psync); enc-s0 passes (e) in both protocols -> BASE = enc-s0 (current code, self_force_scope encoders, no causal
match), the clean-code successor of sibag-s2. The causal training match does not help on this recipe (psync worse).
