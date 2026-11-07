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
