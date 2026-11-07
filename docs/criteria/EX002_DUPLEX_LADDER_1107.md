# EX002 in-stream duplex ladder -- written 2026-11-07 20:5x, before any duplex run (supersedes the G1 ladder of DUPLEX_1107.md
# for the platform; components and gates unchanged)

Base: the EX002 causal-match model of 3a (czm; 2000-step lane first, the ladder runs on the 6000-step model once it exists).
Input after t_i: the deployment-faithful pause (digital silence; G1 dev twin: the frozen R0causal passes every gate with it and
fails with the 'end' input). Benchmark: ex002_interrupt.py (476 dev interruptions, 162 utterances, 40 recordings), metrics
eval_interrupt_ex002.py vs the EX002 human tails (criteria EX002_INTERRUPT_1107.md). All generations on DLC A800.

Duplex fine-tune (one model for steps 1-2): from the base, cfg + ctl + ctl_flow + mask_cond, sc + yield_cf_p 0.3 (counterfactual
YIELD at a stream-grid time inside speech, silence after it, damped-retraction target, free bridge yield_rest_s 1.4) + p_mask 0.3;
same steps as the base lane. Tail positives (GT post-speech motion with a YIELD token) are NOT used in v1: the G1 tail list does not
map onto EX002 recording items (different packaging); disclosed; v2 if v1 fails on naturalness.
Ladder (each row adds one component; the same 476 interruptions):
  0 B3c (base model, silence)            1 + yield token (duplex model, --yield-token)
  2 + rest insertion: flagged known rest latents (mask_cond) from t_i + T on
  3 + end head T (planner head re-trained on the duplex model's own continuations of train-side interruptions)
  4 + funnel layer (per chunk, causal)    5 + CBF (EX002 limits / velocity / MuJoCo self-collision QP, per chunk)
Bridge-jerk module candidates for step 2 (semantic agent's G1 evidence: insertion yields 99.6 % but v3 31.6 %):
  (a) soft ramp of the insertion weight over 0.3-0.6 s before the known span (RePaint w < 1 rising to flagged w = 1)
  (b) mask-span training on rest-pose spans (bridge into rest in-distribution)
  (c) insertion + the per-chunk funnel residual low-pass        (d) the end head chooses T
Gates (per row; the full system must pass): yield >= 90 %, v3 <= 10 %, unsafe <= 1 %, RKD below the best post-hoc method on the
same base model (R0causal_p / funnel re-selected on EX002 dev with silence), paired recording bootstrap excluding 0; uninterrupted
quality (bench_ex002 psync FGD v2) not worse than the base by more than its seed range. Reaction latency (event -> first changed
frame, wall clock, idle GPU) reported separately. Failing rows: negatives row + diagnosis + next module (standing rule).
