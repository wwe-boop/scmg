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

## REVISION 2026-11-08 12:0x (coordinator; written before any run of this ladder): event-machinery ladder on enc-s0
Base: enc-s0 (sibag recipe + self_force_scope encoders, EX002_SFSCOPE_1108.md). Decoder = a parameter (BENCH_VAE; vae-r3 now, r3d if
it passes). Interruption benchmark, input after t_i (digital silence), metrics and human refs as EX002_INTERRUPT_1107.md.
Rungs (one model for L1-L3, quick lane = 2000 steps from enc-s0, seed 0, same recipe + scope encoders):
  L1  yield token only: a YIELD control token through the existing ctl route (cfg ctl / ctl_flow), trained on post-speech tails of
      the TRAIN split (GT motion after the speech offset, YIELD at the offset) mixed into the base data; inference --yield-token.
  L2  L1 + rest-keyframe insertion: target = the speaker's rest pose (human_ref R_sil, 34 joints), reached with B's 9b adaptive
      minimum-jerk ramp (exp/ctl-port, reused, not forked), held with no release back to the generator until the next turn.
  L3  L2 + the deployed safety chain with ISC (3c-4b + interrupt-aware commit).
Comparison rows on the same base and chain: Funnel + ISC, R0causal + ISC (EX002_INTERRUPT_1107.md ISC RESULT), B3 + ISC.
Gate (the interruption rule): yield3d >= 90 %, v3 <= 10 %, unsafe (after the safety layer) <= 1 %, then the lower RKD-post; reported
for L1 / L2 on the raw and the ISC chain, L3 = the full chain; uninterrupted psync FGD v2 of the duplex model within the enc seed
range (95.1-109.0).
Leakage rule (the tails are also the human reference): the 757 retargeted tails (433 recordings) are split by recording
(crc32(recording) % 2): half A is the training positive set, half B the RKD / rest reference; EVERY row of this ladder and of the
comparison rows is re-scored against half B only (the earlier full-set numbers stay as they are, labelled).
L1 addendum (12:3x, before any L1 run): the duplex model = enc-s0 fine-tuned 2000 steps (seed 0, lr 1e-4, warmup 200, the same sc
incl. self_force_scope encoders, BAgger 0.33, semsynth 0.6) on data-v36_1-tail with cfg + ctl + ctl_flow (zero-init routes) and the
YIELD-only token set (tokens-duplex), ctl_focus 0.2 (share of windows placed around a token; the tails are ~0.3 % of the items);
L1 inference = ex002_interrupt.py --yield-token (YIELD at t_i, no look-ahead). Its own uninterrupted psync s0 through 3c-4b is the
v3 / executed-prefix reference of its rows. L2 = rest_ramp_ex002.py on the L1 continuation (waist + arms; head / fingers keep the
continuation); L3 = L2 through 3c-4b + ISC. Comparison rows (Funnel / R0causal / B3 on enc-s0, + ISC) are re-run with the half-B
human reference (rest pose and RKD), so every row of the table uses half B.
