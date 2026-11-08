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
Token-use check (coordinator 12:4x, before any L1 run; G1 negatives row 13: the floor token was ignored):
  same L1 checkpoint, same inputs (digital-silence pause after t_i), three runs on the 476 dev interruptions:
    ON   YIELD at t_i (--yield-token)      OFF  no token      LATE (wrong time)  YIELD at t_i + 1.0 s (--yield-delay 1.0)
  L1 uses the token iff (i) yield3d(ON) - yield3d(OFF) >= 20 pp with a recording-level paired bootstrap 95 % CI excluding 0, AND
  (ii) the wrong-time placement moves the yield: median 3D-yield latency (LATE) - median latency (ON) >= 0.5 s over the
  interruptions that yield in both runs. Otherwise L1 is logged "token ignored" (negatives) and L2 / L3 carry the result.
  Pause-alone check: yield3d(OFF) is reported next to the base enc-s0 B3 (same chain); if OFF alone reaches the yield gate, the
  pause input (not the token) explains the yield, stated as such.
Comparison rows on the half-B reference (13:3x; eval-enc-s0-halfB/EVAL_isc.json; ref-halfB = 375 tails, 151 post-speech windows,
RKD half-split floor 0.179; R0causal / Funnel re-run with the half-B rest pose; enc-s0, vae-r3, 3c-4b + ISC):
  sanity  GT yield3d 6.8 % | B2 100 % | FRONT 0 % (height 98 %) | B1 0 %
  B3 + ISC        yield3d 27.8 % v3 3.2 % RKD-post 0.440 unsafe 0.0 %
  R0causal + ISC  yield3d 97.9 % (0.59 s) v3 0.4 % RKD-post 0.518 unsafe 0.0 %   -> qualifies
  Funnel + ISC    yield3d 89.2 % (1.22 s) v3 4.4 % RKD-post 0.484 unsafe 0.0 %   -> fails yield (< 90 %)
On half B the rule selects R0causal + ISC (Funnel + ISC qualified on the full 757-tail reference with yield3d 91.1 %): the post-hoc
choice depends on the rest reference; the ladder table uses half B for every row as registered.
L2 diagnostic (not a gate; the rest ramp on the BASE enc-s0 continuation without a YIELD token, half-B reference):
  L2diag raw   yield3d 100 % (0.34 s) v3 14.3 % RKD-post 1.397 unsafe 81.7 %
  L2diag + ISC yield3d 100 % (0.36 s) v3 1.1 %  RKD-post 1.368 unsafe 0.0 %
The 9b duration rule (0.3-0.8 s) gives a retraction far faster than the human tails (half-B median yield 1.22 s): RKD-post 1.37 vs
R0causal + ISC 0.518; with ISC it passes yield / v3 / unsafe but loses on RKD.

## L2d = design D3 (pre-registered 11-08 20:5x from refine-logs/FABLE_DESIGN_1108.md D3, before any run; supersedes L2b / L2c)
Kinematically conditioned rest insertion (rest_ramp_ex002.py --version l2d; default = the L2 code path, bit-identical). Post-processing
on the continuation (enc-s0; L1 when available), executed prefix = U as in L2; head (q34 2-3) and fingers (18-33) keep the continuation.
 1 End pose (conditioned): E_A = the half-A human tails (tails/seg_halfA; half B stays the reference): per tail the 12-D end feature
   of yield_v4 (wrists + elbows relative to the hip, mean of the last 25 frames of [off, off + 107)) and the 34-joint mean pose of those
   frames. At an interruption f: the 12-D feature of the executed pose at f; the k = 5 nearest E_A entries; one drawn with the seed
   int(sha1(f"{ci}:{f}")[:8], 16). Waist (0-1) and arms (4-17) are set from its 34-joint pose.
 2 Duration (conditioned): half-A tail duration T_A = the start of the first quiet run (yield_v4 it2 quiet threshold v75 0.397 m/s,
   25 frames; a tail without one is excluded from the fit) regressed (isotonic, increasing) on the wrist travel d = mean over the two
   wrists of |end wrist - wrist at the speech offset| (m); T = fit(d_event) + a residual drawn from the half-A residuals (same seed),
   clipped to [0.5, 2.0] s. Fixed before any model row: if the half-A R^2 of the fit is < 0.1, T is drawn from the unconditional half-A
   T_A distribution instead (logged with R^2).
 3 Brake-then-ramp start: from the executed state at f (position exact, velocity from the 5-frame moving average as L2) every joint of
   waist + arms decelerates with a raised-cosine velocity profile v(t) = v0 (1 + cos(pi t / t_b)) / 2 (acceleration 0 at both ends),
   t_b = max_j pi |v0_j| / (2 a_j) clipped to [0.06, 0.5] s, a_j = the train a95 table (ARM_ACC_P95.json) for the arm joints and the
   EX002RM acceleration limit x 0.5 for the waist; then the quintic (zero end velocity / acceleration) from the braked state to the end
   pose over max(T - t_b, 0.3) s; hold the end pose to the window end.
 4 Deployed chain afterwards: 3c-4b + ISC (as every interruption row).
Controls (same seeds): (a) unconditioned end pose (uniform E_A draw) + the duration rule; (b) conditioned end pose + fixed T = 1.22 s;
(c) quintic C2 start (the L2 start: position / velocity / acceleration matched, no brake) with conditioned pose and duration;
(d) mismatched source: end pose = a 34-joint pose at a random time of a random train item (data-v36_1 train rows, same seed rule) --
the hindsight-token hold pool stores displacement windows, not poses, so random-time train poses are the pose analogue of it.
Quick lane (CPU, <= 5 min): the 60 interruptions of the first 5 dev recordings (ci 0-4), raw chain (no 3c-4b / ISC), half-B reference:
RKD-post, v3, yield3d, outward drift (per wrist: max over [f, f + 15] of the displacement along the unit wrist velocity at f - 1;
max of the two wrists; median / p95 / max), for l2d and (a)-(d). KILL: l2d RKD-post >= 0.518 on this subset, or l2d v3 > 10 %.
Full run (only after the coordinator's go): 476 dev interruptions, deployed chain, half-B reference: yield3d reported (not a sole gate),
v3 <= 10 %, unsafe <= 1 %, RKD-post below R0causal + ISC (0.518) with the recording-clustered paired bootstrap 95 % CI excluding 0;
withdrawal / activity components reported; end-pose 3-NN distance to the half-B end-pose set <= D90 on a share of events >= half B's
own leave-one-out share; RKD-post without its end-height descriptor (kin feature 2) reported beside the full RKD; uninterrupted quality
untouched (post-processing acts only after f); executability 100 %; reaction latency and outward drift (median / p95 / max) vs
R0causal + ISC; controls (a)-(d) in the same table. Raw rows report v3 under the executed-U rule (primary) and the own-prefix rule.
