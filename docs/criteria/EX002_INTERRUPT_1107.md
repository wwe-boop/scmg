# EX002 interruption benchmark (3b) -- written 2026-11-07 20:3x, before any EX002 interruption number

Human references (post-speech tails):
- Source: the G1 candidate list (interrupt-baseline-1105/human/gaps_train_tail/candidates.json: ECHO-G full-take utterance table,
  split train_drop, min_gap 1.2 s, post 3.0 s, VAD 20 dB / 0.1, pre 1.5 s). An EX002-own selection from data-v36_1 items finds
  only 40 (gap >= 2.6 s) / 243 (>= 1.2 s) candidates (dropped utterances make the gaps loud; 60 % of BEAT2 items are -wNN windows
  without take time), so the full-take table is used.
- Allow-list (no test manifest read): keep a tail iff its speaker is a BEAT2 speaker of the data-v36_1 TRAIN split and not an
  EX002 dev speaker (2_scott, 13_lu). 1372 G1 train_drop tails -> 1044 kept (433 recordings); dropped speakers 13_lu, 15_carlos,
  18_daiki, 23_hailing, 24_kexin, 28_tiffnay, 2_scott, 30_katya.
- Retargeting: frozen release06 retargeter (smplx2ex002_3f, BEAT source, near=False), whole recording -> q_full [T50, 37] -> 50 Hz
  frame window [t_off - 1.5, t_off + post] -> learn_index (34); byte-identical environment of dlc_entry.sh; rejections are kept out
  (no backfill). Verified: train items retargeted this way are bitwise equal to their data-v36_1 rows (VERIFY.json).
Metrics (port of scripts/interrupt/eval_interrupt.py to EX002, 50 Hz, window 2.13 s = 107 frames): hands = the two wrist bodies
(children of arm_joint7, exact FK scmg/kinematics.py); rest height / rest pose from the human silence frames (R_sil); yield (both
wrists at rest height + 5 cm and staying), react (first frame a wrist leaves the own uninterrupted output by > 2 cm), abruptness v3
(peak wrist jerk over [t_i, t_i + 1.8 s] / the same of the own uninterrupted output > 1.5), RKD (5 kinematic descriptors vs the
human retraction windows), unsafe (data-v36_1 META joint limits + limits34 velocity, MuJoCo self-collision in
ex002-rm003-v4-20261020/model-r1/scene.xml with structural pairs excluded from GT statistics).
Sanity checks before any method row: (1) RKD human half-split floor; (2) GT tails scored as a method: yield ~= the human rate,
RKD ~= the floor; (3) mismatched GT (another clip's tail spliced at t_i): react / v3 near chance; reported in the same table.
Addendum 11-07 23:0x (before any EX002 interruption metric; found while checking the first generation run dlc1cu0r2jgn1e3a):
frames before t_i are only bitwise equal to the uninterrupted psync output for 344 / 476 interruptions. All 132 differences start
at a frame = 12 mod 16, 1-3 frames before f_i: the VAE decoder looks one latent (4 frames) ahead, so when f_i falls after the
start of the last latent before the restart chunk boundary, re-decoding after the restart rewrites up to 3 already-played frames
(median 9e-4 rad at the first differing frame, up to 0.139 rad). [Correction 11-08 audit (prefix_check_ex002.py, all 476 interruptions of
every raw row: enc-s0 seeds 0 / 1 / 2, r4 base, step-1000 snapshot): <= 4 frames (4 when f_i falls on a latent start); 72.3 % of the
interruptions have a bitwise-equal prefix, 20.0 % differ only in the last 1-3 frames, 7.8 % first differ at f_i - 4, none earlier --
identical shares on every raw row (set by the latent phase of f_i). Max |y[:f] - U[:f]| p95 0.014 rad on enc-s0 (0.197 on r4 base).
Post-processed and deployed-chain rows (R0causal / Funnel post, every ISC row, B1 / B2 / FRONT + ISC) have a bitwise-equal prefix on
476 / 476. Raw rows are reported with v3 under this rule (primary) and with the continuation's own prefix y[:f] on both sides
(diagnostic): enc-s0 s0 31.3 / 17.4 %, s1 28.6 / 16.2 %, s2 32.1 / 18.1 %, r4 base 92.4 / 74.4 %, snapshot 55.5 / 41.0 %.] Rule: B3 rows are scored on the EXECUTED prefix -- the method's
own uninterrupted output before f_i, its continuation from f_i on (eval_interrupt_ex002.py --prefix executed, default; the file
as written = --prefix generated, diagnostic only). The G1 interruption runs share this decoder property (not re-scored here).
Addendum 11-08 01:1x (protocol fixes before any reported EX002 interruption number; found by a pipeline smoke run on the 458 tails
retargeted so far (of 1044), PARTIAL human refs, scratch only, not reported):
1. unsafe / collision: the raw scene.xml contact test is not a self-collision model for this robot (no contact exclusions: arm
   link5-link7, wheel-chassis, finger-hand-base touch in every frame; after excluding pairs present in >= 50 % of GT frames the dev
   GT itself was 76 % unsafe). Rule: collision = the retargeter's exact WBC-pair check (release06 smplx2ex002_3f.robot.EX002
   .clearance: runtime-rm003 349 rows (309 x 10 mm, 32 same-hand finger x 2 mm, 8 arm/hand-vs-head x 50 mm), sound GJK predicate,
   the check that certified the training data; names34 == its CONTROLLED order, asserted), every frame (stride 1). Smoke: dev GT
   1.3 % unsafe (velocity), mismatched GT 98.5 %. eval_interrupt_ex002.py --collision wbc (default); scene = diagnostic.
2. RKD reference: only 5 of 176 complete post-speech windows (partial set) satisfy the retraction rule (wrists above rest + 5 cm
   at the offset and back at rest in the window; G1: 79 / 460), so the retraction-only reference is degenerate (half-split floor
   2.56). Rule: RKD vs ALL human post-speech windows (--rkd-ref all, default; partial-set floor 0.124); the retraction subset is
   reported with its n only.
3. B2 settle time T_r: the EX002 median over the few retraction windows is not usable (0.88 s from 5 windows). Rule: T_r = 1.8 s,
   the G1 median yield time of the SAME BEAT2 post-speech tails (79 retraction windows, interrupt-baseline-1105 dev_final _meta)
   (--tr 1.8); the EX002 median is reported next to it.
Coordinator to confirm 1-3 before the final run (all tails retargeted).
Addendum 11-08 01:4x (coordinator confirmed 1-3 with conditions; before any reported number):
- collision columns: primary "retargeter WBC clearance (ours)" (release06 EX002.clearance, every frame); second column "MuJoCo
  scene, structural-contact-limited" = scene.xml contacts minus parent-child links (116 pairs) minus pairs in >= 50 % of dev GT
  frames (13 more); partial-set smoke: dev GT 1.3 % (WBC) vs 76 % (scene), i.e. the scene column is reported only as the
  structural-contact-limited diagnostic.
- RKD-post: vs all human post-speech windows; not comparable with G1's retraction-window RKD (stated in the paper). The
  retraction-window count is re-reported with all 1044 tails; RKD-retract is a secondary column only if it reaches >= 30.
- T_r = 1.8 s (G1, same tails), the EX002 human median next to it.
Addendum 11-08 03:xx (coordinator, after the qualitative review found B3 hands held in front at waist height scored as "yielded";
before any reported EX002 interruption number): PRIMARY yield = both wrists within 10 cm (3D, base-frame FK) of the human rest wrist
positions (R_sil of the tails) and staying so for >= 0.5 s (25 frames) inside the window; latency = first frame of that hold
(icommon_ex002.yield3d). The height-only yield is a secondary column. Sanity control FRONT: damped (T_r) to a fixed pose with both
wrists within 5 cm of rest height but far from the rest positions (dev GT frame with the largest smaller-wrist distance; partial
set: recording 37 frame 1200, 0.49 m).
Sanity (partial human ref, 458 / 1044 tails; scratch, not reported as results): B2 3D 100 % (height 100 %) PASS; FRONT 3D 0 % (height
97.6 %) FAILS as required; GT continuing to speak 3D 8 % (height 48 %) mostly fails. Pipeline smoke on the method rows: B3 of the
selected base 3D 39 % vs height 91 %, czm quick lane 41 % vs 86 % -- the height metric overstated retraction.
Addendum 11-08 09:1x (written before any enc-s0 interruption number; coordinator: choose R0causal vs Funnel by the registered rule):
this file registers no R0causal-vs-Funnel rule. The registered interruption gates of the duplex criteria (DUPLEX_1107.md: yield
>= 90 %, abruptness v3 <= 10 %, unsafe <= 1 %, then the lower RKD) are adopted for the EX002 post-hoc choice, scored on the DEPLOYED
chain (generator -> post method -> 3c-4b streaming safety projection, safety_int_ex002.py): primary yield = yield3d, v3, unsafe =
the retargeter WBC column after the safety layer. A method qualifies iff it passes all three; among qualifying methods the lower
RKD-post wins; if neither qualifies, both rows are reported and none is selected. Raw (pre-safety) rows are reported next to them.
Disclosure: the sibag-s2 rows (eval-final-sibag, raw chain) were seen before this addendum (R0causal yield3d 97.3 % / v3 6.7 % /
RKD 0.593 / raw unsafe 42.7 %; Funnel 93.9 % / 41.6 % / 0.616 / 36.3 %); no enc-s0 number has been read.
Addendum 11-08 10:0x (protocol fix of the deployed-chain rows, found in the first enc-s0 run, before the R0causal / Funnel choice):
the first run fed the safety layer z = U[:f_i] + continuation, so the layer's chunk holding f_i used continuation frames while the
eval splices the executed prefix (projected U) up to f_i: every post-safety velocity violation (B3 7, R0causal 17, Funnel 5 of
476) sits exactly at f_i (offset 0), and the layer's own stream stays <= 0.995 x vmax. The layer commits whole 16-frame chunks, so the
chunk holding f_i was projected from U before the interruption: rule = the continuation enters the safety layer at the first chunk
boundary >= f_i (safety_int_ex002.py). The raw rows are unchanged; the safe rows are re-run.
RESULT 2026-11-08 10:2x, base enc-s0 (eval-enc-s0; raw rows EVAL.json of the first run, safe rows EVAL_safe.json of the fixed
chain; 476 dev interruptions; human ref = the 757-tail set; continuation enters the safety layer at the next chunk boundary: median
added delay 7 frames = 140 ms, mean 7.3, max 15; yield / latency are computed on the executed stream, so they include it):
  sanity: GT unsafe 1.3 %; B2 yield3d 100 %; FRONT 0 % (height 94 %); B1 0 %; RKD human half-split floor 0.142
  raw    B3 yield3d 32.4 % v3 31.3 % RKD 0.525 unsafe 56.5 % | R0causal 98.1 % / 5.9 % / 0.542 / 43.3 % | Funnel 91.5 % / 40.8 % / 0.586 / 38.5 %
  safe   B3 31.6 % / 8.8 % / 0.510 / 0.0 % | R0causal 98.2 % (latency 0.60 s) / 10.3 % / 0.566 / 0.0 % | Funnel 91.1 % (1.19 s) / 12.0 % / 0.555 / 0.0 %
Rule (09:1x addendum): R0causal fails v3 (10.3 % > 10 %), Funnel fails v3 (12.0 %); neither qualifies -> no selection, both reported.
The safety layer removes every unsafe frame (0.0 % WBC / position / velocity on all three rows) and lowers B3 / Funnel abruptness,
but raises R0causal's v3 from 5.9 % to 10.3 % (its retraction is braked / re-timed by the projection).
## Module ISC (pre-registered 11-08 10:2x, before any run): interrupt-aware safety commit
At the interruption the safety layer re-plans the committed chunk from f_i instead of executing it to the boundary: state = the
executed frames up to f_i - 1 (history) and the still-unexecuted committed frames f_i .. boundary - 1 followed by the committed braking
tail (the recursive-feasibility candidate: already verified, consistent with the history, ending at rest); the frames f_i ..
boundary - 1 are projected again with the post method's continuation as targets (a partial chunk, same QP, same exact checks,
cut cap 8, qdldl, tail_smooth 1e-3, predictive rows / carry off = 3c-4b), then normal chunks resume on the 16-frame grid. The
feasibility argument of 3c-4 holds (the candidate admits every relaxed row), so no fallback is introduced. Rows: B3 / R0causal /
Funnel safe with ISC; the same rule (yield3d >= 90 %, v3 <= 10 %, unsafe <= 1 %, then RKD-post); disclosure: the numbers above.
ISC RESULT 2026-11-08 11:4x (eval-enc-s0/EVAL_isc.json; 1428 streams, 0 fallback; the executed prefix equals the projected U):
  B3 + ISC       yield3d 31.6 % v3 3.2 % RKD-post 0.475 unsafe 0.0 %
  R0causal + ISC yield3d 98.2 % (latency 0.59 s) v3 0.4 % RKD-post 0.551 unsafe 0.0 %
  Funnel + ISC   yield3d 91.1 % (latency 1.19 s) v3 4.4 % RKD-post 0.519 unsafe 0.0 %
Rule (09:1x addendum): both R0causal and Funnel qualify (yield3d >= 90 %, v3 <= 10 %, unsafe <= 1 %); lower RKD-post -> Funnel (0.519
vs 0.551). Selected post-hoc method on the deployed chain with ISC: Funnel. (Base enc-s0 on the vae-r3 decodes; not frozen until
the r3d decoder decision.)
Check (11-08 11:5x): the safe and the ISC rows use the same v3 pairing: the counterfactual reference is the same no-interruption
stream, enc-s0 psync s0 through the same 3c-4b layer (runs/ex002-proj4b-enc/psync/s0, passed as B3:<dir>:<ref> in both evals);
only the interrupted streams differ, so 10.3 % -> 0.4 % (R0causal) is not a reference change.
Addendum 11-08 14:xx (coordinator; written before any model row is scored under it): yield metric v4 replaces yield3d as the primary
yield (yield3d kept as a secondary column). Reason: only 20 / 297 (6.7 %) A-half human tails that need a yield reach R_sil within the
tail, so yield3d scores human behaviour near the bottom. Definition, calibration (A half only) and validity gates (B-half tails
>= 85 %, speech continuations GT / FRONT / B1 <= 15 %): scripts/ex002/yield_v4_ex002.py (d0d67b0) and refine-logs/EVAL_PROTOCOL.md
section 6. Iterations of the definition (A half only) are listed here as they happen. Iteration 1 = the d0d67b0 definition.
  v4 iteration 1 audit (A-calibrated v75 0.169 / v95 0.584 m/s, D90 0.131 m): B-half tails 58.3 % (activity 68.9 %, withdrawal 80.1 %)
  -> gate FAIL; GT in speech 0.4 %, FRONT 0.0 %, B1 36.1 % -> B1 gate FAIL (frozen poses at t_i often already lie in the human end-pose
  support). yield3d on the same tail windows: A 17.5 % (4.6 % of the needed), B 16.6 % (10.1 % of the needed).
  v4 iteration 2 (A half only, written before its audit): (i) an event is NEEDED iff the pose at the event (frames f-1..f+1) lies
  outside the A-half end-pose support (3-NN distance > D90); v4 is reported on the needed events (primary) and on all; (ii) the quiet /
  stroke speed thresholds are A-half percentiles chosen as the smallest pair on the grid {75..95} x {95..100} whose activity pass on the
  A-half needed events is >= 90 %. Gates unchanged, applied to the needed events.
  v4 iteration 2 audit (A-picked quiet / stroke percentiles 90 / 100 -> 0.397 / 1.608 m/s; D90 0.131 m): needed events only (primary):
  A 62.1 % (29 needed), B 38.1 % (42 needed), GT in speech 7.6 %, FRONT 0 %, B1 0 %; all events: A 86.4 %, B 75.5 %, GT in speech 8.8 %,
  FRONT 0 %, B1 36.1 %. -> the human gate (>= 85 %) fails on B under either reading: v4 is NOT validated. Per the coordinator's cap no
  v3 unless calibrated on A and confirmed on a fresh set: a fresh confirmation set is being built (dev-speaker 2_scott / 13_lu post-
  speech tails with gap >= 2.6 s from the G1 candidate list, never among the 757; 101 candidates, 28 recordings; tails_fresh).
  Until a definition passes on B and on the fresh set, yield3d and the v4 components (activity, withdrawal) are reported numbers only.

## Metric M1: paired stroke-suppression yield (PSS) -- pre-registered 11-08 21:0x from refine-logs/FABLE_DESIGN_1108.md M1, before any
## model row is scored under it (validity gates first; no single yield gate: PSS is reported with withdrawal and RKD, never alone)
s(t) = the larger wrist speed (exact FK, as yield_v4 feats, m/s). A stroke = an onset where s rises through v_s from below and stays
above v_s for >= 5 frames (0.1 s); rho(window) = strokes per second. v_s = the median over the half-A human tails of the peak of s in
their pre-offset speech window; rho_min = the p10 of the half-A pre-offset stroke rates (both fixed numbers from half A only).
Windows (50 Hz): post = [f + 15, f + 107) (0.3 .. 2.13 s); reference window = for a MODEL the same frames of its own uninterrupted
continuation U (the v3 reference); for a HUMAN tail the preceding speech [max(0, off - 92), off) (92 frames when available; a tail with
fewer than 50 pre-offset frames is excluded and counted). PSS = 1 - rho(post) / max(rho(ref), rho_min); an event passes iff PSS >= 0.5.
Validity gates (half A calibrates; nothing else is looked at before they are written):
 (1) human tails pass on >= 85 % of the half-B tails AND >= 85 % of the fresh dev-speaker tails (tails_fresh; reported per speaker);
 (2) dev GT continuing to speak (window inside speech at each of the 476 interruption points: post = [f + 15, f + 107) of GT, ref =
     [f - 92, f) of GT) passes on <= 15 %;
 (3) mismatched reference (the half-B tails scored against another half-B tail's pre-offset window, fixed derangement by sort order)
     moves the half-B pass rate by <= 10 pp.
 Iteration (at most 2, logged): if (1) or (2) fails, only v_s (the half-A percentile of the per-tail peak, grid 30 / 40 / 50 / 60 / 70)
 or the post-window skip (0.2 / 0.3 / 0.5 s) changes; the pick uses half A only: among the grid points whose half-A tail pass rate is
 >= 0.90, the highest v_s percentile, then the longest skip; then (1)-(3) re-run. If the gates still fail after 2 iterations, PSS is
 recorded as an invalid metric (negatives) and not used.
Known limits (recorded): humans are paired with their own preceding speech, models with their own counterfactual continuation -- not
the same pairing; B3 (generator continues) has PSS ~ 0 by construction; B1 / FRONT pass PSS by construction and are rejected only by
withdrawal / RKD.
M1 amendment 11-08 21:2x (found in the half-A calibration, before any validity number): with v_s = the half-A median per-tail peak
(0.335 m/s), 56.5 % of the 154 half-A pre-offset speech windows (median 1.5 s) contain no stroke, so their p10 rate is 0 and the
registered rho_min = p10 = 0 leaves PSS undefined when the reference window has no stroke. rho_min is set instead to one stroke per
post window (1 / 1.84 s = 0.543 / s), i.e. a reference without strokes counts as one stroke: an event then passes iff its post window
has no stroke. Everything else (v_s rule, windows, gates 1-3, iteration rule) unchanged.
