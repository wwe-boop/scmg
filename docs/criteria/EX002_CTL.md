# EX002 controllability quick lanes: pass criteria (written 2026-11-07 before any EX002 run)

Platform: EX002 PVT1 + RM003 v4 three-finger hands (data-v36_1, 34 joints, 50 Hz; fixed base). Dev = the 380 val clips of
data-v36_1 (fixed v3.2 BEAT2 dev; speakers 2_scott / 13_lu, never in train). The manifest v3.2 test split is NOT used; it will be
registered in TEST_USAGE before a final EX002 method is scored on it.

## Official pointing protocol (2610.03105 v2, re-read 2026-11-07, quotes from §4.2)
- Ray: "At the apex â detected by Score_a, we cast a semi-infinite ray from the wrist along the elbow-to-wrist direction of the
  annotated pointing hand. This forearm direction provides a simple, reproducible proxy for arm-based pointing, without modelling
  finger articulation or gaze."
- Score_b: "Let d_miss be the ray's distance to the target's axis-aligned bounding box (zero on intersection) and d_centroid be the
  wrist-to-centroid distance. The extent-aware angular error θ_eff = arctan(d_miss / d_centroid) is mapped to
  Score_b = exp(−θ_eff² / 2σ²) with a fixed decay scale σ = 20°."
- Score_a: frozen logistic-regression apex detector (±3-frame positives, trained on recordings disjoint from the evaluation);
  e_t = |â − a_ref|, d_t = max(0, e_t − τ), Score_a = exp(−d_t² / 2σ_t²), τ = 10 frames (0.33 s), σ_t = 15 frames (0.5 s) at 30 fps.
- STG = Score_a × Score_b per event, averaged.
Port to EX002: scmg/ex002_ctl.py exact FK (base frame); wrist = arm_joint7 child body, elbow = arm_joint4 child body; seconds
instead of frames (τ 0.333 s, σ_t 0.5 s); target box side 0.20 m (as on G1). The INDEX-FINGER ray (index base x_2_1 -> tip x_2_3)
is OUR extension, reported next to the official forearm-ray numbers, never instead of them.

## Commanded-event evaluation (all 380 dev clips; does not need GT timing)
Per clip, events at random times (seeded, ≥ 1.5 s apart, ≥ 1 s from the clip ends): alternately an AIM command (hand uniform,
target = a train-label target drawn from the train target distribution of that hand) and a FINGER command (extended-finger counts
per hand drawn UNIFORMLY from {0..3}: the train distribution is 76 / 15 / 8 / 1 % for 0 / 1 / 2 / 3 on the left hand, so drawing
from it would make "do nothing" score 0.61; uniform counts make the command informative). Runs per model and inference seed
(0 / 1 / 2, per_clip): none (no tokens), given, shuffle (each command's target / counts replaced by another clip's, seeded).
Follow check: under shuffle, the score against the GIVEN command must exceed the score against the original command.

## Metrics (labels in the tables)
- Pointing (official protocol): Score_a / Score_b / STG with the forearm ray; a_ref = the command time; apex detector trained on
  the EX002 train-split BEAT2 deictic segments (apex = max wrist distance from rest, as on G1). Index-ray STG = ours.
- Finger count hit (OUR metric): per finger command, both hands' extended-finger counts at the command time ± 0.2 s (majority over
  the frames) == the commanded counts (per-hand hit and both-hands hit). Sanity on dev GT (run 2026-11-07): GT vs its own labels
  1.000; vs permuted labels 0.609 (left) / 0.678 (right) = chance Σp²; vs the same clip 1 s away 0.697 / 0.727 (drops).
- FGD v2 / FKD (OUR validated metrics; caption: GT self 0, same speaker other clip 1.59, other speaker 13.77, GestureLSM 172,
  EMAGE 279 on FGD v2); ΔBA-equivalent = scmg.evaluate beat_align ratio.
- Executability (public): MuJoCo scene.xml (ex002-rm003-v4-20261020/model-r1) joint limits, velocity limits, self-collision.
- pair_win on the 43-clip GT-event set (OUR metric, secondary row only; sanity GT = max / mismatched GT ≈ 0.5 before reporting).

## Gates (3-seed means, given vs none of the SAME model)
1. STG(given, forearm ray) − STG(none, same commands) ≥ +0.10, and the shuffle follow check holds;
2. finger count hit (both hands) ≥ 0.9, and the shuffle follow check holds;
3. FGD v2 ≤ control + δ, δ = max(0.03 x control, 2 x SD of the control's 3 seed FGDs) — the control (no-token runs of the warm
   start sibag-s2) is run first and δ is written into this file before any candidate is scored;
4. executability after CBF 100 % of clips;
5. default paths bitwise unchanged (unit tests).

## Exact evaluation protocol of lane 1 (for EVAL_PROTOCOL; recorded 2026-11-07 before the candidate numbers)
- Code: exp/text-path 9036faa (release 20261107-181129-ex002-ctl). Data: data-v36_1 (SPLIT sha256 a4a86440…5118), dev = val split,
  43 clips = 380 utterances (2_scott 220, 13_lu 160; 0 train items of these speakers). Test: not used.
- Commands: PLAN_dev.json (control-token-1105/ex002/PLAN_dev.json): 1091 events (559 aim, 532 finger); per clip seed
  crc32(clip) + 7; shuffle derangement seed 20261107. Train target pools / labels: control-token-1105/ex002/tokens
  (ex002_ctl_tokens.py, 39,540 train items, 89,514 aim + 89,538 finger labels; train count distribution left 0.548 / 0.287 /
  0.121 / 0.044, right 0.573 / 0.263 / 0.123 / 0.042).
- Inference: offline H = inf, 10 Euler steps, CFG 2.0, per_clip seed = seed + crc32(clip), seeds 0 / 1 / 2.
- Apex detector: tokens/APEX.json (ex002_apex.py; 1,060 train deictic events, 106,000 frames, 10,507 positives).
- Sanity (dev GT, 2026-11-07): pointing on the 38 dev GT deictic events with targets on the GT forearm ray at the GT apex:
  Score_a 0.999, Score_b 0.996, STG 0.995; the same events with another event's target: Score_b 0.427, STG 0.427. Finger count:
  see above (own 1.000 / permuted 0.609 · 0.678 / 1 s away 0.697 · 0.727). FGD v2 GT self 0 (1.4e-10); smoke 3 clips: sibag-s2 152.
- FGD v2: frozen encoders CKPT/zehan/gesture/fgd-ex002-v2/human_seed{0,1}/best.pt (code USERS/zehan/fgd-ex002-v2-20261102/code),
  mean of the two encoders, whole clips (50 Hz, 34 joints) vs GT; FKD = scmg.evaluate kin_feats / frechet.

## Module 2 (pre-registered 2026-11-07 19:20, before any lane-1 candidate number): joint-space hand-shape override
Motivation: finger count is a static hand state; the latent path's ceiling is the VAE round trip (dev GT both-hands frame
agreement 0.945, per hand 0.982 / 0.962). Module: ex002_ctl_run.py infer --finger-override HAND_SHAPES.json (default off): at every
finger command the 8 finger joints of each hand are blended to the canonical shape of the commanded count (median of 130,920 train
human frames with exactly that count; each median shape reproduces its count) over [t − 0.4, t + 0.4] s with 0.15 s cosine ramps;
arms / torso untouched; CBF-QP / executability afterwards. Same plan, seeds, gates (count hit ≥ 0.9; FGD v2 ≤ control + δ;
executability 100 % after CBF); boundary jerk of the finger joints at the ramps reported. Runs on the lane-1 model (or sibag-s2 for
the finger-only row) with modes given / shuffle.
