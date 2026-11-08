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

## δ fixed from the control (2026-11-07, before any candidate number)
Control = sibag-s2, mode none, seeds 0 / 1 / 2 on the 43 dev clips (control-token-1105/ex002/lane1/ctrl):
FGD v2 91.13 / 87.15 / 95.32 (mean 91.20, SD 4.08), FKD 1.357 / 1.412 / 1.384; against the plan's commands (no tokens given):
forearm STG 0.254 / 0.253 / 0.254, both-hands finger hit 0.056 / 0.051 / 0.060.
δ = max(0.03 × 91.20, 2 × 4.08) = 8.17 → gate 3: FGD v2 ≤ 99.37 (mean of 3 seeds).

## Lane 1 result (one line) and lane 2 (pre-registered 2026-11-07 19:40, before any lane-2 number)
Lane 1 (ex002ctl, dlcbbo4y3r2de98k): STG given 0.574 vs none 0.301 (+0.273, passes; shuffle follows 0.582 vs 0.368), finger both-hands
hit 0.602 (fails 0.9; shuffle follows 0.617 vs 0.065), FGD v2 given 230.0 / none 270.8 (fails, gate 99.37), FKD 2.73 vs 1.38.
Diagnosis to test: the no-token run already degrades (270.8 vs 91.2), so the fine-tune RECIPE (self-forcing off, no BAgger,
ctl_w 4, ctl_focus 0.5, w_ctl_aux 3) rather than the tokens moves the model off its distribution.
Lane 2 (same plan / seeds / gates / δ):
- rcp  : the sibag-s2 recipe (self_force_p 0.5 / steps 4 / rounds 1, BAgger share 0.33, semsynth 0.6), NO control modules, 2000
         steps — tells whether 2000 more steps of the original recipe keep FGD v2 near the control;
- ctl2 : the sibag-s2 recipe + control modules with light weights (ctl_w 1, w_ctl_aux 1, w_ctl_finger 1, ctl_focus 0.3,
         counterfactual targets / counts 0.5, ctl lr x10), 2000 steps;
- fo   : module 2 (finger override, pre-registered above) on ctl2 and on lane-1 ex002ctl, modes given / shuffle.
Code change for lane 2: scmg/data.py collate pads the control arrays of windows without them (BAgger samples) — default path
(no control tokens) unchanged.
Host rule (from agent A's bench_ex002 cross-check, exp/duplex 71f2e04): the same code gives bitwise-equal generations on one host
type but differs by 2e-4–3e-4 rad between DLC A800 and the local RTX 5090 → every EX002 control / semantics number is generated on
DLC A800 (all lane runs so far are); bench_ex002.py reproduces ex002_ctl_run evaluate to full precision on the same outputs.

## Lane 2 result (one line) and lane 3 (pre-registered 2026-11-07 21:30, before any lane-3 number)
Lane 2 (dlcqbkgzvypnn0hy): rcp (sibag recipe, no control) FGD v2 124.3 / FKD 1.76 — even 2000 more steps of the ORIGINAL recipe
leave the δ band (99.37); ctl2 (light weights) STG 0.419 vs none 0.295 (+0.124, passes, shuffle follows), finger hit 0.497, FGD v2
163 (none 166); finger override (module 2): hit 1.000 on ctl2 and lane 1 (by construction; shuffle: 1.000 vs 0.058 against the
original), FGD v2 +12 (175) on ctl2. Diagnosis: any 2000-step fine-tune of the converged sibag-s2 at lr 1e-4 moves FGD v2 by
+33 or more, before tokens matter.
Lane 3 (module 3: frozen base): ctl3 = sibag-s2 + control modules where the BASE learns at lr 1e-6 and the control modules at 1e-4
(--lr 1e-6, ctl_lr_scale 100), 2000 steps, data recipe of lane 1 (no SF, no BAgger), ctl_w 1, w_ctl_aux 3, w_ctl_finger 1,
ctl_focus 0.5, counterfactual targets / counts 0.5. With no tokens the control routes contribute exactly 0, so mode none ≈ the
control; the gates measure whether the token routes alone reach the gates. Runs: none / given / shuffle and finger-override
given / shuffle x seeds 0-2. Same plan, gates, δ (FGD v2 ≤ 99.37).

## Recipe-drift check (coordinator request, 2026-11-07 21:40)
(a) step-0 identity: sibag-s2 through the fine-tune path (scmg.train, --lr 0, 1 step, the rcp arguments) saves an EMA state that is
    bitwise identical to sibag-s2's (0 of all tensors differ; base / extra cfg equal) -> no loading bug.
(b) EMA: sibag-s2/final.pt is an EMA snapshot (raw vs EMA |Δ|/|θ| = 0.027 at its own step 2000); the continuation restarts the EMA
    from these EMA weights with the same decay 0.999 (raw vs EMA 0.027 at step 2000, like the original). After 2000 steps the EMA has
    moved |Δ|/|θ| = 0.069 from sibag-s2 (raw 0.094) — a real drift, not an EMA artefact. (Intermediate snapshots were not saved.)
(c) recipe parity: rcp vs sibag-s2's stored StageCfg differ only by fields added after 11-04, all at their inactive defaults
    (ogm_ge is gated by ogm_alpha 0; ctl / ret / phys / safe weights 0); base / extra cfg identical; same view, BAgger 0.33, semsynth
    0.6, SF 0.5, warmup 200, lr 1e-4, batch 64; data order differs (training seed 0 vs sibag-s2's 2). Open question tested by
    dlc1mjl4p7zdospj: FGD v2 of the sibling training seeds sibag-s0 / s1 (same recipe) — if they sit near 124, sibag-s2's 91 is a
    favourable training seed and the continuation regresses to the recipe's mean.
Result of dlc1mjl4p7zdospj: sibling training seeds of the same recipe give FGD v2 86.92 (sibag-s0) / 89.41 (sibag-s1), inference
seed 0, vs sibag-s2 91.13 -> the control is NOT a favourable outlier; the +33 of the 2000-step continuation at lr 1e-4 is a real
drift of a converged model under a restarted (warm-up + cosine from 1e-4) schedule. The frozen-base lane 3 is the remedy under test.

## Lane 3 result (one line) and lane 4 (pre-registered 2026-11-07 ~23:00, before any lane-4 number; committed 25734ab 23:04)
Lane 3 (dlc1ygzggbtksmlc): ctl3 none FGD v2 131.0 (the "frozen" base at lr 1e-6 still moved: AdamW normalises the step, every base
weight moves ~lr per step whatever its gradient), given 209.2; STG 0.417 vs none 0.286 (+0.131, passes; shuffle follows 0.415 vs
0.332); finger hit 0.326, override 1.000 at FGD v2 214. Frozen-base evidence: the token routes alone add +78 FGD v2 (given 209 vs
none 131) under the dense plan. Lanes 1-3 keep their verdicts (all fail gate 3).
Rationale for the new quality gates (coordinator 2026-11-07 ~22:55): an obeyed command is by design not a speech-driven gesture, so
the harm to bound is the harm to the motion OUTSIDE commands and at a realistic command rate; the dense plan (one command per
1.5-2.0 s) is a stress test of obedience, and the lane-3 +78 shows a dense whole-clip FGD gate is infeasible for any obedient model.
Lane 4 (module 4: true freeze): ctl4 = sibag-s2 + control modules, `scmg.train --train-only ctl_` (every parameter whose name does not
start with ctl_ has requires_grad False; smoke: trainable 1.10M / 117.79M, 0 base tensors changed after 5 steps at lr 1e-3), control
modules at lr 1e-4 (--lr 1e-4, ctl_lr_scale 1), 2000 steps, otherwise the lane-3 recipe. Expected and checked: mode none reproduces
the control (sibag-s2 none) up to the A800 host noise, because the base is bitwise sibag-s2 and the control routes add exactly 0
without tokens.
Runs (seeds 0 / 1 / 2, DLC A800): dense plan PLAN_dev.json none / given / shuffle and finger-override given / shuffle; sparse plan
PLAN_dev_sparse.json (`ex002_ctl_run.py plan --gap 6.0`, 301 events, one per ~6.4 s; gap 1.5 reproduces PLAN_dev.json bitwise)
given / shuffle and finger-override given / shuffle.
Outside-command FGD v2 (ours): scripts/ex002/ex002_fgd_outside.py — frames within ±R s of every command of the plan are masked; a
32-frame window is kept only if it holds no masked frame, for the generated clip and its GT alike; R = 0 reproduces evaluate's
FGD_v2 (checked on the control: 91.13 / 87.15 / 95.32 exactly). Control values (sibag-s2 none, lane1/ctrl, scored 2026-11-07 ~22:58
before any lane-4 run), δ = max(0.03 x mean, 2 x SD) as before:
| plan / R | windows (of 6,031) | control seeds 0/1/2 | mean | δ | gate |
|---|---|---|---|---|---|
| dense / 1.0 s | 83 (8 clips have none) | 165.82 / 165.68 / 163.52 | 165.01 | 4.95 | 169.96 |
| dense / 0.5 s | 537 | 99.24 / 89.05 / 94.71 | 94.33 | 10.21 | 104.54 |
| sparse / 1.0 s | 3,563 | 91.92 / 87.22 / 95.95 | 91.70 | 8.74 | 100.44 |
| sparse / 0 (whole clip) | 6,031 | 91.13 / 87.15 / 95.32 | 91.20 | 8.17 | 99.37 |
Quality gates (both must pass, 3-seed means; for the configuration claimed — with and without the finger override, both reported):
(i) SPARSE plan, FGD v2 on windows more than 1 s from any command ≤ 100.44. Decision (coordinator 2026-11-07 ~23:02, made on the
    control numbers above only, no candidate output existed): the coordinator's first text put (i) on the dense plan, but the dense
    ±1 s set keeps only clip heads / tails (83 of 6,031 windows, 8 of 43 clips have none) and the control itself scores 165.0 there
    vs 91.2 whole-clip (small-sample bias of a 32-d covariance from 83 windows), so it is not measurable; the sparse ±1 s set keeps
    3,563 windows (69 % of frames).
(ii) sparse plan, whole-clip FGD v2 ≤ 99.37.
Report-only: dense whole-clip FGD v2; dense outside ±0.5 s (reference 104.54); dense outside ±1 s (reference 169.96, flagged
UNRELIABLE: 83 windows, 8 clips with none); FKD. If the finger gate needs the override, gates (i) / (ii) are also measured on the
sparse override runs; override and plain are reported separately. Sanity of the new metric: ex002_fgd_outside.py at R = 0 equals
evaluate's FGD_v2 to full precision (control, above); the job re-scores the control on DLC as a cross-check.
These gates are committed (this commit) before any lane-4 output is scored.
Obedience gates unchanged (dense plan): STG(given) − STG(none) ≥ +0.10 with the shuffle follow check; finger hit ≥ 0.9 (override
allowed, reported both ways) with the shuffle follow check; boundary jerk of the override ramps reported; executability after CBF
100 % of clips. The sparse-plan STG / finger hit are reported too.
Executability / ramp jerk tool: scripts/ex002/ex002_exec.py — CBFFilter (scmg/safety.py limits34, margin 0.03 rad) per clip; a clip
is executable if after the CBF no frame violates position / velocity / acceleration (jerk is best-effort in the CBF, reported as a
frame rate); ramp jerk = |3rd difference| x 50^3 of the 16 finger joints on the frames 0.4 < |t - t_cmd| <= 0.55 s of every finger
command (the override's ramps), the same frames in every run. Smoke on old lane outputs (seed 0, dense plan): control / ctl2 given /
ctl2 + override all 100 % executable after CBF; ramp frames over the 1500 rad/s^3 finger jerk limit 4.8 % / 9.4 % / 67 % — the
0.15 s override ramps are too steep; reported, not gated (pre-registered as report-only above).
Plans: PLAN_dev.json sha256 7515bde6…3860, PLAN_dev_sparse.json sha256 76863c5c…aa3f (both in control-token-1105/ex002/).
Script: scripts/ex002/ex002_ctl_lane4.sh (also checks ctl4 none vs control generations, max |diff|).

## Lane 4 result (dlc6eiyvft9aw7i8, LANE_DONE 2026-11-08 00:11; 3-seed means)
- True freeze verified: ctl4 none generations == control (sibag-s2 none) bitwise on all 43 clips x 3 seeds (max |diff| 0).
- Quality gates FAIL: (i) sparse, FGD v2 outside +-1 s 113.45 (gate 100.44; control 91.69); (ii) sparse whole clip 113.35 plain /
  114.73 override (gate 99.37). Report-only: dense whole clip 199.9 / 198.1; dense outside +-0.5 s 153.0 (ref 104.5); outside +-1 s 218.0
  (unreliable). FKD dense given 4.07 vs none 1.38.
- Obedience: STG dense given 0.400 vs none 0.254 (+0.147, passes; shuffle follows 0.398 vs 0.325 against the original); sparse 0.358 vs
  0.261. Finger hit 0.288 plain (fails) / 1.000 with the override (passes; shuffle follows 1.000 vs 0.058). Executability after CBF
  100 % of clips (all runs). Override ramp frames over the finger jerk limit 0.55-0.58 (control 0.04-0.05; report-only).
Diagnosis: with the base bitwise frozen, the token routes alone move the motion FAR from the commands (+21.8 FGD v2 on windows more
than 1 s from every command, sparse plan) -> the control routes act globally (planner / chunk attention sees every token), not locally.
## Module 5 (pre-registered 2026-11-08 00:20, before any module-5 number): command-local blend at inference (default off)
y = (1 - w) * y_none + w * y_given per joint, w(t) = max over the plan's commands of a window that is 1 for |t - t_cmd| <= 0.6 s and
falls to 0 by a cosine ramp at 0.9 s (all inside the +-1 s mask); y_none / y_given = the same model and per-clip seed without / with
tokens (ctl4 lane-4 outputs, so no new generation); finger override (module 2) applied after the blend for the "fo" rows. Outside
+-0.9 s the motion is exactly the no-token generation (= the control, bitwise), so gate (i) holds BY CONSTRUCTION and is not counted as
evidence; the test is gate (ii) (sparse whole clip <= 99.37) with the obedience gates unchanged (STG +0.10 with shuffle follow, finger
>= 0.9 override allowed, executability after CBF 100 %). Evaluation: the unchanged ex002_ctl_run evaluate + ex002_fgd_outside.py +
ex002_exec.py on the blended arrays (post-processing of the DLC generations; scored locally, flagged; pai_train login expired).
## Module 5 result (2026-11-08 ~00:35; blended lane-4 DLC generations, scored locally on the RTX 5090)
- Gate (ii) sparse whole clip PASSES: 94.81 plain (94.61 / 89.89 / 99.95) and 96.68 with the override (gate 99.37); gate (i) equals the
  control exactly (by construction, not evidence). Sparse FKD 1.44 vs none 1.38. Executability after CBF 100 %.
- Obedience FAILS on pointing: dense STG 0.301 vs none 0.254 (+0.047 < +0.10; lane 4 unblended +0.147); sparse 0.314 vs 0.261.
  Finger hit with the override 1.000 (shuffle follows 1.000 vs 0.058); plain 0.288 / 0.350.
Diagnosis: the obeyed pointing gesture (approach + apex + return) is not contained in +-0.6 s of the command time; cutting it to the
blend window removes most of the STG gain. Dense whole clip 183 (report-only).
## Module 5b (pre-registered 2026-11-08 ~00:45, before any 5b number): asymmetric command-local window
Diagnostic used to set it (not a gate metric): mean arm-joint distance |y_given - y_none| around the sparse aim commands (seed 0)
peaks at +0.0..+0.4 s (1.06-1.11 rad vs ~0.5 baseline) and stays elevated to about +1.0 s (pointing hold), rising from -0.4 s.
Window: w = 1 on [t - 0.3, t + 0.9] s, cosine ramps of 0.3 s -> 0 outside [t - 0.6, t + 1.2] s (aim and finger alike). The ramp past
+1.0 s now reaches windows of gate (i), so gate (i) is real evidence again. Same gates, same scoring scripts (ex002_local_blend.py
gains --pre / --post; the symmetric module-5 call is unchanged).
## Module 5b result (2026-11-08 ~01:00; blended lane-4 DLC generations, scored locally on the RTX 5090 -> re-score on DLC once the
## pai_train login is back; host differences of scoring are ~1e-4, generation was on A800)
- Gate (i) sparse outside +-1 s: 91.72 (91.94 / 87.22 / 96.00; gate 100.44) PASS (real evidence: the +1.2 s ramp end enters it).
- Gate (ii) sparse whole clip: 94.52 plain (94.28 / 89.46 / 99.83), 96.32 with the override (gate 99.37) PASS.
- STG dense: 0.357 vs none 0.254 = +0.103 (gate +0.10) PASS, thin margin; per seed 0.353 / 0.358 / 0.360; shuffle follows (0.353 vs
  0.308 against the original). Sparse 0.338 vs 0.261.
- Finger hit: override 1.000 PASS (shuffle follows 1.000 vs 0.058); plain 0.288 (fails) -> the passing configuration needs module 2.
- Executability after CBF 100 % PASS. Override ramp frames over the finger jerk limit 0.54-0.58 (report-only; the 0.15 s ramps).
- Report-only: dense whole clip 181.5 / 183.9; FKD sparse 1.44 (none 1.38), dense 2.84.
Verdict: ctl4 (true freeze) + module 5b (asymmetric command-local blend) + module 2 (finger override) passes every pre-registered
lane-4 gate on dev. Caveats: STG margin 0.003; the 5b window was set from a dev diagnostic (|y_given - y_none| profile, disclosed above);
offline only (needs commands >= 0.6 s ahead and both a none and a given generation, 2x compute); override ramp jerk to fix (longer
ramps) before any robot use; dev only, test untouched.
## Module 5b confirmation on fresh plans (coordinator 2026-11-08 ~01:15; frozen before the job starts)
Reason: the 5b window was chosen from a dev diagnostic on the same lane-4 generations and plans its gates are scored on, after a failed
module 5 on the same data; it passed by +0.003. 5b counts only if it passes on FRESH plans and NEW generations.
- Fresh plans: `ex002_ctl_run.py plan --plan-seed 1108 --shuffle-seed 20261108` (defaults 7 / 20261107 reproduce PLAN_dev.json /
  PLAN_dev_sparse.json bitwise, checked): PLAN_dev_fresh_dense.json (--gap 1.5; 1096 events, 559 aim; sha256 43d1bb20…7bb0) and
  PLAN_dev_fresh_sparse.json (--gap 6.0; 301 events, 161 aim; sha256 5b638eb4…46d7).
- New generations on DLC A800: the lane-4 checkpoint (ctl4/train/final.pt), inference seeds 3 / 4 / 5, modes none / given / shuffle on
  both fresh plans (15 runs).
- 5b exactly as frozen: w = 1 on [t - 0.3, t + 0.9] s, 0.3 s cosine ramps (ex002_local_blend.py --pre 0.3 --post 0.9, inner 0.6 / outer
  0.9 give the 0.3 s ramp width), finger override (HAND_SHAPES.json, hold 0.4 s, 0.15 s ramps) after the blend for the "fo" rows.
- Gates on the fresh plans (seeds 3-5): STG(given) - STG(none) >= +0.10 with the shuffle follow check; finger hit >= 0.9 (override
  allowed); executability after CBF 100 %; quality: (i) sparse FGD v2 outside +-1 s and (ii) sparse whole clip <= C + δ, where C and
  δ = max(0.03 C, 2 SD) come from the no-token runs (ctl4 none = sibag-s2 bitwise) of seeds 3-5 on the same fresh-plan masks (the
  same formula as before; the old absolute thresholds 100.44 / 99.37 are reported too).
- Reported alongside: the old-plan A800 re-score of 5b (selection evidence), the unblended lane-4 rows on the fresh plans, and the
  recording-clustered bootstrap CI of the STG gain (scripts/ex002/ex002_stg_boot.py, 43 recordings resampled, all seeds together,
  10,000 draws) on both the old and the fresh dense plans.
## Module 6 (pre-registered 2026-11-08 ~01:15; runs only if 5b fails on the fresh plans): temporally gated control routes
Lane 4 shows the control routes act globally (+21.8 FGD v2 more than 1 s from every command with a bitwise-frozen base). Module 6
makes them local by construction, in training and inference alike: each control token's contribution (its attention value in ctl_text
and its FiLM / additive term in ctl_flow) is multiplied by a per-frame window w_k(t) = 1 on [t_k - 0.3, t_k + 0.9] s with 0.3 s cosine
ramps and 0 elsewhere, so frames outside every window receive exactly zero from the control routes (the no-token motion there up to
the planner's chunk context, which is checked and reported). Default off, bitwise unchanged when off (unit test); trained like lane 4
(true freeze, 2000 steps); same gates on fresh plans.
## Module 5b confirmation result (dlci2zmkrgtpnpye, LANE_DONE 2026-11-08 01:21; all generated and scored on DLC A800)
Old plans, A800 re-score (selection evidence): identical to the local 5090 scores at the reported precision (STG 0.357 vs 0.254,
gate (i) 91.72, gate (ii) 94.52 / 96.32, finger override 1.000, executability 100 %) -> no appendix row needed. Bootstrap (43
recordings, 10,000 draws): forearm STG gain +0.103 [+0.084, +0.122] (P(gain < 0.10) = 0.37; per seed +0.099 / +0.105 / +0.106);
index ray +0.083 [+0.067, +0.100]; unblended lane 4 +0.147 [+0.124, +0.168].
FRESH plans (seed 1108) and NEW generations (seeds 3-5), 5b as frozen:
- STG dense (forearm, official): 0.355 vs none 0.241 = +0.114 (per seed +0.108 / +0.108 / +0.127), bootstrap 95 % CI [+0.094, +0.135],
  P(gain < 0.10) = 0.08 -> PASS; shuffle follows (0.353 vs 0.296 against the original). Index ray (ours) 0.342 vs 0.244 = +0.098
  [+0.082, +0.114]. Sparse 0.315 vs 0.242. Unblended lane 4 on the fresh plans +0.153 [+0.132, +0.175].
- Finger hit: override 1.000 (shuffle follows 1.000 vs 0.082) PASS; plain 0.310.
- Gate (i) sparse outside +-1 s: 84.75 (82.23 / 88.83 / 83.18) <= C + δ = 84.73 + 7.13 = 91.86 PASS (old absolute 100.44 also met);
  unblended 101.62.
- Gate (ii) sparse whole clip: 93.29 plain (91.27 / 96.73 / 91.88), 94.91 with the override (92.68 / 98.45 / 93.60) <= C + δ =
  89.31 + 5.70 = 95.01 PASS (override margin 0.10; old absolute 99.37 met); unblended 109.83.
- Executability after CBF 100 % PASS (all runs). Report-only: override ramp frames over the finger jerk limit 0.61-0.66 (none 0.04,
  5b plain 0.03-0.10); dense whole clip 177.4 / 179.8 (none 89.3); FKD sparse/dense see JSON.
VERDICT: module 5b (+ module 2 for fingers) PASSES on fresh plans and new generations. Margins: STG CI lower bound 0.094 (< 0.10:
the gate is met by the mean, not by the CI); gate (ii) with the override 0.10 below its threshold. Module 6 is not triggered.
## Port to the causal-match base (pre-registered 2026-11-08 ~01:40, before the port job)
Base: czm-s12000-lr1e-4-w0.1 (agent A's selection, ex002-sweep-1107/SELECTION.txt; train/final.pt). Code: branch exp/ctl-port = exp/text-path
+ exp/duplex (clean merge, test_ex002_ctl ALL_OK), release 20261108-013330-ex002-ctl-port (8feb5ee). Python = the image's
/usr/local/bin/python with PYTHONPATH = release (as the sweep job), so the no-token runs can be compared bitwise with A's bench outputs.
- Training: the czm recipe (cfg adsc; sc causal_front / causal_spk / causal_text, w_phys 0.1, phys_dyn_w 0, sf0; --causal-dir
  pre-train-slim2m, semsynth 0.4) + control modules (cfg ctl / ctl_flow; lane-4 control weights), TRUE freeze --train-only ctl_, lr
  1e-4, 2000 steps, seed 0.
- New fresh plans (never used before): plan seed 1109 / shuffle seed 20261109: PLAN_dev_port_dense.json (1080 events, 551 aim; sha256
  8807a8db…3dd3) and PLAN_dev_port_sparse.json (297 events, 160 aim; sha256 d9d86fa9…bff9); inference seeds 0 / 1 / 2.
- Protocols: offline (ex002_ctl_run infer = bench offline) AND psync (bench_ex002 infer --protocol psync with the plan's tokens,
  commands visible 1.0 s before their time: --ctl-lookahead 1.0; smoke on speech_server GPU1: the tokens change the psync output).
  The gates apply to BOTH protocols, each with its own C and δ = max(0.03 C, 2 SD) from that protocol's no-token runs (seeds 0-2) on
  the same fresh masks. No-token identity check for both: ctl-port none vs A's czm offline / psync s0-s2 outputs (max |diff|).
- Configuration: 5b as frozen (w = 1 on [t - 0.3, t + 0.9] s, 0.3 s cosine ramps) + finger override.
- Finger-override comparison row (coordinator 01:35, rule fixed now): (a) frozen cosine 0.15 s ramps; (b) minimum-jerk quintic profile
  with 0.3 s ramps (--fo-ramp 0.3 --fo-profile minjerk). Report finger hit, ramp over-limit share (frames hold < |t - t_cmd| <= hold +
  ramp, 1500 rad/s^3) and gate (ii) for both. Rule: if (b) keeps finger hit >= 0.9 AND gate (ii) (per protocol), (b) becomes the default
  override of the final configuration.
- Gates (per protocol): STG(given) - STG(none) >= +0.10 with the shuffle follow (bootstrap CI reported); finger hit >= 0.9 (override);
  (i) sparse FGD v2 outside +-1 s <= C + δ; (ii) sparse whole clip <= C + δ; executability after CBF 100 %.
## Qualitative observations (qual-1108/ex002, coordinator review 2026-11-08 ~02:15; the 5b metric verdict stands)
- Pointing elevation: the obeyed gesture gets the azimuth roughly right but the forearm stays near waist height; for targets above
  the shoulder the ray undershoots (aim1, STG 0.78: ~20-30 deg low by eye); horizontal targets (shuffle case of aim1) look right. STG
  credits approximate directions (mismatched GT already scores 0.427).
- Counting is not communicative: the commanded count is achieved in the finger joints, but the hands hang at hip height (finger0,
  L1 R2); no viewer reads it as "showing 2". The finger metric counts extension states, not a presented count.
## Aiming error report (ours; scripts/ex002/ex002_aim_error.py; fresh dense plan, seeds 3-5, 1677 aim events, 71 % of targets above
## the shoulder; angle between the ray at the apex frame and wrist -> target; elevation signed, negative = below the target)
| run | forearm total median / p90 | azimuth median | elevation median (target above / below shoulder) | index total median | wrist - shoulder z |
|---|---|---|---|---|---|
| none | 50.3 / 100.5 | 36.7 | -14.0 / +32.7 | 45.8 | -0.23 m |
| lane 4 unblended given | 28.3 / 66.4 | 21.6 | -2.4 / +21.8 | 28.6 | -0.22 m |
| 5b given | 35.0 / 80.2 | 25.0 | -5.6 / +24.0 | 33.3 | -0.23 m |
| 5b shuffle (vs shuffled command) | 35.6 / 76.9 | 25.8 | -6.4 / +23.0 | 33.0 | -0.23 m |
Reading: the median wrist stays 0.23 m below the shoulder in every run (the arm is not raised); the directional error is dominated
by azimuth (25 deg) plus a sign-dependent elevation bias (targets below the shoulder are pointed ~24 deg too high); median total error
35 deg vs the 20 deg sigma of Score_b. These are dev numbers on sibag-s2 lane 4; the czm port gets the same report.
## Module 7 (pre-registered 2026-11-08 ~02:25, runs on the czm port base after the port job): aim-direction module
Training-side change (default off, bitwise when off): the ray loss (ctl_ray_loss) is split into azimuth and elevation components
with separate weights (w_ray_az, w_ray_el; el weight 3x), plus a shoulder-raise term when the commanded target is above the shoulder
(hinge on wrist z < shoulder z - 0.05 m inside the command window). Same true freeze / steps / recipe as the port.
Gates (fresh plans, seeds 0-2, offline and psync, each against its own no-token C / δ): PRIMARY median forearm angular error <= 25 deg
(port value reported for reference) AND |median elevation error| for above-shoulder targets <= 5 deg; STG +0.10 with shuffle follow;
quality gates (i) / (ii) with 5b; executability after CBF 100 %.
## Module 8 (pre-registered 2026-11-08 ~02:25, czm port base): present-count
At a finger-count command the arm goes to a presentation pose: an AIM token for the commanded hand toward a frontal presentation target
(the median wrist->forward direction of train-GT frames whose hand is raised (wrist z >= shoulder z - 0.20 m) with >= 1 extended
finger, K-half speakers), issued at the count command's time, plus the finger override (minimum-jerk 0.3 s if the port rule selects it,
else cosine 0.15 s); blended with 5b. Gates: count hit >= 0.9; hand height >= chest (wrist z >= shoulder z - 0.20 m) on >= 80 % of
command frames (+-0.4 s); palm-normal angle to the forward direction (+x, toward the listener; normal from the cross product of
index-base and third-finger-base vectors from the wrist) median <= 45 deg; quality gates (i) / (ii); ramp jerk reported.
Module 8 target, fixed before any module-8 generation (tokens/PRESENT.json, ex002_present.py target, 3000 train items, 163k / 186k
raised-hand frames): the presentation target follows the POINT convention, so it is built from the median FOREARM direction
(elbow -> wrist) of the raised-hand frames (the "wrist -> forward" wording above, made exact): left (0.76, -0.22, 0.62), right
(0.77, 0.26, 0.58) = forward, toward the midline, 35-38 deg up; target = rest shoulder + 2.25 m x direction. (The shoulder -> wrist
direction, computed first, points 23 deg DOWN because the raised set allows the wrist 0.20 m below the shoulder; a target from it
would lower the hand, so it was not used.)
Module 7 code: scmg/losses.py StageCfg w_ctl_ray_el / w_ctl_raise (default 0 = bitwise unchanged path), run with w_ctl_aux 3,
w_ctl_ray_el 6 (elevation 3x in total), w_ctl_raise 5.
## Port result (dlcnd7d3f2iwdfi1, LANE_DONE ~02:50 2026-11-08; DLC A800; fresh plans 1109, seeds 0-2)
- Identity: ctl-port no-token == czm bitwise, offline and psync, 43 clips x 3 seeds (max |diff| 0).
- STG gain (forearm, official) FAILS on both protocols: offline 0.355 vs none 0.285 = +0.070, bootstrap [+0.055, +0.085] (per seed
  +0.064 / +0.067 / +0.081); psync 0.367 vs 0.293 = +0.074 [+0.060, +0.088]. Shuffle follows (offline 0.359 vs 0.310; psync 0.369 vs
  0.320). Unblended lane-4-style given: offline +0.098, psync +0.096 (below +0.10 too). The czm no-token baseline is higher (0.285 /
  0.293 vs sibag 0.241-0.254), which eats the margin. Index ray +0.048 / +0.061.
- Finger hit with the override 1.000 on both protocols (cosine 0.15 s and minimum-jerk 0.3 s alike); plain 0.253 / 0.225.
- Gate (i): offline 70.92 <= 70.91 + 18.50 = 89.42 PASS; psync 92.08 <= 92.03 + 14.21 = 106.24 PASS.
- Gate (ii): offline 74.36 plain / 74.86 cos / 75.23 minjerk <= 67.83 + 16.06 = 83.89 PASS; psync 101.68 / 103.30 / 103.95 <= 91.01 + 16.09
  = 107.10 PASS. (czm's own seed spread is wide: offline 74.99 / 69.35 / 59.15.)
- Executability after CBF 100 % (all runs). Ramp over-limit share: cosine 0.15 s 0.55-0.60, minimum-jerk 0.3 s 0.010-0.019 (no-token
  0.016-0.029).
- Override rule (pre-registered): minimum-jerk keeps finger hit >= 0.9 and gate (ii) on both protocols -> it becomes the DEFAULT override.
VERDICT: the port passes quality, finger, executability and identity, and FAILS the pointing gate (+0.07 < +0.10, CI excludes 0.10) on
both protocols. Next: module 7 (aim direction, pre-registered) on this base, gated on the median angular error and STG.
## czm vs sibag-s2 stillness (coordinator request 2026-11-08 ~03:00; static_share.py, dev 43 clips, per_clip seeds 0-2)
| run | still-frame share | mean wrist speed (m/s) | wrist amplitude (m, per-axis std) | elbow flare |
|---|---|---|---|---|
| dev GT | 0.019 | 0.411 | 0.120 | 0.261 |
| sibag-s2 offline s0 / s1 / s2 | 0.031 / 0.027 / 0.027 | 0.328 / 0.338 / 0.341 | 0.097 / 0.101 / 0.099 | 0.268 / 0.268 / 0.261 |
| czm offline s0 / s1 / s2 | 0.140 / 0.140 / 0.148 | 0.209 / 0.207 / 0.206 | 0.084 / 0.085 / 0.083 | 0.258 / 0.253 / 0.255 |
| czm psync s0 / s1 / s2 | 0.200 / 0.195 / 0.181 | 0.166 / 0.173 / 0.173 | 0.074 / 0.076 / 0.074 | 0.234 / 0.241 / 0.236 |
czm is clearly more static than sibag-s2 (still share x5-7, wrist speed -38 % offline / -50 % psync, amplitude -15 % / -25 %) although its FGD v2
is lower (offline 67.8 vs 91.2): FGD v2 does not penalise under-motion enough on dev; the judge's "static / held" impression is real.
## Modules 7 / 8 result (dlc1gin7d7ehjneq, LANE_DONE 04:10 2026-11-08; czm base, fresh plans 1109, seeds 0-2, DLC A800)
Module 7 (ray elevation 3x + shoulder-raise loss, true freeze):
- PRIMARY gates FAIL: median forearm angular error offline 28.5 deg / psync 29.1 deg (gate <= 25; port 31.0 / 31.1, no-token 40.7 / 38.8);
  median elevation error for above-shoulder targets offline +6.9 / psync +7.8 deg (gate |.| <= 5; port +2.9 / +0.1) -> the arm now
  overshoots upward; median wrist - shoulder z -0.10 m (port -0.19, no-token -0.21).
- STG passes with a wide margin: offline 0.472 vs 0.285 = +0.188 [+0.165, +0.210], psync 0.462 vs 0.293 = +0.169 [+0.151, +0.187];
  shuffle follows (0.451 vs 0.338; 0.455 vs 0.344). Azimuth median 23.8 / 22.4 deg (port 24.0 / 22.8) -> azimuth unchanged.
- Quality gates pass: (i) offline 70.92 <= 89.42, psync 92.06 <= 106.24; (ii) offline 73.38 / 73.35 (minjerk) <= 83.89, psync
  87.35 / 88.64 <= 107.10. Finger override 1.000; executability after CBF 100 %. Report-only dense whole clip 290 / 325.
- Verdict: FAILS its pre-registered primary gate (angular error) although it raises STG from +0.07 to +0.19; the remaining error is
  azimuth (~23 deg) and a now-positive elevation bias.
Module 8 (present-count: frontal presentation aims + minimum-jerk override on the port model):
- Count hit 1.000 (both protocols) PASS.
- Hand height >= chest (wrist >= shoulder - 0.20 m): mean share of command frames offline 0.63 / psync 0.66 (port 0.26 / 0.22); events-
  hands with >= 80 % of frames raised 0.41 / 0.44 -> FAIL (gate >= 80 %).
- Palm-normal angle to forward (sign-agnostic) median 77.8 / 75.4 deg (port 79.0 / 78.0) -> FAIL (gate <= 45): the aim raises the
  forearm but does not turn the palm.
- Quality: (i) passes (70.96 / 92.13); (ii) FAILS offline 93.59 > 83.89, psync 114.14 > 107.10 (two simultaneous POINT tokens per
  count command move the whole clip). Ramp over-limit 1.7-2.5 % (minjerk).
- Verdict: FAIL. A presentation pose needs a palm / wrist-orientation target (or a library pose), not only a forearm direction.
## Modules 7b / 8b (pre-registered 2026-11-08 ~04:20; queued for the base agent A re-selects with liveliness guards; NOT started)
Common: true freeze (--train-only ctl_) on the re-selected base, its own recipe, 2000 steps; new fresh plans (seed 1110 / shuffle
20261110), inference seeds 0-2, offline AND psync, each protocol against its own no-token C / δ; 5b blend; minimum-jerk override.
Liveliness guard added to the quality gates (dev, per protocol, scripts/video/static_share.py on the whole clips): still-frame share
<= 2 x dev GT (0.019 -> <= 0.038), mean wrist speed >= 0.75 x GT (0.411 -> >= 0.308 m/s), wrist amplitude >= 0.75 x GT (0.120 ->
>= 0.090 m); for the 5b + override configuration, given runs on the sparse plan.
- 7b (aim direction, second try): ray loss weighted toward azimuth (w_ctl_aux on the full 1 - cos term as before plus an azimuth term
  1 - cos(az(ray) - az(target)) at 2x), elevation term weight 2x (w_ctl_ray_el = 2 x w_ctl_aux), raise weight halved (w_ctl_raise 2.5).
  Gates: median forearm angular error <= 25 deg AND |median elevation error, above-shoulder targets| <= 5 deg AND STG gain >= +0.10
  with the shuffle follow; quality (i) / (ii) + liveliness guard; executability after CBF 100 %.
- 8b (present-count, second try): ONE POINT token per count command (the hand with the larger commanded count; ties -> right) toward
  the frontal presentation target, plus a wrist / palm orientation target: the median wrist joint angles (arm_joint5-7) of the library
  presentation pose (train-GT frames, K-half speakers, hand raised and >= 1 extended finger) blended in joint space over the command window
  with the minimum-jerk ramp (like the finger override; arm joints 1-4 untouched). Gates: count hit >= 0.9; hand height >= chest
  (wrist >= shoulder - 0.20 m) on >= 80 % of command frames; palm-normal angle to forward median <= 45 deg; quality (i) / (ii) +
  liveliness guard; ramp jerk reported.
## 7b / 8b run setup (2026-11-08 ~04:40; coordinator: base = sibag-s2 after agent A's corrected liveliness selection; gates exactly as cfc519d)
Fresh plans seed 1110 / shuffle 20261110: PLAN_dev_m78b_dense.json (1089 events, 559 aim; sha256 2b5e232a…e44e), PLAN_dev_m78b_sparse.json
(303 events, 164 aim; sha256 c15b311f…af5). 7b = lane-4 recipe + w_ctl_ray_az 6 / w_ctl_ray_el 6 / w_ctl_raise 2.5 (true freeze, 2000 steps).
8b = the lane-4 control model ctl4 (sibag-s2 true freeze) with ex002_present.py plan1 (one token, present_hand) + tokens/PRESENT_WRIST.json
(library wrist pose: left (0.050, 0.003, 0.356), right (0.069, -0.012, -0.339) rad for arm_joint5-7, 163k / 186k frames) + minimum-jerk override.
Code exp/ctl-port (merge of exp/text-path 98169eb); liveliness guard = scripts/video/static_share.py (NORM_DEV.json) on the given runs.
## Modules 7b / 8b result on sibag-s2 (dlctt3a4rulv5svl, LANE_DONE ~06:46 2026-11-08; DLC A800; fresh plans 1110, seeds 0-2)
Liveliness guard reported both ways: (P) protocol guard = EVAL_PROTOCOL §5 / agent A's lively_floor_ex002.py (exp/duplex 5d6f017, read
by path, unchanged; 13 moving upper-body points; decides pass / fail: still <= 0.033, wrist speed >= 0.308, amplitude >= 0.090, held <= 10 %,
stutter <= 1.129); (S) my pre-registered static_share.py guard (6 points: still <= 0.038, speed >= 0.308, amp >= 0.090). Check: (P) on the
no-token runs reproduces A's sibag-s2 numbers (offline 0.023 / 0.336 / 0.099, psync 0.022 / 0.360 / 0.101; dev GT 0.017 / 0.411 / 0.120).
Module 7b (azimuth term 2x, elevation 2x, raise 2.5; true freeze on sibag-s2):
- PRIMARY FAIL: median forearm angular error offline 33.2 / psync 35.2 deg (gate 25; unblended 26.0 / 29.5; no-token 49.5 / 50.9);
  elevation above-shoulder offline -6.1 / psync -8.2 deg (gate |.| <= 5); azimuth 26.1 / 27.6 deg.
- STG passes: offline 0.404 vs 0.245 = +0.158 [+0.137, +0.177]; psync 0.383 vs 0.242 = +0.142 [+0.122, +0.161]; shuffle follows
  (0.403 vs 0.319; 0.382 vs 0.312).
- Gate (i) passes: offline 90.76 <= 100.65, psync 94.56 <= 102.48. Gate (ii) FAILS: offline 100.95 plain / 101.94 override > 99.37; psync
  107.69 / 109.76 > 101.07. Finger hit (override) 1.000; executability after CBF 100 %.
- Liveliness (sparse given, 5b + override): (P) offline 0.021 / 0.340 / 0.097 / held 0 / stutter 1.065, psync 0.020 / 0.360 / 0.098 / 0 / 1.059
  -> pass; (S) still 0.025-0.029, speed 0.333-0.363, amp 0.094-0.099 -> pass. Dense plan (report): amplitude 0.076 / 0.077 (< 0.090).
- Verdict: FAIL (angular error, elevation, gate (ii)).
Module 8b (one presentation token + library wrist pose + minimum-jerk override on ctl4):
- Count hit 1.000 PASS. Presenting hand raised (wrist >= shoulder - 0.20 m): mean share of command frames offline 0.46 / psync 0.45 (no-token
  0.20 / 0.16), events with >= 80 % raised 0.28 / 0.28 -> FAIL (gate 80 %). Palm-forward angle (presenting hand) 71.6 / 71.4 deg (no-token
  75.1 / 73.3) -> FAIL (gate 45).
- Gate (i) passes (90.74 / 94.54); gate (ii) offline 95.02 <= 99.37 passes, psync 102.59 > 101.07 FAILS. Ramp over-limit 3.3-4.7 %.
- Liveliness (sparse given): (P) offline 0.019 / 0.346 / 0.096 / 0 / 1.054, psync 0.019 / 0.361 / 0.097 / 0 / 1.053 -> pass; (S) pass. Dense
  (report): amplitude 0.061 / 0.059.
- Verdict: FAIL (hand height, palm angle, psync gate (ii)).
## Module 9 (pre-registered 2026-11-08 ~07:40, before any module-9 number): keyframe insertion with causal re-conditioning
Coordinator diagnosis after 7b / 8b: tokens alone cannot hit precise geometry on this arm; pointing and presenting have explicit geometric
targets, so solve for them. Post-processing + re-conditioning, no retraining (ports to a new base unchanged). Base sibag-s2; tokens = the
lane-4 control model ctl4 (true freeze). New fresh plans seed 1111 / shuffle 20261111 (dense + sparse); seeds 0-2; offline AND psync, each
against its own no-token C / δ (the no-token runs are sibag-s2 bitwise).
Method (scripts/ex002/ex002_keyframe.py, default off; bench_ex002.py --inpaint-dir, default off):
1. keyframe per command, damped least squares (Gauss-Newton + Levenberg damping, 20 iterations) on the commanded arm's 7 joints only, exact
   FK (scmg/kinematics), joint limits = META bounds -/+ 0.03 rad, sphere-proxy self-penetration as a residual (weight 10), regularisation
   sqrt(0.05) (q - q_seed), seeded from the token-only pose at the command time. point: forearm ray (elbow -> wrist) toward the target;
   present (finger commands; hand with the larger count, ties -> right): wrist at shoulder + (0.30, 0.06 toward the midline, -0.12) m and
   palm normal (cross(index base - wrist, third base - wrist)) along +-x (sign of the seed). A keyframe with penetration > 0 after the solve
   is REJECTED (no insertion for that event; rejection share reported).
2. insertion in joint space: minimum-jerk ramps of 0.3 s, hold [t - 0.1, t + 0.6] (point) / [t - 0.4, t + 0.4] (present); finger override
   (minimum-jerk 0.3 s) after.
3. causal re-conditioning: the inserted frames are encoded by the VAE and inpainted (RePaint-style latent inpainting already in
   scmg/infer.py) on the insertion spans with cosine-ramped weights while the SAME model / tokens / seed regenerate the clip (bench_ex002
   --inpaint-dir, offline and psync) -> the generator's history after release is the inserted motion; the keyframes are then re-applied in
   joint space on the re-generated output (re-solved from its pose) for exact geometry. Final = re-conditioned output (no 5b blend; the
   continuation after release is the point of the re-conditioning).
Rows reported: token-only (5b + minimum-jerk override, no IK), IK post-processing without re-conditioning (5b + IK), module 9 (re-conditioned).
Gates for module 9 (per protocol; primary as 7b / 8b; the angle is easy with IK, so the side gates are the real test):
- pointing: median forearm angular error <= 25 deg; |median elevation error, above-shoulder targets| <= 5 deg; STG(given) - STG(none) >= +0.10
  with the shuffle follow (shuffle runs get keyframes toward the SHUFFLED targets).
- present: count hit >= 0.9; presenting hand raised (wrist >= shoulder - 0.20 m) mean share of command frames >= 0.80; palm-normal angle to
  forward (presenting hand) median <= 45 deg.
- quality (i) sparse FGD v2 outside +-1 s <= C + δ; (ii) sparse whole clip <= C + δ.
- ramp jerk: share of ramp frames over the limits34 jerk limits <= token-only + 2 percentage points, for the finger ramps (ex002_exec.py
  --ramp 0.3) and for the arm insertion ramps (ex002_keyframe.py release, same frames in both rows).
- executability after CBF 100 % of clips.
- liveliness: protocol guard (A's lively_floor_ex002.py: still <= 0.033, wrist speed >= 0.308, amplitude >= 0.090, held <= 10 %, stutter <= 1.129)
  decides; my static_share guard reported; on the sparse given runs; dense plan report-only (the 7b / 8b dense amplitude drop to 0.061 noted).
- release continuity: median over insertions of the peak arm-joint acceleration norm in +-0.1 s around the release <= the p95 of the same
  quantity over all dev GT frames (ex002_keyframe.py release).
Plus 2-3 qualitative renders (point and present) through the Isaac renderer.
Module 9 run setup (before the job): plans PLAN_dev_m9_dense.json (1112 events, 567 aim; sha256 9a883716…53e2), PLAN_dev_m9_sparse.json
(304 events, 164 aim; sha256 58bf0bc8…9e31); code exp/ctl-port b7941af (bench_ex002 --inpaint-dir 74a6f99, default off; lively_floor_ex002.py
from exp/duplex unchanged), release 20261108-071120-ex002-m9. Smoke on 3 old-plan clips (post-processing only, before the job; disclosed): 18 %
of point / 39 % of present keyframes rejected for self-penetration; release-acceleration median 56 (IK post-processing) vs 26 token-only vs
human p95 31.9; arm ramp jerk over-limit 0.53 vs 0.27.
Module 9 clarification (coordinator, 2026-11-08 ~07:20, before any module-9 job result; not a change): a REJECTED keyframe counts as a failure in
every primary gate -- the angle error, STG, chest height and palm angle are computed over ALL command events of the plan; a rejected event is
scored from its actual output (no insertion there, i.e. the token-only / re-generated motion). No event is ever dropped from a denominator
(this is how ex002_aim_error / evaluate / ex002_present already work: they iterate over the plan's events). The reject share is reported
per protocol as its own row.
## Module 9 result (dlccwprez8uqhqyg, LANE_DONE 10:05 2026-11-08; sibag-s2 + ctl4; fresh plans 1111, seeds 0-2; DLC A800)
Rows: tok = 5b + minimum-jerk override (token-only), ik = 5b + IK insertion (no re-conditioning), m9 = re-conditioned + IK (the module).
Rejected keyframes (scored from their actual output everywhere): offline ik 6.7 % aim / 19.4 % present, m9 8.9 % / 23.2 %; psync ik 7.8 % / 18.1 %,
m9 9.8 % / 23.4 %.
PRIMARY (all PASS for m9, both protocols):
- pointing: median forearm angular error offline 23.4 / psync 23.9 deg (gate 25; ik 27.8 / 27.9, tok 35.0 / 37.7, none 50.7 / 51.7); elevation
  above-shoulder -0.5 / -0.9 deg (gate +-5; tok -6.6 / -11.8); STG 0.426 vs 0.245 = +0.181 [+0.150, +0.215] / 0.454 vs 0.232 = +0.222 [+0.192,
  +0.253]; shuffle follows (0.452 vs 0.270; 0.469 vs 0.287). Token-only gain +0.106 [0.088, 0.125] / +0.094 [0.077, 0.112].
- present: count hit 1.000; presenting hand raised 0.818 / 0.823 of command frames (gate 0.80; tok 0.18 / 0.09); palm angle 1.0 / 1.0 deg
  (sign-agnostic; tok 75.8 / 75.1).
SIDE GATES (m9):
- quality (i) FAIL: offline 111.0 > 100.29, psync 124.8 > 102.61 (ik 93.0 / 98.1 pass; tok 93.0 / 98.1). (ii) FAIL: offline 111.8 > 99.37, psync
  127.2 > 101.07 (ik 96.6 passes offline, 105.8 fails psync; tok 96.6 / 105.8).
- ramp jerk FAIL: finger ramps over the limit 0.072 / 0.100 vs tok 0.035 / 0.044 (> +2 pts); arm insertion ramps 0.551 / 0.565 (dense) vs tok
  0.228 / 0.231.
- executability after CBF 100 % PASS.
- liveliness, protocol guard (decides; sparse given): offline FAIL (still 0.029, wrist speed 0.303 < 0.308, amplitude 0.0899 < 0.090, held 0,
  stutter 1.068), psync PASS (0.028 / 0.323 / 0.092 / 0 / 1.051); ik passes both (0.022 / 0.343 / 0.099; 0.022 / 0.363 / 0.100). My guard:
  offline speed 0.303 fails, psync passes. Dense (report-only): m9 still 0.052 / 0.059, speed 0.251 / 0.265, amplitude 0.074 / 0.076.
- release continuity FAIL: median peak arm acceleration around release 54.4 (dense) / 47.5 (sparse) offline, 55.3 / 49.8 psync vs human p95
  29.8 (tok 20.1 / 19.7; ik 55.3 / 53.5).
VERDICT: module 9 passes every primary gate (geometry solved) and FAILS the side gates: quality (i) / (ii), ramp jerk, release continuity, and the
offline liveliness guard. Re-conditioning by latent inpainting does not remove the release jump (54 vs ik 55) and costs ~18 FGD v2 (111 vs 93);
the IK insertion itself (0.3 s ramps, 0.6 s hold) produces arm accelerations ~1.8x the human p95 at release.
Qualitative (qual-1108/ex002-m9; aim0 / aim1 / present0; front + 45 deg): in aim0 the forearm ray (red) runs parallel to the command direction
(yellow) in the ik / m9 panels; in present0 the right hand comes up to the face / chest with the palm forward but crosses toward the midline and
partly covers the face in the front view.
## Module 9b (pre-registered 2026-11-08 ~10:40, before any 9b number; coordinator after the m9 sheets)
Base: enc-s0 (agent A, exp/duplex 652af8c / b365eaa: sibag recipe + self_force_scope "encoders", selected under the frozen rule, 3 seeds pass
the liveliness guard). Tokens: control modules trained on enc-s0 with the lane-4 recipe and the true freeze (--train-only ctl_, 2000 steps);
with no tokens the control model equals enc-s0 bitwise (checked in the job against fresh enc-s0 no-token runs). New fresh plans seed 1112 /
shuffle 20261112 (dense + sparse); seeds 0-2; offline AND psync.
Changes vs module 9 (ex002_keyframe.py insert --version 9b; the m9 code path unchanged):
1. NO latent-inpainting re-conditioning: IK insertion on the generated stream (5b-blended token output). The m9 numbers (sibag-s2, f184a6f) are
   quoted as the reference row (not re-run).
2. Adaptive ramps in AND out: per insertion, T = max(0.3, sqrt(5.77 * max_j |Δq_j| / a95_j)) over the commanded arm's joints (Δq = keyframe -
   the generated pose at the ramp boundary), capped at 0.8 s; a95_j = per-joint p95 of |second difference| x 50^2 over TRAIN-split human items of
   data-v36_1 (tokens/ARM_ACC_P95.json; never dev). Both ramps are quintic Hermite segments on the arm joints: ramp-in from the generator's state
   (position, velocity, acceleration) at the ramp start to the keyframe at rest; release from the keyframe at rest to the generator's own state
   (position, velocity, acceleration) at the ramp end -- not to a frozen pose. Hold as m9 ([t - 0.1, t + 0.6] point, [t - 0.4, t + 0.4] present).
3. Present target without midline crossing: wrist in front of the SAME-side shoulder, nominal shoulder + (0.30 forward, 0.05 outward, -0.12) m
   (below the chin: the chin is above the shoulder), palm normal along +-x. Self-collision inside the solve (penetration residual weight 100);
   fallback grid around the nominal point (outward offset {0.05, 0.10, 0.00} x height {-0.12, -0.06, -0.18} m, tried in that order: never inward
   of the shoulder); reject only if every grid point leaves penetration > 0 (rejected events still count as failures in every gate). Point: as m9
   but with the same penetration weight 100 and no grid (the target is a direction).
Gates (same as m9, per protocol, the enc-s0 tok row = reference for the relative gates): pointing median forearm angular error <= 25 deg,
|elevation error above-shoulder| <= 5 deg, STG gain >= +0.10 with the shuffle follow; present count hit >= 0.9, presenting hand raised >= 0.80
of command frames, palm angle <= 45 deg; quality (i) / (ii) <= C + δ (enc-s0 no-token C / δ per protocol); ramp jerk (finger and arm ramps)
<= tok + 2 pts; executability after CBF 100 %; protocol liveliness guard (decides) + my guard on the sparse given runs, dense report-only;
release continuity: median peak arm acceleration around the release <= the dev-GT p95 (as m9). Reject share per protocol reported.
Note: on sibag-s2 the tok row itself failed (ii) on psync (105.8 > 101.07); the enc-s0 tok numbers are reported first.
Quick check before the DLC run: 3-clip post-processing smoke on existing sibag-s2 m9 tok outputs (disclosed). Renders after: aim + present,
front + 45 deg, selection rule committed before rendering.
Module 9b quick check (disclosed, before the DLC run): ARM_ACC_P95.json (train a95 per arm joint 9 / 9 / 9 / 12 / 15 / 4.9 / 7.4 rad/s^2 left,
same right: the retargeted data sits at its own acceleration limits); 3-clip post-processing smoke on the sibag-s2 m9 tok5bplain outputs (plan
1111, offline seed 0): rejected keyframes 2.5 % point / 2.6 % present (2.6 % used a fallback grid point); median ramp Tin / Tout 0.39 / 0.45 s
(point), 0.77 / 0.77 s (present); release peak acceleration median 27.1 (tok 27.8; human p95 29.8), p95 254 (tail); arm ramp jerk over-limit 0.033
(tok 0.294). Plans for the run: PLAN_dev_m9b_dense.json (1092 events, 556 aim; sha256 a0336b83…12b2), PLAN_dev_m9b_sparse.json (302 events,
163 aim; sha256 2ad29827…5eed).
## Module 5c (pre-registered 2026-11-08 ~10:30, before any 5c gate number; coordinator after the 9b release-tail diagnosis)
Bug fixed: 5b composes the per-command windows by max(); on the dense plans (about 72 % of command gaps in 1.2–1.8 s) a ramp-out crossing
the next ramp-in gives w a corner (derivative jump), and the blended pose gets a cusp (see the 9b release-tail diagnosis above).
- 5c: w = 1 − Π_i (1 − w_i) on frames where ≥ 2 windows are non-zero; elsewhere the single window, as in 5b. This is C1, keeps every
  command's own window [t − 0.3, t + 0.9] s with 0.3 s raised-cosine ramps, and does not fill gaps. ex002_local_blend.py --compose smooth;
  the default (max) is unchanged.
- Unit test scripts/ex002/test_blend5c.py, ALL_OK before any run:
  (A) the default equals the 5b weight at 366aed0 bitwise on every dev plan;
  (B) on 349 clip-plans without overlapping windows (every sparse plan) 5c equals 5b bitwise;
  (C) on 167 overlapping clip-plans, max |w''| is 107.5 ≤ the two-window bound 1.5 π²/r² = 164.5 (+5 % margin: 172.7), while 5b reaches 521.7.
  Disclosed before the gates: on smoke clip 0, arm frames over 100 rad/s² are 2.6 % with 5b, 0.77 % with 5c, 1.2 % in the none stream.
- Validation (post-processing only): on the enc-s0 generations of the m9b-enc job (dlc127d7nvk060w9: ctlenc control modules,
  plans 1112 dense / sparse, seeds 0–2, offline AND psync).
- Gates: the original 5b gates, per protocol, against enc-s0 none on the same plan masks:
  - STG(given) − STG(none) ≥ +0.10 on the dense plan, with the shuffle follow check (recording bootstrap CI reported);
  - finger hit ≥ 0.9 with the override;
  - quality (i) sparse FGD v2 outside ±1 s, and (ii) sparse whole clip, both ≤ C + δ (δ = max(0.03 C, 2 SD) of the none seeds);
    on the sparse plans 5c is 5b bitwise (test B), and they are recomputed as a check;
  - executability after CBF 100 %.
- Report-only, both plans, 5b vs 5c vs none vs GT:
  - share of arm frames over 100 rad/s², p99, and the p95 of the peak over random 0.2 s windows;
  - release continuity (median / p95 / max);
  - arm ramp jerk;
  - dense-plan whole-clip FGD v2.
- Hypothesis "the 5b blend caused tok's psync (ii) failure on sibag-s2": (ii) is scored on the sparse plan, where no windows overlap
  (0 of 261 gaps on PLAN_dev_m9_sparse are < 1.8 s), so 5c equals 5b there and the prediction is NO change (105.8 stays). This refutes
  my earlier suggestion before any run. Checked anyway by recomputing tok psync (ii) with 5c on m9-sibag (expected bitwise 105.8). The
  dense-plan whole-clip FGD v2 of tok psync, 5b vs 5c, is reported as the place where the bug can act.
- Downstream: after the m9b-enc run, every tok / ik / 9b row moves to 5c.
5c run setup (2026-11-08 ~10:50, before any 9b or 5c number): post-processing job scripts/ex002/ex002_ctl_m5c.sh (exp/ctl-port), started after
the m9b-enc job ends.
- Rows: tok5c (5c + minimum-jerk override) and m9b5c (9b insertion, exactly as pre-registered in 12f8351, on 5c plain instead of 5b plain).
  m9b5c is scored with the full 9b gate set, the tok5c row being its reference. The 5b rows of m9b-enc are reported next to them.
- Identity checks: on the sparse plan, 5c vs 5b, and m9b5c vs m9b (expected bitwise).
- sibag-s2: m9-sibag tok with 5c on both plans (the (ii) check above, and the dense whole-clip FGD v2); release median / p95 / max
  re-scored for the m9-sibag tok / ik / m9 rows.
Generator acceleration-spike diagnosis (read-only, numbers only; coordinator 11-08; scripts/ex002/diag/spikes_where.py on speech_server;
JSON control-token-1105/ex002/diag/SPIKES_WHERE_1108.json).
- Correction: the "enc-s0 none random 0.2 s window p95 147" quoted earlier came from 6 clips of seed 0. Over all 43 dev clips × 3 seeds it is
  50.3 offline / 57.1 psync (GT 27.5).
- Magnitude (arm, max over arms of the joint-norm acceleration, 0.2 s windows with stride 0.1 s):

  | stream | window p95 | ratio to GT | frames > 100 | p99 | max |
  |---|---|---|---|---|---|
  | EX002 GT | 27.5 | 1.0x | 0.10 % | 28.2 | 5762 (a GT glitch frame) |
  | enc-s0 none offline | 50.3 | 1.8x | 0.66 % | 84.3 | 331 |
  | enc-s0 none psync | 57.1 | 2.1x | 0.69 % | 86.8 | 314 |
  | sibag-s2 none offline | 46.7 | 1.7x | 0.54 % | 79.0 | 337 |
  | sibag-s2 none psync | 64.4 | 2.3x | 0.82 % | 93.4 | 449 |

  Frames above the GT p99: 6.3 / 7.9 % (enc-s0) and 6.0 / 9.5 % (sibag-s2), against 1 % in GT. Resampled to 30 fps the ratios stay
  (enc-s0 45.7 / 51.9 vs GT 26.5).
- Location:
  - Phase: a 4-frame (one latent) pattern. Relative spike rate by frame within the latent is about 1.14–1.19 / 1.05 / 0.69–0.75 / 1.05–1.07
    (frame 0 / 1 / 2 / 3; GT is flat, 1.00 ± 0.01). No extra rate at the 320 ms chunk boundary (phase 0 of 16) or at the streamer's emission
    boundary (phase 12): phases 0 / 4 / 8 / 12 are equal within ±0.06.
  - Joints: the spikes sit on arm joints 4 and 5 (enc-s0: L 14 / 17 %, R 11–13 / 21–22 %) and joint 3 (7–9 %). GT's few spikes sit on the
    wrist joints 5–7.
  - Fingers and head/waist: per-joint p99 is fingers 60–65 vs GT 45, head/waist 10–12 vs GT 4.5.
  - Speech vs silence: rate of frames above the GT p99 is 5.4 % in speech vs 14.1 % in silence offline, and 7.2 % vs 14.2 % psync
    (GT 0.9 % vs 1.5 %). Silence is 2.0–2.6x spikier, against 1.6x in GT. Speech covers 90 % of frames.
- G1 (joint space, arms 25:39 of robot_repr, 30 fps, same window rule): G1 GT dev window p95 130 (frames above 100: 3.0 %). czm6kws dev 114
  offline / 105 psync (0.88x / 0.81x GT). The czmws test predictions (read-only, predictions only, no GT; logged in TEST_USAGE.jsonl) are
  110 / 119 (0.85x / 0.91x). Frames above the G1 GT p99: 0.5–1.2 %. So on G1 the generator is not spikier than its human data; the
  excess is EX002-relative: EX002 GT is about 5x smoother than G1 GT (27.5 vs 130), and the generator does not reach that smoothness.
VAE isolation (read-only, numbers only; scripts/ex002/diag/vae_diag.py on speech_server GPU1; JSON diag/VAE_DIAG_1108.json). Threshold for
"spike" = GT p99 = 28.1 rad/s².
- A. Dev GT (43 clips) → vae-r3 encode (mu) → decode:
  - window p95 32.9 vs GT 27.5 (1.20x); p99 35.9 vs 28.1; frames above the GT p99 2.9 % (GT 1 %); max 217;
  - per-frame-in-latent rate 1.29 / 1.06 / 0.47 / 1.18 (GT flat 1.00 ± 0.01): the same 4-frame pattern as the generator, stronger;
  - joint split L j4 17 % / j5 16 %, R j4 13 % / j5 31 %, the same signature as the generator (GT spikes sit on wrist joints 5–7);
  - silence 5.4 % vs speech 2.6 %;
  - reconstruction error: mean |Δq| 0.032 rad (upper body), 0.048 rad (arms).
- B. enc-s0 offline seed 0, 8 clips (0, 6, ..., 42), committed latents captured. The streamed output matches the enc-s0 files bitwise.
  GT on the same clips: window p95 28.7.
  - streamed (1-latent look-ahead, as emitted): 45.5 (1.59x GT), frames above the GT p99 5.6 %, phase 1.13 / 1.07 / 0.77 / 1.04;
  - one-shot decode of the same latents: 45.7, identical pattern; max |diff| to streamed 4e-4 rad, so the streaming does not add spikes;
  - decode without look-ahead (next latent replaced by a copy): 357, phase 1.9 / 0.07 / 0.04 / 1.99. The look-ahead is needed, and it is used;
  - streamed output re-encoded and decoded (projection onto the codec manifold): 38.9 (1.35x), frames above the GT p99 4.8 %,
    phase 1.16 / 1.06 / 0.69 / 1.08. This removes about 40 % of the excess over GT.
- Reading:
  - The decoder makes the 4-frame pattern and the joint-4/5 signature: both are already in the GT reconstruction.
  - The magnitude splits into a decoder part (GT reconstruction 1.20x) and a generator part. Generator latents are partly off-manifold:
    projection takes 1.59x down to 1.35x, which is still above the GT reconstruction's 1.20x.
## Decoder-only fine-tune r3d (pre-registered 2026-11-08 ~11:30, before any r3d run; coordinator approved after the VAE isolation)
Goal: remove the decoder's 4-frame acceleration pattern without touching the latent space.
- Recipe. Init vae-r3. Encoder parameters and latent statistics are frozen; this is asserted at save, as in scripts/echog_g1/vae_decoder_ft.py
  (c67ede4), so every EX002 generator stays valid without retraining. Data: data-v36_1 TRAIN split only. Script:
  scripts/ex002/vae_decoder_ft_ex002.py.
- Loss. The r3 loss (rec + 0.3 vel + 0.1 acc in normalised units; the KL is constant), plus physical-unit velocity and acceleration
  matching on the moving upper body (joints 2–33: head, arms, fingers; waist / pedestal excluded): relative L1,
  mean |Δ^k ŷ − Δ^k x| / mean |Δ^k x| in rad/s and rad/s², k = 1, 2, weights 1 / 1. lr 1e-4, 200-step warmup, batch 64 × 256 frames, EMA 0.999.
- Arms:
  - A: z = mu + σ·eps from the frozen posterior;
  - B: as A, but on 50 % of the batches z = mu + 0.3·eps (latent-error robustness, aimed at the off-manifold generator latents).
- Schedule. Quick lane of 2000 steps per arm. An arm goes on to 10k steps only if its quick-lane checkpoint passes gates 1–2. The final
  candidate is the 10k checkpoint, and every gate is evaluated on it.
- Gates. Dev only; "GT" = the 43 dev clips (enc-s0 offline gt_*.npy); spike threshold = GT p99 = 28.1 rad/s²; statistics as
  scripts/ex002/diag/spikes_where.py.
  1. Reconstruction (mu decode of the dev GT) not worse than r3 by more than 2 %, on both the normalised L1 over all joints and the mean
     |Δq| over the upper body. Same script, same clips; r3: 0.032 rad upper body.
  2. Reconstructed GT window p95 ≤ 1.2x GT (≤ 32.96); reconstructed frames above the GT p99 ≤ 1.5 % (r3 2.9 %); flat latent phase, all four
     per-frame-in-latent rates within 0.9–1.1 (r3 0.47–1.29).
  3. On regenerated enc-s0 none outputs (bench_ex002 infer --vae r3d, offline AND psync, seeds 0–2, per_clip, same as A's enc-s0 runs):
     - window p95 ≤ 1.3x GT (≤ 35.7) per protocol (r3: 50.3 / 57.1);
     - FGD v2 and FKD not worse than r3 by more than one seed SD (3-seed means; r3 = A's enc-s0 runs re-scored in the same job);
     - liveliness: the protocol guard (lively_floor_ex002.py, pooled seeds) and my guard both pass.
- Adoption rule:
  - one arm passes all gates: it becomes the EX002 decoder, and every EX002 table re-runs on its outputs;
  - both pass: take the lower gate-3 window p95 (mean of the two protocols);
  - neither passes: keep r3 and log a negatives row.
- Reported for the winner: per-frame-in-latent rates, arm-joint split, and one before / after render (same clip, r3 vs r3d decode,
  front view).
- Cost: one 4-GPU pod, about 1.2 h for both arms.
## Module 9b result (dlc127d7nvk060w9, LANE_DONE 12:09 2026-11-08; enc-s0 + ctlenc; fresh plans 1112, seeds 0-2; DLC A800)
IDENTITY: the control model's none output equals A's enc-s0 runs bitwise (6/6 runs, max |diff| 0).
Rows: tok = 5b + minimum-jerk override (reference); m9b = 9b insertion on 5b plain. m9 (sibag-s2, f184a6f) is quoted as a reference row.
Rejected keyframes: aim 2.0 % offline / 2.4 % psync; present 0.3 % / 0.3 %; fallback grid used for 1.4 % / 1.3 % of presents. Median Tin / Tout:
aim 0.40 / 0.46 s (offline), 0.42 / 0.46 s (psync); present 0.71 / 0.70 s, 0.71 / 0.69 s.
tok on enc-s0 (reference, reported first):
- STG gain +0.095 [+0.077, +0.115] offline, +0.084 [+0.070, +0.100] psync (below +0.10);
- angle 33.2 / 34.6 deg; elevation above shoulder -3.5 / -8.8 deg; presenting hand raised 0.21 / 0.11; palm 74.6 / 75.7 deg;
- (i) 86.96 / 107.04 (pass); (ii) 92.25 > 90.15 offline and 118.27 > 114.34 psync (FAIL on both protocols);
- release median 18.9 / 20.2 (dense), 20.3 / 21.4 (sparse).
PRIMARY (m9b, offline / psync):
- pointing angle (median forearm angular error at the apex frame): 29.0 / 28.2 deg, gate 25: FAIL (m9 23.4 / 23.9; tok 33.2 / 34.6;
  none 45.6 / 46.4);
- elevation above shoulder +1.5 / -0.7 deg (gate ±5): PASS;
- STG +0.122 [+0.095, +0.153] / +0.127 [+0.102, +0.154]: PASS; shuffle follows (0.365 vs 0.275; 0.391 vs 0.276);
- present count hit 1.000: PASS; presenting hand raised 0.824 / 0.799 (gate 0.80): offline PASS, psync FAIL by 0.001; palm 2.0 / 2.0 deg: PASS.
SIDE GATES (m9b):
- quality (i) sparse outside ±1 s 86.97 ≤ 92.16 / 107.04 ≤ 112.74: PASS;
- quality (ii) sparse whole clip 92.15 > 90.15 / 116.40 > 114.34: FAIL (tok fails too: 92.25 / 118.27);
- ramp jerk: finger ramps over the limit 0.037 / 0.045 = tok; arm ramps 0.042 / 0.039 dense and 0.024 / 0.022 sparse
  (tok 0.213 / 0.213, 0.173 / 0.182): PASS;
- executability after CBF 100 %: PASS;
- liveliness, sparse given:
  - protocol guard PASS / PASS (offline still 0.022, speed 0.314, amplitude 0.095; psync 0.026 / 0.316 / 0.094);
  - my guard PASS / PASS (0.029 / 0.314 / 0.095; 0.032 / 0.316 / 0.094);
  - dense (report-only): protocol guard fails on speed / amplitude (0.290 / 0.075; 0.273 / 0.073);
- release continuity, median peak arm acceleration around release, dense 27.1 / 27.8 and sparse 24.5 / 24.1 ≤ 29.8: PASS.
  Report-only p95: dense 226 / 242 (the 5b dense-plan cusps, root cause row 47), sparse 67.2 / 60.0 (tok 73.0 / 72.8). Max: from the 5c job.
VERDICT: module 9b FAILS the primary pointing-angle gate (both protocols), presenting-hand raised on psync (0.799), and quality (ii)
(both protocols; tok fails it too). It passes STG, elevation, count, palm, quality (i), ramp jerk, executability, both liveliness guards
and release continuity (median).
Diagnosis of the angle failure (offline seed 0, 197 aim events of 15 clips, read-only):
- At the middle frame of the IK hold the median forearm angular error is 0.9 deg (IK residual median 4.2 deg).
- The frozen apex detector (APEX.json, logistic regression with positive weights on wrist speed) never chose a frame inside the hold:
  75 % of apex frames fall before it, 25 % after, and the median error at those frames is 31.4 deg.
- The hold is exactly static (speed 0), which the detector scores low. The gate (apex rule) is unchanged.
## r3d result (dlc1p9g16zayjbbt, LANE_DONE 12:13 2026-11-08): both arms stop after the quick lane (2000 steps)
Gates 1-2 on the quick-lane checkpoints (encoder identical to r3 in both arms, asserted):

| | A (posterior) | B (+ latent noise) | r3 | threshold |
|---|---|---|---|---|
| normalised L1 | 0.1365 (+7.0 %) | 0.1616 (+26.6 %) | 0.1276 | gate 1: ≤ +2 % |
| upper-body mean abs (rad) | 0.0362 (+12.8 %) | 0.0476 (+48 %) | 0.0321 | gate 1: ≤ +2 % |
| reconstructed GT window p95 | 25.8 | 22.9 | 32.9 | gate 2: ≤ 32.96 (GT 27.5) |
| frames above the GT p99 | 1.09 % | 0.67 % | 2.87 % | gate 2: ≤ 1.5 % |
| latent-phase rates | 1.19 / 1.05 / 0.73 / 1.03 | 1.14 / 1.05 / 0.76 / 1.05 | 1.29 / 1.06 / 0.47 / 1.18 | gate 2: all in 0.9–1.1 |

VERDICT: both arms FAIL gate 1 (reconstruction) and the flat-phase part of gate 2. Both lower the spike magnitude below GT (smoother than
the human data). The 4-frame pattern weakens but stays (frame 2 at 0.73 / 0.76). By the adoption rule, r3 stays the EX002 decoder; no
10k run and no gate 3. Module 9c runs on r3.
## Module 9c (pre-registered 2026-11-08 ~13:20, before any 9c run; coordinator after the 9b result and the train-statistics check)
Base: enc-s0 + ctlenc, decoder r3 (r3d not adopted). Input: the m9b-enc generations blended with 5c (tok5c = 5c + override is the
reference row). 9c changes exactly two things relative to 9b; everything else (keyframe IK for aim, ramps, timing, gates) is as 12f8351.
1. Pointing hold motion replayed from human data:
   - Pool: tokens/HOLD_POOL_TRAIN.npz (sha256 c35f2424…; scripts/ex002/hold_pool.py), TRAIN split only.
   - point pool: 2947 entries from 3000 sampled pointing tokens (of 89,514; seed 1114). Each entry is the wrist displacement of the token's
     hand over [t − 0.1, t + 0.6] s (36 frames, the aim-hold window).
   - control pool: 2947 windows of the same length from train speech spans, at least 1.0 s from any pointing token, random hand.
   - Pool statistics, written before any run (mean wrist speed over the window, median [IQR]): point 0.160 [0.066, 0.271] m/s,
     end displacement 5.7 cm; control 0.260 [0.117, 0.423] m/s, 9.6 cm. (Source check, window [t − 0.3, t + 0.6]: along-ray push-out
     5 mm [0, 23], perpendicular drift 4.7 cm [1.8, 9.5]; HOLD_STATS_TRAIN.json a5f96b11.)
   - Sampling rule: pool index = sha1("<clip>:<t:.3f>:9c") mod pool size, the same rule for both pools. If the pool entry's hand differs
     from the command hand, the lateral (y) axis is mirrored.
   - Per-frame IK during the hold: the wrist target is w_k + dw[τ] (w_k = the 9b keyframe wrist), and the forearm ray is kept on the
     command target (residual as 9b + wrist-position residual). IK at every 3rd hold frame plus the last one, warm-started from the
     previous key, 5 iterations, pen_w 100; a key with self-penetration is replaced by the previous key; cubic interpolation between keys.
     The ramps match the hold's own velocity / acceleration at both ends (quintic, as 9b).
   - Rows: m9c uses the point pool; m9cctl, the control, uses the control pool through the same pipeline.
2. Present reachability: a hard-weighted lower bound on wrist height (wrist z ≥ shoulder z − 0.15 m, hinge weight 100) inside the IK,
   alongside self-collision. A grid point is accepted only if collision-free and within the bound (1 mm tolerance); the fallback grid is
   as 9b. The event is rejected (counted as a failure everywhere) only if every grid point fails. Present holds stay static.
- No anticipation change (point and present timing as 9b).
- Gates: identical to 9b, per protocol, rejected events scored from their actual output. Reference = tok5c. Primary: apex-based
  angle ≤ 25 deg; elevation ±5; STG ≥ +0.10 with shuffle follow; count ≥ 0.9; presenting hand raised ≥ 0.80; palm ≤ 45. Side: (i) / (ii)
  ≤ C + δ; ramp jerk ≤ tok5c + 2 pts; executability 100 %; protocol + my liveliness guard (sparse given); release median ≤ 29.8.
- Report-only: hold-window forearm angle (median over hold frames), apex-in-hold share, hold wrist speed vs the pool statistics
  (scripts/ex002/ex002_hold_report.py); release p95 / max; held-key share; the 5c effect on (ii) via tok5c vs the 5b tok row of m9b-enc.
- Quick check first: a disclosed 3-clip smoke (timing and pipeline only). Then the DLC run (scripts/ex002/ex002_ctl_m9c.sh). Renders follow
  with a selection rule committed before rendering.
Module 9c quick check (disclosed, before the DLC run): 3 clips (offline seed 0, dense plan; m9b-enc generations, 5c plain), on speech_server CPU.
- m9c: aim 38 events, rejected 2.6 %, held (penetrating) keys 0.05 per event, wrist-target error median 1.9 cm.
- m9cctl: same, held keys 0.24 per event, wrist-target error 3.0 cm.
- present: 38 events, 0 rejected, 0 height rejects, grid point 0 for all, wrist − shoulder z median −0.129 m.
- Default-path check: the 9b insertion with the new code reproduces the m9b-enc output of clip 5 bitwise.
- Cost: about 500 s per clip per row on speech_server, about 3x 9b. The insert stage therefore runs on CPU pods split by NSH / ISH, then
  the eval stage runs on a 4-GPU pod.
## Module 5c result (dlc150dvyjw8nthi, LANE_DONE 13:46 2026-11-08; post-processing of the m9b-enc generations; DLC)
IDENTITY:
- Sparse plan: 5c equals 5b bitwise (tok5c, tok5cplain, m9b5c vs m9b; 43/43 clips × 3 seeds × 2 protocols; sibag-s2 too).
- Dense plan: 13/43 clips bitwise; the rest differ, max |diff| 0.35–0.51 rad, at the window crossings.
Original 5b gates on enc-s0, tok5c (tok5b of m9b-enc in brackets), offline / psync:
- STG gain +0.099 [+0.080, +0.118] / +0.088 [+0.071, +0.105] (tok5b +0.095 / +0.084): FAIL (< +0.10 on both). Shuffle follows (0.365 vs
  0.311; 0.350 vs 0.309).
- finger hit 1.000 (override): PASS.
- (i) 86.96 ≤ 92.16 / 107.04 ≤ 112.74: PASS.
- (ii) 92.25 > 90.15 / 118.27 > 114.34: FAIL, bitwise = 5b. Without the override (tok5cplain): 90.05 ≤ 90.15 / 115.30 > 114.34.
- executability 100 %: PASS.
VERDICT: 5c fails the original 5b gates on enc-s0 (STG and (ii)), exactly where 5b fails them on enc-s0; on the sparse plan it is 5b
bitwise. On sibag-s2 the 5b gates had passed (fresh-plan confirmation).
Blend-artifact report (dense plan, offline / psync):
- arm frames over 100 rad/s²: tok5b 1.30 / 1.32 % → tok5c 0.88 / 0.86 % (none 0.66 / 0.69 %, GT 0.10 %);
- random-window p95: 185 / 186 → 90 / 89 (none 50 / 57);
- max: 1307 / 1405 → 405 / 305;
- dense whole-clip FGD v2: 178.7 / 216.2 → 179.0 / 216.5 (none 86.3 / 109.0);
- finger ramp frames over the jerk limit (dense): 0.038 / 0.046 → 0.029 / 0.039.
(ii) hypothesis on sibag-s2: tok5c psync 105.80 = tok5b 105.80 (offline 96.61 = 96.61), bitwise as predicted: the blend bug did not cause
the sibag-s2 tok psync (ii) failure. Dense whole-clip FGD v2 on sibag-s2: 179.35 → 179.73 / 202.49 → 202.90.
Release continuity, median / p95 / max (rad/s²), every row:

| row | dense, offline | dense, psync | sparse, offline | sparse, psync |
|---|---|---|---|---|
| sibag tok5b | 20.1 / 80.6 / 220 | 21.6 / 80.6 / 261 | 19.7 / 72.6 / 184 | 22.0 / 75.8 / 184 |
| sibag ik | 55.3 / 99.2 / 220 | 55.9 / 103.3 / 261 | 53.5 / 108.1 / 186 | 55.8 / 112.9 / 204 |
| sibag m9 | 54.4 / 98.7 / 299 | 55.3 / 104.0 / 343 | 47.5 / 112.3 / 172 | 49.8 / 111.0 / 192 |
| enc tok5b | 18.9 / 78.1 / 231 | 20.2 / 75.1 / 165 | 20.3 / 73.0 / 208 | 21.4 / 72.8 / 222 |
| enc m9b | 27.1 / 226.1 / 1044 | 27.8 / 241.8 / 1235 | 24.5 / 67.2 / 195 | 24.1 / 60.0 / 133 |
| enc tok5c | 18.8 / 77.2 / 231 | 20.1 / 75.0 / 165 | = tok5b | = tok5b |
| enc m9b5c | 27.1 / 92.5 / 217 | 27.5 / 90.5 / 229 | = m9b | = m9b |

m9b5c with the 9b gates (dense rows change, sparse rows = m9b):
- angle 28.3 / 27.7: FAIL;
- elevation +1.4 / -0.8: PASS;
- STG +0.128 [+0.102, +0.158] / +0.131 [+0.105, +0.160]: PASS;
- presenting hand raised 0.824 / 0.799: FAIL psync; palm 2.0: PASS;
- (i): PASS; (ii): FAIL (= m9b);
- arm ramps 0.030 / 0.028 (tok5c 0.225): PASS;
- executability: PASS; liveliness sparse: PASS;
- release median 27.1 / 27.5: PASS.
Same verdict as 9b.
Qualitative (qual-1108/ex002-m9b; rule ctl-port d5fcb52: plan 1112 dense, offline seed 0, sha1-ranked aim0 / present0 / aim1; panels
tok5b / m9b / m9b5c; front + 45 deg; frames at the command time in aim0.f.png / present0.f.png):
- present0 (fingers L2 R2, right hand presents): in the m9b and m9b5c panels the right hand is raised in front of the right side of the
  chest with the palm forward. It does not cross to the face (m9 did). In tok5b the hands stay low at the hips.
- aim0 (left hand): the red forearm ray runs close to the yellow command direction in the m9b / m9b5c panels and further from it in
  tok5b.
- At the command time m9b and m9b5c look the same.
## Why the token path does not transfer to enc-s0 (read-only diagnosis, numbers only; coordinator after the 5c result)
1. Where the (ii) cost sits (sparse plan, seeds 0-2; scripts/ex002/diag/ii_split.py; diag/II_SPLIT_1108.json). Windows (32 frames,
   stride 16) are split into inside (overlap ±1 s of a command, 41 % of windows) and outside. Offline / psync:

   | run | whole = (ii) | outside | inside-only | swap |
   |---|---|---|---|---|
   | none | 86.29 / 109.02 | 86.83 / 107.06 | 89.21 / 115.72 | — |
   | tok5cplain | 90.05 / 115.30 | 86.84 / 107.10 | 105.29 / 139.04 | 86.29 / 109.05 |
   | tok5c (override) | 92.25 / 118.27 | 86.84 / 107.10 | 112.02 / 148.12 | 86.29 / 109.05 |

   - Swap = the whole-clip FGD with the inside windows taken from the none run. It returns to the none value, so the entire (ii) excess
     (+5.96 / +9.25 over none) sits in the command windows; outside windows equal none's.
   - The override's share of the excess: +2.20 / +2.97 (37 % / 32 %).
   - Per-window Mahalanobis d² to the GT window Gaussian, inside windows: none 42.5 / 44.4, tok5cplain 41.5 / 41.9, tok5c 42.6 / 42.9.
     The token windows are not individually atypical; the inside excess is a distribution-level effect (lower spread near commands).
2. STG on enc-s0 vs sibag-s2 (diag/raw):
   - Unblended (raw) given runs, aim angle (median, offline / psync): sibag ctl4 29.9 / 32.5, enc ctlenc 30.1 / 32.0 (none 50.7 / 51.7 vs
     45.6 / 46.4). The raw aiming is the same on both bases.
   - Raw STG gain: sibag +0.144 [+0.121, +0.168] / +0.135 [+0.116, +0.156]; enc +0.113 [+0.093, +0.133] / +0.099 [+0.082, +0.116]
     (plans 1111 vs 1112).
   - In STG terms: the enc none STG is higher (0.257 / 0.260 vs 0.245 / 0.232) and the enc given STG slightly lower
     (0.370 / 0.358 vs 0.389 / 0.367).
   - The blend costs about 0.04 of gain on sibag (raw +0.144 → tok5b +0.106) and about 0.015 on enc (+0.113 → +0.095–0.099).
   - Training: the ctl_ loss curves are the same within noise (steps 1800-2000: ctl_ray 0.1015 vs 0.1008, ctl_finger 0.2216 vs 0.2221).
   - self_force_scope: both control trainings used self_force_p = 0 (self_forced 0.0 in every log line), so the 'encoders' scope never
     acted on the ctl_ modules. The scope only shaped the enc-s0 base itself.
## Module 10 (pre-registered 2026-11-08 ~14:30, before any module-10 run; coordinator approved the draft with two additions)
Base enc-s0 + ctlenc, decoder r3, 5c blend, plans 1112, seeds 0-2, offline AND psync. Reference row tok5c (m5c-enc).
(a) Override shapes from a human library.
  - Library: tokens/HAND_LIBRARY_TRAIN.npz (scripts/ex002/hand_library.py), TRAIN split only. For each hand h and count n (0..3): up to
    2000 frames (seed 1115, every 5th train frame) whose frozen finger-count detector count (ex002_ctl.finger_counts) is n; stored as
    the 8 finger joints of that hand.
  - Each finger command and hand draws one entry by sha1("<clip>:<t:.3f>:10a:<h>") mod the library size and uses it in place of the
    single HAND_SHAPES template. Hold 0.4 s and 0.3 s minimum-jerk ramps are unchanged (finger_override --fo-library; default off).
  - Circularity: the library is filtered by the same detector that scores the finger-hit gate, so that gate is close to guaranteed for
    (a) and is not evidence for it.
  - Independent check, reported: per finger, the flexion joints of the sampled shape are classified extended or curled by joint angle,
    by the nearest of the HAND_SHAPES all-extended (h_3) and all-curled (h_0) template values for that finger (Euclidean over the
    finger's flexion joints), not by the detector's curl-fraction threshold. Agreement = share of shapes whose angle-classified
    extended count equals n. Reported per (h, n) over the library and over the sampled events.
(b) Control modules trained longer on enc-s0.
  - Continue ctlenc (step 2000) with the same recipe (true freeze --train-only ctl_, self_force_p 0, lr 1e-4, 200-step warmup) for +1000
    steps (= 3000).
  - Quick check: raw (unblended) STG gain at 3000 vs 2000 on all 43 dev clips, offline, seed 0, dense plan 1112. The none run is the
    frozen base, bitwise the m9b-enc none_s0. The 2000-step given run is m9b-enc gen/d_given_s0.
  - Continue to 4000 steps only if the gain at 3000 exceeds 2000 by ≥ +0.005. Otherwise (b) stops at the quick check and is reported
    as a negative.
  - The final (b) checkpoint (3000 or 4000) regenerates given / shuffle runs (offline + psync, seeds 0-2, plans 1112).
- Rows: tok5c (reference), tok10a, and, if (b) continues, tok10b and tok10ab. All with 5c, the minimum-jerk override and the same plans.
- Gates (the original 5b gates, per protocol, against enc-s0 none on the same masks):
  - STG(given) − STG(none) ≥ +0.10 on the dense plan, with the shuffle follow check (bootstrap CI reported);
  - finger hit ≥ 0.9 (for (a) not evidence, see above);
  - quality (i) sparse outside ±1 s and (ii) sparse whole clip ≤ C + δ;
  - executability after CBF 100 %.
- Report-only: the inside / outside / swap split of (ii) (scripts/ex002/diag/ii_split.py); liveliness guards; finger ramp jerk; the
  independent shape check.
- Adoption: a row that passes every gate becomes the token path for 9c and later modules; otherwise a negatives row.
Module 10 setup (before any module-10 run):
- Library: tokens/HAND_LIBRARY_TRAIN.npz (sha256 2e73ca18…), 2000 entries per (h, n).
- Independent joint-angle check over the library: agreement 0.62 / 0.61 / 0.56 / 1.00 (left, n = 0..3) and 0.67 / 0.52 / 0.60 / 1.00
  (right).
- Over the sampled dense-plan events (1072 hand-commands): 0.72 overall, 1.00 for n = 3, 0.55–0.69 for n = 0..2.
- Reading: the detector counts a finger as extended only if it is fully straight (curl fraction < 0.15). The nearest-template angle
  rule calls partly curled human fingers extended when they are closer to the straight template, so the two rules disagree on graded
  human shapes.
- The template override path is bitwise unchanged (clips 0 and 7 vs m5c-enc tok5c).
- Job script: ctl-port ex002_ctl_m10.sh.
## Label provenance (audit pass-1 finding, 2026-11-08 ~15:10; applies to every EX002 control section above)
EX002 pointing commands, the CTL.json 'aim' tokens and the dev plans' targets are HINDSIGHT relabels, not human pointing.
- How they are built (scripts/ex002/ex002_ctl_tokens.py): random times ≥ 1.5 s apart; hand = the wrist farther from rest; target =
  wrist + U[1.5, 3] m along the forearm ray at that time.
- They measure controllability (does the robot follow a commanded direction), not the reproduction of human pointing.
- The 9c "point" pool (HOLD_POOL_TRAIN.npz) and HOLD_STATS_TRAIN.json are built from these tokens, so they are random-time windows of
  the more-extended hand, not pointing holds. The 9c point-vs-control contrast is mainly hand choice.
- Real human pointing in the train split: the 1060 BEAT2 deictic events of EVENTS.json. They have no 3D targets. APEX.json was trained
  on all of them; none of their stems is a dev item or clip (checked: overlap 0).
Hold statistics side by side, median [IQR]:

| source | window | push-out A (mm) | perpendicular drift (cm) | wrist speed (m/s) |
|---|---|---|---|---|
| hindsight tokens (2947) | [t − 0.3, t + 0.6] | 5 [0, 23] | 4.7 [1.8, 9.5] | 0.161 [0.068, 0.267] |
| real deictic (1060) | [t − 0.3, t + 0.6] | 4 [0, 19] | 11.0 [4.9, 18.2] | 0.289 [0.173, 0.445] |
| real deictic (1060) | [t − 0.1, t + 0.6] | 2 [0, 12] | 9.3 [3.7, 16.9] | 0.281 [0.150, 0.438] |

Real-deictic end along-ray value: −37 mm [−105, −1] (onset window), i.e. a retraction.
## Module 9c' (pre-registered 2026-11-08 ~15:15, before any 9c' run; coordinator decision after the provenance finding)
9c exactly as 1b19ea9, with one change: the point pool is built from the 1060 real train deictic events.
- Pool: tokens/HOLD_POOL_DEICTIC_TRAIN.npz (sha256 9761af44…; scripts/ex002/deictic_pool.py). Window [t − 0.1, t + 0.6] s at the
  annotated time; hand = the larger wrist-path diameter in [t − 1, t + 1] s (the APEX.json rule). Point pool speed 0.281 [0.150, 0.438] m/s.
- The control pool is unchanged (the m9cctl row of m9c-enc is reused, linked; same inputs, same pool).
- Run: insertion of the point row only (row name m9c inside m9cp-enc), then the 9c eval stage. Same gates as 9b / 9c, reference tok5c.
- 9c (registered) finishes as is. Its result will state that "point" means hindsight windows and that the point-vs-control contrast is
  mainly hand choice.
