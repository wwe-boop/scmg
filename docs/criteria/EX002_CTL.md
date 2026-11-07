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
