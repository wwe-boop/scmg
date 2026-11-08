# SemREPA quick lane: pre-registration (agent B, 2026-11-09, written before any teacher or planner number)

Design source: refine-logs/SEM_BRIDGE_1109.md §2 (main agent). This lane is G1 dev only. It never reads val_common (the G1 test)
or the sealed sets.

## 1. Teacher T (a frozen semantic motion encoder; NOT the pair_win recogniser)
- Data: TRAIN ids of the held-out-speaker G1 view views/echog-g1-v1-hos (12,838 items; dev speakers 2_scott / 6_carla absent).
  Motion: 50 Hz view q, channels 22..38 (waist + both arms, 17), physical units.
- Frame labels (BEAT2 human annotation, sem_en, the mapping of scripts/semantic/g1_build_sem.py: echog.recording + start_sec):
  per frame, class in {none 0, beat 1, deictic 2, iconic 3, metaphoric 4} = the deictic / iconic / metaphoric segment that
  contains the frame (the latest-starting one when segments overlap), else a beat_align segment, else none. An item without a
  sem file: class loss masked.
- Word target: the frozen e5 token features of the model's own text cache (scmg/text-e5-echog-g1-v1, non-causal), mean of the
  tokens whose time lies in the window [c - 0.8 s, c + 0.8 s], L2-normalised. No timed token in the window: e5 term masked.
- Architecture (differs from the recogniser's dilated conv + 2-layer transformer on 2.4 s at 30 fps): 1.6 s window = 80 frames
  at 50 Hz, window mean removed, divided by the train std, + first differences x 5 (34 channels); patch embedding Linear over
  4-frame patches (20 tokens, d 192) + learned positions; 4 pre-norm Transformer layers, 4 heads, FFN 384, dropout 0.1. Heads:
  per-patch 5-class logits (label = the class of the patch's centre frame); window embedding e = LayerNorm(mean of the
  tokens) (192), then Linear(192 -> 768) for the e5 cosine.
- Loss: class cross-entropy (weights (n_max / n_c)^0.5, label smoothing 0.05) + 1.0 x (1 - cos(e5 head, e5 target)).
- Training: seed 7 (the recognisers use 0 / 1 / 2), AdamW lr 5e-4, weight decay 0.05, batch 512 random window centres inside
  train items (edge-held), amplitude scale U(0.85, 1.15) + 0.01 noise, 6000 steps, OneCycle (5 % warm-up). The final
  checkpoint is used. No dev data in training or selection.
- REPA target: T(window) = L2-normalised e (192-d).
- Sanity on dev, run once before any planner training (the 400 dev items, data-sem-hos dev events). These are teacher gates:
  S1 GT vs mismatched windows: for every dev semantic event (deictic / iconic / metaphoric) at its segment midpoint,
     cos(e5 head(own GT window), own e5 target) > cos(e5 head(another dev clip's GT window at that clip's event time), own e5
     target). The pair accuracy must be >= 0.60 (chance 0.5). The other clip comes from a derangement with seed 20261109.
  S2 semantic detection: AUC of p(deictic) + p(iconic) + p(metaphoric) (mean over the window's patches) at dev semantic event
     midpoints vs at the dev background beat times. It must be >= 0.60.
  Reported, not gated: dev 5-class balanced patch accuracy (chance 0.2); the S1 pair accuracy with the e5 target swapped
  instead of the motion.
  If S1 or S2 fails, one pre-stated revision is allowed (window 2.4 s = 120 frames, 30 patches). If that also fails: negatives
  row, no planner lane.

## 2. Planner loss (default off; bitwise)
- Config semrepa (bool, ADDABLE false -> true), semrepa_dim 192, semrepa_block 2. The hidden state is the output of planner
  block 2 of 8, before p_norm; in the per-target causal pass (_plan_as_of) it is token (k, k). Projector = Linear(d, 2d) -
  SiLU - Linear(2d, 2d) - SiLU - Linear(2d, 192), as in REPA. It is initialised from a private torch.Generator (seed 1109), so
  the global RNG stream (data order, noise, dropout) is the same as the control's.
- StageCfg w_semrepa = lambda = 0.5 (fixed; the only value), semrepa_teacher = path of T.
  L = L_recipe + lambda x mean over valid chunks k of (1 - cos(proj(h_k^(2)), T(GT window centred on chunk k))).
  The term uses the main teacher-forced planner pass (the one whose h feeds the flow loss). GT window = the physical GT q of
  the training window (batch q x qs + qm), 80 frames centred on the chunk centre, edge-held inside the training window. A chunk
  is valid iff all 16 of its frames are valid. T is frozen (eval, no grad, fp32).
- Inference is unchanged: neither the projector nor T is used.
- Tests (scripts/tests/test_semrepa.py, must print ALL_OK before any lane):
  (1) the default Config / StageCfg gives a bitwise-identical loss dict to HEAD on a fixed batch;
  (2) semrepa = true with w_semrepa = 0 gives a recipe loss bitwise-identical to off (the capture changes no numerics, RNG untouched);
  (3) the REPA term alone gives non-zero gradient to planner blocks 1-2, the projector and the planner input, and exactly zero to
      planner blocks 3-8 and the Local Flow;
  (4) the teacher's output on a training-window GT reproduces the standalone teacher on the same frames (max abs diff < 1e-5).

## 3. Quick lane (G1 dev; DLC, one pod)
- Arms, same release and training seed 0, hos-base warm start, 2000 steps:
  - ctrl = the czmws recipe as czm2kws (cfg {"adsc": true}; sc p_stream 0.67, p_async 1, p_async_seg 0, self_force_p 0,
    causal_front / causal_spk / causal_text, w_safe 300; causal dir pre-train-slim);
  - semrepa = ctrl + cfg "semrepa": true + sc w_semrepa 0.5, semrepa_teacher T.
- Evaluation: the 400 D-spk dev clips, official ECHO-G evaluator (scripts/echog_g1/bench.sh), CFG 2.5, 10 steps, per_clip
  inference seeds 0 / 1 / 2. Protocols: offline; offline no-text (--modality audio); psync.
- pair_win: recognisers rec-sem-hos-seed0..2 (semantic-baseline-1105/g1), data-sem-hos dev split; mean over 3 recognisers x 3
  seeds (sem_multiseed.py, with recording-clustered bootstrap CIs on the paired differences). Sanity rows are scored first:
  GT (must be clearly > 0.5) and mismatched GT (each dev stem gets another dev clip's GT, derangement seed 20261109; must
  satisfy |pair_win - 0.5| <= 0.03). If a sanity row fails, pair_win is not reported.
- Text use = pair_win(offline) - pair_win(offline no-text), per arm.
- FGD = official FGD. Liveliness = scripts/echog_g1/g1_liveliness.py (still share, wrist speed, wrist amplitude vs dev GT),
  reported for both arms.
- czmws seed range (czm2kws isodev-1107, offline / psync seeds 0-2): offline FGD 1.448 / 1.477 / 1.468, psync 1.671 / 1.674 /
  1.710. If ctrl's own seeds fall outside it, the range is widened to include ctrl's seeds (stated in advance).
- Kill (any one):
  K1 pair_win(semrepa offline) - pair_win(ctrl offline) < +0.02;
  K2 semrepa mean FGD above the upper end of the range, offline (1.477) or psync (1.710). Below the range is not a kill;
  K3 text use(semrepa) <= text use(ctrl).
  Reported, not gated: per-class pair_win, bootstrap CIs, psync pair_win, liveliness. A liveliness drop of more than 10 % vs
  ctrl in wrist speed is flagged.
- Pass -> full run only on main's go: 3 training seeds, a second independently trained recogniser for scoring, and a blind
  judge.
- Risk stated in advance: T and the pair_win recognisers share the BEAT2 label source, so a pair_win gain can partly reflect
  label agreement rather than perceived semantics. It is reported next to the blind-judge check.

## Amendment 1 (main 11-09, written before any teacher-sanity or planner number): quick lane first
- First run = a true quick lane. Both arms train 2000 steps, as in section 3. Evaluation covers 100 dev clips: QUICK100 = numpy
  default_rng(20261109).choice of the sorted 400 dev stems (A/dev_stems.txt), 100 without replacement, sorted. It uses per_clip
  seed 0 only, offline + offline no-text. psync is not run.
- pair_win: the same recognisers, restricted to the events of QUICK100 (g1_recognizer.py score --allow-missing). The GT and
  mismatched-GT sanity rows use the same stems (mismatched = a derangement inside QUICK100, seed 20261109). Mean over the 3
  recognisers, seed 0.
- K2 range on QUICK100: the official FGD of czm2kws (isodev-1107 offline seeds 0 / 1 / 2) re-evaluated on QUICK100 with the
  same evaluator (bench.sh eval, no new inference). It is widened to include ctrl if ctrl falls outside. Kill if semrepa's
  offline FGD is above the upper end. K1 (Δpair_win < +0.02) and K3 (text use not above ctrl) are unchanged.
- Liveliness on QUICK100, reported.
- Only a pass leads to the 400-clip x 3-seed lane of section 3, unchanged.

## Result of the teacher gate (11-09 01:4x) and amendment 2: teacher v2 (written before any v2 number)
- v1 failed the gate twice (negatives row 60): S1 0.537 (1.6 s) / 0.563 (2.4 s) < 0.60; S2 0.707 / 0.678 passed. Diagnosis: the
  raw e5 window vectors are anisotropic (cos ~ 0.86 between any two windows), and the cosine-regression head learns the mean
  direction (cos own 0.863 = cos mismatched 0.863). As registered, no planner lane ran on v1.
- v2 changes only the word term. Target = L2-normalise(mean window e5 - mu), with mu = the mean e5 token feature over the timed
  tokens of the hos TRAIN items. Loss = symmetric InfoNCE between the normalised e5 head and the targets of the batch (in-batch
  negatives, temperature 0.07 fixed, windows without a target excluded), weight 1. Class CE, architecture (1.6 s), data, seed 7
  and the 6000 steps are unchanged.
- The same gates apply, computed with the v2 target definition: S1 >= 0.60 and S2 >= 0.60. One pre-stated revision (2.4 s)
  is allowed. If that also fails: negatives row, and SemREPA stops at the teacher.

## Result of teacher v2 (11-09 02:0x)
S1 0.581 (1.6 s) / 0.576 (2.4 s) < 0.60, so the gate FAILED; S2 0.745 / 0.699. As registered, SemREPA stops at the teacher
(negatives row 60). Sanity files: CKPT/zehan/gesture-scmg/semrepa-1109/SANITY{,2}_w{80,120}.json.

## Amendment 3 (main 11-09, written before any number of this variant): class-only teacher, SemREPA-C
- Finding kept as a result: on held-out speakers, GT arm motion picks its own words over another clip's words only 54-58 % of the
  time (S1, four teachers above).
- Teacher T3 = the v1 recipe (1.6 s, seed 7, 6000 steps) trained on the 5 frame classes only (e5 weight 0). Its REPA target is
  T3's window embedding e, the "semantic vs beat" representation. Own gate: S2 >= 0.60 on dev; S1 is not gated (no word head).
  If T3 fails S2, the pre-stated fallback is the v2 1.6 s teacher's embedding (T2_w80_s7, S2 0.745 already measured).
- Planner lane = amendment 1 (QUICK100 = g1-quick-1109/QUICK100.txt, sha256 f48ad687...; per_clip seed 0; offline + offline
  no-text; 2000 steps; same arms ctrl / semrepa; lambda 0.5, block 2). Kills K1 / K2 / K3 unchanged.
- Goodhart risk, stated in advance: the pair_win recognisers are trained on the same BEAT2 class labels as T3. A pair_win gain
  is therefore not sufficient on its own. A pass needs K1 together with the class-agnostic metrics: FGD within the range (K2),
  text use up (K3), and a liveliness wrist speed no more than 10 % below ctrl (here a gate, not only a flag). A pair_win-only
  gain is reported as "insufficient".
