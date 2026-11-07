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
(median 9e-4 rad at the first differing frame, up to 0.139 rad). Rule: B3 rows are scored on the EXECUTED prefix -- the method's
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
