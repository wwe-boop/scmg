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
