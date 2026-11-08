# G1 encoder-only self-forcing quick lane: pre-registration (agent B, 2026-11-09, before any run)
Source: refine-logs/FABLE_THEORY_1109.md §1.2 and main's queue item. G1 dev only (no val_common, no sealed sets).

- Arms: the czmws recipe as czm2kws (hos-base warm start, 2000 steps, training seed 0; cfg {"adsc": true}; sc p_stream 0.67,
  p_async 1, p_async_seg 0, causal_front / causal_spk / causal_text, w_safe 300; causal dir pre-train-slim). All three arms
  come from one release:
  sf0  = self_force_p 0 (the same run as the SemREPA ctrl arm);
  full = self_force_p 0.5, self_force_steps 4, self_force_rounds 1, self_force_cfg 0, mix 0, noise 0, scope 'all';
  enc  = the same with self_force_scope 'encoders' (the EX002 enc-s0 SF settings).
- Evaluation: QUICK100 (g1-quick-1109/QUICK100.txt, sha256 f48ad687...), per_clip seed 0, official ECHO-G evaluator (CFG 2.5,
  10 steps), offline + psync. Liveliness = scripts/echog_g1/g1_liveliness.py, normalised by the 400-clip dev GT: still share,
  wrist speed, wrist amplitude, each vs GT. Stutter = scripts/echog_g1/beat_phase.py: max / min of the 10-bin chunk-phase beat
  histogram vs GT. The history-source ablation (§1.2 iii) is not part of the quick lane.
- czmws range on QUICK100 = czm2kws isodev-1107 seeds 0-2 re-evaluated on QUICK100 (offline and psync), widened to include sf0
  if sf0 falls outside.
- Kill (any): Ka enc wrist-speed ratio (vs GT, psync) not above full's; Kb enc FGD above the upper end of the range, offline or
  psync.
- Reported, not gated: still share and amplitude, stutter of enc vs sf0 and GT, dBA / dDiv / dJerk, and full vs sf0.
- Pass -> the 3-seed DLC lane on main's go (the same hos dev protocol, 400 clips, 3 training seeds x per_clip 0 / 1),
  with "confirmed cross-platform" iff psync FGD <= sf0 within the seed range and the stutter at the sf0 level.

## Quick-lane numbers: filed only (descoped by the user 11-09 02:2x; dlc23ffnvfw7kain finished 02:25)
QUICK100 seed 0, official FGD offline / psync: sf0 (ctrl) 1.636 / 1.763; full 1.569 / 1.589; enc 1.594 / 1.751. czm2kws QUICK100
range: offline [1.690, 1.712] -> widened to include sf0 [1.636, 1.712]; psync [1.733, 1.900]. Wrist speed (m/s) offline / psync:
sf0 0.207 / 0.197, full 0.239 / 0.267, enc 0.210 / 0.306; GT 0.236. Still share psync: sf0 0.068, full 0.040, enc 0.015; GT 0.036.
Stutter max / min, offline / psync: sf0 1.26 / 1.30, full 1.57 / 1.72, enc 1.30 / 1.30; GT 1.21.
Read against the registered kills: Ka is not hit (enc's psync wrist ratio 1.29 x GT, above full's 1.13 x). Kb is not hit either:
enc is inside the range on both protocols. Enc keeps the sf0-level stutter (full SF's is 1.57 / 1.72), but its psync FGD is
worse than full's. No verdict and no follow-up (descoped).

## Verdict against the registered kills (11-09, user: "备选的G1可以推进"; scored from the filed numbers above, nothing re-run)
- Ka (enc wrist-speed ratio vs GT, psync, not above full's): enc 0.306 / 0.236 = 1.29x, full 0.267 / 0.236 = 1.13x -> NOT hit.
- Kb (enc FGD above the upper end of the range): offline 1.594 <= 1.712 and psync 1.751 <= 1.900 -> NOT hit.
-> QUICK LANE PASS (seed 0, QUICK100).
Caveats, stated with the verdict:
- enc is above GT on psync liveliness: wrist speed 1.29x GT, still share 0.015 vs GT 0.036. Offline it is below GT (0.210 = 0.89x).
- full SF has the best psync FGD (1.589 vs enc 1.751, sf0 1.763), but the worst chunk stutter (1.57 / 1.72 vs GT 1.21). enc keeps
  sf0's stutter (1.30 / 1.30).
- One training seed and one inference seed on 100 clips; the psync czm2kws range is wide (1.733-1.900).
- History-source ablation (§1.2 iii) not run: the inference has no teacher-forced-history mode, so it is not cheap.
- pair_win not read (invalid on 100 clips).
