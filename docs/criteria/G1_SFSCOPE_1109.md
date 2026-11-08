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
