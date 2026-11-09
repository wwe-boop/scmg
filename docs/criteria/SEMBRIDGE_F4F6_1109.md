# SEM_BRIDGE quick lanes F4 / F6 (agent B, 2026-11-09, written before any run)
Source: refine-logs/SEM_BRIDGE_1109.md §1, families "text-motion latent alignment" (SemGes / MAG) and "contrastive negatives
inside the flow" (SemConFlow); main's specification 11-09. G1 dev only.

## Modules (default off; scripts/tests/test_f4f6.py ALL_OK on exp/ctl-port before the run)
- F4: Config f4_align (ADDABLE) creates f4_txt = Linear(d, 256) and f4_mot = MLP(CHUNK x z -> d -> 256, SiLU) from a forked RNG
  (seed 1110). StageCfg w_f4 = 0.1 (fixed). Per training chunk k (centre c_k), the text embedding is the mean of the planner's
  text tokens (after text_in, the planner's own stream incl. the ADSC visibility) that are visible to chunk k and timed within
  [c_k - 0.8, c_k + 0.8] s. The motion embedding is f4_mot of the GT chunk latents (the normalised VAE latents of the target
  chunk). Loss = symmetric InfoNCE over all valid (b, k) pairs of the batch (at least one word, all frames valid), in-batch
  negatives, tau 0.07. Computed after every recipe term (recipe terms and RNG unchanged, test 3).
- F6: StageCfg w_f6 = lambda = 0.05 (fixed), f6_margin = 2.0. A second planner + Local Flow pass runs with mismatched conditions:
  text tokens, text mask / visibility / times, audio encoding / mask and the Local Flow audio window are taken from batch item
  b - 1 (roll). Speaker, history, noise, x_t, t and stream state are the item's own. Per instance i: e_pos_i / e_mis_i = the
  supervised velocity MSE under the own / mismatched conditions. Term = mean_i relu(2.0 x e_pos_i (detached) - e_mis_i): it pushes
  the mismatched prediction away from u_t until its error is twice the matched one, then stops (the margin prevents
  divergence). The ratio e_mis / e_pos is logged. Computed last (test 4).

## Quick lane (DLC, one pod, 2 GPUs)
- Arms f4 and f6 = the czm2kws recipe (hos-base warm start, 2000 steps, training seed 0, w_safe 300, causal_front / spk /
  text, pre-train-slim) + the module. The control is the ctrl arm of zehan-scmg-g1-quick-1109 (dlc23ffnvfw7kain): same recipe and
  seed, same evaluation; the default path is bitwise between the two releases (test 1). The GT / mismatched-GT pair_win sanity
  rows and the czm2kws QUICK100 FGD range come from that job as well.
- Evaluation: QUICK100 (g1-quick-1109/QUICK100.txt, sha256 f48ad687...), per_clip seed 0, offline + offline no-text
  (--modality audio), official evaluator (CFG 2.5, 10 steps). pair_win = rec-sem-hos-seed0..2 on the QUICK100 events, mean of
  the 3 recognisers. Text use = pair_win(offline) - pair_win(no-text). Liveliness = g1_liveliness.py (reported).
- Kill, per arm (any one): K1 pair_win - pair_win(ctrl) < +0.02; K2 offline FGD above the upper end of the czm2kws QUICK100
  range (widened to include ctrl if ctrl falls outside); K3 text use not above ctrl's.
- Prior expectation for F4 is low: SemREPA S1 found word identity only weakly recoverable from arm motion (0.54-0.58).
- Pass -> the 400-clip x 3-seed lane on main's go.
