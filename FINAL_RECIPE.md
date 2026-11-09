# SCMG — final confirmed implementation (paper baseline, 2026-11-09)

This is branch `paper/final-1109`: exp/ctl-port + exp/duplex + exp/text-path + exp/safe-proj, merged. On 11-09 it was re-merged with exp/ctl-port 76d6281 and exp/text-path bd0e6b0 to bring in the explicit-semantics path.
- Every file equals the version at one of the four branch tips. There was 1 conflict, scripts/ex002/ex002_local_blend.py, resolved to the ctl-port version, which already contains the 5c port.
- The verdict scripts reproduce every reported verdict on this tree:
  - scripts/eval/beat_ex002.py (base selection, safety, interruption, yield metrics);
  - scripts/eval/beat_ex002_ctl.py (control / keyframes);
  - scripts/eval/beat_g1.py (G1 main table).
- Every non-final module stays in the tree behind a default-off switch, with its negative result in docs/ and the ops negatives table.

## G1 (Unitree G1, ECHO-G benchmark: main table)
- Model czmws: causal training match (czm) + 6000 steps + self_force_p 0 + w_safe 300. Release 20261106-084856 (5a3a96f).
- Protocols:
  - offline;
  - psync: a word becomes visible after it is spoken; audio prefix only; 160 ms look-ahead.
- Seeds: per_clip inference seeds 0 / 1.
- Safety: CBF-QP layer (scmg/safety.py); executability on vc_every8.
- Results: refine-logs / runs/FINAL_TABLE_G1_ABL_1108.{md,json}, with ablations (ncws, train / inference no-text and no-audio). The paired bootstrap vs czmws is pending.

## EX002 (three-finger humanoid, data-v36_1, vae-r3, 50 Hz)
- **Base enc-s0.** Current code + `self_force_scope encoders`, sibag recipe, 2000 steps, 3 training seeds (EX002_SFSCOPE_1108.md). Causal training match is not used on EX002 (it failed the rule).
- **Safety layer 3c-4b.** Recursive-feasible chunk projection: qdldl, tail_smooth 1e-3, memoised linearisation, cut cap 8 (EX002_CBF_1108.md).
  - Gates 1–3 pass, zero fallback, bitwise reproducible.
  - Gate 4 latency fails (p99 184 ms on 1 CPU thread).
- **Interruption.** R0causal retraction + 3c-4b + ISC (interrupt-aware safety commit), evaluated against the half-B human tails (EX002_INTERRUPT_1107.md, EX002_DUPLEX_LADDER_1107.md). There is no single yield gate: the yield metric is invalid (EVAL_AUDIT_1108.md).
- **Control (token path) tok10ab.** ctl modules trained 4000 steps (true freeze) + human hand-shape library override + 5c blend (EX002_CTL.md, module 10).
- **Keyframe insertion 9c''.** Real-deictic hold replay + release-time search + signed palm fix. NOT final: it fails only the apex-angle gate. The port to tok10ab is running.
- **Duplex ladder** (L1 YIELD token, L2 rest keyframe): in progress, not final.
- **Explicit semantics (canonical path, final).** scripts/ex002/expl_canon.py. Lexicon → IK keyframes / count override:
  - SELF: chest keyframe;
  - YOU / POINT: point keyframe;
  - NUMBER: count + present keyframe, 1.2 s hold;
  - BIG / SMALL: keyframes.
  - Rate rule: ≤ 1 per sentence, ≥ 3 s spacing, SELF only on an object or possessive.
  - Full dev appropriateness: SELF 0.98, YOU 0.97, POINT 0.84, NUMBER 1.00, BIG 0.74, SMALL 0.50. FGD v2 is not worse, executability is 100 %, and the renders have been checked.
  - Criteria: docs/criteria/EXPLICIT_SEM_1109.md, amendments 2–6.

## In-network innovation search: stopped by the user on 11-09
- Five lanes were killed: P1 yield_cf, P1b-SW, P2 sem_cf (fails G3 FGD by 0.97), Design B (rollout-scored) and W-low. See the negatives rows and docs/criteria/P2*_1109.md.
- Design A (keyframe-conditioned rollout, exp/keyframe) and the stage-2 gated LoRA (exp/glora) were descoped before they gave any result. They are NOT merged.
- All of this code stays default-off.

## Evaluation
- refine-logs/EVAL_PROTOCOL.md (draft, not frozen) and refine-logs/EVAL_AUDIT_1108.md. The audit's reading rules are binding.
- Win rules in code: scripts/eval/beat_*.py.
- Novelty: refine-logs/NOVELTY_CHECK_1109.md. Designs: refine-logs/FABLE_DESIGN_1108.md.
- EX002 test-set runs: NOT done yet (protocol freeze first).
