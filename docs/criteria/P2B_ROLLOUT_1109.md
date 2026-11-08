# Design B quick lane: rollout-scored sem_cf (agent B, 2026-11-09; written before any run)
Source: refine-logs/FABLE_ROUND5_1109.md §2 B + main's instruction. EX002 dev only. It isolates cause (1): the outcome loss scored on
the one-step conditional-mean estimate instead of a sample.

## Claim (1) verified in code (exp/ctl-port scmg/losses.py, before the change)
- l.968: x1hat = torch.where(known[..., None], x1i, xt + (1 - t[..., None]) * v) -- the one-step estimate at the training noise
  level t.
- l.983-984: yfull = vae.decode(...x1hat...); y = yfull[:, Hx * STRIDE:(Hx + L) * STRIDE].
- l.1011: out.update(sem_desc_loss(y.float() * qs + qm, ...)). The sem_cf outcome loss is computed on the decoded one-step estimate.

## Change (default off; bitwise test)
StageCfg sem_rollout_steps (0 = off). With k > 0, the sem_desc outcome is computed on a decoded k-step Euler rollout of every
training chunk's target latents, starting from noise: CFG with weight sem_rollout_cfg = 2.0 (the inference sampler at reduced
steps). The conditional branch keeps the gradient (it reaches the ctl_ modules). The unconditional branch (null audio and text in
the planner and the flow, as the self_force CFG branch) is detached: an approximation, stated. Context latents = the instance's
known context. With self_force_p 1 that context is the model's own no-grad generated history (self-forced). The decode follows
the one-step path's decode exactly, with the rollout in place of x1hat. Every other loss is unchanged (still on the one-step
estimate).

## Quick lane (speech_server GPU1; 10 recordings ci 0, 4, ..., 36; offline seed 0; frozen enc-s0, --train-only ctl_)
- Arm R (rollout): the P2 recipe + self_force_p 1 (scope encoders; history self-generated) + sem_rollout_steps 4, 600 steps.
- Arm O (one-step, matched compute): the same recipe with sem_rollout_steps 0. Its steps = round(600 x s/step(R) / s/step(O)),
  with s/step(O) measured on a 50-step probe of the same configuration (matched wall-clock).
- Reference: the P2 model (quick-lane results and the full-run realisation numbers).
- Loss weighting under the rollout: frame weights = the valid-frame indicator (no training-noise-level trust weight, which belongs
  to the one-step estimate).
- Note (A, Step 0 pre-read): P2's one-step training sem_desc fell only 1.39 -> 1.34, while the dev rollout E(given) is 1.04. By
  Fable's Step-0 rule this points to causes 2 / 4 rather than cause 1; the lane runs as ordered by main.
- Rows per arm: offline none / given (oracle) / swapped, the P2 quick-lane plans (PLAN_oracle of p2-quick-1109).
- Primary measures: realisation ratio (realised / commanded amp_h and vpeak_h, median over the 62 events), descriptor error E
  given vs none, swap follow, FGD v2 of given (10 recordings) vs none.
- Kills (main): (K1) R's realisation ratio not above P2's full-run 0.65 (amp) / 0.70 (vpk) by >= 0.10, i.e. R needs amp >= 0.75 AND
  vpk >= 0.80; (K2) R's FGD v2 (given, 10 recordings) worse than P2's quick-lane given row (138.72). Reported: the same for arm O
  (the cause-1 isolation is R vs O at matched compute), E reduction, swap follow, liveliness.
