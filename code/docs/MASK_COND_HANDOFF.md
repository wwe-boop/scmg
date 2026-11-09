# Mask-conditioned Local Flow and flagged known latents: handoff note (2026-11-07, exp/text-path)

Use this to reuse the switch without editing these files. Everything is off by default. With the switches off, the
training and inference paths are bitwise identical to before. Tested by test_ctl.py A/B against HEAD.

## Switches
| Where | Name | Meaning |
|---|---|---|
| model `Config` (`scmg/model.py`) | `mask_cond: bool = False` | Adds `f_mask`, a zero-initialised `nn.Parameter(d)`. `flow_velocity(..., ins=None)` adds `ins[..., None] * f_mask` to the Local Flow input at flagged positions. With `ins=None`, or without `f_mask`, nothing changes. |
| `scmg/train.py` | `ADDABLE` contains `mask_cond` | You can switch it on when fine-tuning from any checkpoint with `--cfg '{"mask_cond": true, ...}'`. The new parameter starts at 0. |
| `StageCfg` (`scmg/losses.py`) | `p_mask: float = 0.0`, `mask_span: tuple = (0.5, 2.0)` | This is the probability per sample. It places one random clean latent span of `mask_span` seconds anywhere in the training window. In every chunk instance (context or target positions) the span's latents are known: clean `x1`, `t = 1`, flagged by `ins`, and not in the flow loss. The RNG draws happen only when `p_mask > 0`. |

## Checkpoint
- Dev quick lane trained with the sf0 recipe: hos-base warm start, 2000 steps, `--cfg '{"adsc": true, "mask_cond": true}'`, `--sc` = sf0 plus `"p_mask": 0.3`.
- Paths: local `~/bulk/ctl-1105/local/mc/train/final.pt`, CPFS `/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/mc/final.pt` (same size, 932926803 B).
- Training log: `~/bulk/ctl-1105/local/mc/train.log`. At step 2000: flow 0.599, dev_flow 0.836.

## Placing flagged known latents in the Streamer (`scmg/infer.py`)
`Streamer(..., inpaint=dict(lat=..., w=..., seed=...))`:
- `lat`: `[N, z]` target latents on the GLOBAL latent grid (latent i = frames [4i, 4i+4) at 50 Hz), normalised with the
  generator's `(mu - model.lat_mean) / model.lat_std`. Encode frames with `vae.encode(((frames - q_mean) / q_std)[None])`.
- `w`: `[N]` weights in [0, 1]. `seed`: seed of a separate noise generator, so the main noise stream is unchanged.
- On a `mask_cond` model, positions with `w >= 0.999` become clean known latents with `ins = True`, exactly as in training.
  Positions with `0 < w < 1` are inpainted RePaint-style. The start value is `w eps + (1 - w) x`. After each Euler step the value becomes
  `w (t' lat + (1 - t') eps) + (1 - w) x`.
- On a model without `mask_cond`, every `w > 0` position uses the RePaint rule. At `w = 1` the committed latent equals the target exactly.
- You can set or replace `self.inpaint` at any tick, for example at the interruption. Copy the format, including `g`, a torch
  Generator, from the constructor: `scripts/semantic/interrupt_mc.py` `IntStreamer._apply` does exactly this for the rest
  pose. It encodes the last 1.28 s of emitted frames plus the held rest target. The global latent of encoding latent j is
  `len(self.lat) - 1 - len(hist) / 4 + j`, because emitted frames lag the committed latents by one latent. It sets `w = 1`
  from latent `ceil((t_i + T_r) * 12.5)` on.
- All CFG branches receive the same `ins`, because the insertion is a constraint, not a condition.

## Tests
- `scripts/semantic/test_maskcond.py`:
  - with `p_mask 0` the losses are bitwise equal to the base model;
  - with `p_mask 1` the losses are finite and the gradient reaches `f_mask`;
  - without insertion, offline and engine inference are bitwise equal to the base model.
- `scripts/semantic/test_lib.py`:
  - zero weights give bitwise default sampling;
  - locality: nothing changes before the first window that can see the weights;
  - weight-1 committed latents equal the target exactly.
- `scripts/semantic/test_ctl.py --ctl <ctl-tokens-hos> --orig <git archive HEAD>`: the default training and inference paths are bitwise
  equal to HEAD.
- Environment: `SCMG_J=39 SCMG_NO_GLITCH=1 SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror SCMG_TEXT_CAUSAL=<causal e5 cache>`.

## Adding it to the 6000-step final recipe
You can add `p_mask` and `mask_span` to that recipe's `--sc` as they are, and `mask_cond` to its `--cfg`. The block only adds known
positions after the recipe's own `known` mask is built: outpaint, `p_noctx`, bagger, `has_ctx`. It needs no other field. Three
interactions to know:
1. With `ctx_noise_p > 0`, inserted CONTEXT positions can be drawn as noisy context like any other known context position.
   Inserted TARGET positions stay clean.
2. With self-forcing, context latents are the self-forced history. The flag marks positions, not values, so the inserted
   context then holds the self-forced latent.
3. Only the main `flow_velocity` call of the flow loss receives `ins`. The extra sem-head pass (cfg `sem_head` with
   `sem_w`) does not. That is harmless for the flow loss, but those branches see the inserted positions as plain known positions.

Agent A's causal-input ("psync") protocol lives in another worktree. `interrupt_mc.py` uses this branch's engine protocol. If the
twin model is trained with causal inputs, port only the `inpaint` handling into that Streamer.
