# Inference seed behaviour of every benchmarked method (2026-11-06, read-only code check)

Baselines are run as their official implementations do. The SCMG protocol changed on 11-06: from then on every new SCMG evaluation uses `--seed-mode per_clip` (exp/divba ebb03ae).

| method | stochastic at inference? | where the seed is set | do clips of one run share noise? | seeds we ran |
|---|---|---|---|---|
| **ECHO-G official** (`echo_g.inference`) | Yes: initial flow noise, then a deterministic Euler ODE (`flow.py:86-130`) | Once per clip: `torch.Generator().manual_seed(stable_seed(stem, seed))`, with `stable_seed = sha256(f"{stem}:{seed}")[:8] % (2^31-1)` (`inference.py:353-356`, `utils.py:31-33`) | **No**. Independent per clip; does not depend on batching or order | 0, 1 on val_common; 0..19 on the 300-clip subset (Multimodality) |
| **ECHO-G, our stream wrapper** (official_offline / stream_prefix) | Yes, same noise | `stream_infer.py:152-154`: the same per-clip `stable_seed(stem, seed)`. Each window takes slices of that clip's noise; `--offline` equals `echo_g.inference` | **No** | 0, 1 |
| **SemTalk** | **No**: codes and gate are argmax, model in eval mode (`semtalk_infer.py:187-188, 201, 144`) | `torch.manual_seed(0)` once per process (`semtalk_infer.py:309`). It has no effect on the output | n/a (deterministic) | one run. "seed0" in the directory name is only a label |
| **EMAGE** (`emage_audio`) | **No**: argmax decoding (`modeling_emage_audio.py:398-401, 445-448`; `emage_infer.py:83-86`) | none | n/a (deterministic) | one run. "seed0" is only a label |
| **RoboGesture** | Yes: `torch.randn(1,30,41)` per 30-frame block, 50 Euler steps (`model.py:358-359`) | **Reset at every utterance**: official `MotionGenerator.generate` calls `torch.manual_seed(0)` at its start (`model.py:326-328`). Our `rg_generate.py` changes only the seed value | **Yes**. Every clip replays the same noise sequence (same as SCMG's old `run` mode) | 0, 1 on the 300-clip subset |
| **GestureLSM** (shortcut-reflow) | Yes: `torch.randn` per window on the global RNG (`LSM.py:292`) | Official: once per process (`set_random_seed`, `random_seed` default 2021), and the state advances across clips. **Our wrapper differs**: it reseeds each clip with `seed*1000003 + crc32(stem)` (`gmr-repro/scripts/lsm_infer.py:176`, `lat_emage_lsm.py:118`) | **No** (official: no either, but depends on clip order) | 0 on val_common; 0..19 on the 300-clip subset |
| **SCMG (ours)**, `--seed-mode run` (old default; all results before 11-06) | Yes: flow noise (plus SDE noise if `--sde-eta` is set) | A new `torch.Generator().manual_seed(seed)` per clip (`generate()` → new `Streamer`) | **Yes**. Every clip starts from the same generator state, so chunk k gets the same initial noise in every clip | 0, 1 |
| **SCMG (ours)**, `--seed-mode per_clip` (11-06 onward) | same | `seed + crc32(stem)` per clip (`infer_test.py clip_seed`) | **No**. Independent and reproducible; does not depend on clip order, the other clips or the shard (`scripts/tests/test_seed_mode.py`) | 0, 1 |

**Why it matters.** Under shared noise (SCMG `run` mode, RoboGesture), one inference seed is effectively a single noise sample for the whole set, not 400 independent ones. The czm6k dev control (old mode) shows how large the seed-to-seed swing is:

| metric | seed 0 | seed 1 |
|---|---|---|
| psync spectrum-shape L1 | 0.299 | 0.121 |
| offline Div | 1.372 | 1.572 |
| psync BA | 0.587 | 0.629 |
| psync JerkLW | 38.7 | 42.0 |

**Not determinable from code:**
- Whether RoboGesture's Mimi streaming encoder draws RNG between blocks.
- GPU-level nondeterminism (cuDNN / atomics) of the deterministic methods.
- Which `random_seed` value the official GestureLSM demo config ends up with. `seed: 42` in its yaml is the training key.

**GestureLSM decision (coordinator, 11-06): not rerun, disclosed in the appendix.**
- Official GestureLSM seeds the RNG once per process (`utils/other_tools.py:746-753` via `demo.py:575`) and lets it advance across clips.
- Our wrapper reseeds each clip (`gmr-repro/scripts/lsm_infer.py:176`).
- In both cases the noise of each clip is an independent draw from the same distribution, so the generated distribution is identical. Only the particular random numbers differ, and fairness is unaffected.

**Draft appendix paragraph.** EMAGE and SemTalk decode their VQ codes by argmax and draw no noise at inference, so each clip has exactly one output. ECHO-G seeds its initial flow noise per clip with a hash of (clip name, sample seed), so clips receive independent noise regardless of batching; our streaming variant draws its per-window noise from the same per-clip generator. The official RoboGesture generator resets the global RNG to a fixed seed at the start of every utterance, so all clips of a run replay the same noise sequence; we keep this and vary only the seed value. GestureLSM officially seeds once per process and lets the generator advance across clips; our wrapper instead reseeds each clip with seed·1000003 + CRC32 of the clip name. Both give independent, identically distributed noise per clip, so the output distribution is the same. For SCMG, all reported evaluations use a per-clip protocol: each clip's noise seed is the run seed plus the CRC32 of its name. This replaces an earlier setting in which every clip of a run started from the same generator state.
