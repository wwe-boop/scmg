# Div / BA / perceived-quality candidates: acceptance criteria (fixed 2026-11-06, before any candidate run; joint version 11-06, still before any candidate result)

The coordinator approved these on 11-06. They replace the first draft's "|ΔBA| down >= 30%" criterion.

## Why the criteria changed (diagnosis: `diag_beats.py`, czm6k dev, 400 clips)

**Official BA is mostly beat density.** BA against the clip's own audio onsets, circularly shifted (`BA_null`), is:

| | official BA | BA_null | BA lift |
|---|---|---|---|
| GT | 0.6311 | 0.6268 | +0.0043 |
| offline | 0.6003 | 0.5975 | +0.0028 |
| psync | 0.5895 | 0.5880 | +0.0015 |

The clip-bootstrap SE of the lift is 0.0007. "Raise dev BA" therefore means "add beats".

**The density gap reverses between splits.** On dev we are below GT. On test, psync BA is 0.595 against GT 0.534, so we are above it. This is the same speaker-energy trap as Div (design doc §7.28): the model outputs one speaker-average energy, and the dev and test speakers differ in energy.

**On dev there are no extra beats and no chunk artifact:**
- Beats within ±1 frame of a 320 ms boundary: 19% for us, 20.7% for GT, about 21% by chance.
- Instead the motion is too smooth. Speed power above 2 Hz is 0.25–0.5× GT.
- Share of spectral power above 2 Hz: GT 26.3%, offline 17.4%, psync 14.8%.

## Protocol

- **Data:** dev split only (D-spk view `echog-g1-v1-hos`, 400 dev clips, `causal_g1.sh dev_env`). The test set is not touched.
- **Protocols and seeds:** psync is the main protocol and offline is secondary. CFG 2.5 and 10 steps, unless the candidate itself changes steps. Inference seeds are 0 and 1, and every criterion is evaluated on the 2-seed mean.
- **Control:**
  - Zero-training candidates: czm6k dev with default inference.
  - Training candidates: a 2000-step lane from hos-base with the same causal recipe, the same seed and the new loss off.
- **Evaluation:** official `eval_g1_motion_cls.py` (FGD, Div, BA, Jerk), `diag_beats.py` (BA lift, BA_null, spectrum), `executability.py` and `safety_run.py` with the `cbf_qp` layer (K = 10).

## Criteria: joint version (user 11-06; replaces the per-criterion version above, fixed before any candidate result is read)

BA, Div and smoothness are optimised jointly. All numbers are on dev, psync, as the mean of inference seeds 0 and 1. Offline numbers are reported alongside.

**Three objectives**

| | objective | better if |
|---|---|---|
| T1 sync | BA lift = BA − BA_null (onsets circularly shifted, K = 20) | higher |
| T2 clarity | spectrum-shape L1 distance to GT (`diag_beats.py` `shape_l1`) | lower |
| T3 amplitude | gap = \|Div_psync − Div_offline\| | lower |

- For T3, the amplitude constraint is \|Div − Div_control\| ≤ 0.05 per protocol.
- Absolute Div is never tuned on dev (§7.28).

**Noise unit u_i, one per objective, computed from the control's own runs only and frozen in `NOISE_UNITS_<control>.json` before any candidate result is read:**
- u_i = 2 × sqrt(SE_boot² + (x_seed0 − x_seed1)² / 4).
- SE_boot is the clip-bootstrap SE (2000 resamples, from `diag_beats.py`) of the control's metric. For the T3 gap, SE_boot = sqrt(SE_psync² + SE_offline²) of Div.

**Admission.** A candidate is admitted only if all of the following hold:
1. All guards pass:
   - G1: FGD ≤ control + 0.03 (psync).
   - G3: \|JerkLW − GT\| ≤ control's value. The raw ΔJerk = JerkLW − GT is reported for both protocols.
   - G4: after CBF-QP (`cbf_qp`, K = 10):
     - clip executability is not lower than control;
     - mean correction RMSE ≤ 1.2 × control;
     - fallback-frame share ≤ control + 0.5 pp;
     - correction p99 / max are reported.
   - G5: causal unit tests and the candidate's own tests are ALL_OK. The default path is bit-identical.
   - Amplitude constraint: \|Div − Div_control\| ≤ 0.05 for both psync and offline.
2. No objective is worse than control by more than one noise unit u_i.
3. At least one objective reaches its planned improvement:
   - T1: lift ≥ control + 0.0015, or
   - T2: shape_l1 ≤ 0.7 × control.

**Ranking.** Admitted candidates are ranked by score = Σ_i (improvement_i / u_i) over T1, T2 and T3, with the sign chosen so that improvement is positive. The highest score wins. A tie (score difference < 1) goes to the lower psync FGD.

**Other rules**
- **Combinations** (e.g. decoder spectrum loss + (b')) run as their own quick lane and are judged by the same rules.
- **Decoder-only candidates** (decoder fine-tuned with a multi-band speed-spectrum loss, encoder frozen) must also keep the dev VAE round trip within limits: joint-angle RMSE ≤ 1.05 × the current decoder (decft-w3d0m05), and contact sliding not worse. The generator is unchanged, so the control is the same generator with the current decoder.
- **Controls:**
  - zero-training candidates (16 steps, noise scale, the stochastic sampler `--sde-eta`): czm6k dev with default inference;
  - training candidates: the 2000-step hos-base lane with the same recipe and the new loss off;
  - decoder candidates: the generator under test with decft-w3d0m05.

- **Reported but not used for selection:** official BA and Div, beats/s, the phase histogram, per-speaker numbers.
- **Offline:** offline numbers are reported. A candidate that passes on psync but fails C1 or C2 in the wrong direction offline is reported as "psync only".
- **Training candidates (all default-off, 2000 steps):**
  - (a') two-sided spectrum-shape matching on the one-step x0 estimate;
  - (b') onset-conditioned beat loss. Onsets are taken only from **training-set** audio. Inference needs no extra input.
  - (c) chunk-boundary consistency is dropped: there is no boundary artifact to fix.
- **Rendering:** each passing method gets 6 dev clips rendered side by side (control / candidate / ECHO-G), official MuJoCo for G1, under `/mnt/bulk/zehan/renders/`.
