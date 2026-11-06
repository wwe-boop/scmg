# Div / BA / perceived-quality candidates: acceptance criteria (fixed 2026-11-06, before any candidate run)

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

## Criteria (a candidate passes only if all hold, psync)

| | quantity | pass if |
|---|---|---|
| C1 synchrony | BA lift = BA − BA_null (shifted-onset null, K = 20) | lift ≥ control lift + 0.0015 **and** BA_null ≤ control BA_null + 0.01, so the gain cannot come from density |
| C2 sharpness | L1 distance between the normalised band-power vectors (0-1/1-2/2-4/4-6/6-10/10-15 Hz) of the mean mmae-normalised upper-body speed, pred vs GT. The measure is scale-free, so it is independent of energy. | distance ≤ 0.7 × control (control psync 0.299, offline 0.239) **and** total power ≤ 1.10 × control |
| G1 FGD | official FGD | ≤ control + 0.03 |
| G2 Div | official Div | \|Div − Div_control\| ≤ 0.05, **and** the psync − offline Div gap is not larger than control's |
| G3 Jerk | official JerkLW | \|JerkLW − GT\| ≤ control's \|JerkLW − GT\|. The raw ΔJerk (JerkLW − GT) is also reported. |
| G4 safety | after CBF-QP (`cbf_qp`, K = 10): clip-level executability, plus the correction amount (RMSE mean / p99 / max vs the raw output) and the fallback-frame share | executability is not lower than control. Mean correction RMSE ≤ 1.2 × control. Fallback share is not higher than control + 0.5 pp. More safety-layer intervention counts as a side effect. |
| G5 tests | causal unit tests (`test_causal_stream`, `test_causal_train`, `test_causal_inc`) and the candidate's own unit tests | all ALL_OK. Every training option is default-off, with bitwise-unchanged training and inference when off. |

- **Reported but not used for selection:** official BA and Div, beats/s, the phase histogram, per-speaker numbers.
- **Offline:** offline numbers are reported. A candidate that passes on psync but fails C1 or C2 in the wrong direction offline is reported as "psync only".
- **Training candidates (all default-off, 2000 steps):**
  - (a') two-sided spectrum-shape matching on the one-step x0 estimate;
  - (b') onset-conditioned beat loss. Onsets are taken only from **training-set** audio. Inference needs no extra input.
  - (c) chunk-boundary consistency is dropped: there is no boundary artifact to fix.
- **Rendering:** each passing method gets 6 dev clips rendered side by side (control / candidate / ECHO-G), official MuJoCo for G1, under `/mnt/bulk/zehan/renders/`.
