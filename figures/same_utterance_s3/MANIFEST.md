# Same-utterance figure, s3 version (2026-11-09)

Pipeline: the same as ../same_utterance/MANIFEST.md, with these changes.

## Input
- Sentence: s3, "在这个展厅里，我们一共准备了三个互动体验区，欢迎大家逐一体验。"
- Inputs are built by build_inputs.py.
- Barge-in at t_i = 4.32 s, the end of "欢迎". The audio is zero from t_i to 6.50 s.

## EX002 (SCMG)
- U is renders/final-1109/work/ex002/canon/3.npy. It has the NUMBER = 3 keyframe at 2.12 s, from amendment 6 (finger hold 0.2 / 1.0 s).
- Causality check: the cut replay equals the uninterrupted replay bitwise up to frame 220 (4.40 s).
- r0causal (ref-halfB-v2), then ISC 3c-4b. The ISC step reports 0 fallback frames.

## Renders
- All rows use a front view.
- Crops are 450 × 450 px, with the head top at the same height. G1 is cropped at (145, 110, 595, 560) and EX002 at (165, 170, 615, 620).
- At head-top to shoulder, both robots measure about 100–110 px in their frames.
- Inset: the counting hand at 2.6 s, crop (455, 185, 615, 345).

## Wrist speed (m/s)
| row | mean, 0.5–4.32 s | mean, 5.32–6.4 s | max, 5.32–6.4 s | max, whole clip |
|---|---|---|---|---|
| ECHO-G | 0.122 | 0.072 | 0.171 | 0.721 |
| RoboGesture | 0.031 | 0.055 | 0.078 | 0.407 |
| SCMG | 0.345 | 0.033 | 0.117 | 0.943 |
