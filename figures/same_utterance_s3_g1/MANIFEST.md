# Same-utterance figure, s3, all rows on G1 (2026-11-09)

Rows 1–2 and the inputs are the same as ../same_utterance_s3.

## Row 3: SCMG on G1
- Model: czmws-s0 (release 20261106-084856) with decoder decft-w3d0m05.
- Inference: scripts/echog_g1/infer_test.py --protocol psync, CFG 2.5, 10 steps, per_clip seed 0.
- Inputs: the cut s3, built by renders/final-1109/code/g1_build_tts.py under the same stem name as the uninterrupted run, chinese_tts_final1109_s3_utt_0000.
- Retraction: scripts/interrupt/blend_retract.py with the frozen R0causal recipe --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1.
  - Prefix: the uninterrupted psync output of renders/final-1109/work/g1.
  - Human reference: human_ref_v2_train_drop.json.
  - f_i = 130 (30 fps).
- Safety: scripts/echog_g1/safety_run.py --layers cbf_qp --rg-pairs. The run reports 0 fallback chunks and outputs 194 frames.
  - For rendering, real_num_frames was set to 194.
- Rendering: official renderer, same camera as rows 1–2, same crop.

## Facts recorded during the run
- Before f_i, the continuation run is not bitwise equal to the uninterrupted run.
  - The q50 stream differs from frame 12.
  - The 30 fps output differs from frame 1, with a max of 0.0018 rad before frame 125.
  - The figure uses the uninterrupted frames before f_i, as R0causal defines.
- The SCMG G1 output includes root yaw. In the frames around and after t_i, the body is turned about 45° away from the fixed camera.
  - ECHO-G keeps the root near the start pose.
  - RoboGesture has a fixed root.

## Wrist speed of SCMG on G1 (m/s)
| mean, 0.5–4.32 s | mean, 5.32–6.4 s | max, 5.32–6.4 s | max, whole clip |
|---|---|---|---|
| 0.348 | 0.182 | 0.388 | 1.516 |
