# Same-utterance figure (2026-11-09)

## Input
- Sentence: s1 of renders/final-1109/sentences.json, recorded with TTS robot_service_v1.
- Barge-in at t_i = 5.28 s, which is the end of "随". After t_i the audio is zero up to 7.50 s, and the unspoken words are withdrawn.
- G1 baselines get a 16 kHz wav (audio/s1int.wav).
- ECHO-G additionally gets the words that end at or before t_i (utterances.jsonl).
- EX002 replays a recorded TTS stream. The stream is cut the same way: pcm is set to zero after t_i, and anchors after t_i are dropped.

## Systems
| row | robot | how it was run |
|---|---|---|
| ECHO-G | G1 | Official code and weights (best.pt, audio + text). Run with echo-g-extract-conditions (audio, text, merge) and echo-g-infer, seed 0, 8 steps, on zehan-server. |
| RoboGesture | G1 | Official weights. rg_one.py runs the same calls as baseline-repro-1104/scripts/rg_generate.py, with the parity check on sample_06. It uses the official filtered branch. The output is converted to G1 39-d with scripts/baselines_g1/rg_to_g1.py (225 frames). |
| SCMG | EX002 | See the steps below. |

SCMG steps:
1. Run the enc-s0 deploy worker (scripts/tts_deploy_replay.py) on the cut stream. This gives the continuation.
2. Take the uninterrupted frames from renders/final-1109/work/ex002/canon/1.npy. These are explicit semantics with the YOU keyframe at 3.56 s.
3. Run scripts/ex002/retract_post_ex002.py r0causal with --human ref-halfB-v2.
4. Run scripts/ex002/safety_int_ex002.py --isc, which is 3c-4b. It reports 0 fallback frames.

Causality check: the cut replay equals the uninterrupted replay bitwise up to frame 268, at 5.36 s.

## Renders
- G1: official ECHO-G MuJoCo renderer, run with --fixed-camera --distance 2.0 --elevation -8.
- EX002: Isaac Sim, 3/4 view.
- Keyframes are taken at 1.00, 3.88, 5.28, 6.00 and 7.20 s.

## Wrist speed (wrist_speed.json)
- For each frame, the faster of the two wrists.
- G1 uses the official g1_forward_kinematics with a fixed root, at 30 Hz.
- EX002 uses scmg/kinematics through lively_floor_ex002.pts, at 50 Hz.

| row | mean speed, 0.5–5.28 s (m/s) | mean speed, 6.28–7.4 s (m/s) | max speed, 6.28–7.4 s (m/s) |
|---|---|---|---|
| ECHO-G | 0.085 | 0.135 | 0.325 |
| RoboGesture | 0.028 | 0.244 | 0.378 |
| SCMG | 0.607 | 0.050 | 0.129 |
