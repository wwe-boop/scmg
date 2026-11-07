# §7.55 semantic-line quick lanes: pass criteria (written 2026-11-07 before any run of these modules)

Protocol (all three items): our architecture, sf0 recipe (stream training, p_async 1, p_async_seg 0, self_force_p 0), hos-base warm
start, 2000 steps, held-out-speaker dev (400 clips); inference offline, CFG 2.5, 10 steps, `--seed-mode per_clip`, inference seeds 0/1/2.
Control = sf0 under the same protocol (FGD 1.447 / 1.453 / 1.452 → 1.447; pair_win 0.530; no-text 0.527 / FGD 1.563).
Recognisers: rec-sem-hos-seed0..2 (train split only); pair_win pooled over seeds, recording-clustered bootstrap (sem_multiseed.py).
Text sensitivity: text_sensitivity.py upper-body d(own, swap) / d(own seed k, own seed k+1) and the semantic-window
r_text − r_audio (sem_window_sens.py), per seed, averaged.

A lane passes when ALL hold:
1. pair_win(full) − pair_win(sf0 full) > 0 and the 3-recogniser recording-bootstrap intervals do not cross 0, OR
   pair_win(full) − pair_win(own no-text) ≥ +0.10 with intervals above 0 (the standing criterion 2);
2. text sensitivity (swap ratio) above sf0's;
3. FGD ≤ 1.447 + 0.05 = 1.497 (mean of 3 seeds);
4. ΔBA within the sf0 seed range ± its spread (sf0 per_clip ΔBA seeds: see MULTISEED / benchmark files);
5. default paths bitwise unchanged (unit tests), causality test passes.
Passing lanes go to the long run (6000 steps, training seeds). Failing lanes are reported with the numbers.

Items:
1. semantic-first temporal stream (cfg sem_time): causal time-lattice Transformer over timed words (Gaussian prior on frame time − word
   time), per-chunk semantic state added zero-initialised to the planner input and the Local Flow condition.
2. prosody-only audio (sc / inference switch audio_prosody_only): the 80 mel columns of feat89 zeroed in training AND inference.
3. controllability: pointing (STG, arXiv 2610.03105) and counts ("tap / raise N times"; count labels from wrist-speed peaks of the GT;
   metric = count hit rate). Extra criterion: STG with given targets − STG without ≥ +0.10; count hit rate ≥ 2 × chance.

## Item 1 on ECHO-G (written 2026-11-07 before the run; scripts/echog_g1/echog_semt.py, echog_semt_lane.sh)
Module: SemTimeEG = the same time-lattice cross-attention (Gaussian prior on frame time − token centre, sigma 0.25–2 s, lag ±0.5 s,
2 layers, d 256) over the official Qwen text tokens, output zero-initialised and ADDED TO THE DiT INPUT (before all SGDiT blocks);
0 for the official null text. Control = the identical fine-tune without the module.
Protocol: official G1 checkpoint, 2000 optimizer steps (batch 12 x accumulation 4 = 48 clips / step, lr 1e-4, 100-step warm-up,
cosine to 0, official loss / condition dropout 0.1 / temporal weight 0.5 / EMA 0.999 / clip 1.0 / wd 1e-4), training stems = official
train_drop minus the hos dev speakers (english_2_scott, english_6_carla). Caveat: the official initial checkpoint was trained on all of
train_drop and has seen the dev clips; this is the same for both lanes, so only the RELATIVE comparison is read.
Evaluation: hos dev 400 clips, official sampler (8 Euler steps, guidance 1, noise stable_seed(stem, seed)), seeds 0/1/2, own / notext
(official null text) / swap (dev swap map = SCMG infer_test.py derangement, centres rescaled), rec-sem-hos-seed0..2.
Passes when ALL hold (control = egctrl):
1. pair_win(egsemt own) − pair_win(egctrl own) > 0 with the recording-bootstrap intervals not crossing 0, OR
   pair_win(egsemt own) − pair_win(egsemt notext) ≥ +0.10 with intervals above 0;
2. swap text sensitivity of egsemt above egctrl's (mean of 3 seeds);
3. FGD(egsemt own) ≤ FGD(egctrl own) + 0.05 (means of 3 seeds);
4. ΔBA within the egctrl seed range ± its spread;
5. echog_semt.py test ALL_OK (zero-init identity, null text exact, gradients reach the module).

## Item 3 on SCMG: pointing + counts (written 2026-11-07 before the run; scripts/semantic/ctl_count_g1.sh)
Count labels (g1_ctl_counts.py, scmg/ctl.py count_np): non-POINT annotation tokens with a 0.1–4 s span get count = strokes of the
span's dominant hand (larger wrist bounding-box diagonal): local maxima of the 4-frame-smoothed 50 Hz wrist speed above every neighbour
within ±5 frames and above the APEX.json action speed (0.592 m/s), capped at 4 ("4 or more"). Dev: 857 tokens, distribution
0.356 / 0.350 / 0.163 / 0.070 / 0.061, chance agreement Σp² = 0.284, majority 0.356. The evaluation applies the SAME rule to the
generated 50 Hz motion.
Models: ctlcnt (new: ctlcf recipe + count slot, soft-count loss w 1, counterfactual counts p 0.5, 2000 steps) and ctlcf (existing,
4000 steps, pointing only). Inference seeds 0/1/2, per_clip, offline, CFG 2.5, 10 steps; modes none / gt (400 dev clips, official
benchmark), shuffle, and for ctlcnt nocount / count_shuffle (annotation clips).
Pass (3-seed means), per model:
P1 pointing: STG(gt, given targets) − STG(none, scored against the same dev targets) ≥ +0.10, and under shuffle the STG against the
   GIVEN targets exceeds the STG against the dev GT targets (the given target is followed);
P2 counts (ctlcnt): count hit (gt) ≥ 2 × 0.284 = 0.569, and under count_shuffle the hit against the GIVEN counts exceeds the hit
   against the dev GT counts; reported next to the nocount hit (empirical baseline from audio / text alone);
P3 quality: FGD of the none AND gt runs ≤ 1.497; ΔBA within the sf0 seed range ± its spread;
P4 tests: test_count.py ALL_OK (labels reproduced by the evaluation rule, soft limit = hard count, training reaches the modules),
   test_ctl.py ALL_OK (default paths bitwise vs HEAD).
Item 3 on ECHO-G is a separate step (needs a token path into the official SGDiT); not part of this run.

## Item 2 on ECHO-G: Wav2Vec2 -> prosody (written 2026-11-07 before the run; echog_semt.py --audio prosody, echog_pros_lane.sh)
The official audio input (Wav2Vec2, 1024) is replaced by the 8 prosody columns of the SCMG view (onset, prominence, energy, logf0,
voiced, beat, beat_left, beat_right; normalised, 50 Hz -> 30 fps nearest frame) through a NEW LayerNorm(8) + Linear(8, 768) whose
weight starts at 0 and bias at the official projection of the dropped (zero) audio, i.e. training starts from the official model with
null audio (max |velocity diff| 7e-5 at init, GEMM-shape round-off). Same recipe, data, sampler, seeds and recognisers as item 1 on
ECHO-G; control = egctrl of that run (Wav2Vec2 kept). Passes when ALL hold: 1. pair_win(egpros own) − pair_win(egctrl own) > 0 with the
recording-bootstrap intervals not crossing 0, OR pair_win(egpros own) − pair_win(egpros notext) ≥ +0.10 with intervals above 0;
2. swap text sensitivity above egctrl's; 3. FGD(egpros own) ≤ FGD(egctrl own) + 0.05; 4. ΔBA within the egctrl seed range ± spread;
5. echog_semt.py test ALL_OK.

## Pointing-only lane on SCMG at the 2000-step protocol (written 2026-11-07 before the run; scripts/semantic/ctl_point_g1.sh)
ctlpt = the ctlcf recipe without counts (adsc + ctl + ctl_flow, ctl_w 4, w_ctl_aux 3, ctl_field_drop 0.2, ctl_cf_p 0.5, ctl_lr_scale 10,
ctl_focus 0.5, tokens ctl-tokens-hos), hos-base warm start, 2000 steps, training seed 0. Inference seeds 0/1/2, per_clip, offline,
CFG 2.5, 10 steps; none / gt on the 400 dev clips, shuffle on the annotation clips. Passes when ALL hold (3-seed means):
1. STG(gt run, given targets) − STG(none run, same dev targets) ≥ +0.10;
2. under shuffle, STG against the GIVEN targets > STG against the dev GT targets;
3. FGD(gt run) ≤ 1.497;
4. ΔBA (BA_signed_gap) of the gt run inside the sf0 range ± 0.003, i.e. |ΔBA| ≤ 0.0120 (sf0 seeds −0.0075 / −0.0062 / −0.0090;
   a value closer to 0 than −0.0032 counts as inside: it is nearer the ground truth).
If it passes, ctlpt is the controllability row.
