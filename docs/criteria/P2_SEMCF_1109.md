# P2 "sem_cf": counterfactual exemplar-descriptor SEM tokens, EX002 (agent B, 2026-11-09, written before any run)
Source: refine-logs/FABLE_INTERACTION_1109.md §2 P2 + main review. EX002 dev only (data-v36_1 val, 43 recordings); never test.

## 1. Descriptor d (16-D, scmg/sem_desc.py; one torch function used on GT, generations and in the loss)
Window W = [t - 0.6, t + 0.6] s around the event time (61 frames, 50 Hz). Exact EX002 FK (scmg/ex002_ctl points / points_t):
wrist and shoulder of both hands. h = the commanded hand (GT: the larger wrist-path diameter in W, the APEX.json rule);
o = the other hand; r_t = wrist_h - shoulder_h.
 1 amp_h = 2 sqrt(mean |r_t - mean r|^2)       2 amp_o (same, other hand)       3 disp = |mean_t (r_t - r_0)|
 4 vpeak_h = 0.05 log mean exp(v_t / 0.05)   (v = wrist_h speed, m/s)          5 vmean_h          6 vmean_o
 7 tpeak = softmax(v / 0.05)-weighted time in W, scaled to [-1, 1]                8 z_end = mean r_z of the last 5 frames
 9 z_mean = mean r_z      10 x_mean = mean r_x (forward)      11 y_abs = mean sqrt(r_y^2 + 1e-4) (lateral)
12 iw = mean |wrist_L - wrist_R|      13 path = sum |r_t - r_{t-1}|
14-16 dir = unit(mean_t (r_t - r_0))  (3 components)
D(d) = [z-scored 1-13 (mean / std over the train exemplar bank), dir 14-16]. Error E(a, b) = |D(a) - D(b)|^2 / 16; on the
direction dims this is 2 (1 - cos) / 16 (unit vectors).
Metric validation on the dev events, before any model number (gate: both must hold, otherwise the metric is revised once and then
the lane stops): (V1) GT motion vs its own descriptor E = 0 (max); (V2) GT motion vs a mismatched descriptor (derangement across
recordings, seed 20261109) = E_mis, the chance level; GT motion with the window moved +-1 s vs its own descriptor must lie between
0 and E_mis, and E_mis >= 3x that shifted error is NOT required (reported).

## 2. Exemplar bank and the deployable inference rule (fixed before any run)
Bank = the TRAIN semantic events of EVENTS.json (semantic-baseline-1105/ex002-sem/data; cls in deictic / iconic / metaphoric,
split train, 5316 events), each with (keyword kw, class, hand h, descriptor of its GT window). Deployable path from the TTS text:
every word of the utterance whose g1_lexicon category maps to a semantic class (scmg.ctl.LEX2SEM) gets a SEM token at
(word onset - 0.2 s). Its descriptor (+ hand) comes from the bank: entries with the same keyword (lower-case word); else the same
lexicon category; else the same class. Among the candidates, entry sha1("<clip>:<t:.3f>:sem") mod n is used. Tokens closer than
1.2 s to a previous one are dropped (greedy in time).
Oracle rows (upper bound / evaluation): the dev GT semantic events (EVENTS split 'test' = the 43 dev recordings, 247 events),
token time = the event's keyword onset - 0.2 s (nearest aligned occurrence of kw within t +- 1 s; none: t - 0.2 s), descriptor =
the GT descriptor of that event.

## 3. Training (frozen enc-s0, --train-only ctl_, 2000 steps)
- Token type 'sem' (appended after 'yield': every existing id unchanged), group 0, hand in value slot 3, descriptor in a new
  16-D channel ctl_d (cfg ctl_desc: Linear(16 -> d) on the token embedding; default off = the old model bitwise).
- Training tokens = the bank events of the train items: time = keyword onset - 0.2 s (same rule as the oracle), segment
  [a, b] = W of the event, value = its own GT descriptor + hand.
- Counterfactual (sem_cf_p 0.5 per sample): every kept SEM token of the sample gets another token's descriptor + hand
  (roll over the batch's SEM tokens). Imitation is lifted on [a - 0.3, b + 0.3] of the counterfactual tokens (the existing
  ins_cf path). Outcome loss w_sem_desc 1.0: E(decoded window, given descriptor) for every kept SEM token visible to the chunk of
  its window start, decoded window = the target halves of the sample's chunks (as ctl_count_loss), frames all valid and >= 4
  chunks of history.
- Recipe otherwise = the module-10 ctl recipe (ctl + ctl_flow, ctl_w 1, ctl_field_drop 0.2, ctl_focus 0.5, ctl_lr_scale 1),
  no pointing / finger terms; base enc-s0 (ex002-sfscope-1108/enc-s0/train/final.pt) frozen.
- Tests (scripts/tests/test_semcf.py ALL_OK before the run): default path bitwise vs HEAD; ctl_desc on with no SEM tokens =
  off; descriptor numpy == torch; the outcome term reaches ctl_ modules only; the none path of the trained model is bitwise
  enc-s0.

## 4. Protocol rows
- offline (H = inf, final text), the main quick-lane row;
- psync (the deployment row, tokens arrive with their word: anchor arrival), reported;
- TTS-sync (declared as its own row): psync, but every SEM token is visible and timed 0.5 s before its time (the robot's TTS
  schedule is known ahead; --ctl-lookahead 0.5), reported.

## 5. Quick lane (10 dev recordings ci 0, 4, 8, ..., 36, offline, seed 0, per_clip)
Runs: none, given (oracle descriptor), swapped (each oracle token gets another recording's descriptor + hand, seed 20261109),
retrieval (the deployable rule of §2). Metric = mean E over the oracle events of the 10 recordings (generated window at the
event's GT time t).
Kill (any): (K1) E(given) not below E(none) by >= 30 % of E(none); (K2) swapped not following: E(swapped run, swapped
descriptor) >= E(swapped run, original descriptor).
Reported: retrieval-row E vs its own retrieved descriptors and vs the oracle descriptors; FGD v2 / liveliness guard of the 10
recordings (none vs given); the TTS-sync row on the same 10.
Secondary (only on main's go after a pass): pair_win full - no-text >= +0.10 with CI (the §7.45 bar). Risk stated in advance:
pair_win may not move even if the descriptors are realised (the recogniser measures class timing, not kinematics); the result
would then be "controllable semantic kinematics", not "better semantics".

## Metric validation result (11-09 02:2x, sem_bank.py on speech_server; before any model number)
Bank 5316 train events, 247 dev events. Descriptor mu / sd in sem-1109/SEM_BANK.json. Dev GT: V1 own E = 0.000; V2 mismatched
(derangement across recordings) E = 2.556; GT window moved +-1 s vs own = 1.768, between 0 and E_mis as registered -> metric
valid. The keyword had no aligned occurrence within t +- 1 s for 2231 / 5316 train and 103 / 247 dev events; for those the token
time falls back to t - 0.2 s (as registered).

## Quick-lane result (11-09 03:08; speech_server GPU1, release 20261109-022226-p2-quick-1109 = exp/ctl-port cab4b18)
Output: CKPT/zehan/gesture-scmg/control-token-1105/ex002/p2-quick-1109 (SEM_EVAL_offline.json, SEM_EVAL_psync.json,
BENCH_EVAL.json, NONE_BITWISE.json). Training 2000 steps, frozen enc-s0, 1.11 M trainable (ctl_). The train-batch sem_desc
fell from 1.39 to about 1.2. 10 recordings, 62 oracle events. GT sanity on these events: own 0.000, mismatched 2.023.
- offline: E none 1.868, given 1.042 (reduction 44.2 %: K1 not hit, needs >= 30 %); swapped run vs swapped d 1.086 vs vs
  oracle d 1.642 (K2 not hit, it follows). VERDICT PASS.
- none path bitwise == the stored enc-s0 offline s0 outputs on all 10 recordings (max abs diff 0).
- reported rows: psync given 1.329 vs psync none 1.737 (-23.5 %). TTS-sync (look-ahead 0.5 s) 1.104 (-36.4 % vs psync none).
  Deployable retrieval (151 tokens: 114 by keyword, 37 by category, 0 by class): vs its own retrieved descriptors 0.968 (149
  windows); vs the oracle descriptors 1.740 (none 1.868).
- side (10 recordings, noisy): FGD v2 none 124.2, given 138.7, retrieval 171.5; FKD 2.03 / 2.03 / 2.38. Liveliness guard: given
  passes; none fails wrist amplitude (0.082 vs 0.75 x GT 0.083); retrieval fails wrist speed and amplitude (0.253 / 0.069).
- note: the bank median lag from (keyword onset) to the segment midpoint is 0.0 s, because 42 % of the bank tokens use the
  t - 0.2 fallback. The retrieval "own" windows are therefore centred on the word onset.

## Round 4b pre-check: is the descriptor predictable from deployable causal inputs? (written 11-09 before computing it)
Source: refine-logs/FABLE_ROUND4B_1109.md §1 + main review. CPU only, no generator. scripts/ex002/sem_precheck.py.
- Samples: the 5316 train bank events (fit) and the 247 dev events (score). Target = standardised D (16), scored with E.
- Causal features at the token time t_tok (keyword onset - 0.2 s):
  - e5: the prefix-causal e5 (models/text-e5-causal-v36_1) of the last 5 timed tokens with time <= t_tok + 0.2, mean. PCA to 32
    dims, fitted on train.
  - lexicon category one-hot of the keyword (15).
  - prosody: the 8 prosody columns of feat89, mean and std over [t_tok - 2, t_tok] (16).
  - hand in use: per hand, mean wrist speed and mean wrist height minus shoulder height over [t_tok - 1, t_tok] (4), from the
    GT history (the deployed generator's own history plays this role).
  - All features z-scored on train.
- Models: ridge (alpha in {0.1, 1, 10, 100, 1000}) and k-NN (k in {5, 20, 50}, Euclidean on the z-scored features, mean of the
  neighbours' D). Hyper-parameters and the ONE model to score are chosen by 5-fold recording-grouped CV on train only. Dev is
  scored once.
- Baselines (dev E): marginal mean (train mean of D); marginal sample (a random train D, mean over 20 draws); class mean via the
  keyword's lexicon category -> LEX2SEM class (deployable).
- Reported: R^2 per component on dev; within-class / total variance ratio per component on train (oracle class).
- Gate (the "gain" threshold): PASS iff the CV-chosen model's dev E <= 0.90 x E(marginal mean), AND the recording-clustered
  bootstrap 95 % CI (2000 resamples of the 43 dev recordings) of 1 - E(model) / E(marginal mean) lies above 0.
  PASS -> Design A quick lane (pre-registered separately). FAIL -> finding "the semantic descriptor is not inferable from
  deployable causal inputs on BEAT2", written here and as a negatives row; no MDN run.

## Round 4b pre-check RESULT (11-09 03:2x; sem-1109/PRECHECK.json; release 20261109-031432-p2-precheck-1109)
- Model chosen by train CV: ridge alpha 100 (CV E 0.709 vs the marginal mean's 0.876; k-NN 0.79-0.90).
- Dev (247 events): E ridge 1.096; marginal mean 1.459; marginal sample 2.322; lexicon-class mean 1.453.
  Reduction vs the marginal mean = 24.9 %, bootstrap 95 % CI [17.3, 30.1] % -> GATE PASS (>= 10 %, CI above 0).
- Where the gain comes from (dev R^2): amp_h 0.33, amp_o 0.45, vmean_h 0.43, vmean_o 0.52, path 0.43, disp 0.27, z_mean 0.23,
  iw 0.24; but vpeak 0.09, tpeak 0.02, z_end 0.01, x_mean 0.00 and dir 0.00 / 0.00 / 0.09. The predictable part is
  "how big / fast", not "which shape / direction".
- Within-class / total variance on train is 0.98-1.00 for every component: the BEAT2 class carries almost no descriptor
  information (consistent with the lexicon-class mean = the marginal mean).
- Caveat stated with the result: the hand-in-use features come from the GT motion of the preceding 1 s (the deployed
  generator's own history plays this role). Speed / amplitude persistence can explain the amplitude R^2. Post-hoc diagnostic,
  not a gate, running: the same model with text + lexicon + prosody only, and with history only.
- Post-hoc diagnostic (not a gate; same model selection by train CV):
  - text + lexicon + prosody only: ridge 1000, dev E 1.404, i.e. -3.7 % [1.9, 5.1] vs the marginal mean (R^2 amp_h 0.08,
    vmean_h 0.07);
  - motion history only: k-NN 50, dev E 1.094, i.e. -25.0 % [20.7, 29.3].
  The whole registered gain comes from the preceding 1 s of motion (amplitude / speed persistence), which the generator already
  conditions on. From the deployable TEXT + AUDIO inputs the descriptor is predictable by only ~4 %, below the 10 % bar the
  pre-check used. Reading: the registered gate passed, but for the purpose it was meant for ("can a deployable source beat the
  marginal on WHAT gesture to make") the answer is no. Design A would mostly learn persistence. Design A NOT run (main 11-09: agreed); negatives row 61.

## Visual check of the P2 renders (main, 11-09 03:4x; renders qual-1109-p2, rule fec7a8d)
The descriptor effect is real but visually subtle: given and none look alike (hands near the waist, small changes in wrist height
and spread). The realised amplitude / peak speed fall well short of the command (ev2: commanded amp 0.19 / vpk 0.84, realised
0.09 / 0.28; ev0 swapped: commanded vpk 1.07, realised 0.34). ev2 is "my" (deictic): no panel moves a hand toward the chest.
Reading: P2 shows a controllable statistic, not a recognisable gesture. The P2 full run stays as mechanism evidence (lower
priority) with a report-only "realisation ratio" (realised / commanded amp_h and vpeak_h). The explicit-semantics canonical path
(EXPLICIT_SEM_1109.md amendment 1) is the deliverable for "appropriate gesture".

## Full run + controls (main's go 11-09; registered before running; mechanism evidence)
- Main model = the quick-lane model (p2-quick-1109/train, training seed 0). All 43 dev recordings x per_clip seeds 0 / 1 / 2.
  Rows: offline none / given (oracle) / swapped / retrieval; psync given; TTS-sync given (psync + look-ahead 0.5 s). none psync
  = the stored enc-s0 psync outputs. Oracle plan = all 247 dev events; retrieval plan = the §2 rule on all 43 recordings.
- Gates (offline, pooled over the 3 seeds, recording-clustered bootstrap 2000):
  G1 E(given) <= 0.70 E(none), with the CI of the reduction 1 - E(given) / E(none) above 0;
  G2 swap follow: E(swapped run, swapped d) < E(swapped run, oracle d), with the CI of the difference below 0;
  G3 FGD v2 given <= none + delta, delta = max(0.03 x none, 2 x SD of none's 3 seeds);
  G4 liveliness guard on given;
  G5 the none path is bitwise the stored enc-s0 offline outputs (43 x 3).
- Reported: psync and TTS-sync E vs psync none; retrieval E vs its own and vs the oracle descriptors; realisation ratio
  (realised / commanded amp_h and vpeak_h, raw units, per event, median and IQR; given and swapped rows). pair_win is not computed
  (sanity rows unreliable on these sets; main's reading rule).
- Controls (same 43 x 3 seeds, offline given + swapped; each trained like the main model, 2000 steps, frozen enc-s0):
  c1 plain: sem_cf_p 0 and w_sem_desc 0 (descriptor token, flow loss only, imitation kept: the C3 / outcome ablation);
  c2 onset: token time = keyword onset (t_tok + 0.2 s), in training and in the plan;
  c3 late: token time = keyword onset + 0.2 s (t_tok + 0.4 s).
  Reading (reported, not gated): the mechanism needs the counterfactual + outcome terms iff c1's E(given) reduction is below the
  main model's by more than its CI; the lead matters iff the main reduction exceeds both c2's and c3's.
