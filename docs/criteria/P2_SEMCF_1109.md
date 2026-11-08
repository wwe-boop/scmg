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
