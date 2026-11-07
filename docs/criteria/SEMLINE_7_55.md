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
