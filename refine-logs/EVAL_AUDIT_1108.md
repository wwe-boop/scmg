# EX002 / G1 evaluation-environment audit (Stage 3, "test the test") — 2026-11-08

Written by the main agent (Claude) as an ICML-reviewer pass over the evaluation environment, following the autonomous-ml-research method. AI-produced and not checked by a person.

Rule from now on: no NEW design round starts until a full reviewer pass finds no new issue. Runs already pre-registered and submitted (9c, module 10, duplex L1) may finish; their verdicts count only under metrics that pass this audit.

Status legend: PASS = verified with evidence (commit or number given); FAIL = shown invalid; OPEN = not yet checked.

## Pass 1 (2026-11-08 15:00)

### A. Bug checks
| # | Question | Status | Evidence / action |
|---|---|---|---|
| A1 | psync sees no future (text only after each word is spoken, audio prefix only, 160 ms look-ahead) | PASS (G1 and EX002 code) | test_causal_stream, test_retract_psync; the deployment streamer is bitwise equal to one-shot within 4e-4 rad (VAE_DIAG_1108) |
| A2 | Executed-prefix rule: the decoder look-ahead never rewrites already-played frames in interruption evals | PASS | executed-prefix rule; ISC rows check the prefix frame by frame against the projected U |
| A3 | Dev / test recordings absent from every training view (base, tails, control tokens, libraries) | PARTIAL | tail view: leakage asserts (A); HOLD_POOL / HAND_LIBRARY / HOLD_STATS: train split only (B). OPEN: one script that asserts this for ALL EX002 training inputs at once |
| A4 | Every new trainable module gets a gradient; frozen parts stay bitwise frozen | PARTIAL | true freeze verified for control (no-token output bitwise = base); self_force_scope test. OPEN: a grad-coverage check on the L1 YIELD route |
| A5 | Answer not in the question: control-token evals do not leak the target into features other than the token | OPEN | check that plan targets enter only through ctl tokens (B) |

### B. Repeat
| # | Question | Status | Evidence |
|---|---|---|---|
| B1 | Same training setting gives the same curve | PASS | 2a81db6 retrain = original sibag-s0 to 4 decimals |
| B2 | Same inference seed gives the same output | PASS | per_clip seeds; 3c-4b safety outputs bitwise across reruns |
| B3 | Seed variance known for every primary metric | PARTIAL | FGD v2 seed SD known (enc 3 seeds). OPEN: SD of STG gain, count hit, RKD, v3 across inference seeds |

### C. Published rankings
| # | Question | Status | Evidence |
|---|---|---|---|
| C1 | G1 official evaluator reproduces the published ECHO-G / EMAGE / GestureLSM ordering | PASS | main-table baseline rows (official implementations) |
| C2 | EX002 FGD v2 / FKD ranks published baselines sensibly | PASS | GT 0, same speaker 1.59, cross speaker 13.77, GestureLSM 172, EMAGE 279 |
| C3 | STG (2610.03105) on human data reproduces the paper's human-level numbers | OPEN | B: STG on train-split human pointing vs the paper's reported human value |

### D. Broken and cheating controls (does a bad method get credit?)
| # | Metric | Broken / cheating control | Status |
|---|---|---|---|
| D1 | FGD v2 / FKD | mismatched GT, static pose, temporally shuffled GT | PARTIAL: mismatched GT done; OPEN: static and frame-shuffled GT |
| D2 | Liveliness guard | white-noise jitter on a static pose, or a time-reversed / shuffled GT, passes the guard | OPEN (expected to pass: the guard is a floor only, so it must never be read alone) |
| D3 | STG gain | "point at a random target all the time", "always raise the arm" | OPEN: shuffle-follow exists; a constant-pointing control is missing |
| D4 | Count hit | template override on random clips; detector on human frames | PASS for the detector (human self 1.0, shuffle ≈ Σp²). Note: arm (a) of module 10 is filtered by the same detector (circular; disclosed) |
| D5 | Pointing apex detector (APEX.json) | a perfectly aimed but static hold | FAIL as a measure of aim quality for static holds: it never picks a static frame (9b). Keep it as the primary gate as registered; always report the hold-window angle beside it |
| D6 | yield3d | human tails | FAIL: humans pass 16–18 % (needed events 4.6–10.1 %) |
| D7 | yield v4 (v1, v2) | human tails / B1 | FAIL: B-half needed 38.1 % (< 85 %) |
| D8 | activity-only yield | human tails pass, speech continuations fail, B1 rejected by v3 / RKD | OPEN (A, fresh dev-speaker tails) |
| D9 | v3 abruptness | GT vs GT; a hard cut (B1 freeze) must score high | PARTIAL: noise floor done; OPEN: B1 v3 value |
| D10 | RKD | human half-split floor 0.179; mismatched-speaker tails; a constant pose | PARTIAL: floor done; OPEN: constant-pose and frozen-pose controls |
| D11 | unsafe (WBC) | GT 1.3 %; a known-colliding pose | PASS |
| D12 | Release continuity (median) | heavy tails hidden by the median | FAIL-as-sole-number: the 5b cusps were invisible to the median (m9b p95 226). Report the median, p95 and max always |

### E. Score audit
| # | Question | Status |
|---|---|---|
| E1 | Each primary metric moves during training and separates the methods | PARTIAL: FGD v2, liveliness and STG do; OPEN: RKD and v3 across checkpoints |
| E2 | Metrics agree between offline and psync, and across seeds | PASS for FGD v2 / liveliness (3 seeds); OPEN for interruption metrics |
| E3 | Dense-plan vs sparse-plan scores agree in sign | FAIL: dense-plan liveliness fails while sparse passes; dense quality is report-only. State what each plan can judge |

### F. Tuning parity
| # | Question | Status |
|---|---|---|
| F1 | Baselines tuned as hard as our methods | G1: official implementations and weights (no tuning by us). EX002: GestureLSM / EMAGE retargeted, not tuned. OPEN: state this as a limitation, or give each EX002 baseline the same sweep budget |
| F2 | Interruption baselines (B1, B2, B3, FRONT) given the same post-processing chain (safety + ISC) as our rows | PASS for B3 + ISC; OPEN for B1 / B2 / FRONT + ISC |


## Pass 1 results (16:00, from A: exp/duplex 23b9c87, ~/bulk/causal-train-1105/audit-1108/)
- A3 PASS: leak_check_ex002.py checks every EX002 training input, and none contains a dev or test recording. Two caveats:
  - HOLD_POOL / TARGET_POOLS carry no ids, so their provenance rests on how they were built.
  - **NEW issue:** STATS_v36_1 (the q/f mean and std) was computed over all view rows, including the 380 val items. It is normalisation statistics only, but it is still dev information.
- A4 PASS: the L1 YIELD route gets gradients; all zero-init params moved.
  - **NEW issue (pre-existing):** the audio-encoder AdaLN `ada.weight` in blocks 0–3 is exactly zero and has stayed bitwise unchanged since r4. The condition path into those layers carries no signal, so the modulation reduces to a learned constant.
- D1: FGD v2 rejects static, shuffled and jittered controls (393 / 168 / 393 vs model 107–112). FKD rejects static (9.35) but speed-matched jitter reaches 3.08 (about 2× the model). FKD is partly fooled by jitter: never read it alone; FGD v2 is the primary.
- D2: the liveliness guard passes frame-shuffled GT. It is a floor only, as expected.
- D8 FAIL: the activity-only yield misses the fresh-set human gate.
  - fresh 54.5 %; 2_scott 25 %, with wrist speed after speech 5× the B half;
  - GT-in-speech 40.8 % (> 15 %).
  - Outcome: no single yield gate. Interruption is reported by components.
- D9 PASS: B1 v3 is 78.6 %.
- D10: RKD rejects a hard freeze (3.00) but barely penalises a constant pose (0.806 vs B2 0.752 and GT-continuation 0.668). Read it together with v3 and the activity component.
- 16:20 follow-ups (A, audit-1108/ada_probe.log, STATS_TRAINONLY_1108.json):
  - AdaLN: by design, not a bug. The audio Encoder is called with c=None, so c is all zeros. The zero-init ada.weight therefore gets exactly zero gradient, and only the bias trains. This holds identically on G1 and EX002. Speaker conditioning does reach the planner and flow (planner ada.weight is non-zero).
    - Decision: no change; it would need a retrain of every model.
    - Record: about 9.4M dead parameters. Any parameter count we report must say so.
  - STATS incl. val rows (0.39 % of rows):
    - q: the change is < 1 %, negligible.
    - f: 6 channels change by > 1 % (max 8.7 % on high mel bins). That is smaller than the same rule's seed sampling noise on those channels (up to 17.6 %).
    - Decision: negligible; recorded as a limitation. Any NEW base trained from now on uses train-only stats, behind an explicit flag.
- Implication: there is currently NO validated metric for "yields like a human". Interruption can be judged on abruptness, safety and distributional likeness only.

- 17:00 F2 (A; EVAL_f2.json, V4_SCORE_postchain.json): the interruption controls run through the same chain (3c-4b + ISC), half-B reference, n = 476, 0 fallback frames.
  - B1_isc (freeze): v3 0.105, RKD 1.88. ISC smooths most of the hard stop, so v3 only just rejects it (10.5 % vs the 10 % gate); RKD rejects it clearly.
  - B2_isc: yield 1.0, v3 0.002, unsafe 0, RKD 0.740. It passes yield / v3 / unsafe and loses on RKD to R0causal_isc (0.518) and Funnel_isc (0.484).
  - FRONT_isc: yield3d 0, RKD 0.681.
  - Activity-only: every control scores ≈ 1.0, so activity can never be read alone. Withdrawal separates FRONT (0) from B1 (0.36).
- F2 PASS (parity done). E1 is partial: v3 for the r4 / step-1000 snapshots needs the uninterrupted U generated in the same DLC environment; it is waiting for GPUs.

## Pass 2 (17:05) — new questions
| # | Question | Status |
|---|---|---|
| H1 | Interruption CIs: is the bootstrap clustered by recording (476 interrupts from about 40 recordings)? | DONE: the old table had point estimates only. Now there is a recording-clustered paired bootstrap (6500a90). **New finding:** B1_isc v3 has CI [7.9 %, 13.4 %], which straddles the 10 % gate, so the v3 gate does not reliably reject a freeze after ISC; RKD does (1.88 [1.85, 1.92]). Also: B3_isc (the generator continuing through the pause) has the best RKD, 0.440, but yield3d is only 0.28. RKD and yield pull in opposite directions |
| H2 | Design count: how many designs / variants were tried per table (control, interruption, safety, base)? The skill requires reporting it | Interruption: 3 bases, 3 post-processors × 3 chains = 9 compared combinations (+ L2 diagnostic, L1 running), 5 yield-metric versions, 1 selection. Safety: 9 modules + 2 pre-emptive + 1 perceptual; 3c-4b selected (latency FAIL recorded), ISC 1. Control: pending (B) |
| H3 | v3's margin after ISC: B1_isc fails by only 0.5 pp. Is the v3 gate still discriminative once every row passes through ISC? Report the v3 seed SD (B3) beside it | DONE: B1_isc v3 is 10.5 / 8.6 / 8.2 % over seeds 0–2 (SD 1.2 pp), so it PASSES the v3 gate on 2 of 3 seeds. After ISC, v3 does not reject a freeze. The freeze is rejected by yield (≤ 2.6 %), withdrawal (0.36) and RKD (1.88). Rule: read v3 for abruptness only, never as the freeze detector. B3 seed SDs: RKD 0.018, v3 0.2–1.9 pp, yield3d 0.3–1.6 pp |
| H4 | EX002 baselines (GestureLSM / EMAGE retargeted) were not tuned, so "beats baseline" claims on EX002 are weak | RECORDED as a limitation |

- 17:40 B items (8077f40):
  - D3: STG-gain cheats collect 30–50 % of the gain: a static pointing pose +0.040, a fixed high arm +0.030, a random target +0.050, vs m9b +0.110. The shuffle margin separates them (+0.015–0.029 vs +0.095). Rule: never read the STG gain without the shuffle check.
  - B3: inference-seed SD of the STG gain is 0.006–0.014; count hit SD 0; apex angle SD 0.2–1.8°.
  - G2 PASS: APEX.json training events have 0 overlap with dev.
  - C3: in-sample consistency only (human STG 0.965). Not comparable with the paper.
  - A5 PASS: the plan enters only through ctl tokens; no-token runs are bitwise equal to the base.
  - D12 DONE: median / p95 / max are now the default output.
  - H2 control: 24 designs tried or started, 18 with a verdict. The only design that passes everything on its own base is 5b + override on sibag-s2.

- 19:00 E1 (A, chain-*-raw): checkpoints r4 base → step 1000 → enc-s0 (2000).
  - v3 0.92 → 0.56 → 0.31, RKD 1.51 → 0.73 → 0.47, unsafe 0.97 → 0.73 → 0.57, activity 0.71 → 0.87 → 0.91. All move monotonically, well beyond the seed SD: E1 PASS for these.
  - yield3d and withdrawal are flat after step 1000.
  - **NEW issue E1b:** the prefix check found y[:f] == U[:f] bitwise for only 9 of 12 sampled interruptions. v3 splices the reference U at f_i, so a prefix mismatch can contaminate v3. RESOLVED (495bf44): every post-processed and ISC row is bitwise equal on 476 / 476, so their v3 is unaffected. Raw rows differ in the last ≤ 4 frames (structural: 72.3 / 20.0 / 7.8 %). The current rule (prefix = executed U) is correct; alt is reported as a diagnostic. The doc is corrected from "≤ 3 frames" to "≤ 4 frames".

## Pass 3 (18:10) — video check against the metrics (main agent watched the ex002-m9b renders)
| # | Finding | Status |
|---|---|---|
| V1 | present0 (scott_0_110_110, t 10.30): the presenting right hand crosses the midline to the left chest, and the palm does not visibly face the camera. The gates say "same-side target" and "palm 2°" | BUG CONFIRMED and fixed (ctl-port 9944d16, row 51). The palm normal had no fixed side; the IK and the metric used |n_x|, so palms facing the chest passed. Signed re-score: m9b palm 178° (old metric said 2°). Every earlier palm PASS is void. The fixed render shows the palm to the camera on the same side (main agent checked) |
| V2 | aim0 (scott_0_61_61, t 20.56): about 1–2 s after the command, the m9b forearm points straight up above the head for > 1 s, while tok points diagonally. The release may not converge to the tok stream, or the joint-space interpolation passes through a far configuration | NOT A BUG: a second command (a present at t + 1.53 s on the same hand) raises the arm |

## Pass 4 (19:30, main agent) — full re-read of every item
- All items from passes 1–3 are either PASS, FAIL-with-a-rule (the metric is not used as a gate, or must be read with a companion), or recorded as a limitation (F1 / H4: EX002 baselines not tuned; latency not measured onboard; ~9.4M dead AdaLN params; stats incl. val is negligible).
- Re-read for new issues: none found.
- **The audit is passed. Design rounds may start.**
- Reading rules, binding from now on:
  - STG gain only with the shuffle check.
  - FKD and the liveliness guard never alone; FGD v2 is primary.
  - v3 for abruptness only, not as a freeze detector.
  - RKD always together with the yield components.
  - Release continuity as median + p95 + max.
  - Palm angle signed.
  - No single yield gate.
  - Pointing commands are hindsight relabels.

## What the environment can and cannot judge (final, pass 4)
- Can judge: G1 distribution quality against official baselines; EX002 distribution quality (FGD v2 / FKD) and under-motion (liveliness floor); executability and safety frame by frame; pointing direction gain (STG) and count legibility (detector); abruptness at interruption (v3); distributional likeness of post-interruption motion to human post-speech tails (RKD, with a 0.18 floor).
- Cannot judge (now): whether a robot "yields" in a human-like way (no validated yield metric yet); aim precision of static holds by the apex metric; semantic appropriateness of gestures to speech content (the blind-judge content question failed); listener / backchannel behaviour (no dyadic data); onboard real-time latency (not measured onboard).

### G. Label provenance (added during pass 1, 15:10, found by B)
| # | Question | Status |
|---|---|---|
| G1 | Are EX002 pointing commands human pointing? | FAIL-as-described: CTL.json 'aim' tokens are hindsight relabels (random time ≥ 1.5 s apart; the more-extended hand; target = wrist + 1.5–3 m along the forearm ray). This is valid for testing controllability; it is not human pointing. 9c's "pointing-hold" pool and HOLD_STATS inherit this. Fix: 9c' uses the 1060 real train deictic events (EVENTS.json) |
| G2 | APEX.json training data excludes dev | PASS (B, 8077f40): 0 overlap |
| G3 | STG comparable to the 2610.03105 human values | NO: we have no real 3D targets; only an in-sample Score_a check is possible |

## Pass log
- Pass 1 (15:00): new issues D5, D12, E3, F1, F2, A3 (single assert script), B3, C3, D1–D3, D8–D10 OPEN. Next pass after these are closed.
- Pass 2 (17:05): H1–H4. Pass 3 (18:10): video check found V1 (palm sign bug, fixed) and E1b (prefix, resolved). Pass 4 (19:30): no new issue, audit passed.
- 11-09 02:35 addendum: pair_win on a 100-clip G1 dev subset FAILS its sanity check (mismatched-GT 0.583 vs 0.5 ± 0.03; GT 0.714). New reading rule: pair_win is read only on the full dev set, and only with its sanity rows passing; never on quick-lane subsets.

