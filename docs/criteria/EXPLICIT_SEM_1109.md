# Explicit semantics -> appropriate gesture (agent B, 2026-11-09; user goal via main; written before any data is looked at)
Goal: when the speech carries EXPLICIT semantics, the robot makes an APPROPRIATE gesture. Fewer actions overall. EX002, dev only.
Model: the trained P2 sem_cf model (p2-quick-1109/train/final.pt: frozen enc-s0 + SEM token modules), no retraining.

## 1. Explicit-category lexicon (fixed now; from scripts/semantic/g1_lexicon.py, restricted to explicit, gesturable meanings)
Word matching is g1_lexicon.words (en: whitespace tokens, lower-case; zh: longest match). A word gets the category below if
g1_lexicon gives it the source category AND it is in the restricted list.
- SELF   (source 'self'):   en i me my mine myself i'm i've i'll i'd;  zh 我 自己
- YOU    (source 'you'):    en you your yours yourself yourselves you're you've you'll you'd;  zh 你们 你 您
- POINT  (sources 'this' + 'there'): en this here these there those;  zh 这 那
- NUMBER (source 'number'): en two three four five six seven eight nine ten twenty thirty first second third both;
                            zh 两 三 四 五 六 七 八 九 十 第
- BIG    (source 'big'):    en big huge large giant massive enormous wide whole entire;  zh 巨大 大 整个
- SMALL  (source 'small'):  en little small tiny bit few narrow slightly;  zh 小 一点
('that' is excluded because it is mostly a conjunction. 'round' is not in the source lexicon and is not added. Finger shape is not
part of the 16-D descriptor, so NUMBER is judged on arm kinematics only, stated.)

## 2. Appropriateness metric (validated before any model number)
- Exemplars: every occurrence of a category word in the TRAIN recordings of data-v36_1 (all speakers except the dev ones). Window W
  = [onset - 0.6, onset + 0.6] s around the word onset, inside the recording. Hand = the larger wrist-path diameter in W. The
  descriptor D (scmg/sem_desc.py, standardised with the P2 bank mu / sd).
- Classifier: multinomial logistic regression (class-balanced weights, L2 C chosen by 5-fold recording-grouped CV on train
  among {0.01, 0.1, 1, 10}) from D to the 6 categories. Appropriateness of an event = 1 if the argmax is the word's category.
  The appropriateness rate is the mean per category; the overall rate is the mean over the kept categories (balanced).
- Sanity on the dev GT (all 43 dev recordings, every category-word occurrence), per category: own-label rate >= chance + 20 pp
  (chance = 1 / K, K = number of categories); and the overall rate with labels shuffled within dev (20 permutations) within
  chance +- 5 pp. A category failing its own-label check is DROPPED (registered as a drop in this file). The classifier is then
  refitted on the kept categories, and the check is repeated once on them (no further drops: if a kept category fails the
  second check, the metric is reported as failed).
- If fewer than 2 categories survive: the metric is invalid, the lane stops, and that is the finding.

## 3. Deployable path (no retraining)
At every category word visible under the protocol: a SEM token at (onset - 0.2 s). Greedy in time: a token closer than 1.2 s to the
previous kept one is dropped ("fewer actions"). Descriptor SAMPLED from the category's distribution: a Gaussian fitted to the
category's train exemplar D (mean, covariance with 0.1 x identity shrinkage), one draw per token (seed sha1("<clip>:<t:.3f>:expl")).
The direction part is renormalised to unit length. Hand drawn with the category's train P(right hand). Token type SEM as in P2.

## 4. Quick lane (the 10 P2 quick-lane recordings ci 0, 4, ..., 36; seed 0)
- Runs: offline none (= the existing p2-quick none_s0, bitwise enc-s0), offline given (deployable plan), psync none, psync given,
  TTS-sync given (look-ahead 0.5 s).
- Scored events: every category-word occurrence of the kept categories in the 10 recordings (whether or not it got a token).
  Appropriateness of the generated window (dominant hand of the generated window, as for GT).
- Gate: appropriateness(given) - appropriateness(none) >= +20 pp (offline, balanced over the kept categories), with the
  recording-clustered bootstrap 95 % CI (2000 resamples) excluding 0. psync / TTS-sync are reported.
- Side gates (10 recordings, reported with the gate): FGD v2 whole-clip and outside +-1 s of the tokens (ex002_fgd_outside.py,
  i / ii), liveliness guard (lively_floor_ex002 via bench_ex002 eval), executability (ex002_exec.py on the token plan).
- Then renders of 4 events, one each for a SELF, YOU, NUMBER and BIG word if they exist among the scored events (the first by
  sha1("<clip>:<onset:.3f>:explrender")). Panels: GT / none / given; front + 3/4; the category, the posterior and the
  descriptor are printed.

## Amendment 1 (main 11-09 03:3x; written BEFORE the sanity result was read): fallback "canonical explicit gestures"
Per category, the data-driven path (§2-4) is used only if that category passes the §2 sanity check. Otherwise the category uses
the canonical path below. Both are reported.
- Mapping onto the command machinery that already passes its validated gates (EX002 tok10ab model: pointing tokens + the
  module-10 finger override with human-library shapes; keyframe insertion where noted):
  - SELF: a point command aimed at the robot's own sternum (target = the sternum point of the robot's rest pose in the base
    frame, offset 0.10 m forward), hand = the free hand (the hand with the lower wrist speed over the previous 1 s of the none
    run; ties -> right);
  - YOU: a point command forward to the listener position (base frame (1.5, 0, 1.3) m);
  - POINT (this / here / these -> down-front (0.8, 0, 0.6) m; there / those -> forward-side (1.5, +-0.8, 1.2) m, the side of the
    free hand);
  - NUMBER: the finger-count command (count = the number word; > 3 -> 3, 'both' -> 2, ordinals first / second / third -> 1 / 2 / 3,
    zh 两 -> 2, ...) with the module-10 override on the free hand, the other hand count 0;
  - BIG: a bimanual spread keyframe (both wrists to (0.45, +-0.45, 1.15) m, palms in) held 0.4 s; SMALL: a narrow close-hands
    keyframe (both wrists to (0.40, +-0.08, 1.10) m) held 0.4 s. Both use ex002_keyframe.py IK insertion with the 9c'' release
    search; this is a new keyframe kind, default off with a bitwise test.
- Trigger: under TTS-sync at word onset - 0.2 s; under psync at the onset itself; 1.2 s spacing (greedy in time).
- Geometric success metrics, validated FIRST on synthetic positives / negatives (positives: the commanded keyframe inserted into
  a GT clip; negatives: the unedited GT clip; each metric must separate them with >= 90 % accuracy, else it is revised once):
  - SELF: min over [t - 0.3, t + 0.9] of the commanded wrist-to-sternum distance <= 0.15 m;
  - YOU / POINT: the official STG Score_b (forearm ray vs the target box) at the apex-detector frame, success = Score_b >= 0.5;
  - NUMBER: the count hit (both hands, the validated finger-count metric);
  - BIG: max wrist separation in [t - 0.3, t + 0.9] >= none's + 0.15 m; SMALL: min separation <= 0.20 m while both wrists are
    above 0.9 m.
- Gate per category: success(given) - success(none) >= +20 pp, with the recording-clustered bootstrap CI excluding 0 (TTS-sync
  row primary for the canonical path, psync reported). Side gates: FGD v2 (i) / (ii) and the liveliness guard.
- Renders: 4 events ("me", "you", "three", "big"; the first by sha1("<clip>:<onset:.3f>:explrender") among the scored
  occurrences of each word), panels GT / none / given, front + 3/4 view.

## Amendment 2 (11-09 04:0x): one metric / positive revision, triggered by the synthetic validation (registered allowance)
Before any canonical generation number. A 2-recording validation run (28 commands) gave: SELF tp 0.00 (the aim-style IK positive
never brought the wrist near the chest: wpos_err 0.44 m); YOU tp 0.875 / fp 0.625 (acc 0.625: GT forearm rays often point forward
within 23.5 deg of the forward box); SMALL tp 0 (IK fell 0.2 m short: the orientation residual and the q0 regulariser).
Revisions (this is the one allowed revision per metric):
- Keyframe IK (BIG / SMALL and the SELF chest keyframe): present-style wrist position, orientation term off (palm_fix with
  or_scale 0), 3 chained IK rounds x 30 iterations (the regulariser re-centred each round). Checked on 3 recordings: BIG reaches
  (0.42-0.45, +-0.44, 1.145); SMALL separation 0.17-0.19 m; SELF chest wrist 0.12-0.14 m from the sternum reference.
- SELF gets two canonical variants, both reported: (a) the aim command at the sternum target (as registered); (b) a chest keyframe:
  the commanded hand's wrist to (0.25, 0.05 toward the midline, 1.10) m, held 0.4 s. The SELF validation positive is (b).
- YOU / POINT success (revised): forearm-ray angle to the target box <= 15 deg AND |wrist - shoulder| >= 0.40 m (arm extended),
  both at the apex-detector frame within t +- 1 s.
SELF, BIG, SMALL and NUMBER metrics are unchanged. Validation is re-run on all 10 recordings; a metric still below 90 % is
reported as failed and its category is not gated.

## Data-driven path: sanity RESULT (11-09 04:1x; expl-1109/EXPL_SANITY.json)
Exemplars: 68,217 train / 668 dev category-word occurrences (train: SELF 17,157, YOU 16,721, NUMBER 15,090, POINT 13,817,
SMALL 2,838, BIG 2,594; dev: SELF 412, YOU 110, POINT 73, NUMBER 42, SMALL 16, BIG 15). Classifier train CV balanced
accuracy 0.254 (C 10; chance 0.167). Dev GT own-label rates: SELF 0.150, YOU 0.155, POINT 0.123, NUMBER 0.333, BIG 0.200,
SMALL 0.063; the gate is chance + 20 pp = 0.367. ALL SIX categories FAIL. Balanced 0.171, shuffled 0.169. Under the registered
rule fewer than 2 categories survive -> METRIC INVALID: on BEAT2, the arm kinematics at explicit words do not identify the
word category (consistent with S1 and the 0.99 within-class ratio). The first build crashed at the empty refit; the numbers
come from the same exemplars, re-run with the fixed code (--reuse).
-> Every category uses the canonical path (amendment 1 / 2).

## Amendment 3 (main 11-09 04:2x; a rate rule fixed by design, BEFORE any canonical generation number)
The first canonical run was stopped during its metric validation (no generation number existed). New trigger rules:
- at most 1 explicit gesture per sentence (sentence = one data-v36_1 item / utterance), the first eligible word wins;
- minimum spacing 3 s between any two explicit gestures (replaces 1.2 s);
- SELF triggers only on en "me / my / myself / mine" and zh "自己" or "我" directly followed by "的" (possessive); never on the
  subject "I" / a bare "我";
- YOU, POINT, NUMBER, BIG and SMALL keep their words.
Command counts per category are reported after the rule; categories with fewer than 5 commands are reported but not gated.
Everything else (amendments 1-2: targets, metrics, synthetic validation, gates, renders) is unchanged.

## Canonical quick-lane RESULT (11-09 04:4x; expl-canon3-1109, release 20261109-040558-expl-canon3-1109 = exp/ctl-port ed55fca)
10 recordings, seed 0, amendment 3 rates (35 commands). Metric validation (synthetic positives / negatives, all 10 recordings): SELF
acc 0.95, YOU 0.94, NUMBER 1.00, POINT 1.00 (n 1), SMALL 1.00 (n 2) pass; BIG 0.50 (n 1) fails -> BIG not gated (it also has < 5
commands).
TTS-sync row (primary), success none -> given (delta, bootstrap CI):
- SELF (n 10): aim-token variant 0.00 -> 0.00 (FAIL); chest-keyframe variant 0.00 -> 1.00 (+1.00 [1.00, 1.00], PASS).
- YOU (n 16): 0.19 -> 0.06 (-0.125 [-0.29, 0.00], FAIL). The tok10ab point token does not produce an extended arm aimed within
  15 deg of the listener box.
- NUMBER (n 5): 0.00 -> 1.00 (+1.00 [1.00, 1.00], PASS; module-10 finger override).
- not gated (< 5 commands): POINT 0/1, BIG 1/1, SMALL 1/2.
psync row: the same pattern (SELF chest 1.00, NUMBER 1.00, YOU 0.07 -> 0.07, POINT 1/2, BIG 1/1, SMALL 1/2).
Side gates (10 recordings): FGD v2 none 114.83 / TTS-sync 115.64 / TTS-sync + chest 115.36 / psync 113.27; outside +-1 s of the
commands: none 122.73, TTS-sync 122.73 (identical: the edits are local). FKD 1.83 / 1.83 / 1.82 / 1.80. Liveliness guard PASS on
all rows (still 0.019, wrist 0.30-0.31 m/s, amp 0.093-0.095).
Verdict per category: SELF PASS (chest keyframe only), NUMBER PASS, YOU FAIL. POINT / BIG / SMALL are reported, not gated.
