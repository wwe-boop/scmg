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
