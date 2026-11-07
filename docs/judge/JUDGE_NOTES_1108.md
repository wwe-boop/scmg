# Main-agent judging, EX002 dev, rounds 1-2 (11-08) -- format findings (written by the material builder from the judge's results)
- Material: round1 (32 pairs, 3x4 front frames + subtitles; English subtitles fixed 11-08 ~03:00, KEY unchanged) and round2 (24 pairs:
  control layout with 45 deg view and ~3x hand insets, dense keyword strips, czm vs sibag + "frozen" question). Builders:
  scripts/judge/make_pairs.py, make_round2.py (exp/text-path).
- Content question (2, "matches the subtitles"): FAILED validity in both formats -- GT vs mismatched GT picked GT 4 / 9 over both rounds
  (12 still frames + subtitles in round 1; 8 frames over +-1 s around keywords in round 2). Dropped from the judging protocol.
- Naturalness, czm vs sibag-s2 (no token, offline seed 0): sibag 6, czm 4, tie 2 over 12 pairs scored so far (n = 18 planned with
  round 2). Consistent with the quantitative still-share finding (czm 5-7x more still; EX002_CTL.md).
- Counting: finger counts could not be read reliably even with the ~3x hand insets at this resolution -> no judging of finger commands
  from stills; count claims rest on the joint-space metric and video review.
- Silent pairs: p00 / p29 segments are silent (selected under the pre-fix word timing) -> judged as silent, excluded from content.
- No round 3 for now.
