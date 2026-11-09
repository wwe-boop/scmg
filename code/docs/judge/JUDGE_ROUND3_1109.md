# Judge round 3 (EX002 current pipeline) — notes
- Builder: scripts/judge/make_round3.py (text-path fbde260; selection rule committed before rendering). 24 pairs; sheets in
  ~/bulk/judge-1109/round3/pairs (copy at CKPT/zehan/gesture-scmg/judge-1109/round3). KEY kept outside the sheet directory.
- Main-agent judgements committed BEFORE the KEY was sent: ~/bulk/judge-1109/round3/judgements/MAIN_JUDGE_R3.json,
  sha256 13b6c4243fc8c0846a5efceeb0014122bcac4e9b22c00ca5058eb0065c5f9b99 (recorded here before unblinding).
Unblinded after the sha256 commit (6ef6b2f). KEY (A / B):
- base: r00 enc/sib, r01 sib/enc, r02 sib/enc, r03 enc/sib, r04 enc/sib, r05 sib/enc, r06 sib/enc, r07 enc/sib, r08 enc/sib, r09 sib/enc
  (enc = enc-s0 none, sib = sibag-s2 none).
- ctl: r10 tok10ab/tok5c, r11 tok5c/tok10ab, r12 tok10ab/tok5c, r13 tok5c/tok10ab, r14 tok10ab/tok5c, r15 tok5c/tok10ab, r16 tok10ab/tok5c,
  r17 tok5c/tok10ab, r18 tok10ab/tok5c, r19 tok5c/tok10ab.
- valid: r20 / r21 GT/GT; r22 = r00 swapped (sib/enc); r23 = r10 swapped (tok5c/tok10ab).
Scores (main-agent judge):
- base, natural: enc-s0 preferred 3, sibag-s2 1, tie 6. Decisive share 3/4, binomial 95 % CI [0.19, 0.99] — not significant.
- ctl, natural: tie 10 / 10. ctl, command: tie 10 / 10. The judge could not see a command-execution difference in the sheets ("neither
  visibly points along the command line"; "finger counts not readable in the close-ups").
- validity: r20 / r21 GT vs GT both tie (PASS). r22 vs r00: enc-s0 preferred in both after the swap (consistent). r23 vs r10: tie in both
  (consistent).
Reading: the sheet format separates the bases weakly and does not resolve tok10ab vs tok5c (both are 5c blends of the same base; the
commanded aim is often small in the 6 s sheet). Pair set (3) (tok10ab + 9c'' vs tok10ab), with visible IK holds, follows when the port
results exist.
