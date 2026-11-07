# Video -> EX002 motion: robot-side re-gate and leg-region fix (quick checks, written 2026-11-07 23:15 before any run)

Coordinator approval 2026-11-07 ~23:00 (user request: "看有没有神经网络动作重定向能够利用 DLC 然后将大量的真实视频转为动作，比如 B 站和 Seamless"):
CPU-only quick checks (c)1 and (c)2; no DLC scale-up until the gates below pass and the result is reported. Secondary to the EX002
control line. No neural retargeter is built (audit: the repair gate, not retargeting speed, limits the yield).

## Legitimacy of changing a curation gate
The repair gate is a TRAINING-DATA curation rule. Changing it is legitimate only while every evaluation is untouched: the dev / test
splits of data-v36_1 (dev = 2_scott / 13_lu, 43 clips), the sealed test sets (testset-primary-20260929, heldout-t1test-sealed-20260929:
never read), the FGD v2 encoders, the BEAT2-derived training view and every criteria file stay exactly as they are. Video items only
ever enter TRAINING data; no video item is added to any evaluation set, and speakers of the dev split are not involved (video
sources are disjoint from BEAT2 speakers).

## (c)1 Robot-side re-gate of the existing Seamless RESULTs (5,245 items, already retargeted by release06; no re-retargeting)
Rule R0 (current): reject if the repair moved the hand > 5 cm from the human hand on > 20 % of frames. Candidate rules, each scored
separately and cumulatively (hours gained over R0, per rule):
- R1 executable: after the CBF-QP (scmg/safety.py limits34, margin 0.03 rad) no frame violates position / velocity / acceleration;
- R2 no self-collision: MuJoCo EX002 model, fraction of frames with any non-adjacent robot-robot contact = 0 (or <= the rate of the
  R0-passing items, reported both ways);
- R3 wrist inside the torso-box / reach limits used by the retargeter;
- R4 repair allowed only near the body: the > 5 cm repair frames must lie where the human hand is inside the near-body zone of the
  retargeter (hands-on-lap / hands-on-torso), elsewhere R0 still applies.
Accept = R1 and R2 and R3 and R4. Report hours gained per rule and in total, and spot-render 20 newly accepted clips (random, seeded)
next to the source video; the visual check is recorded per clip (pass / fail with reason).

## (c)2 Leg-region fix rerun (200 Seamless + 100 Bilibili R0-rejected items, seeded random sample)
Rerun with the Embody leg-region fix (release07, region-6 near-body remap off) on CPU; plus 50 R0-PASSING Seamless items as the
reproduction control. Report pass rate under R0 and under the accept rule above, and the repair distance distribution.

## (c)3 Gates before any scale-up (all must pass; written before the runs)
G1 reproduction: on the 50 R0-passing items, the fixed retargeter matches release06 within mean |Δq| <= 2 deg over the arm and
   finger joints (the fix must not change what already works);
G2 wrist out-of-range rate of newly accepted items <= that of the R0-passing items + 2 percentage points (frames);
G3 self-collision frame rate (MuJoCo) of newly accepted items <= that of the R0-passing items + 1 percentage point;
G4 executability after CBF 100 % of newly accepted items;
G5 spot render: >= 18 / 20 clips pass the visual check;
G6 (only after G1-G5, needs approval for its DLC run) FGD v2 (dev, 3 inference seeds) of a 2000-step model trained with the added
   items vs the same-recipe control without them: <= control + δ, δ = max(0.03 x control, 2 x SD of the control's 3 seeds); more
   speakers preferred (5 -> 20 speaker result).
