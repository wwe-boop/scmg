# Video -> EX002 motion: robot-side re-gate and leg-region fix (quick checks, written 2026-11-07 23:08 before any run)

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

## Result of (c)1 (2026-11-07 23:20-23:50; speech_server CPU, 4 processes; outputs CKPT/zehan/gesture-scmg/video-regate-1107)
Code: scripts/video/regate_seamless.py (per item), regate_agg.py, regate_render_prep.py, regate_human_panel.py. The Seamless
RESULTs were made by release07-headroom (frozen release06 engine + release07 postprocess), not release06 itself.
- 5,245 items / 27.93 h, 0 errors. R0 passes 459 / 2.39 h; R0 rejects 4,786 / 25.54 h.
- Among the R0 rejects: R1 (CBF executable) 4,786, R2 (exact WBC GJK self-collision free + physical_pass) 4,786, R3 (no position
  violation) 4,786 — the retargeter already certifies every stored float32 frame, so the robot-side rules R1-R3 reject nothing;
  R4 (> 5 cm repair only on near-body frames, far-frame repair <= 20 %) keeps 4,409 / 23.59 h.
- Accept rule: 4,868 / 25.98 h (+4,409 / +23.59 h over R0, x10.9); no R0-passing item fails it.
- G1: not applicable to (c)1 (no re-retargeting; the outputs are release07's own).
- G2: wrist near-limit frame rate new 0.0000 vs R0-pass 0.0000 -> pass.
- G3: self-collision frame rate (exact check) new 0 vs R0-pass 0 -> pass.
- G4: executable after CBF 100 % (4,409 / 4,409) -> pass.
- G5: 18 / 20 pass (threshold 18) -> pass at the threshold. Fails: a hands-clasped-at-chest clip (robot keeps the hands apart at
  the waist) and one raised-hand clip (uncertain from thumbnails, counted as fail). Per clip: G5_VERDICTS.json, videos pair/.
- Fidelity / content of the newly accepted items (reported, not gated): near-body fraction 0.85 (R0-pass 0.46); robot-wrist error
  vs target 4.64 cm on far frames (R0-pass 1.40) and 12.2 cm on near frames (4.18); 11 / 20 rendered clips are static or mostly static
  (seated, hands on the thighs; the robot holds its hands at the hips with the elbows out, because the pedestal has no lap); arm
  posture vs the data-v36_1 train view: elbow flare median 0.29 vs 0.21 shoulder widths, elbow angle 93 vs 72 deg, wrist height
  -1.04 vs -0.75 shoulder widths. 214 new items have a whole-trajectory safety back-off (amplitude scaled toward rest).
- Seamless's own split: of the 4,409 new items 3,380 / 18.19 h are Seamless-train (165 speakers), the rest Seamless dev / test;
  Seamless-train without back-off: 3,221 / 17.32 h.
## (c)2 not run (evidence already on disk)
The near-body remap diagnostic seamless-eval-20261102 (40 Seamless items, release07 with near=False for ALL regions, a superset of
the leg-region fix): strict pass 2 / 40 -> 4 / 40, wrist error stays 10-20 cm on the rejects. The leg-region fix therefore cannot
recover more than about that, and the robot-side rule of (c)1 already keeps 92 % of the R0 rejects; the 300-item rerun (~30 core-h)
would not change the decision.

## (c)1 status (coordinator 2026-11-07 ~23:40): curation gate validated physically; training value untested; raw data off-line
G6 was pre-registered (control / S-full / S-up at a 15 % human-mass share via --anchor-share; expected ~4 h usable after the utterance
pipeline) but NOT run: 2,602 of the 3,221 selected parents sit in 294 raw tars that were moved to OSS (user 11-03 / 11-04 quota moves),
and the utterance pipeline needs the wav + transcript inside the tars. Built on the 619 parents still on disk (newdata-1103 rules
unchanged, scripts/video/sea_build/{select_near,extract,features,gate}.py; CKPT/zehan/gesture-scmg/video-regate-1107/build):
868 utterances extracted -> 865 with features (3 low-energy CAM++ rejects) -> 774 after the subwindow R4 gate = 0.647 h, 43 speakers,
383 parents (below the 2 h threshold -> not submitted). OSS restore (option 2) is the user's decision.
Still-frame share (ours, scripts/video/static_share.py: 6 upper-body FK points, speed / data-v36_1 dev-GT mean speed, still = mean
< 0.15; NORM_DEV.json) and mean elbow flare (shoulder widths): dev GT 0.019 / 0.261; sibag-s2 dev seed 0 0.031 / 0.268; data-v36_1
train human sample (600 items) 0.169 / 0.296; the built Seamless utterances 0.166 / 0.293 — at the utterance level (own speech) the
built set is not more static or more elbows-out than the training data (the static look of the rendered 20 s windows comes from
listening / resting parts that the utterance pipeline drops).

## Option 3 pilot (approved 2026-11-07 ~23:45, pre-registered before any pilot run)
100 WHOLE Seamless tracks (all 20 s windows of every valid run >= 5 s, convert_seamless 'all' mode) from tars still on disk,
Seamless train split only, participants not in Seamless dev/test, not in data-v36_1 (no Seamless there) and not among the speakers
already built above (one track per participant, seeded 20261107). Same frozen release07 retargeter -> (c)1 robot-side gate (R1-R4 and
no safety back-off) -> newdata-1103 utterance pipeline (extract / features / subwindow R4 gate) on one DLC CPU pod (<= 88 cores, no GPU).
Report: usable hours per 100 tracks, speakers, near-body fraction, still-frame share (frame level, definition above) and the share of
utterances with still share >= 0.5, 10 spot renders.
Scale-up rule (coordinator): proceed to a larger run only if usable >= 1 h per 100 tracks AND the still-frame share <= 0.60;
G6 runs only once >= 5 h are built; no larger run before that. Training data only; every evaluation set untouched.
Note (coordinator 2026-11-08): the utterance-level still-frame share (built Seamless 0.166 vs data-v36_1 human train 0.169; elbow
flare 0.293 vs 0.296) removes the main risk raised in (c)1 (static / elbows-out data); the pilot reports the same two numbers.
Pilot smoke (speech_server, 2 tracks / 22 windows, release 20261107-234556-video-pilot): whole chain ran; 21 / 22 windows pass the
(c)1 gate; 16 utterances / 44.5 s kept (projection ~0.6-0.7 h per 100 tracks).
## Option-3 pilot result (dlc1i5g16zc45pk2, 00:44-02:51 2026-11-08; outputs CKPT/zehan/gesture-scmg/video-regate-1107/pilot)
- 100 whole tracks (one per participant) = 1255 windows / 6.35 h retargeted, 0 physical rejects; (c)1 rule: R0 passes 191 / 0.92 h, new
  accepts 955 / 4.90 h (R1-R3 reject nothing; R4 keeps 955 of 1064 rejects), back-off windows excluded (29 + 29).
- Utterance pipeline: new accepts 1014 utterances / 0.811 h / 87 speakers; the R0-passing windows were added in a supplementary CPU pass on
  speech_server (pilot-r0: 197 utterances / 0.158 h; the pilot driver had only selected new accepts) -> USABLE 0.969 h per 100 tracks,
  1211 utterances, 90 speakers.
- Still-frame share of the usable utterances 0.161 (0.146 for the new accepts; data-v36_1 human train 0.169, dev GT 0.019); utterances
  with still share >= 0.5: 12 %; near-body fraction of the accepted windows 0.70; elbow flare 0.284.
- Scale-up rule (pre-registered): usable >= 1 h per 100 tracks AND still share <= 0.60 -> FAILS on hours (0.969 < 1.0; still share passes).
  No larger run. G6 stays deferred (built so far 0.647 h + 0.969 h = 1.62 h < 5 h).
Pilot spot renders (10 newly accepted windows, seed 20261108; robot Isaac front + human stick figure; ~/bulk/video-regate-1107/pilot/pair,
sheets): 9 / 10 pass (fail: P3100 holds the arms crossed at the chest throughout, the robot keeps the hands low at the hips); gestures with
the hands away from the body are followed (P0506, P1313, P0359A); 5 / 10 are static or mostly static windows (hands on the thighs or hanging
at the sides; the robot shows the elbows-out rest or arms down) -- the utterance pipeline then keeps only the speaking parts (still share 0.16).
