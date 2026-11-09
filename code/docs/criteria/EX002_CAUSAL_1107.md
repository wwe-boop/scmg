# EX002 causal-training-match base (3a) -- written 2026-11-07 19:xx, before any EX002 causal run

Data / protocol (frozen here; evaluator entry to be confirmed by the coordinator before the first benchmark):
- View dataset/zehan/gesture/train-v1-20261101/views/data-v36_1 (SPLIT.json: train 73,237 items = manifest train; val = the fixed v3.2
  BEAT2 dev, 380 items, speakers 2_scott / 13_lu, never trained on; test never exported -- not touched).
- Robot EX002 PVT1 + RM003 v4 three-finger, 34 joints at 50 Hz (SCMG_J default), VAE trained/vae-r3/final.pt (z 16).
- Causal inputs (scripts/echog_g1/precompute_causal.py, code exp/duplex fc61397 copied to ex002-causal-1107/code):
  words = the view's alignment.json units (zh characters / en words, onset_s / offset_s; prep_causal_ex002.py), audio = each
  item's manifest audio16k.wav, fixed prosody statistics = causal_feat.prosody_stats over 400 train items (seed 0),
  CAM++ = causal-train-1105/campplus (the G1 encoder); grades 1 / 2 / 4 / 8 s + whole item; token end times from text-e5-v36_1.
  Checks before training (load_causal): whole-item mel == view mel (f16 ulp), whole-item CAM++ == SPK_EMB within 1e-3 (measured
  2-3e-4 on the first items; the G1 tolerance 1e-5 does not hold on EX002 -- disclosed), coverage >= 0.99.
- Recipe (proposed, coordinator to confirm): INIT dlc-scmg-train-r4-1102/base/best.pt, cfg {"adsc": true}, sc = sf0 (p_stream 0.67,
  p_async 1.0, p_async_seg 0, self_force_p 0) + causal_front / causal_spk / causal_text + the EX002 safety loss w_phys 0.03,
  phys_dyn_w 0 (limits34 hinge + sphere-proxy self-collision, = the EX002 deployment recipe sibag-coll-s1); seed 0.
  Quick lane 2000 steps, then 6000 steps. Control: the same without the causal flags (ncz).
- Evaluation: the 380 dev items, offline + psync (psync = causal inputs from the arrived PCM every tick, words once ended,
  CAM++ grades, 160 ms look-ahead, words-dir / audio-dir of prep_causal_ex002.py), per_clip inference seeds 0 / 1 / 2, CFG 2.5,
  10 steps; quality = FGD v2 / FKD (ex002-echogeval-1105 definitions, ours, validated -- labelled as such); executability in MuJoCo
  scene.xml (ex002-rm003-v4-20261020/model-r1) with ex002_3f_spec.json limits.
- Reading (the G1 analogue): the causal match must lower psync FGD v2 vs ncz with non-overlapping seed ranges; offline must not
  rise by more than 5 % of ncz offline. Otherwise: one negatives row, diagnose, next module.

Addendum 11-07 20:30 (coordinator; before any run): all EX002 numbers come from DLC A800 generations (bench_ex002.py is bitwise
equal to ex002_ctl_run on one host; across GPU types generations differ by ~3e-4 rad, so hosts are never mixed within a table).
Recipe confirmed (INIT r4 base, sf0 + czm + w_phys 0.03 / phys_dyn_w 0; 2000 then 6000 steps) plus the isolation lane ncz.
Data finding: EX002 view clips are RECORDINGS (train 55.8k recordings / 73,237 items; dev 43 / 380), so the causal precompute is
per recording (precompute_causal.py --recording: items' audio16k + alignment units concatenated in offset order, per-item token
ends); smoke on 2 dev recordings: training cz_pro == inference CausalInputs prosody bitwise, mel within the f16 ulp; the
load-time CAM++ check (single-item clips only) uses atol 1e-3 (measured 2-3e-4).
Evaluator: scripts/ex002/bench_ex002.py (exp/duplex 71f2e04): FGD v2 (two frozen encoders, mean) + FKD over the 43 dev recordings;
sanity: GT-as-generation 0 / 0, identical to ex002_ctl_run on the same outputs (sibag-s2 91.1268 / 87.1537 / 95.3214).
Coverage note (11-07 21:5x, after the train step of lane run dlckiwbuzns3km4w, before any bench number): the load_causal log line
"cover 18171 / 55802 clips" of release 20261107-2210-ex002-czm counted only the item-keyed entries (single-item clips). True
coverage of pre-train-slim2m on the data-v36_1 train split = 18,171 item entries + 37,631 recording entries = 55,802 / 55,802
clips (100 %), every entry with frames == the clip frames (recomputed from INDEX.json + train.json with the View grouping and the
causal_rows lookup); the training gate --causal-min-cov 0.99 uses this total. The log line now prints both parts (exp/duplex).
