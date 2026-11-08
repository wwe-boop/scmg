# EX002 design round after the 11-08 audit — Fable design memo (design only, no runs)

Written 2026-11-08 by the Fable design agent on the user's request. Inputs: EVAL_AUDIT_1108.md (reading rules and can / cannot-judge are binding), EVAL_PROTOCOL.md, EXPERIMENT_PLAN.md, SCMG_NEGATIVE_RESULTS_1107.md (rows 1–51), SCMG_IMPROVE_DESIGN_1104.md §7.42–7.66, the EX002 criteria files (CTL, INTERRUPT, DUPLEX_LADDER, CBF, SFSCOPE), the memory rules, `scmg/vae.py`, `scmg/model.py`, `scmg/losses.py`, and the EX002 collision table `ex002-model-r1/runtime-rm003.json`. Everything here is a proposal; no number below is a result. Nothing was run.

Conventions used throughout:
- "Gate" = pass / fail decided before the run; "report-only" = shown, never decides.
- Side-gate bundle **S** (the audit's reading rules): quality (i) sparse outside ±1 s and (ii) sparse whole clip ≤ C + δ against the no-token enc-s0 run on the same masks; liveliness guard (still ≤ 2× GT, wrist speed and amplitude ≥ 0.75× GT, moving upper body incl. fingers, pedestal excluded; offline and psync) read as a floor only; executability after the deployed safety layer 100 % of frames (retargeter WBC clearance, position, velocity); release / insertion continuity as **median + p95 + max**; FGD v2 primary, FKD never alone; STG only with the shuffle-follow check and the constant-pointing cheat controls (D3 of the audit); palm angle signed; v3 for abruptness only; RKD never without the yield components; no single yield gate.
- Statistics: recording-clustered paired bootstrap (2000), 95 % CI; inference seeds per_clip 0 / 1 / 2; selection on the 380-clip dev set (2_scott, 13_lu) only; the test split is touched once per final method after a TEST_USAGE entry.
- Every switch is default-off with a bitwise test that the default path is unchanged (same test style as `test_blend5c.py`, `test_selfforce_scope.py`, `test_palm.py`).
- Quick lane = ≤ 5 min: for post-processing designs, 2–3 dev clips or the 2 diagnostic recordings (ci 7, 37) on CPU; for training designs, 2000 steps on one local 5090 (true-freeze ctl_ modules train in a few minutes; a fine-tune of the base is ~5 min at the sibag batch).
- "Base-changing" is flagged where a design alters the enc-s0 generator or the decoder, because then every EX002 table re-runs.

## 0. Ranking (expected value per unit cost)

| rank | id | problem | one line | runs on | base-changing | cost |
|---|---|---|---|---|---|---|
| 1 | D1 | 4 safety latency | Block-separable chunk projection: two 8-joint finger QPs + one 18-joint arm/waist/head QP, same exact WBC certificate, same recursive-feasibility candidate | CPU | no | ~0 GPU-h, ~6 CPU-h |
| 2 | D2 | 5 acceleration spikes (+ probe for 6) | Causal polyphase equaliser: a per-phase (t mod 4) ridge-fitted FIR on the decoder output, fitted on train round-trips, applied before blend / keyframes / safety | CPU fit, GPU decode | decoder post-layer: every EX002 number changes, but a re-decode only | ~1 GPU-h |
| 3 | D3 | 3 interruption | Kinematically conditioned rest insertion (L2d): brake-then-ramp start, duration and end pose sampled from half-A human tails conditioned on the current pose, through ISC | CPU | no | ~0 GPU-h |
| 4 | D4 | 2 pointing / presenting | 9d: differential-IK hold replay (per-frame Jacobian tracking), unit-consistent orientation weights with a hard wrist-height box, same-hand command arbitration | CPU | no | ~0 GPU-h |
| 5 | D5 | 1 token control | Module 6 revived on enc-s0: temporally gated control routes trained with the window (no inference blend), ctl_flow-only variant | 5090 quick, DLC full | no (true freeze) | 2–6 GPU-h |
| 6 | D6 | 7 semantics | Stroke-lead word timing: word times shifted −0.2 s at train and inference under a declared TTS-sync protocol row; late-shift control | 5090 quick, DLC full | yes (fine-tune of the base) | 3–12 GPU-h |
| — | P6 | elbows | No design yet: a 10-minute attribution diagnostic decides whether it is the VAE or the sampler; both fixes are base-changing | GPU minutes | — | — |
| — | P8 | backchannel | Skipped: no dyadic data locally; nothing can be validated | — | — | — |
| M1 | metric | 3 yield | Paired stroke-suppression (PSS): self-normalised stroke-rate ratio, interrupted vs own uninterrupted continuation, validated on human tails vs their own preceding speech | CPU | — | — |
| M2 | metric | 1 quality (ii) | Obedience floor: a GT-with-human-deictic-splice control fixes the (ii) gate an obedient system can reach | CPU | — | — |

Problem 1 gets no new generator design beyond D5 because (a) module 10 is already queued for exactly it, (b) the STG shortfall (+0.099 / +0.088 vs +0.10) is inside the inference-seed SD of the gain (0.006–0.014, audit B3), so a design "for the last 0.01" cannot be distinguished from noise without the CI rule, and (c) the (ii) excess that is not the override is a property of obedience itself (M2), not a defect to engineer away.

---

## D1. Block-separable chunk projection (safety latency, problem 4)

**Problem.** 3c-4b passes gates 1–3 with zero fallback and is bit-reproducible, but p99 184 / 171 ms per chunk (1 thread, 9950X) vs the 100 ms gate. About 5 % of chunks need 2–5 QP solves, each 45–60 ms (qdldl, 41 frames × 34 joints with the 25-frame braking tail → ~1400 primal variables plus rows). The raw violations are 0.4–2 mm intrusions into the 2 mm **same-hand finger-pair** margins (EX002_CBF_1108 3c-2 diagnosis). Predictive rows and 4-worker checks did not help (rows 42): the solve, not the check, is the bottleneck.

**Mechanism.** The collision table (`runtime-rm003.json`, 349 pairs; read today) splits cleanly by joint dependence:
- 32 same-hand finger pairs (2 mm): the relative pose of two links on the same hand depends only on that hand's 8 finger joints. Independent of the arm.
- 317 remaining rows (10 mm arm–arm / arm–torso / arm–hand / hand–hand / finger–other-arm; 8 × 50 mm arm/hand–head): depend on the 18 arm + waist + head joints; those involving a finger link depend weakly on that hand's fingers too (finger links are ≤ a few cm long).

Replace the one 34-joint chunk QP by three QPs on the same 41-frame horizon with identical position / velocity / acceleration / jerk rows and the same terminal braking tail:
1. **Finger blocks** F_L, F_R (8 joints each): collision rows = the 16 same-hand pairs of that hand, linearised as now (frames + midpoints, cutting planes from the exact test). Solved first, in parallel (two threads).
2. **Arm block** A (18 joints): all 317 mixed rows, with the finger joints fixed at the finger-block solution (their Jacobian columns dropped; the row offsets are evaluated at the new finger trajectory). Cutting planes from a failed exact test on a mixed pair go to A.
3. **Certificate unchanged:** the exact WBC test (`EX002().clearance`, GJK) on the combined pose certifies every frame and midpoint; a chunk that fails after the cap executes the verified plan (same fallback as 3c-4b).
4. **Recursive feasibility unchanged in kind:** the shifted verified plan + hold is a full 34-joint trajectory; each block relaxes its own rows to admit its sub-trajectory (l_row ← min(l_row, a_row·x_shift) with a_row of that block). A's mixed rows are linearised at the finger solution, not at the candidate fingers, so A's admission of the candidate's arm part is exact for the relaxed rows but the exact test may still fail on a mixed pair → one more cut round, or the fallback. Gate 2 (zero fallback) therefore stays a real test.
5. Order variant (chosen by cut-round count on the two diagnostic recordings only, as in 3c-5): fingers-first (above) vs arm-first with fingers fixed at the candidate.

Everything else = 3c-4b: qdldl, tail_smooth 1e-3, memoised linearisation, cap 8, predictive / carry off. Switch `--blocks fingers,arm` (default `none` = 3c-4b bit-identical).

**Why it should work.** qdldl on these banded KKTs scales super-linearly in the primal dimension; 1394 → 738 + 2 × 328 variables cuts the dominant factorisation by roughly 2–3× per round, and the rows that trigger the extra rounds (finger pairs) now live in an 8-joint QP that costs a few ms per solve, so the 5 % multi-solve chunks stop dominating p99. Standard block / consensus decomposition of MPC QPs (e.g. ADMM-based OSQP splitting, or decomposition by subsystem in distributed MPC) is the closest prior; the difference is that here the blocks are made exactly separable by the kinematic tree (finger-pair rows do not depend on arm joints), so no consensus iteration is needed and the exact certificate stays whole-body.

**Combination value.** Pure engineering of the existing layer; composes with ISC (the partial-chunk re-projection uses the same QP) and with every row that runs through 3c-4b.

**Pre-registration draft.**
- Gates (EX002_CBF_1108 gates, unchanged): 1 WBC-clear / position / velocity 100 % on all 6 runs (offline / psync × seeds 0–2) and all 1428 interruption streams; 2 zero fallback frames; 3 dFGD v2 and dFKD within ±1 raw seed SD, jerk-over-table 0 %; 4 idle per-chunk p99 ≤ 100 ms, single process, load reported; reproducibility: second psync s0 run bit-identical.
- Side: liveliness change vs raw reported; correction RMSE vs 3c-4b reported (expected small but non-zero: outputs are **not** bit-identical to 3c-4b).
- Controls: 3c-4b on the same inputs (same machine, same session) as the reference timing; "finger blocks only, arm rows kept in a 26-joint block" as the attribution row.
- Reading rules: latency only idle and only on one declared host; 3c-6 (4 workers) numbers in the same table.

**Quick lane (≤ 5 min, CPU).** Replay the captured chunk QPs of ci 7 and ci 37 (370 chunks, the 3c-5 diagnostic set) with `--blocks`: outputs p50 / p95 / p99 per order variant, cut-round shares, WBC-clear on every frame, fallback count. Kill: any WBC violation, any fallback, or p99 on these 370 chunks not below 0.7 × the 3c-4b p99 on the same chunks.

**Cost / risks.** CPU only (~6 CPU-h for the 6 runs + 1428 streams). Risk: mixed-row cuts bounce between a cut round in A and a finger change → more rounds, not fewer (the quick lane shows it). Metric cheating: none (the certificate is the exact test, not the QP). Guard: gate 1 is per frame on every output, as now.

---

## D2. Causal polyphase equaliser for the vae-r3 decoder (acceleration spikes, problem 5; probe for problem 6)

**Problem.** EX002 arm-acceleration window p95 ≈ 2 × GT with a 4-frame latent-phase pattern (phase ratios 1.19 / 1.05 / 0.73 / 1.03); the decoder alone accounts for 1.2×; a decoder fine-tune with derivative losses traded 7–27 % reconstruction and left the pattern (row 49). The pattern period equals the `ConvTranspose1d(c, c, 4, stride=4)` stride in `vae.py`: a polyphase artefact (each of the 4 output phases is produced by a different sub-filter with its own gain / response).

**Mechanism.** A linear, causal, per-phase FIR applied to the decoded motion x̂ ∈ R^{T×34}:

y[t] = Σ_{k=0}^{K} W_{φ(t),k} ⊙ x̂[t−k] + b_{φ(t)},  φ(t) = t mod 4, K = 7 (two latents of history), W diagonal per joint (34 × 4 × 8 = 1088 weights + 136 biases) or per joint group (shared across the 7 arm joints of one side).

Fit by ridge least squares on **train-split** VAE round-trips (encode → decode of GT): target = GT frames, input = the round-trip output, λ chosen on a train-internal hold-out (no dev). Closed form; minutes. Initialise at identity (W_{φ,0} = 1) so the fit can only move away from the current decoder where it reduces error. Plugs in `infer.py` directly after `vae.decode`, before 5c / keyframes / finger override / safety projection; applies identically to offline, psync and interruption streams. Switch `--dec-eq PATH` (default off = bit-identical). A per-phase affine-only variant (K = 0) is the attribution control.

**Why it should work.** The pattern is phase-locked to the stride, which is exactly what a polyphase equaliser corrects (Odena et al. 2016 "Deconvolution and checkerboard artifacts"; Kinoshita & Kiya ICASSP 2020 fix it with a fixed smoothing layer *at training time*; Wave-U-Net replaced strided transposed convs with interpolation after observing the same periodic buzz in 1-D). The difference: a post-hoc linear layer fitted to the frozen decoder, so no VAE retrain and no new generators. It is linear, so it cannot invent motion; it can only redistribute gain across phases and frames. The sampler's share of the excess (the remaining ~1.0×) is out of scope and must be stated.

**Combination value.** Reduces the acceleration load that the chunk QP and the 9-series ramps see, so D1 and D4 both benefit. If adopted it becomes part of the decoder definition for every EX002 row (flag: re-decode of all stored latents; generators unchanged).

**Pre-registration draft.**
- Primary gates (dev, on round-trips of dev GT and on enc-s0 generations offline + psync, seeds 0–2): (a) round-trip reconstruction L1 ≤ r3 − 0 % (not worse; the r3d lane's +2 % tolerance is not granted since a least-squares fit should improve it); (b) arm-acceleration window p95 ratio to GT on **generations** falls by ≥ half of the decoder share (from ~2.0× toward ≤ 1.8×) with the per-phase ratios all within [0.9, 1.1] on round-trips; (c) FGD v2 and FKD on generations within ±1 raw seed SD (FGD v2 may improve; an improvement is reported, not required).
- Side bundle S; in particular the liveliness guard must not move by more than the seed SD (an equaliser that damps motion is rejected here even if (b) passes).
- Controls: K = 0 affine-only; a phase-agnostic FIR (same K, W shared across φ) — if it does as well, the effect is not polyphase and the design is relabelled; "encode-decode re-projection" row (the existing 40 % remover) as the comparison.
- Reading rules: FKD never alone; still-share and wrist speed beside every acceleration number.

**Quick lane (≤ 5 min).** Fit on 2000 train items (CPU, seconds), evaluate on 10 dev clips: reconstruction L1 before / after, per-phase acceleration ratios, window p95 ratio on the enc-s0 offline s0 decodes of those clips. Kill: reconstruction worse, or per-phase ratios not flattened (max/min > 1.15 after the fit).

**Cost / risks.** ~1 GPU-h (re-decode of 6 runs + interruption streams). Risk: the generated latents are off-manifold relative to round-trips, so the fitted filter may under-correct on generations (then (b) fails honestly). Cheating risk: a filter that low-passes everything lowers acceleration and could lower FGD v2 (which under-reads under-motion) — guarded by (a) on round-trips and by the liveliness side gate.

**Probe for problem 6 (no design, 10 minutes GPU).** Report the elbow amplitude ratio to GT for: dev GT round-trip (VAE share), enc-s0 offline seeds 0–2 (sampler share), CFG 1.0 / 2.0 / 3.0, and with / without the equaliser. If the round-trip already loses elbow amplitude, the fix belongs to a VAE v4 (interpolation upsampling + `ch_w` on the elbow channels; base-changing, not recommended now). If the round-trip is ≈ 1.0, the loss is the sampler's mean-regression on a high-variance joint; the only non-cheating remedies are training-side (base-changing). Either way no inference-time rescaling of a joint is proposed: it would be optimising a metric by construction.

---

## D3. Kinematically conditioned rest insertion, L2d (interruption, problem 3)

**Problem.** Post-processing rows trade RKD against yield: B3 (generator continues) RKD 0.440 / yield3d 0.28; R0causal + ISC 0.518 / 0.98; Funnel + ISC 0.484 / 0.89; the L2 diagnostic rest ramp 1.37 (0.67 s ramps vs human 1.22 s median yield). Human end poses are broad; our rows collapse to the per-joint median rest pose. In the first 0.3 s after an interruption the arm continues outward (momentum + the C2 start of the quintic). L2b (human-sampled duration) and L2c (human-sampled end pose) are planned; D3 is the conditional version of both plus a different start.

**Mechanism** (post-processing on the L1 or enc-s0 continuation, `rest_ramp_ex002.py --version l2d`, default off):
1. **End pose, conditioned.** From the **half-A** human tails (training positives; half B stays the reference), take the end-pose set E_A (12-D wrist / elbow relative to hip, plus the 34-joint pose). For an interruption at f with current upper-body pose p_f, pick the k = 5 nearest E_A entries in the 12-D space and sample one with a per-event seed (sha1 of clip:t). Waist and arms are set from it; head and fingers keep the continuation (as L2).
2. **Duration, conditioned.** Fit on half A a monotone regression T(d) of tail duration to first quiet frame on the wrist travel distance d between the pose at speech offset and the end pose (a Fitts-like relation); sample T = T(d) + residual drawn from the half-A residuals. Floor 0.5 s, cap 2.0 s.
3. **Brake-then-ramp start.** Instead of a quintic matched to the continuation's velocity *and* acceleration (which must continue outward), use a two-piece profile: a braking segment with per-joint deceleration at the train a95 table (ARM_ACC_P95.json, train split), reducing the outward velocity to zero, followed by the quintic to the end pose over the remaining T. Jerk continuity at the junction is kept by a 0.1 s raised-cosine on the acceleration profile.
4. Then the deployed chain: 3c-4b + ISC.

**Why it should work.** Holler et al. 2023 (Phil. Trans. B, 378:20210473) coded gesture phases at turn ends: transitions were as likely after a post-stroke hold or a retraction as with no gesture, and the authors conclude that gesture completion / retraction is *not* a turn-yielding cue, while ongoing preparation / stroke *holds* the turn. That matches our observation that humans 2 s after speech often have not returned to rest, and argues for ending in a human-like *settled* pose, not the median rest pose. Conditioning on the current pose shortens travel, which lowers the speed / displacement descriptors toward the human windows (RKD) without the late pinning that broke rows 15 and 30. The brake-then-ramp attacks the 0.3 s outward drift directly. Closest prior: our own L2 / L2b / L2c and the G1 tr2hd outpainting (row 15); the difference is conditional sampling + a braking start instead of a C2 match.

**Combination value.** Sits on L1 (YIELD token) and under ISC unchanged; the end pose can also seed the Funnel's tube centre r(t) (a fourth row: Funnel toward the sampled pose).

**Pre-registration draft.**
- Interruption rule rows (half-B reference, 476 dev interruptions, deployed chain): yield3d (reported; no longer a sole gate per the audit), v3 ≤ 10 % (abruptness only), unsafe ≤ 1 %, RKD-post lower than R0causal + ISC (0.518) with the paired recording bootstrap excluding 0; withdrawal and activity components reported; the end-pose 3-NN distance to the **half-B** end-pose set ≤ D90 on a share of events at least equal to half B's own leave-one-out share.
- Side: uninterrupted quality unchanged (post-processing only acts after f); executability 100 %; reaction latency (event → first changed frame) reported; outward drift = wrist displacement along the pre-interruption velocity direction over [f, f + 0.3 s] reported as median / p95 / max, against R0causal + ISC.
- Controls isolating the new parts: (a) L2c-unconditioned end pose (random E_A sample) with the same duration rule; (b) conditioned end pose with the fixed 1.22 s duration; (c) quintic C2 start instead of brake-then-ramp; (d) **mismatched-source control**: end poses sampled from the hindsight-token hold pool instead of E_A (the metric must prefer E_A, otherwise RKD is reading "any human pose").
- Anti-cheat on RKD: E_A is half A; the score is half B; RKD without its end-pose descriptor is reported beside the full RKD so that a direct end-pose match cannot carry the result alone.

**Quick lane (≤ 5 min, CPU).** 60 interruptions (the first 5 recordings) through l2d and the 4 controls, raw chain: RKD-post, v3, yield3d, drift median / p95 / max. Kill: RKD-post ≥ 0.518 on this subset, or v3 > 10 %.

**Cost / risks.** Zero GPU; ISC runs are CPU (minutes). Risk: the yield3d column drops (end poses are not R_sil) — acceptable under "no single yield gate", must be stated. Risk: the Fitts regression is weak on 375 tails (report R²; fall back to the unconditional half-A duration distribution if R² < 0.1, decided before any model row is scored).

---

## D4. 9d: differential-IK hold replay, unit-consistent orientation weights, same-hand arbitration (pointing / presenting, problem 2)

**Problem.** 9c'' (2 clips) gives angle 11–17°, raised 1.0, signed palm 3°, but (a) the palm fix cut the STG gain to +0.11 / +0.095 because the present pose of a neighbouring same-hand command sits inside the aim's ±1 s apex window; (b) the wrist is 5 cm above target because dimensionless unit-vector terms (weight 1) compete with a metre-valued position residual (weight 1); (c) the replayed hold speed is half the human pool's (IK every 3rd frame, 5 iterations); (d) the apex detector cannot see static holds (audit D5, kept as registered).

**Mechanism** (`ex002_keyframe.py insert --version 9d`, default off; 9c'' paths bitwise):
1. **Differential-IK hold replay.** During the hold, instead of IK keys every 3rd frame + cubic interpolation, integrate per frame Δq = J⁺_W (w_{k} + dw[τ] − FK(q)) + N(J) z, with the forearm-ray constraint folded into the task Jacobian (ray direction rows weighted by the target distance so they are in metres), damping λ = 0.02, joint-limit and penetration residuals as now; one Gauss-Newton step per frame, warm-started. This tracks the replayed human displacement at its own speed (expected hold speed ≈ pool 0.28 m/s instead of 0.13).
2. **Unit-consistent present IK.** Orientation terms (signed palm forward, fingers up) scaled by r = 0.08 m/rad (the palm half-width: 1 rad of palm error ≈ 8 cm of fingertip displacement), so position and orientation residuals share units; the wrist-height target becomes a two-sided box [−0.14, −0.10] m relative to the shoulder with hinge weight 100 (the same hinge machinery as the 9c lower bound), and the collision residual stays at 100. Expected: wrist within ±2 cm of −0.12 while the signed palm stays ≤ 10°.
3. **Same-hand arbitration.** When a present command on hand h starts within ±1.0 s of an aim hold on hand h, and the other hand's commanded count is ≥ 1, the other hand presents (the finger override still sets both hands' counts, so the finger-hit metric is unaffected). Otherwise keep h. This is a scheduler rule in the insertion script, not a plan change; the plan and its sha256 stay as registered.
4. Report-only split added to the STG table: aim events **with** vs **without** a same-hand command inside ±1 s. This isolates the dense-plan interaction without changing the gate.

**Why it should work.** (1) is textbook resolved-rate / closed-loop inverse kinematics (Whitney 1969; Siciliano & Khatib handbook ch. on differential kinematics): the per-frame Jacobian step tracks velocity, which the key-and-interpolate scheme cannot. (2) is the standard dimensional scaling of task-space residuals (position in metres, orientation via a characteristic length). (3) follows from the 9b render check (aim0: the "arm up" was the next same-hand present). The difference from 9c'' is only these three mechanics; the pool, the IK solver core, the ramps, the release search and the gates are unchanged.

**Combination value.** Runs on the 9c'' outputs; composes with D2 (less acceleration going into the ramps) and D1 (no change to the safety layer).

**Pre-registration draft.**
- Primary (9b / 9c gates, unchanged, 3 seeds, offline + psync, plans 1112): apex-based median forearm angle ≤ 25°; |elevation| ≤ 5°; STG(given) − STG(none) ≥ +0.10 with shuffle follow **and** the D3 cheat controls (constant pointing +0.040, fixed high arm +0.030, random target +0.050 must stay below the method's shuffle margin); count hit ≥ 0.9; presenting hand raised ≥ 0.80; signed palm median ≤ 45° and palm-forward share reported.
- Side bundle S; release continuity median ≤ 29.8 **plus p95 and max** reported; hold-window angle and hold wrist speed vs the deictic pool (median [IQR]) reported; wrist z − shoulder z median and IQR reported; IK residual reported; rejected-event share reported and counted as failures.
- Controls: 9c'' as is (reference); 9d with (1) only; (1)+(2); (1)+(2)+(3); arbitration applied but IK keys as 9c'' (isolates (3)); **IK insertion on the no-token stream** (none + 9d) — if this row matches the given + 9d row on every primary gate, the token path is not load-bearing for geometry and its (ii) cost is avoidable.
- Reading rules: STG never without the shuffle check; the apex metric is primary as registered, the hold-window angle beside it.

**Quick lane (≤ 5 min, CPU).** The same 2 clips as 9c'' (seed 0, offline / psync), all five rows above: hold speed, wrist height, signed palm, STG gain, angle, release median / p95 / max. Kill: hold speed not ≥ 0.2 m/s with (1), or wrist height not within ±2 cm with (2), or any gate metric worse than 9c'' on both clips.

**Cost / risks.** Zero GPU; per-frame IK is ~35 frames × 1 GN step per event, negligible. Risk: per-frame Jacobian steps can drift into self-penetration faster than the keyed solver; the penetration residual and the exact check after insertion (already in the pipeline) catch it, and rejected events count as failures. Cheating risk: arbitration could be read as "choosing the easy hand"; the rule is fixed before the run and the finger-hit metric is hand-specific, so nothing is dropped.

---

## D5. Module 6 revived on enc-s0: temporally gated control routes (token control, problem 1)

**Problem.** On enc-s0 the 5b / 5c inference blend loses STG gain (+0.099 / +0.088) and (ii) fails by +2 / +4 with the whole excess inside the command windows; lane 4 showed the ungated routes act globally (+21.8 FGD v2 more than 1 s from any command). Module 6 was pre-registered on 11-08 01:15 with the condition "runs only if 5b fails on the fresh plans"; 5b passed on sibag-s2 then, and now fails on enc-s0 — the reopen condition is met.

**Mechanism** (as registered, two additions). Each control token's contribution — its attention value in the planner text stream (`ctl`) and its FiLM / additive term in `ctl_flow` — is multiplied by w_k(t) = 1 on [t_k − 0.3, t_k + 0.9] s with 0.3 s raised-cosine ramps, composed across tokens with the 5c rule w = 1 − Π(1 − w_i), in training and inference alike; frames outside every window receive exactly zero from the control routes. Additions: (a) a **ctl_flow-only** variant (token not entered into the planner stream at all, `--ctl-route flow`), since the planner's chunk attention is where the global leakage was diagnosed; (b) the window is applied to the *loss masks* too (ctl_ray / ctl_finger only on frames inside the window), so the ctl_ modules are never rewarded for moving far frames. Training = lane-4 recipe (true freeze `--train-only ctl_`, lr 1e-4, 2000 steps, self_force_p 0, counterfactual targets); no inference blend; finger override and 9d insertion as before.

**Why it should work.** With the base bitwise frozen, (i) holds by construction and the generator produces one coherent approach–apex–return inside the window instead of a splice of two streams, which is where 5b/5c lost 0.015–0.04 of STG gain (diagnosis in EX002_CTL "Why the token path does not transfer"). Time-local conditioning via gated FiLM is the standard form of temporally localised control in diffusion / flow motion models (e.g. keyframe- and span-conditioned motion in-betweening / control nets); the difference is the hard window tied to a command time inside a strictly causal streaming generator with a frozen base. It does not address the obedience floor on (ii) (see M2).

**Combination value.** Replaces the 5c blend as the token path for 9d / module 10; everything downstream unchanged.

**Pre-registration draft.**
- Gates = module 10 gates: STG gain ≥ +0.10 on the dense plan with shuffle follow and bootstrap CI excluding +0.10 − SD (state the CI; a point estimate at the gate inside the seed SD is "at the gate", not a pass); finger hit ≥ 0.9 (override); (i) and (ii) ≤ C + δ on the sparse plan (with the M2 floor reported beside (ii)); executability 100 %.
- Side bundle S; inside / outside / swap split of (ii) (ii_split.py); ramp jerk; dense plan report-only; no-token output bitwise = enc-s0 (true freeze check).
- Controls: ungated lane-4 recipe on enc-s0 (= ctlenc, exists); gated both routes; gated ctl_flow-only; gated with the inference 5c blend on top (should change nothing if the gate works: reported as a consistency check).
- Reading rules: STG with the shuffle check and the cheat controls; FGD v2 primary.

**Quick lane (≤ 5 min, one local 5090).** 2000 steps true-freeze ctl_ training (ctl_ modules only), then offline seed 0 on 10 dev clips, dense plan: raw STG gain, outside-window max |y_given − y_none| (must be exactly 0 — the construction check), ctl_ray loss curve vs ctlenc. Kill: outside-window difference ≠ 0, or raw STG gain on the 10 clips < ctlenc's raw gain on the same clips − 0.02.

**Cost / risks.** 2–6 GPU-h (full: 2 variants × 3 inference seeds × offline / psync on DLC). Risk: the hard window truncates the approach (the diagnostic showed given − none rising from −0.4 s); the window start could be widened to −0.5 s *only* by re-registering before the run. Cheating: none specific; the frozen base guarantees the none row.

---

## D6. Stroke-lead word timing under a declared TTS-sync protocol (semantics, problem 7)

**Problem.** pair_win 0.547 vs no-text 0.491 vs GT 0.680 vs mismatched GT 0.522; 76–79 % of human semantic strokes start 0.19–0.23 s *before* the word; semantic heads, lexicon and interval tokens failed (rows 2, 10, 11, 22–25). Under psync the word becomes visible only after it is spoken, so even a perfect model is late by the word duration plus the lead.

**Mechanism.** The robot speaks through its own TTS, which produces the text and per-word timestamps before the audio is played; the TTS-sync protocol makes a word visible at (word onset − Δ) instead of at its offset. This is deployment-faithful for a TTS robot and must be declared as its own protocol row (psync stays the main streaming row; the audit's A1 "no future" rule is about *recorded* speech). Implementation: (a) data: a shifted view of the word times, t_word ← t_word − Δ, Δ = 0.2 s (the measured human lead), applied to the planner text stream times and to `sem_time` / word x-attn priors; (b) training: fine-tune enc-s0 for 2000 steps with the shifted times (flag `--word-shift -0.2`, default 0 = bitwise); (c) inference: the Streamer admits a word when t ≥ onset − Δ, which needs the TTS timestamps — in the dev evaluation these are the BEAT2 word alignments, exactly as psync uses them today; (d) a **late-shift control** (+0.2 s) and a **psync-visibility control** (shift in training only, psync at inference) isolate whether the gain is from the lead or from the retrain. Literature: Ferstl / Kucherenko-line timing studies and the 2001.09326 framework report gesture onsets 0.35–0.45 s before the aligned speech (pointing 0.38 s); Holler & Levinson's prediction account (gesture precedes to enable prediction). Closest prior: §7.46 anticipatory planning; the difference is that no predictor is trained — the lead comes from the TTS clock, which a robot legitimately has.

**Flag: base-changing.** It is a fine-tune of enc-s0; if adopted as the EX002 base every EX002 table re-runs. It is therefore offered as a *separate protocol row* first (the fine-tuned model is evaluated on both psync and TTS-sync), not as a base replacement.

**Pre-registration draft.**
- Primary: pair_win(full) − pair_win(no-text) on the TTS-sync row ≥ +0.10 with the recording-clustered CI excluding 0 (the §7.45 criterion), and pair_win(TTS-sync) > pair_win(psync) of the same model with CI excluding 0; the late-shift control must not show the gain.
- Side: FGD v2 / FKD within the enc seed range on psync (the retrain must not cost quality); liveliness; dBA-equivalent reported; executability unchanged.
- Sanity on the metric first (audit rule): pair_win GT = max, mismatched GT ≈ 0.5, reported in the same table.
- Controls: Δ ∈ {0 (retrain only), −0.2, +0.2}; psync inference of the −0.2 model.

**Quick lane (≤ 5 min, one 5090).** 2000-step fine-tune with Δ = −0.2, then pair_win on the 43-clip GT-event set, offline, seed 0, for the shifted and the Δ = 0 retrain. Kill: pair_win(−0.2) − pair_win(0) < +0.02 on this subset, or FGD v2 above the enc seed range.

**Cost / risks.** 3 GPU-h quick + ~9 GPU-h full (3 Δ × 3 training seeds would be 27; start with one training seed and three inference seeds). Risk: the recogniser's pair_win window may penalise early strokes — check that its window allows a stroke before the word (§7.46 note) *before* any model is scored, and if the window must change, change it for every method at once and log it. Cheating risk: visibility of future text is the exact thing psync forbids; the TTS-sync row is labelled, never mixed into the psync column, and the TTS timestamps used are the same alignments psync already consumes.

---

## Problems not given a design

- **Problem 6 (elbows 0.68–0.81 × GT).** The attribution probe in D2 comes first (10 GPU-minutes). Both plausible fixes (VAE `ch_w` on elbow channels with an interpolation-upsampling decoder; or a training-side per-joint velocity-moment regulariser on the decoded chunk, a new `w_` term in `losses.py`) retrain the VAE or the base and therefore re-run every EX002 table. An inference-time rescaling of elbow angles is not proposed: it is a direct optimisation of the amplitude metric and FGD v2 would be the only judge of whether it looks human.
- **Problem 8 (backchannel / listening).** Skipped: there is no dyadic data locally, so no validity gate can be set for any metric and no training positive exists; the Seamless restore is a user decision.
- **Problem 1 beyond D5.** Module 10 is queued; the STG shortfall is inside the inference-seed SD; the non-override (ii) excess is addressed by M2 (a gate problem, not a model problem).

---

## Metric work still needed (validity gates set before any model row is scored)

### M1. Paired stroke-suppression yield (PSS) — a self-normalised, paired "stops gesturing" measure
**Why.** Every yield definition so far (yield3d, v4 i1 / i2, activity-only) failed because humans after speech neither return to rest nor stop moving at a fixed absolute speed; the Holler 2023 phase coding says the turn-relevant cue is the *absence of new preparation / stroke*, not retraction. Activity-only failed because GT-in-speech also has quiet stretches (40.8 %); the fix is to normalise against the subject's own speaking behaviour and to pair against the counterfactual continuation.
**Definition.** For a stream y and an interruption at f, strokes = onsets where the larger wrist speed rises through v_s from below and stays above for ≥ 0.1 s; v_s = the p50 of per-utterance peak wrist speed on the **half-A** human speech windows (one fixed number). Stroke rate ρ(y, [a, b]) = strokes per second. PSS(y) = 1 − ρ(y, [f + 0.3, f + 2.13 s]) / max(ρ(U, [f + 0.3, f + 2.13 s]), ρ_min) where U is the same model's uninterrupted continuation (already the v3 reference) and ρ_min = the half-A speech stroke rate p10 (prevents division by a silent counterfactual). Pass per event: PSS ≥ 0.5. Reported with the withdrawal component and RKD, never alone (FRONT and B1 both pass PSS by construction and are rejected by withdrawal / RKD, as the audit already requires).
**Validity gates (half A calibrates; half B and the fresh dev-speaker set confirm; written before any model row):** (1) human tails: PSS of the human post-speech window against the *same recording's preceding 2.13 s of speech* (as its U) passes on ≥ 85 % of half-B tails and of the fresh set; (2) GT continuing to speak (window inside speech vs the preceding speech window) passes ≤ 15 %; (3) mismatched U (another recording's continuation) moves the pass rate by ≤ 10 pp on human tails (the measure must not depend on which U it is paired with, beyond the subject's own rate). If (1) or (2) fails, change only v_s or the 0.3 s skip on half A, log the iteration, re-run (2) and (3).
**Known limits.** Pairing with "own preceding speech" for humans vs "own counterfactual continuation" for models is not the same pairing; the audit must record it. B3 (generator continuing through the pause) is where PSS is informative: its U *is* its continuation, so PSS(B3) ≈ 0 by construction and a method that truly yields separates from it.

### M2. Obedience floor for quality gate (ii) — a human-splice control
**Why.** Audit E3 and the ii_split diagnosis show the whole (ii) excess sits inside command windows and that it is a spread effect: an obedient system that points when told necessarily has lower spread in those windows than GT at random times. The gate "(ii) ≤ no-token C + δ" therefore has no guarantee of being reachable by *any* obedient system, human-like or not. Row 36 already noted this for the dense plan; the sparse plan is also affected (none inside-only 89 vs tok 105).
**Definition.** Control stream H = dev GT motion with, at every command time of the sparse plan, a **real train-split human deictic window** (EVENTS.json, retargeted, the hand chosen by the plan's hand rule) inserted in joint space with the 9b adaptive ramps (point) or a train-split human presenting-like window (fingers extended, hand raised, from the same hand library source as module 10a) for finger commands; everything outside the insertions is GT. Score (ii)_H = FGD v2 of H vs GT, 3 seeds of window sampling. Register the gate as (ii) ≤ max(C + δ, (ii)_H + δ) and report (ii)_H in every control table.
**Validity gates.** (a) H with zero insertions scores 0 (GT self); (b) H with hindsight-token windows (random-time windows of the more-extended hand) instead of real deictic windows must score ≤ (ii)_H (a less marked insertion raises FGD less); (c) (ii)_H on the dense plan ≫ the sparse value (the gate stays sparse-only, as the audit decided). If (ii)_H ≤ C + δ, the current gate stands and the control is simply reported.
**Known limits.** Retargeted human windows carry the retargeter's own artefacts; the control measures "human motion spliced", not "human obeying", and must be labelled so.

---

## Cross-cutting pre-registration notes
- Order of runs: D1 and D2 quick lanes first (CPU / minutes, no GPU queue), then D3 and D4 (post-processing on existing generations), then M1 / M2 calibration (half A) in parallel, then D5 and D6 quick lanes on the local 5090s, each with its kill rule logged before launch. Full DLC lanes only for designs whose quick lane passed, one node, GPUs filled per the submission rules.
- Every design keeps the default path bitwise (unit test) and names its switch; no design touches training data outside the train split; dev speakers never enter a pool, library, regression or filter fit (D2 fits on train round-trips; D3 on half A; M1 / M2 calibrate on half A / train).
- Latency numbers (D1) only on an idle declared host; onboard not measured, stated.
- Reports state measured results and gate verdicts only; no paper framing.

---
## Main-agent review (11-08 20:10)
- Accepted for quick lanes: D1, D2, D3, D4, D5, M1.
- **M2 is report-only.** Using (ii)_H to raise the gate, after the token path has failed (ii), would loosen a gate post hoc. (ii)_H is reported beside (ii) in every control table; the registered gate stays as it is.
- **D6 is deferred.** It changes the base and adds a new protocol row. It runs only after D1–D5, and only as a separate labelled TTS-sync row.
- D3 supersedes the planned L2b / L2c. Its controls (a) and (b) cover both.
- Owners:
  - A: D1, D2, D3, M1.
  - B: D4 (after the full 9c'' run), D5 (after module 10), M2 (report-only).
- Every design runs its ≤ 5-minute quick lane with its kill rule first. Full runs need my go.
