# Certified streaming projection with interrupt-aware commit — statements and proofs

Written 2026-11-09 by the main agent (Claude). The draft comes from FABLE_THEORY_1109.md §1.1; every statement below was checked line by line against the code it describes:
- `scmg/safety_proj.py`: `ChunkQP`, `ProjectionLayer._project / step`, `diff_ok`, `pos_ok`, `static_safe`;
- `scmg/safety_proj_ex002.py`: the EX002 limits and collision model;
- `scripts/ex002/safety_int_ex002.py`: `isc_stream`.

Configuration: EX002 3c-4b, i.e. K = 16, B = 13, `recursive=True`, qdldl, tail_smooth 1e-3, cut cap 8.

AI-written; not yet checked by a person.

## 1. Setting and definitions
- Joint configuration q ∈ R^J. Frames at 50 Hz. The generator streams raw chunks r_c ∈ R^{K×J}, c = 0, 1, …. A raw chunk is arbitrary: no assumption on the generator.
- **Safe set S** (what "safe" means here; every term is the exact test in the code):
  - (S1) position: lo ≤ q_t ≤ hi for every executed frame (`pos_ok`);
  - (S2) derivatives: |Δ^k q_t| ≤ b_k for k = 1, 2, 3 on every finite difference of the executed sequence (`diff_ok` with `lim.step_bound(k)`; velocity, acceleration and jerk limits in per-frame units);
  - (S3) collision: `coll.collides(q_t) = False` for every executed frame, and for every midpoint ½(q_{t−1} + q_t) (`midpoints = True` on EX002). `collides` is the exact test of the robot's collision model: the WBC pairs with their margins.
- **Verified plan.** Given an executed history H (the last 3 frames), a sequence P = (p_1, …, p_m) is *verified* if:
  - [H; P; hold] satisfies S1–S3, where hold repeats p_m;
  - P ends at rest: p_m = p_{m−1} = p_{m−2}.
  Because P ends at rest, P followed by any number of copies of p_m is again verified.
- **Layer state.** After chunk c the layer stores a plan P_c of length ≥ B. One step (`ProjectionLayer.step`, `_project`) does this:
  1. Build the QP over N = K + B frames: position rows with margin, derivative rows with shrink 0.995, terminal-rest rows, and linearised collision rows (`lin_rows`).
  2. With `recursive=True`, relax every collision row to l ← min(l, a·x_ref), where x_ref = P_{c−1} padded with hold to N frames.
  3. Solve. Verify the solution x with S1–S3 on [H; x; hold] (`pos_ok`, `diff_ok`, `collides` on frames and midpoints).
  4. On failure, add cutting planes and solve again, up to the cap.
  5. **Accept:** if some x passes, execute x[:K] and store P_c = x[K:].
  6. **Fall back:** otherwise execute P_{c−1}[:K] (padded with hold) and store P_c = P_{c−1}[K:], padded with hold.

## 2. Results
**Assumption A0 (initial condition).** The stream starts at rest at a pose q_0 (in code, q_0 = static_safe(first raw frame) when there is no history) that satisfies S1 and S3, and P_{−1} = (q_0, …, q_0). In code, `static_safe` returns a contact-free pose inside the limits or raises an error. A plan of identical frames has zero differences, so P_{−1} is verified.

**Theorem 1 (invariance: zero violation for all time).** Under A0, for every raw input sequence r_0, r_1, … (arbitrary, including adversarial) and every solver output (including failures), the executed stream satisfies S1–S3 at every frame and midpoint, and after every chunk the stored plan P_c is verified with respect to the executed history.

*Proof (induction on c).* Hypothesis: P_{c−1} is verified with respect to H_{c−1}.
- (Accept) The executed frames are x[:K], which passed S1–S3 on [H; x; hold]. So the executed prefix is safe, and its differences with H are inside the bounds. The new plan P_c = x[K:] satisfies S1–S3 on [x[:K]; x[K:]; hold], and the terminal rows give rest. So P_c is verified with respect to H_c.
  - Note: the terminal rows are QP constraints and are not re-checked. If they were violated within solver tolerance, the hold extension in `diff_ok(H, [x; hold])` would still certify S2. Rest is used only for the padding argument below, and padding is with copies of the last frame, which `diff_ok` already checked.
- (Fall back) The executed frames are P_{c−1}[:K], with P_{c−1} padded with hold when shorter. P_{c−1} is verified with respect to H_{c−1}, so its first K frames satisfy S1–S3 together with H_{c−1}. The remainder, plus hold, is a suffix of a verified sequence and is therefore verified with respect to the new history.
- Neither case uses any property of the QP, the linearisation, the cutting planes or the solver. They only produce *proposals*. The certificate is the exact test in the accept case, and the inherited verified plan in the fallback case. ∎

**Corollary 1 (anytime safe stop).** At every chunk boundary the layer holds a verified plan of length ≤ B that ends at rest. Executing it, then holding, stops the robot within B frames (B = 13 → 0.26 s on EX002) inside S, with no optimisation.

**Proposition 2 (recursive feasibility of the proposal QP).** With `recursive=True`, if P_{c−1} was produced by an accepted QP solution, possibly shifted by fallbacks, then x_ref satisfies every row of chunk c's QP, up to the solver's feasibility tolerance:
- position and derivative rows: x_ref is a sub-sequence of an earlier accepted QP solution, extended by holds. Those satisfy the same margin and shrink bounds. The differences across the junction with H are rows of that earlier QP, and the hold frames add zero differences;
- terminal rows: x_ref ends in hold;
- collision rows: relaxed to admit x_ref by construction.
Hence the QP is feasible whenever the stored plan comes from an accepted solution, and solver-reported infeasibility can only be numerical. *Exception:* the very first plan P_{−1} = hold at `static_safe(q)` is not a QP output. Its position margin (1e-4 in `static_safe`) is larger than the layer's 1e-5, so it also satisfies the rows. *Record:* qp_fail = 0 on all 3c-4 / 3c-4b runs.

**Theorem 2 (interrupt-aware safety commit).** Let an interruption arrive at frame f inside the committed chunk c (frames c_0 … c_0 + K − 1). `isc_stream` does three things:
1. It keeps the executed frames x[:f − c_0].
2. It sets the stored plan to x[f − c_0:K] ++ P_c, i.e. the committed but unexecuted frames, followed by that chunk's verified tail.
3. It runs a step on the partial chunk y[f : c_0 + K] of the new (retraction) target.

Then Theorem 1, Corollary 1 and Proposition 2 hold unchanged for the partial chunk and all later chunks.

*Proof.* The sequence x[f − c_0:K] ++ P_c is a suffix of the accepted, certified sequence x ++ P_c, taken after the executed prefix x[:f − c_0]. So it is a verified plan with respect to the truncated history, and the induction of Theorem 1 continues from frame f. Proposition 2 applies because the plan is again a suffix of an accepted QP solution. ∎
- Consequence: the retraction can start at f, not at the next chunk boundary, with zero violation and no new fallback mode.
- *Record:* 1428 ISC streams, 0 fallback frames, unsafe 0.0 %; v3 for R0causal 10.3 % (no ISC) → 0.4 % (ISC).

## 3. What the results do NOT say (must be stated in the paper)
1. **Discrete time.** S3 is checked at frames and midpoints (100 Hz equivalent), not along the continuous path between them.
2. **Model-relative.** "Collision-free" means free under the robot's collision model (EX002 WBC pairs and margins; G1 MuJoCo geometry). The guarantee is only as good as the model, and the paper must report the model's calibration against MuJoCo.
3. **Kinematic only.** S2 bounds finite differences of the commanded joint trajectory. Tracking error of the low-level controller and dynamics are outside the theorem.
4. **Zero fallback is empirical**, not proved. Theorem 1 only guarantees that a fallback is safe. Fallbacks change the motion: it brakes to rest. Their share is reported (0 on 3c-4b dev runs).
5. **Latency is not guaranteed.** The per-chunk p99 is 184 ms on one CPU thread, which misses the 100 ms target. Under Corollary 1, a missed real-time deadline can be handled by executing the stored plan, which keeps safety. That deadline-fallback mode is not implemented or measured yet; stating it as a capability needs a run.
6. **A0 is required.** The stream must start at rest at a verified pose.

## 4. Experiments that support the statements (planned or run)
| statement | evidence | status |
|---|---|---|
| Thm 1 | 0 unsafe frames, 6 × 43 dev recordings + 1428 interruption streams | done (dev) |
| Thm 1, generator independence | stress test: spikes, a colliding target, a velocity ramp injected into the raw stream | queued (A) |
| Prop 2 | qp_fail = 0 on every 3c-4 / 3c-4b run | done |
| Thm 2 | ISC rows, 1428 streams | done |
| Contrast with prior filters | same-detector comparison vs the RoboGesture official filter (per-frame one-step QP, holds on failure) and a MIRA-style hold | running (A) |
| Deadline fallback (remark 5) | not implemented | open |
