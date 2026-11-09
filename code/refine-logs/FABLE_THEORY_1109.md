# SCMG — what can carry the paper as a method / theory innovation (Fable, design only, 2026-11-09)

Inputs read: NOVELTY_CHECK_1109.md (incl. the full-text notes on RoboGesture, MIRA, InteractGesture), FINAL_RECIPE.md (paper/final-1109), EVAL_AUDIT_1108.md (binding), FABLE_DESIGN_1108.md and the outcomes of its quick lanes (negatives rows 56–59; module 10 passed), the negatives tables (rows 1–59), `scmg/safety_proj.py` (ChunkQP, ProjectionLayer, fallback logic), `scmg/losses.py` (w_phys), EX002_CBF_1108.md, EX002_INTERRUPT_1107.md, EX002_SFSCOPE_1108.md, §7.6 of the design history (w_phys arm audit). No runs, no code changes, no sub-agents. Every statement below about *our* numbers is a quotation of a recorded dev / test number; every "would show" is a prediction, not a result.

---

## 1. Judgement of the existing candidates as method innovations

Summary table (details in 1.1–1.4):

| candidate | can carry a formal statement? | proof / analysis feasible in days | decisive extra experiment | reviewer risk | verdict |
|---|---|---|---|---|---|
| Recursive-feasible chunk projection + ISC | **Yes**: an invariance theorem (zero violation for all time, independent of generator and solver), an anytime safe-stop corollary, and an interruption-time re-planning theorem | Yes: ~2 pages, induction on chunks; all ingredients are in `safety_proj.py` | Same-detector comparison vs a RoboGesture-style per-frame one-step QP and a MIRA-style hold-last-pose, on the same streams; a stress test with adversarial raw targets | "This is a predictive safety filter (Wabersich & Zeilinger)" — true in spirit; the delta must be explicit. Latency gate not met | **Core method 1** |
| Encoder-only self-forcing | Partly: a consistency / gradient-routing proposition (the generator's objective stays the teacher-forced flow-matching objective; only the conditioning adapts to generated histories) + a solvable linear-Gaussian toy showing variance shrinkage under full SF | Yes: proposition in a day; toy in a day | G1 dev run (encoder-only vs full SF vs sf0, 2000 steps, 3 seeds) and the history-source ablation at inference | Found through a bug; single platform; name clash with Self Forcing (video) which avoids the collapse differently | **Core method 2, conditional on the G1 check** |
| Causal input matching | Weak: a covariate-shift statement (deployment loss = training loss when the conditioning distribution is matched) — true but textbook | Yes, trivial | Already done on G1 (1.961 / 1.946 vs 2.072 / 2.049); failed on EX002 | "Obvious"; platform-dependent | **Framework / protocol component**, not a theorem |
| Event tokens + keyframe insertion | Only a small identifiability remark about counterfactual tokens | — | — | InteractGesture, RoboGesture insertion points; token path adds little geometry; angle gate still fails | **Capability demo**, supporting |

### 1.1 Recursive-feasible chunk projection + interrupt-aware safety commit (3c-4b + ISC)

**What the code does (from `safety_proj.py`, read today).** State: the last 3 executed frames H and a *verified plan* P (a braking tail of B frames ending at rest). For a chunk of k raw frames r, a QP over N = k + B frames carries position bounds, velocity / acceleration / jerk difference bounds (including the differences across H), terminal rest rows (x_{N−1} = x_{N−2} = x_{N−3}), and collision half-spaces linearised at boundary points, plus cutting planes added after each exact test; every collision row is relaxed just enough to admit x_ref = shifted P + hold. A solution is accepted only if the **exact** tests pass: position box, difference tables on [H, x, hold], and the exact whole-body clearance (GJK, 349 pairs) on every frame **and midpoint**. If accepted, the first k frames are executed and P ← x[k:]. If not (QP failure, cap reached), the first k frames of the shifted P are executed and P ← P[k:] padded with the hold. ISC: at an interruption at frame f inside a committed chunk, the frames f..boundary are re-projected toward the retraction targets with the committed chunk's remaining frames + its braking tail as the candidate.

**Formal statements it can carry.** Let S be the exact safe set of finite sequences: every frame inside the position box, every k-th difference (k = 1, 2, 3) inside the table, every frame and every midpoint collision-free (clearance ≥ d_safe under the exact predicate), and let a *verified plan* be a sequence that lies in S together with the executed history and ends at rest (three equal final frames).

- **Theorem 1 (invariance / zero violation).** If the initial plan P_0 is verified, then for every chunk and every raw input sequence r_1, r_2, … (arbitrary, including adversarial), the executed stream lies in S for all time, and after every chunk a verified plan exists. *Proof sketch (induction).* Accepted case: x passed the exact tests on [H, x, hold], so x[:k] is safe with the history and x[k:] + hold is a verified plan (rest + hold has zero differences and the same collision-free pose). Rejected case: the shifted P is verified by hypothesis and its first k frames are consistent with H because H consists of frames that were either x[:k] with P = x[k:] (same sequence) or P[:k] (same sequence). No property of the QP, the linearisation or the solver is used — they only produce *proposals*; the certificate is the exact test. ∎
- **Corollary 1 (anytime safe stop).** At every chunk boundary the system can come to rest within B frames (13 frames = 0.26 s on EX002) while staying in S, without any solve. (This is exactly what a barge-in needs.)
- **Proposition 2 (recursive feasibility of the proposal QP).** With the relaxation l_row ← min(l_row, a_row·x_ref), x_ref is feasible for every row (position: P verified; differences: P is consistent with H; terminal: P ends at rest; collision rows: relaxed by construction). So the QP is feasible whenever a verified plan exists; infeasibility can only be numerical. Matches the record: qp_fail 0 in 3c-4 / 3c-4b.
- **Theorem 2 (interruption-time re-planning).** If at an interruption at frame f the candidate is the committed chunk's unexecuted frames f..boundary followed by its braking tail, Theorem 1 and Proposition 2 hold for the partial chunk, so the retraction can be re-planned from f (not from the next boundary) with zero violation and no new fallback mode. Record: 1428 ISC streams, 0 fallback, unsafe 0.0 %, v3 10.3 % → 0.4 % (R0causal).
- **Non-statement (must be said).** Zero *fallback* is empirical (0 in 6 runs and 1428 streams on enc-s0), not a theorem; the theorem gives zero *violation* and bounded-jerk braking on fallback (hold-onset jerk ≤ 0.012 × the table, measured).

**Closest prior and delta.** Predictive safety filters (Wabersich & Zeilinger, 2018 / Automatica 2021) wrap a learning policy with an MPC that keeps a terminal safe set and prove recursive feasibility by the shifted-candidate argument (Mayne et al. 2000). Our delta, to be stated plainly: (i) the certificate is the *exact nonconvex* whole-body clearance (349 pairs, GJK, frames + midpoints), with convex cutting-plane QPs only as proposals, so safety does not rest on a convexification; (ii) chunked commit with a braking tail as the terminal set, so the filter is compatible with a 320 ms streaming generator; (iii) partial-chunk re-planning at barge-in (ISC) that reuses the same candidate; (iv) measured against two robot baselines that lack the property: RoboGesture's per-frame one-step OSQP (collision rows only, no limits, residual 0.13–0.88 %) and MIRA's hold-last-pose on projection failure (a hold from a moving state violates the derivative tables; 2.61 % self-collision).

**Feasible analysis in days.** The two theorems and the proposition above, written with the exact set S and the rest / hold lemma (a held rest pose has zero differences; a collision-free pose stays collision-free under hold). One day.

**Extra experiment that makes it convincing (pre-register; dev only).**
1. Same-detector comparison on the enc-s0 streams (offline / psync, seeds 0–2) and the 476 interruptions: 3c-4b (+ ISC) vs (a) RoboGesture-style per-frame one-step QP with linearised collision rows only (our `CBFQPLayer` with limits off is the nearest existing object; a faithful variant is a day of work), (b) MIRA-style per-frame projection with hold-last-pose on failure, (c) clip. Columns: unsafe frames (retargeter WBC, position, velocity), fallback / hold frames, hold-onset jerk median / p95 / max, FGD v2, FKD, liveliness, p50 / p99. Reading rules: FKD never alone; the violation detector is ours for all rows (the "0 vs 0.13 %" caveat from the novelty check).
2. Stress test of Theorem 1: inject into the raw stream (i) random 0.5 rad spikes on 1 % of frames, (ii) a 1 s colliding target (hands crossing), (iii) a velocity-limit violating ramp. Gate: 0 unsafe frames on every injected stream; fallback share and brake jerk reported. This isolates "independent of the generator".
3. ISC reaction latency: interruption → first changed executed frame, wall clock, idle machine, median / p95.

**Reviewer risk.** (a) "A predictive safety filter with a GJK check": mitigated by the explicit delta and the two robot baselines; (b) latency 184 ms p99 vs the 320 ms stride (it keeps up with the stream but exceeds the 100 ms gate) — report as is, with proposal N1 below as the lever; (c) only EX002 (the G1 layer is the per-frame CBF-QP) — state it.

### 1.2 Encoder-only self-forcing (`self_force_scope = 'encoders'`)

**Observation to formalise.** Three regimes, same recipe, EX002 dev, 2-seed means, offline / psync FGD v2: sf0 172.0 / 212.3, full SF 119.7 / 143.4, encoder-only 86.5 / 105.8; only encoder-only passes the liveliness guard (still 0.024 / wrist 0.334 m/s / amp 0.097 vs GT 0.017 / 0.411 / 0.120). On G1 (fixed code) sf0 was selected because full SF caused the 320 ms stutter; encoder-only was never run on G1.

**Formal account (what is defensible).** Write the chunk model as v_θ(x, t | h, c) with h the history latents and c = E_φ(audio, text, speaker). Teacher forcing minimises the flow-matching loss L_TF(θ, φ) = E_{h~GT}[ℓ(v_θ(· | h, E_φ))]; its minimiser in θ is the true conditional velocity field given GT history (standard flow-matching consistency). Full self-forcing minimises a mixture L_TF + L_SF with L_SF = E_{h~ĥ_θ}[ℓ(v_θ(· | ĥ_θ, E_φ))], where ĥ_θ is generated by the model itself. This is scheduled sampling / DAgger applied to a per-step regression objective; Huszár (2015, "How (not) to train your generative model") shows such objectives are *not* consistent with maximum likelihood: their optimum is pulled toward predictions that are good on average over the model's own (shifted) histories, i.e. toward the conditional mean over a broader history distribution, which lowers conditional variance — in motion, amplitude. Motar (2609.10317) reports exactly this collapse under MSE self-forcing and fixes it with distribution-level (adversarial / DMD) losses. Encoder-only scope keeps ∇_θ L_SF = 0: the generator's objective is exactly L_TF (consistent), while φ receives ∇_φ L_SF, i.e. the condition encoders are trained to make c informative *when the history is the model's own*. Statement: **Proposition 3.** Under the encoder-only scope, the generator's stationary points are those of teacher forcing (no bias from self-generated histories), and self-forcing acts only as a domain adaptation of the conditioning. The *damping* of full SF is then predicted, not proved, by the inconsistency argument.

**Feasible analysis in days.** (i) Proposition 3 as above (half a day). (ii) A linear-Gaussian AR(1) toy with a learned conditioning gain: closed-form or 100-line numerics showing that full SF with a squared loss shrinks the learned autoregressive gain (variance) while encoder-only leaves it at the teacher-forced value (one day). (iii) History-source ablation at inference on the three EX002 models: amplitude and still-share when the history is GT (teacher-forced roll-in) vs generated — the account predicts full SF loses amplitude already under GT history (the generator itself was changed), encoder-only does not.

**Extra experiment that makes it convincing.** G1 dev, 2000-step fine-tunes from the fix-base, 3 training seeds × per_clip 0 / 1: sf0 vs full SF vs encoder-only, ECHO-G official evaluator (FGD, ΔBA, ΔDiv, ΔJerk) + the 320 ms phase stutter measure + liveliness. Pre-registered: encoder-only is "confirmed cross-platform" iff FGD ≤ sf0 within the seed range on psync and the stutter is at the sf0 level; otherwise it stays an EX002 finding. Quick lane first (one 5090, 2000 steps, seed 0).

**Reviewer risk.** Found through an autocast-cache bug (the honest history must be told; it strengthens the "intended ablation" story only if the G1 check passes); the scheme is simple (a gradient mask), so the contribution is the finding + the account, not machinery; name clash with Self Forcing (Huang et al. 2025) — call it "encoder-scoped self-forcing" or "conditioning-only self-forcing" in the text to avoid confusion.

### 1.3 Causal input matching (czm)

The statement "if the conditioning inputs at training are drawn from the deployment's incremental-visibility process, the training loss is the deployment loss" is a covariate-shift identity, not a theorem anyone will credit. Evidence: G1 test 1.961 / 1.946 vs 2.072 / 2.049 (bootstrap pending), offline slightly worse (1.465 vs 1.389 dev); EX002: fails the frozen rule (psync 126.3 vs 109.0). Judgement: this is the **protocol / framework** contribution — "deployment-faithful strict causality including incrementally revealed text" — which the novelty check found neither robot system has. Present it as the framework's input contract with the G1 ablation, not as a method theorem.

### 1.4 Event tokens / keyframe insertion

Counterfactual token training has a one-line identifiability remark (with swapped targets, the only loss-reducing path is to follow the token), but the geometry is now solved by IK (the none + IK control matches the given + IK row on 2 clips), tok10ab passes its gates while 9c'' fails the apex-angle gate, the commands are hindsight relabels (audit G1), and InteractGesture / RoboGesture cover streaming control. Judgement: **capability demo** in the EX002 chapter, no innovation claim.

---

## 2. New method proposals (≤ 2, ranked; deadline = days)

### N1. Infeasibility-aware generator: contract loss on the EX002 base to make the certified layer real-time ("the latency of a certified projection is governed by the raw infeasibility rate")

**Problem it closes.** 3c-4b fails only gate 4 (p99 184 / 171 ms vs 100 ms). The record ties the latency to the raw collision rate: the voided czm pick (raw WBC-violating frames ~3 %) reached p99 77 ms with the memoised 3c-3c layer, while enc-s0 (raw WBC 4.9–7.6 %) needs 2–5 solves on ~5 % of chunks (EX002_CBF_1108, 3c-4b result). D1 (decomposition) was killed. The remaining lever is the generator's raw infeasibility, which the existing EX002 contract loss `w_phys` (joint limits + sphere-proxy self-collision on the decoded chunk, `phys_coll_w`, `phys_dyn_w`) already targets: on the pre-fix lively recipe (sbag, i.e. the encoder-only regime by accident), w_phys 0.03 cut self-contact frames from 1.4–3.8 % to 0.1 % with the upper-arm amplitude ratio 0.73–0.89 vs 0.79–0.89 without it (§7.6). A weight with this recipe exists: `landing-l1-1104/sibag-coll-s1/final.pt` (w_phys 0.03, phys_dyn_w 0; pre-fix code = encoder-only self-forcing by the bisect finding).

**Mechanism.** enc-s0 recipe + `w_phys 0.03, phys_coll_w 1, phys_dyn_w 0` (collision + limits only; the derivative term is left to the projection), `self_force_scope encoders`, 2000 steps, 3 training seeds — i.e. the G1 final-recipe pairing "contract loss + projection" ported to EX002 with the collision term that EX002 needs. Default off (w_phys = 0 is the current path). Then 3c-4b unchanged on top.

**Formal grounding (modest, measurable).** For the cutting-plane loop, the number of solves per chunk is 1 + (number of rounds in which the exact test fails); a round fails only on frames whose raw target (or its linearised neighbourhood) is infeasible. So E[solves per chunk] is a monotone function of the per-chunk raw infeasibility probability, and the p99 of the chunk time is set by the tail of that distribution. Pre-registered *law to test*: p99 and the ≥ 2-solve share are monotone in the raw WBC-violating frame share across the models we hold (czm pick ~3 %, enc-s0 5–7 %, sibag-coll, the new lanes). Not a theorem about the generator; a quantitative statement about the layer that makes the contract loss the designed latency lever rather than a tuning accident.

**Closest prior and delta.** Training-time constraint penalties + inference-time filters are standard (our own G1 w_safe 300 + CBF-QP; RoboGesture trains without constraints and filters per frame). Delta: the penalty is justified by the layer's solve-count law and is paired with a certified, recursively feasible layer whose latency it controls; the penalty's effect on liveliness is gated, not assumed.

**Pre-registration draft.**
- Selection gates (dev, 43 recordings, offline + psync, per_clip 0–2, 3 training seeds): (1) liveliness guard passes (still ≤ 2× GT, wrist speed and amplitude ≥ 0.75× GT, moving upper body incl. fingers); (2) FGD v2 within the enc seed range (offline 85.5–91.2, psync 95.1–109.0) or better, FKD ≤ 1.10 × enc; (3) raw WBC-violating frame share ≤ 3 % (half of enc-s0); (4) 3c-4b on its outputs: gates 1–3 as registered (100 % clear, 0 fallback, jerk 0 %, dFGD / dFKD within 1 seed SD) **and gate 4: idle p99 ≤ 100 ms, 1 thread, same host as the 3c-4b numbers**; reproducibility bit-identical.
- Report-only: upper-arm amplitude ratio (§7.6 showed a cost under sf0), elbow ratio, solve-count distribution, the solves-vs-violation-rate curve over all models held, liveliness of the layer's output.
- Controls: enc-s0 (exists); enc + w_phys 0.1 (the czm sweep value) as the over-regularised arm; `sibag-coll-s1` (existing weight) scored as the no-training check of the direction.
- Reading rules: FGD v2 primary; latency only idle; FKD never alone.

**Quick lane (≤ 5 min, no training needed first).** Step A (measurement only, CPU + one GPU for decoding): run `sibag-coll-s1` offline s0 on ci 7 / 37 and 3 more recordings: raw WBC share, liveliness on those clips, then 3c-4b on them: solves per chunk, p99 vs enc-s0 on the same chunks in the same session. Kill A: raw WBC share not ≤ 0.6 × enc-s0's on these clips, or liveliness fails. Step B (one 5090, ~5 min): 2000-step enc recipe + w_phys 0.03 seed 0, same 5 clips, same numbers. Kill B: as A, or FGD v2 on the 5 clips outside the enc seed range by more than the seed SD.

**Cost.** Quick lanes ≈ 0.2 GPU-h; full 3 seeds × bench + 3c-4b 6 runs ≈ 3 GPU-h + ~6 CPU-h. **Changes the base: yes** (new EX002 base candidate → every EX002 dev table re-runs; the EX002 test set has not been touched, so no test re-run is spent). If it fails gate 4 but passes 1–3, it is still the lower-violation base and the latency stays an open item, reported.

**Risks / cheats.** Amplitude loss (the §7.6 cost) — gated by liveliness and the amplitude ratios; the collision proxy is the sphere model, not the exact WBC — gate (3) is on the exact detector; a model could lower violations by keeping the hands apart in a stiff way — FGD v2 + liveliness + the arm-audit columns catch it.

### N2. Yield-or-stop: the retraction tube as hard rows of the certified projection

**Problem.** Interruption rows are post-processors (R0causal / Funnel) followed by the layer; the layer may brake or re-time the retraction (R0causal v3 5.9 % raw → 10.3 % after the layer, fixed by ISC to 0.4 %). There is no statement about *what the interrupted stream is guaranteed to do*. Also: no validated yield metric exists, so the paper cannot claim human-like yielding; it can claim a guarantee.

**Mechanism.** After an interruption at f, add to every chunk QP (the ISC partial chunk and the following chunks) per-frame box rows on the waist + arm joints: |x_t − r(t)| ≤ e(t), where r(t) is the critically damped retraction toward the rest pose (R0causal's trajectory, already computed causally) and e(t) = e_0 φ(t) + ε is the Funnel's contracting tube (φ critically damped, T_f), with the same candidate relaxation as the collision rows (l ← min(l, x_ref) so Proposition 2 still holds), and the raw generator continuation as the QP target. The generator's texture survives inside the tube; outside it is pulled back by the projection instead of by a tanh compressor; the exact certificate is unchanged. Default off (`--tube`), bit-identical when off.

**Formal grounding.** **Theorem 3 (yield-or-stop).** Under Theorem 1's hypotheses, after an interruption at f, at every chunk boundary either the executed frames lie inside the tube around r(t) (so the stream converges to within ε of the rest pose by T_f), or a verified safe stop is executed (Corollary 1). Hence the interrupted stream is always within S and is either retracting on schedule or stopped within B frames — never continuing to gesture. Proof: the tube rows are hard unless relaxed for x_ref; a chunk is accepted only inside the (possibly relaxed) rows, and a relaxed row admits only x_ref's frames, which belong to the verified stop plan. ∎ Honest limit: when relaxation binds, the robot stops where it is instead of retracting; the share of such chunks is a reported number.

**Closest prior and delta.** The convergence funnel (ours, post-hoc tanh compression), tube MPC (Mayne 2005; Langson 2004) for robust constraint satisfaction, MIRA's hold-at-barge-in (a stop with no retraction and no guarantee). Delta: the tube is inside the certified projection with the shifted-candidate relaxation, which yields the yield-or-stop dichotomy; MIRA gives only "stop".

**Pre-registration draft.** Rows on enc-s0, half-B reference, 476 interruptions, deployed chain with ISC: tube (T_f ∈ {1.8, 2.2}, m_0 ∈ {0.8, 1.0} rad, ε 0.02 — the Funnel's registered grid; the choice rule = the interruption rule: v3 ≤ 10 %, unsafe ≤ 1 %, lower RKD-post with the recording bootstrap excluding 0 vs R0causal + ISC 0.518, yield3d and the components reported, no single yield gate). Report-only: share of chunks where a tube row was relaxed (the "stop" branch), time to within ε of rest, reaction latency, first-0.3 s outward drift median / p95 / max. Controls: Funnel + ISC (0.484 / 89.2 %), R0causal + ISC (0.518 / 97.9 %), B3 + ISC, tube without relaxation accounting (must show qp_fail > 0 — demonstrates why the relaxation is needed).

**Quick lane (≤ 5 min, CPU).** The 41-interruption subset used by L2d (ci 2 / 3 / 5 / 6 / 7), ISC chain: RKD-post, v3, unsafe, relaxed-row share, time-to-ε. Kill: RKD-post ≥ 0.518 on the subset, or v3 > 10 %, or unsafe > 0.

**Cost.** ≈ 0 GPU-h, ~2 CPU-h for the 476 × configurations. **Changes the base: no** (post-processing / projection only). Risk: the tube over-constrains and the "stop" branch dominates (reported as the relaxed share — if > 10 % of chunks, the row is a negative with that diagnosis); cheating risk on RKD: the tube pulls toward the median rest pose like R0causal, so RKD cannot be gamed beyond R0causal's own value, and the dichotomy claim does not depend on RKD at all.

---

## 3. Recommended framing (honest; flags what is unproven)

**Framework.** A deployment-faithful, event-driven streaming gesture generator for a talking humanoid: strictly causal inputs including incrementally revealed TTS text (the input contract; G1 ablation czm vs ncws), a chunked flow generator, and one **time-varying feasible-set projection** that implements both execution safety and interruption (the braking tail is the terminal set; interruption changes the set). Event tokens / keyframes are the capability layer of the same event machinery.

**Method innovation 1 (ready to write; strongest).** *Certified streaming projection with interrupt-aware commit*: Theorem 1 (zero violation for all time, independent of the generator and of the solver), Corollary 1 (anytime safe stop within B frames), Proposition 2 (recursive feasibility of the proposal QP), Theorem 2 (ISC), with the record: 0 unsafe frames and 0 fallback on 6 × 43 dev recordings and 1428 interruption streams, bit-reproducible, v3 10.3 % → 0.4 %. Add N2 (Theorem 3, yield-or-stop) if its quick lane passes — it turns the interruption column from "a post-processor that happens to pass gates" into a guaranteed behaviour, which also side-steps the missing yield metric. Add N1 as the latency lever if its gates pass; otherwise report p99 184 ms with the law and the open gate.

**Method innovation 2 (conditional).** *Encoder-scoped self-forcing*: Proposition 3 + the AR toy + the EX002 three-regime table; promoted to a cross-platform method only if the G1 dev check passes; otherwise it goes to the EX002 chapter as a finding with the honest bug history.

**Not framed as innovations.** Causal input matching (framework component, protocol-level), event tokens / IK keyframes (capability), the interruption benchmark (evaluation contribution with its validity record: no validated yield metric; report abruptness, safety and distributional likeness only, per the audit).

**Still unproven / must be flagged in any text.**
- G1 vs ECHO-G paired bootstrap pending; EX002 test set not run (protocol not frozen).
- Zero fallback is empirical; the theorem gives zero violation.
- Latency gate 4 not met (p99 184 ms, 1 thread, 9950X; not measured onboard).
- Encoder-only self-forcing not shown on G1; its mechanism is an account, not a proof of damping.
- The "0 vs 0.13 % (RoboGesture)" comparison is across detectors until the same-detector baseline is run.
- No validated yield measure (5 definitions + PSS failed the human gates); interruption claims are about abruptness, safety, likeness and — with N2 — a guarantee, never "yields like a human".
- EX002 baselines (GestureLSM / EMAGE retargeted) not tuned; ~9.4 M dead AdaLN parameters; normalisation stats included 0.39 % dev rows (all recorded limitations).

**Suggested order for the next days.** (1) Write Theorems 1–2 / Prop. 2 from `safety_proj.py` (1 day). (2) N1 quick lane A (measurement on `sibag-coll-s1`, no training) and N2 quick lane (CPU) the same day; N1 step B on a 5090 if A passes. (3) Same-detector baselines (RoboGesture-style one-step QP, MIRA-style hold) + the stress test (1–2 days, CPU). (4) G1 encoder-only quick lane (5090, 5 min), then the 3-seed DLC lane if it passes. Every lane pre-registered in its criteria file before launch, dev only, default-off switches with bitwise tests.

---
## Main-agent review (11-09 01:10)
- **Accepted framing:** framework + method 1 (certified streaming projection with interrupt-aware commit) + conditional method 2 (encoder-scoped self-forcing).
- **Caveats that must be stated with Theorem 1:**
  - Safety is certified in discrete time, on the checked frames plus midpoints, at 50 Hz.
  - It holds relative to the collision model used by the exact test (its geometry and margin).
  - It covers joint position / velocity / jerk rows as encoded.
  - It holds relative to the reference initial plan P_0.
  - Zero fallback stays empirical.
- **Order and owners:**
  - A: N2 quick lane (CPU); stress test + MIRA-style hold row folded into the RoboGesture same-detector comparison; the D6 quick lane stays queued after these.
  - B: N1 quick lane A (measurement on sibag-coll-s1, no training); then the G1 encoder-only self-forcing quick lane; after 9c'' / D5.
  - Main: the Theorems 1–2 / Prop. 2 write-up, checked line by line against safety_proj.py.
- **Base change:** N1 changes the base. It is adopted only if the latency and liveliness gates both pass; otherwise it stays report-only.

## Correction 11-09 (agent A, from the N2 quick lane; coordinator)
Theorem 3 (yield-or-stop) as written is not proved. Its proof says "a relaxed row admits only x_ref's frames"; the relaxation actually
used (l <- min(l, x_ref), u <- max(u, x_ref), the same as for the collision rows) widens each tube bound to an interval that CONTAINS
x_ref, so the QP may follow the raw continuation anywhere inside the widened interval. Measured (N2 quick lane, 41 interruptions):
tube rows relaxed in 72.5-77.7 % of post-interruption chunks, the accepted plan outside the unrelaxed tube in 67-75 %, behaviour equal
to B3 + ISC (negatives row 60). Theorems 1-2 are unaffected: their safety claim rests on the exact certificate (GJK test of every frame
and midpoint) and on the verified plan as the candidate, not on the relaxed rows. A literal yield-or-stop needs hard tube rows with the
verified plan executed on infeasibility (N2b, quick lane registered in EX002_INTERRUPT_1107.md).

### Corrected Theorem 3 (yield-or-stop, as implemented by N2b; agent A 11-09, checked against the code path)
Setting: the 3c-4b layer (scmg/safety_proj.py ProjectionLayer, proj_l2, recursive feasibility, cap 8) with N2b tube rows: for every
frame t >= f of an interruption at f, box rows r_j(t) - e(t) <= x_{t,j} <= r_j(t) + e(t) on the waist + arm joints j (q34 0-1,
4-17), r = the R0causal trajectory of the event, e(t) = m_0 phi(t - f) + eps; the tube rows are HARD (ChunkQP.matrices: no x_ref
relaxation when tube['relax'] is False), the collision rows keep the 3c-4b relaxation. Hypotheses of Theorem 1 (verified initial
plan, exact certificate = GJK test of every frame and midpoint, limits as encoded).
Statement. At every chunk step after f exactly one of the following holds:
 (a) the chunk is a newly accepted plan; every frame of it lies inside the tube on the tube joints (to the QP solver's tolerance)
     and inside the safe set S (exact certificate); or
 (b) the chunk executes frames of the stored verified plan (the "stop" branch): the previous accepted plan's remaining frames, which
     end in its braking tail (velocity and acceleration zero at the tail end, terminal rows) followed by a hold.
Hence the interrupted stream is always in S, and on the waist / arms it is, chunk by chunk, either inside the contracting tube
(converging to within eps of r(t) by T_f) or braking to a verified standstill; it never leaves the tube with a newly planned motion.
Proof (code path). _project returns a plan only after (i) qp.solve returns a solution (Clarabel 'Solved'), i.e. a point satisfying
every row of A x in [l, u], including the hard tube rows, to the solver tolerance; and (ii) pos_ok and diff_ok (limits) and the exact
collision test of every frame and midpoint pass. If the QP is infeasible (qp.solve -> None) or the certificate still fails after 8
cutting-plane rounds, _project returns None and step() executes plan[:k] and keeps plan[k:] (padded by its last frame) -- frames of a
plan that was itself certified when it was accepted (Theorem 1). Tube rows are built in _project for absolute frames >= f only
(len(hist) minus the 3 unemitted rest-start frames), so the ISC partial chunk at f is covered. A sequence of (b) steps exhausts the
braking tail and holds its final pose (zero difference rows), which is collision-free (the hold lemma of Theorem 1).
Limits, stated: (a) is to the solver tolerance (the tube rows are not re-checked exactly after the solve); the tube constrains waist +
arms only (head and fingers keep the continuation); "stop" is per chunk, not latched: a later chunk may re-enter (a) when the QP is
feasible again; the guarantee is relative to r(t) (R0causal's trajectory toward the half-B rest pose), not to a human-likeness measure.
Measured (N2b quick lane, 41 events, T_f 1.8, m_0 0.8): (b) occurred in 35 of 382 post-interruption chunks (9.2 %), in 6 events,
median 1.12 s after t_i, all from QP infeasibility (qp_fail 35); unsafe frames 0.
