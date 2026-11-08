# Novelty check — SCMG (2026-11-09, main agent)

Method: the novelty-check skill, run in a degraded form.
- Phase B: 12 web searches (standard and extended).
- Phase C: the cross-model check (Codex MCP) was NOT available in this session, so there is no second-model verdict.
- verify_papers.py was not run. Every paper below was seen as an arXiv / CVF page in the search results; abstracts were read through search snippets, not full texts.
- Treat each delta as "from abstracts". A system-level full read of the 3 closest papers (RoboGesture, MIRA, InteractGesture) is still needed.

## Core claims
| # | claim | novelty | closest prior | delta (from abstracts) |
|---|---|---|---|---|
| 1 | Streaming co-speech motion on a humanoid from streaming TTS speech | LOW as stated | RoboGesture (arXiv 2608.28693, ECCV 2026: streaming listen–respond–gesture on G1 + dexterous hands, MPC safety filter); MIRA (2609.24547: full-duplex, ROSCO streaming diffusion, cancellable committed prefix, Astribot S1); SocialHumanoid (2609.33311); GestureFAR (2609.21576), LiveGesture (2604.10927), Super Star (2608.24909) on avatars | Narrower claim: strict causality including incrementally revealed TEXT (a word is visible only after it is spoken), and a training-time match to the deployment's causal inputs, on the official ECHO-G G1 benchmark. ECHO-G (2609.39575) says it does not stream. Our G1 test: causal match 1.961 / 1.946 vs no match 2.072 / 2.049 FGD (bootstrap pending). Note: removing text does not hurt FGD on G1 |
| 2 | Encoder-only self-forcing (self-forced batches update only the condition encoders) | MEDIUM–HIGH (no prior found) | Self Forcing (2506.08009, video); Motar (2609.10317): self-forcing with MSE collapses toward static, under-animated motion, fixed with adversarial / DMD losses; FloodDiffusion, MotionStreamer (diffusion forcing for motion) | Different fix for the same failure Motar reports: restrict self-forced gradients to the condition encoders. EX002 evidence: FGD v2 86.5 / 105.8 vs full SF 119.7 / 143.4 vs none 172.0 / 212.3, liveliness only for encoder-only. Risks: found through a bug; not yet shown on G1; mechanism not isolated |
| 3 | Recursively feasible chunked QP safety projection + interrupt-aware safety commit | MEDIUM | RoboGesture's MPC filter (per-frame OSQP, collisions 4.16 % → 0.13 %); geometry-aware predictive safety filters on humanoids (2508.11129, CBF-MPC, recursive feasibility in navigation); CBF motion-imitation filters (2604.11447, 2605.25546) | Exact whole-body certificate per frame + recursive feasibility → 0 unsafe frames and 0 fallback on every stream, plus re-planning at barge-in (ISC). Weakness: p99 184 ms (target 100) |
| 4 | Graceful interruption: retraction on barge-in, a human post-speech reference benchmark | MEDIUM | MIRA (barge-in abort pauses speech and HOLDS motion; cancellable 15-frame prefix); NCSOFT SIGGRAPH Asia 2024 (gesture phases for barge-in, no metrics); VideoFDB (yield within 1–1.5 s, speech agents); Holler et al. 2023 (retraction is not a turn-yield cue) | Retract vs hold; a benchmark with a human post-speech reference (abruptness v3, RKD, unsafe), with ISC giving v3 0.4 % / unsafe 0. Risk: no validated "yield" metric (5 definitions failed the human gates) |
| 5 | Event-token control (pointing / counting) + IK keyframe insertion during streaming | LOW–MEDIUM | InteractGesture (2608.25734: streaming keyframe, trajectory and pointing-direction control on BEAT2); RoboGesture (gesture insertion points, dexterous hands); Honda 2010; Robot Deictics 2014 | On a robot with three-finger hands under strict causality, with the safety layer. The token path adds little to geometry (the none + IK control matches it on 2 clips); 9c'' still fails the angle gate |

## Overall
- Score: 5 / 10. Recommendation: PROCEED WITH CAUTION.
- "Streaming robot co-speech gesture" is taken: RoboGesture (ECCV 2026), MIRA, SocialHumanoid. The contribution must be the combination of deployment-faithful strict causality, certified execution and graceful interruption, with each piece measured.
- Reviewers will cite RoboGesture (streaming + MPC safety on G1), MIRA (full-duplex barge-in on a humanoid) and InteractGesture (streaming pointing control).

## Candidate contribution set (proposal for the user; the user decides)
- C1. Deployment-faithful strictly causal streaming on the official G1 benchmark: causal-input training match, measured against ECHO-G and against the no-match ablation.
- C2. Interruptible and certified execution: recursive-feasible chunk projection + interrupt-aware safety commit (0 unsafe, 0 fallback, abruptness 0.4 %), evaluated against a human post-speech reference. Include the honest metric-validity findings.
- C3 (secondary method finding). Encoder-only self-forcing keeps liveliness where full self-forcing damps motion. Needs a G1 check and a mechanism ablation before it can stand.
- Supporting, not core: event-token control and IK insertion (pointing / counting) as a capability demo.

## To do before relying on this
1. Read the full text of RoboGesture, MIRA and InteractGesture; confirm each delta above.
2. A second-model check, if Codex MCP becomes available.
3. Run C3 on G1 (dev) and an isolation ablation.

## Full-text read of the 3 closest papers (11-09 00:40; web-fetch of the arXiv HTML; some passages were tool summaries and are marked so in the agent report)
- **RoboGesture (2608.28693).**
  - Audio only ("text-free"); text is used only upstream (ASR → LLM → TTS).
  - Not strictly causal: each 1 s chunk sees about 1 s of future audio inside its window.
  - The safety filter is per frame with one-step Δq: an OSQP QP with linearised collision constraints only. Joint limits are not in the QP, and velocity is only a cost.
  - No recursive-feasibility or zero-violation claim; infeasibility handling is not stated. Residual self-collision is 0.13 % (SemanticBEAT) / 0.88 % (BEAT), measured on their own MuJoCo primitives (dist < −0.01).
  - No interruption handling. No comparison with ECHO-G. Code and weights not stated.
- **MIRA (2609.24547).**
  - Audio only, causal; predicts 50 frames and commits 15.
  - At barge-in it terminates the motion and holds the last safe pose (or a "fallback policy" that is not described).
  - Evaluation is preemption latency only (466 ms median); there is no measure of the motion after the stop and no human comparison.
  - The safety layer is a per-frame collision-aware IK projection that holds the last pose if projection fails. Self-collision 2.61 %.
- **InteractGesture (2608.25734).**
  - Inference-time guidance on a frozen GestureLSM; not strictly causal (later chunks back-propagate into earlier ones).
  - SMPL-X only, no robot. No pointing-specific metric.
- **Effect on the claims:**
  - C1 is stronger. Neither robot system uses incrementally revealed text, and neither is strictly causal in both modalities (RoboGesture has in-chunk future audio).
  - C2 is stronger: residual collisions vs a 0 / 0 certificate with recursive feasibility; hold vs a measured retraction with a human reference.
  - Control (C5) stays weak in novelty, but ours is on a robot and causal.
  - Caveat: the "0 vs 0.13 %" comparison needs one detector (task 2, agent A).
