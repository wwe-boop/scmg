# Duplex mode: interruption as a first-class streaming event (written 2026-11-07 18:xx, before any duplex run)

User direction 11-07 17:35 ("语义如果可以做进系统 打断也这么做"), coordinator design. Post-processing of finished continuation files
is not the method any more; every component runs inside the chunk loop, strictly causal, default off (bitwise test).

## Preconditions (protocol fix after the failed first test run)
- Selection model = the dev twin of the final recipe: czm6kws (hos-base, 6000 steps, sf0 stream setting, causal_front /
  causal_spk / causal_text, w_safe 300, hos train split; runs/dev/czm6kws). Every parameter below is selected on it (or on its
  duplex fine-tune), never on czm2kws.
- Input after t_i (diagnosis 1b): with --audio end the not-arrived feat rows are normalised 0 = f_mean (raw logmel mean -8.6 =
  median speech level, prosody columns at their means) and the stream state says "audio final"; training never supervises motion
  more than ~0.3 s after an audio end. Real pauses in training are low-energy mel (logmel 5 % quantile -13.0; digital silence
  -13.8). Deployment-faithful input: the stream continues with digital silence after t_i (infer_retract --audio silence, psync:
  PCM zero from t_i, words ended by t_i, speaker vector of the speech heard; test_retract_psync.py section E). Which of end /
  silence is used is decided on czm6kws B3c BEFORE any re-tuning (lower v3 and unsafe, both reported).

## Components (added one at a time; ablation ladder on the 900 dev interruptions, per_clip seed 0)
0. B3c: the model's own continuation (chosen audio mode).
1. + yield token: a control token of a new type YIELD on the control-token path (scmg/ctl.py route, ctl + ctl_flow), arriving at
   the interrupt event. Training (ctlcf recipe): positives = train-side post-speech tails of the labelled tail view whose GT
   returns to rest (kinds retract / low; token at the speech offset); hold tails carry no token (audio ended, not yielding);
   token dropout 0.5; counterfactual placement p_cf: the token at a random time inside speech of an ordinary window, the audio
   after it = digital silence (causal inputs frozen at the token time + silence mel), the flow loss removed after the token and a
   rest loss (decoded waist + arm joints -> R_sil after token + T_rest, weighted like w_ctl_aux) as the only supervision there.
   REOPEN JUSTIFICATION (negatives row 13, floor token ignored): row 13 was a plain token without counterfactual supervision;
   the round-1 pointing tokens were ignored the same way until the ctlcf recipe made following the token the only way to lower
   the loss (STG 0.164 -> 0.308). The yield token is trained with that recipe.
2. + rest insertion: from t_i + T on, the rest pose as flagged known latents (mask_cond Local Flow, exp/text-path 1962440,
   handoff docs/MASK_COND_HANDOFF.md; the Streamer inpaint API ported into the psync RetractStreamer, their files unchanged).
   Training adds cfg mask_cond + sc p_mask 0.3 to the duplex fine-tune.
3. + end head: T per interrupt from the planner end head (R0end recipe, re-trained on the duplex model's own continuations),
   clamped [0.6, 1.8] s.
4. + funnel layer: the convergence tube of funnel_retract.py applied chunk by chunk to the emitted frames (r from the executed
   frame at t_i; T_f = the end-head T + the funnel margin), causal.
5. + CBF: the CBF-QP safety layer (exp/safe-proj cbf_qp) per emitted chunk.

## Gates (dev 900, the duplex dev model; uninterrupted reference = the same model's own no_int run)
- yield rate (R_sil) >= 90 %, abruptness v3 <= 10 %, unsafe <= 1 %, RKD below the best post-hoc method re-selected on czm6kws
  (R0causal grid tr {1.0, 1.4, 1.8} x w0 {0.3, 0.6} x tex {0, 1}; funnel grid tf {1.0, 1.4, 1.8, 2.2} x m0 {0.3, 0.5, 0.8, 1.0},
  smooth 0.1, eps 0.02, alpha 0.6; each under the same three other gates), paired recording bootstrap interval excluding 0.
- Uninterrupted quality must not drop: official ECHO-G evaluator on the 400 dev clips, psync, per_clip seeds 0-2: mean FGD <=
  czm6kws + 0.05 and dBA within czm6kws +- 0.003.
- Reaction latency: event -> first emitted frame that differs from the uninterrupted run, wall clock, on an idle GPU
  (no other process), separately from the yield time; p50 / p95 over the 900 events.
Only the full in-stream system (0-5) goes to the second registered test run, with the post-hoc rows (R0causal / funnel,
re-selected on czm6kws) as comparisons in the same run; the first test run's numbers stay in the appendix unchanged.

## Evaluation protocols (user rule 11-07: public protocols first)
- Uninterrupted quality: ECHO-G official evaluator. Executability: Unitree G1 URDF + MuJoCo (executability.py, the paper's
  contract). Physics tracking with an open-source G1 whole-body controller: when the coordinator names the controller.
- The interruption metrics (yield, v3, RKD, react) have no public protocol: sanity checks before use and released with the
  paper -- GT post-speech tails score as the best (yield / RKD of held-out human halves = floor), mismatched GT (tail windows of
  other clips) score at chance for the paired metrics (v3 / react); reported in the same table.
