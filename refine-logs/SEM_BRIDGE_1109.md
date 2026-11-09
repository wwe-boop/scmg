# Bridging low-dimensional continuous motion and high-dimensional semantics — survey + chosen quick lane (main agent, 2026-11-09)

The survey is from 10 web searches, at the search-snippet / abstract level unless noted. Not reviewed by a person.

## 1. How other fields bridge "low-dimensional continuous output ↔ high-dimensional semantics"
| family | field / paper | mechanism | analogue for us |
|---|---|---|---|
| Semantic intermediate tokens, then continuous decoding | TTS: CosyVoice 1/2/3 (2407.05407): supervised semantic tokens (VQ inside an ASR model), then OT conditional flow matching to mel; CosyVoice 2 adds chunk-aware causal flow | the semantic stage carries content and timing; the flow stage carries the acoustic detail | our planner → local flow is the same split, but our planner tokens are unsupervised |
| Representation alignment of the generator to a frozen semantic encoder | images: REPA (ICLR 2025: align early DiT features to DINOv2, 17.5× faster, FID 1.42); VA-VAE (2501.01423: align the VAE latent to a VFM); video: VideoREPA, MoAlign | semantics are injected as an auxiliary feature-alignment loss; no inference cost | **not done for co-speech / streaming gesture** (closest: MoLingo semantic-aligned motion encoder for text-to-motion; LUMA teacher-free) |
| Gradient insulation between a semantic backbone and a continuous expert | robotics VLA: Knowledge Insulation (π0.5-KI, 2505.23705): the backbone trains on discrete FAST action tokens, the flow-matching action expert reads the backbone through stop-gradient | continuous-expert gradients damage the semantic representation; insulate them | mirrors our finding that self-forced flow gradients hurt (encoder-only SF routes gradients the other way) |
| Text-to-motion-latent alignment | gesture: SemGes (ICCV 2025: cosine text → frozen VQ motion latent), ExpGest, MAG (contrastive motion–text–audio), Semantic Motion Anchors (2605.30608: LLM-verbalised anchors, because direct contrastive alignment "overemphasizes low-level kinematics") | a shared latent for semantics and motion | SemGes-style alignment was not tried in our negatives |
| Sparse semantic stream + gate | gesture: SemTalk (ICCV 2025: rhythm base + sparse semantic motion, frame-level semantic score), DuoGesture (variational-bottleneck gate) | semantics are rare and sparse; gate them in | our semantic head / sem_time failed (rows 2, 22) |
| Contrastive negatives inside the flow | SemConFlow: mismatched audio–text as negatives for the velocity field | the field is repelled from incongruent trajectories | not tried |
| Retrieval of exemplars | RAG-Gesture (CVPR 2025), Semantic Gesticulator | library insertion | tried on G1 (row 31: edge jerk) |

## 2. Chosen quick lane: semantic representation alignment of the streaming planner ("SemREPA")
- **Why this one.**
  - It is the family with the strongest cross-field evidence (REPA, VA-VAE, VideoREPA).
  - It has no counterpart in co-speech or streaming gesture generation (searched).
  - It costs nothing at inference and keeps strict causality: the teacher only sees the GT target chunk at training time.
  - It differs from our failed semantic head. That head predicted a class from the planner; here the target is a dense feature of a frozen semantic motion encoder on the target chunk, aligned to EARLY planner blocks, which the REPA ablations show is where alignment helps.
- **Teacher.** It must NOT be the recogniser used to compute pair_win (Goodhart). Train a separate semantic motion encoder on the TRAIN split only, MoLingo / SemGes-style:
  - a small temporal conv / transformer over 1–2 s motion windows;
  - trained with frame-level BEAT2 semantic labels plus the e5 embedding of the aligned words (cosine), with a different architecture and seed from the pair_win recogniser;
  - frozen afterwards.
- **Loss.** L = L_flow + λ · (1 − cos(proj(h_planner^{(l)}_k), T(GT chunk k))) for an early planner block l ∈ {2, 3}. λ ∈ {0.5}: one value, fixed in advance. Default off; bitwise test on the default path.
- **Quick lane (G1 dev, ≤ 5 min per arm).** czmws, 2000 steps: SemREPA vs a matched fine-tune control.
  - Metrics: pair_win (with the GT / mismatched-GT sanity rows), text use = full − no-text, FGD, liveliness.
  - Kill: pair_win gain < +0.02 over the control, or FGD outside the czmws seed range, or text use not up.
- **Honest risk.** pair_win's recogniser and the teacher share the label source (BEAT2 semantic annotations). A gain can partly reflect label agreement rather than perceived semantics, so we must report it next to the blind-judge check. If it passes, the full run uses 3 seeds and a second, independently trained recogniser for scoring.
