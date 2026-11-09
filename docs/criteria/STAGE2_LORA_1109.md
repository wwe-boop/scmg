# Fable round 5 Design A stage 2: token-gated LoRA, code record (agent B, 2026-11-09; code only, no runs)
Code: exp/glora d8b8e0d (on exp/keyframe 3823be8), scmg/lora_gate.py; tests: scripts/tests/test_lora_gate.py ALL_OK.
- Rank 8, alpha 8, on the Local Flow blocks' sa.q / k / v / o and ff.fc1 / fc2: 36 Linear layers, 0.442 M parameters (0.38 % of the
  base). Parameters model.lora_gate.*, B zero-init, A from a forked RNG (seed 1113). Delta: y = W x + B ((alpha / r) g A x); the gate
  g is applied in the rank-r space before B. Config lora_gate (default off, ADDABLE); flow_velocity(gate=None); StageCfg
  lora_gate_train (A wires the gate into training, the rollout and the Streamer).
- Gate: window_gate = max over the visible KEY tokens of w(position time - t_k). w = 1 on [t_k - 0.6, t_k + 0.8] s, raised-cosine ramps
  of 0.3 s outside (non-zero from t_k - 0.9 s to t_k + 1.1 s), times the token's visibility (KEY_LEAD = 1.0 s).
- Bitwise claim (per instance): a Local Flow instance with no gated position is bitwise the base; switch off and switch on
  without a token are bitwise the old model (tests a, b, d).
- Causal caveat (main 11-09): self-attention mixes the positions of one Local Flow instance. Every frame of an instance that carries
  any gate (including the ramp from t_k - 0.9 s) may change, and that can include frames earlier than the window within the same
  chunk. Measured in test c: gate-0 positions inside a gated instance change by up to 0.136 (random weights, B perturbed). The claim
  "frames before the window are bitwise" holds only for frames whose whole instance carries no gate.
- Latency item: flow-forward overhead +36 % (all instances gated) to +52 % (15 % gated) at M = 1024, bf16 (noisy, launch-bound;
  2 extra small matmuls per wrapped Linear). To be measured on the idle GPU (the latency protocol) only if stage 2 is used.
- Status: held until A's stage-1 verdict; no runs.
- Status (11-09): descoped by the user 11-09, never trained. exp/glora left unmerged.
