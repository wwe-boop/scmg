# W-low quick lane: outcome-loss weight (1 - t) on the P2 pipeline (agent B, 2026-11-09; written before any run)
Source: A's Step 0 (~/bulk/causal-train-1105/audit-1108/step0/, exp/duplex 11a09f1). At flow time t >= 0.5 the one-step x_hat is anchored
to the GT content in x_t, so the adapter cannot change it there. Yet the outcome losses use the weight w_inst = t
(exp/ctl-port scmg/losses.py l.1012: w_inst = t[:, -1]), which puts most of the weight in that region.

- Switch (default off; bitwise): StageCfg sem_w_low. When on, the sem_desc outcome frame weight is fv x (1 - t) (t = the instance's
  flow time) instead of fv x t. Chosen before running: the (1 - t) weighting, not the t <= 0.5 restriction.
- Arm W: the P2 recipe exactly (frozen enc-s0, --train-only ctl_, sem_cf_p 0.5, w_sem_desc 1, 2000 steps, training seed 0)
  + sem_w_low. Control: P2 as is, i.e. the existing quick-lane model p2-quick-1109/train (the same recipe, steps and seed; not
  retrained) and its quick-lane rows on the same 10 recordings (offline seed 0).
- Rows: offline given / swapped with the P2 quick-lane oracle plan; none = the P2 quick-lane none (bitwise enc-s0).
- Primary: realisation ratio amp / vpk (median over the 62 events), E vs none, swap follow, FGD v2 (given, 10 recordings).
- Kill (any): realisation ratio not up by >= 0.10 vs the control on the same 10 recordings (amp AND vpk); or E(given) not better
  than P2's quick-lane 1.042; or FGD v2 (given) worse than P2's 138.7.
- Also reported (no runs): where the same w_inst = t weighting applies. exp/ctl-port losses.py: ctl_ray_loss l.1036, ctl_finger_loss
  l.1046, ctl_count_loss l.1049, phys_reg l.1052 / 1055, recon (cont) l.1016, foot l.1023, contract l.1029. exp/duplex losses.py:
  yield_sw_loss l.1214, yield_settle_loss l.1216, ctl_ray l.1219, finger l.1228, count l.1231.

## RESULT (11-09 08:06; speech_server GPU1, release 20261109-074646-p2w-1109)
10 recordings, 62 events, offline seed 0. W (1 - t) vs the P2 control (same 10 recordings):
- E: none 1.868 -> W 1.061 (-43.2 % [35.0, 50.7]); P2 1.042.
- Swap follow: E_sw 1.113 vs E_oracle 1.648, PASS.
- Realisation, median: amp 0.55 (P2 0.54), vpk 0.57 (P2 0.58).
- FGD v2: 139.76 (P2 138.72, none 124.23); liveliness PASS.
Kills: K1 realisation up by >= 0.10 -> NO (+0.01 / -0.01), HIT; K2 E better than 1.042 -> NO (1.061), HIT; K3 FGD v2 <= 138.7 -> NO
(139.76), HIT -> KILLED. W is indistinguishable from P2. Moving the outcome weight to the low-t steps (where x_hat is not anchored to
the GT) changes neither the realisation nor E. The training sem_desc also stays at about 1.3-1.4, as in P2. Negatives row 66.
No renders (the lane did not pass).
