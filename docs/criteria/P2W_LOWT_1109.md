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
