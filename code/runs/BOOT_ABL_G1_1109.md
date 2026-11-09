# G1 ablations vs czmws (SCMG): recording-level paired bootstrap of official FGD (test val_common, 3 training seeds x 2 inference seeds per row)
n_recordings 349, n_boot 2000, seed 20261106; diff = FGD(SCMG) - FGD(ablation): negative = SCMG better. Resumed 11-09 (boot_resume.py, recording-aggregated latent statistics; per-run FGD identical to the 11-08 pass).

| ablation | protocol | FGD SCMG | FGD ablation | diff [95% CI] | p |
|---|---|---|---|---|---|
| no causal warm start (ncws) | psync | 1.946 | 2.049 | -0.103 [-0.129, -0.078] | 0.000 |
| no text in training (nt) | psync | 1.946 | 1.944 | +0.003 [-0.029, +0.032] | 0.893 |
| no audio in training (na) | psync | 1.946 | 2.097 | -0.151 [-0.265, -0.044] | 0.006 |
| text dropped at inference | psync | 1.946 | 1.983 | -0.036 [-0.062, -0.010] | 0.013 |
| audio dropped at inference | psync | 1.946 | 2.093 | -0.147 [-0.261, -0.035] | 0.009 |
| no causal warm start (ncws) | offline | 1.961 | 2.072 | -0.111 [-0.144, -0.080] | 0.000 |
| no text in training (nt) | offline | 1.961 | 1.859 | +0.102 [+0.044, +0.159] | 0.001 |
| no audio in training (na) | offline | 1.961 | 2.157 | -0.196 [-0.257, -0.137] | 0.000 |
| text dropped at inference | offline | 1.961 | 1.901 | +0.060 [+0.015, +0.106] | 0.007 |
| audio dropped at inference | offline | 1.961 | 2.107 | -0.146 [-0.201, -0.092] | 0.000 |
