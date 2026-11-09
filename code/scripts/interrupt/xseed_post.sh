#!/bin/bash
# cross-seed statistics (11-06): per_clip seeds 0 / 1 / 2 on the dev subset -> R0causal_p and B2_p per seed, v3 scoring,
# paired RKD difference per seed, and the pairing check (frames before t_i of the interrupted run vs its counterfactual)
W=$HOME/bulk/interrupt-1105; I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105; cd $W
PYH=/home/xr-25b7/bulk/echog-official-ro/official/src; S=/home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt; PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python
INT=interrupts_dev.json; HJ=human_ref_v2_train_drop.json; mkdir -p xseed
for s in 0 1 2; do O=xseed/s$s
  ssh zehan-server "cd $I/xseed && tar cf - s$s" | tar xf - -C xseed
  PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/B3c --prefix-from $O/no_int --out-dir $O/R0causal --interrupts $INT --human $HJ --every 3 --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1
  PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/no_int --full-clip --out-dir $O/B2 --interrupts $INT --human $HJ --every 3 --tr 1.8 --tb 0.5 --w0 1.0
  for d in R0causal B2; do for sh in 0 1; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYH ECHOG_OFFICIAL_SRC=$PYH OMP_NUM_THREADS=2 $PY $S/safety_project.py \
    --in-dir $O/$d --out-dir $O/${d}_p300 --interrupts $INT --every 3 --smooth 300 --shard $sh/2 > /dev/null 2>&1 & done; done; wait
  bash ev.sh $INT metrics/xseed_s$s.json 3 B2_p=B3:$O/B2_p300:$O/no_int R0causal_p=B3:$O/R0causal_p300:$O/no_int B3c=B3:$O/B3c:$O/no_int > logs/xseed_s$s.txt 2>&1
  PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/xseed_s$s.jsonl R0causal_p metrics/xseed_s$s.jsonl B2_p > metrics/xseed_s${s}_rkddiff.txt 2>&1
done
$PY - <<'P'
import json, numpy as np, glob, torch, os
W='/home/xr-25b7/bulk/interrupt-1105/'; ks=('yield_rate_R_sil','yield_s_R_sil','abrupt_v3','rkd','unsafe_any','react_s')
R={s: json.load(open(f'{W}metrics/xseed_s{s}.json')) for s in (0,1,2)}
out={}
for m in ('B2_p','R0causal_p','B3c'):
    out[m]={k: dict(mean=float(np.mean([R[s][m][k] for s in R])), sd=float(np.std([R[s][m][k] for s in R], ddof=1)), per_seed=[R[s][m][k] for s in R]) for k in ks}
ints={(r['stem'],r['frac']):r for r in json.load(open(W+'interrupts_dev.json'))['interrupts']}; d=[]
for f in sorted(glob.glob(W+'xseed/s0/B3c/*.pt'))[:150]:
    st,fr=os.path.basename(f)[:-3].split('@'); fi=ints[(st,float(fr))]['f_i']
    a=torch.load(f,weights_only=False)['robot_repr']; u=torch.load(f'{W}xseed/s0/no_int/{st}.pt',weights_only=False)['robot_repr']
    d.append(float((a[:fi-3]-u[:fi-3]).abs().max()))
out['pairing_check']=dict(note='per_clip seed 0: 30 fps frames before f_i - 3 of the interrupted run vs its uninterrupted counterfactual', median=float(np.median(d)), max=float(np.max(d)), n=len(d))
json.dump(out, open(W+'metrics/xseed_summary.json','w'), indent=1)
for m in ('B2_p','R0causal_p','B3c'): print(m, {k: f"{out[m][k]['mean']:.3f}+-{out[m][k]['sd']:.3f}" for k in ks})
print(out['pairing_check'])
P
for s in 0 1 2; do grep -v Warn metrics/xseed_s${s}_rkddiff.txt; done
echo XSEED_POST_DONE
