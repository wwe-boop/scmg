#!/bin/bash
# 11-09 Design B quick lane (exp/text-path docs/criteria/P2B_ROLLOUT_1109.md): arm R (rollout-scored sem outcome, 4-step CFG, self-forced
# history, 600 steps) vs arm O (one-step-scored, same recipe, matched wall-clock); 10 recordings, offline seed 0; realisation ratio, E,
# swap follow, FGD v2.   ex002_p2b_quick.sh <out dir>   env PY, G, BATCH
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; S=$X/sem-1109; P2=$X/p2-quick-1109
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; ENC=$B/ex002-sfscope-1108/enc-s0; CIS=0,4,8,12,16,20,24,28,32,36
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
train() {  # label steps extra_sc
  [ -f $O/$1/DONE.json ] && return 0
  CUDA_VISIBLE_DEVICES=$G $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $ENC/train/final.pt --out $O/$1 --steps $2 --batch ${BATCH:-64} --lr 1e-4 --warmup 60 --workers ${WORKERS:-6} --seed 0 --eval-every 100000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $S/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_desc": true}' \
    --sc "{\"p_stream\": 0.67, \"p_async\": 1.0, \"p_async_seg\": 0.0, \"w_int\": 0.3, \"sem_w\": 2.0, \"self_force_p\": 1.0, \"self_force_scope\": \"encoders\", \"ctl_w\": 1.0, \"ctl_field_drop\": 0.2, \"ctl_lr_scale\": 1.0, \"sem_cf_p\": 0.5, \"w_sem_desc\": 1.0, \"sem_bank\": \"$S/SEM_BANK.json\"$3}" \
    > $O/$1.log 2>&1 && log "TRAINED $1 $2" || { log "TRAIN_FAILED $1"; return 1; }
}
sps() { $PY -c "import json,sys; L=[json.loads(l) for l in open(sys.argv[1]) if '\"s_per_step\"' in l]; v=[x['s_per_step'] for x in L[1:]] or [L[-1]['s_per_step']]; print(sum(v)/len(v))" $1/train.jsonl; }
train R 600 ', "sem_rollout_steps": 4, "sem_rollout_cfg": 2.0' || exit 1
SR=$(sps $O/R); train Oprobe 50 '' || exit 1; SO=$(sps $O/Oprobe)
NO=$($PY -c "print(int(round(600 * $SR / $SO)))"); log "MATCHED s/step R $SR O $SO -> O steps $NO"
train O $NO '' || exit 1
for arm in R O; do for x in "given|given" "shuffle|swapped"; do IFS='|' read -r md lab <<< "$x"; d=$O/gen/$arm/${lab}_s0
  [ -f $d/.done ] || { CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py infer --ckpt $O/$arm/final.pt --protocol offline --seed 0 --cis $CIS --out $d \
      --mode $md --plan $P2/plans/PLAN_oracle.json >> $O/infer.log 2>&1 && touch $d/.done && log "GEN $arm $lab"; }
done; done
for arm in R O; do
  $PY scripts/ex002/sem_eval_full.py --sem $S --plans $P2/plans --gt $P2/offline --out $O/SEM_EVAL_$arm.json none=$P2/offline/none_s0 given=$O/gen/$arm/given_s0 swapped=$O/gen/$arm/swapped_s0 >> $O/eval.log 2>&1 && log "SEM_EVAL $arm"
done
$PY scripts/ex002/sem_eval_full.py --sem $S --plans $P2/plans --gt $P2/offline --out $O/SEM_EVAL_P2.json none=$P2/offline/none_s0 given=$P2/offline/given_s0 swapped=$P2/offline/swapped_s0 >> $O/eval.log 2>&1 && log "SEM_EVAL P2"
for d in $O/gen/R $O/gen/O; do for g in $P2/offline/gt_*.npy; do ln -sf $g $d/; done; done
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none=$P2/offline/none_s0 P2=$P2/offline/given_s0 R=$O/gen/R/given_s0 O=$O/gen/O/given_s0 >> $O/eval.log 2>&1 && log "BENCH_EVAL"
chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
