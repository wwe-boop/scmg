#!/bin/bash
# 11-09 W-low quick lane (exp/text-path docs/criteria/P2W_LOWT_1109.md): P2 recipe + sem_w_low (outcome weight 1 - t), 2000 steps seed 0;
# 10 recordings offline seed 0 given / swapped; control = the P2 quick-lane model and rows.   ex002_p2w_quick.sh <out dir> [wait-for-dir]
set -uo pipefail
O=$1; WAIT=${2:-}; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; S=$X/sem-1109; P2=$X/p2-quick-1109
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; ENC=$B/ex002-sfscope-1108/enc-s0; CIS=0,4,8,12,16,20,24,28,32,36
log() { echo "$(date -Is) $*" >> $O/lane.log; }
[ -n "$WAIT" ] && until grep -q "LANE_DONE\|FAILED" $WAIT/lane.log 2>/dev/null; do sleep 60; done
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
[ -f $O/W/DONE.json ] || CUDA_VISIBLE_DEVICES=$G $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $ENC/train/final.pt --out $O/W --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers ${WORKERS:-6} --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $S/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_desc": true}' \
    --sc "{\"p_stream\": 0.67, \"p_async\": 1.0, \"p_async_seg\": 0.0, \"w_int\": 0.3, \"sem_w\": 2.0, \"self_force_p\": 0, \"self_force_scope\": \"encoders\", \"ctl_w\": 1.0, \"ctl_field_drop\": 0.2, \"ctl_lr_scale\": 1.0, \"sem_cf_p\": 0.5, \"w_sem_desc\": 1.0, \"sem_bank\": \"$S/SEM_BANK.json\", \"sem_w_low\": true}" \
    > $O/W.log 2>&1 && log "TRAINED" || { log "TRAIN_FAILED"; exit 1; }
for x in "given|given" "shuffle|swapped"; do IFS='|' read -r md lab <<< "$x"; d=$O/gen/${lab}_s0
  [ -f $d/.done ] || { CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py infer --ckpt $O/W/final.pt --protocol offline --seed 0 --cis $CIS --out $d \
      --mode $md --plan $P2/plans/PLAN_oracle.json >> $O/infer.log 2>&1 && touch $d/.done && log "GEN $lab"; }
done
$PY scripts/ex002/sem_eval_full.py --sem $S --plans $P2/plans --gt $P2/offline --out $O/SEM_EVAL_W.json none=$P2/offline/none_s0 given=$O/gen/given_s0 swapped=$O/gen/swapped_s0 >> $O/eval.log 2>&1 && log "SEM_EVAL"
for g in $P2/offline/gt_*.npy; do ln -sf $g $O/gen/; done
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none=$P2/offline/none_s0 P2=$P2/offline/given_s0 W=$O/gen/given_s0 >> $O/eval.log 2>&1 && log "BENCH_EVAL"
chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
