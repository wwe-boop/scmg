#!/bin/bash
# 11-09 P2 sem_cf quick lane (exp/text-path docs/criteria/P2_SEMCF_1109.md §3-5): frozen enc-s0 + ctl / ctl_flow / ctl_desc
# (--train-only ctl_), SEM tokens of sem_bank.py, sem_cf_p 0.5, w_sem_desc 1, 2000 steps; plans (oracle / retrieval) for the
# 10 quick-lane recordings; offline seed 0: none / given / swapped / retrieval; psync and TTS-sync (look-ahead 0.5 s) given;
# none-path check vs the stored enc-s0 outputs; descriptor metric + FGD v2 / liveliness of the 10 recordings.
#   ex002_p2_quick.sh <out dir>     env PY, G (GPU id, default 0), WORKERS
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; S=$X/sem-1109
PY=${PY:-python}; G=${G:-0}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; ENC=$B/ex002-sfscope-1108/enc-s0; CIS=0,4,8,12,16,20,24,28,32,36
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) bank $(sha256sum $S/SEM_BANK.npz | cut -c1-16) tokens $(sha256sum $S/tokens/CTL.json | cut -c1-16)"
if [ ! -f $O/train/DONE.json ]; then
  CUDA_VISIBLE_DEVICES=$G $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $ENC/train/final.pt --out $O/train --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers ${WORKERS:-6} --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $S/tokens --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_desc": true}' \
    --sc "{\"p_stream\": 0.67, \"p_async\": 1.0, \"p_async_seg\": 0.0, \"w_int\": 0.3, \"sem_w\": 2.0, \"self_force_p\": 0, \"self_force_scope\": \"encoders\", \"ctl_w\": 1.0, \"ctl_field_drop\": 0.2, \"ctl_lr_scale\": 1.0, \"sem_cf_p\": 0.5, \"w_sem_desc\": 1.0, \"sem_bank\": \"$S/SEM_BANK.json\"}" \
    > $O/train.log 2>&1 && log "TRAINED" || { log "TRAIN_FAILED"; exit 1; }
fi
CK=$O/train/final.pt
$PY scripts/ex002/sem_plan.py --sem $S --cis $CIS --out $O/plans >> $O/lane.log 2>&1 && log "PLANS" || { log "PLAN_FAILED"; exit 1; }
gen() {  # protocol mode plan out [extra]
  [ -f $4/.done ] && return 0
  CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py infer --ckpt $CK --protocol $1 --seed 0 --cis $CIS --out $4 --mode $2 ${3:+--plan $3} ${5:-} >> $O/infer.log 2>&1 \
    && touch $4/.done && log "GEN $4" || log "GEN_FAILED $4"
}
gen offline none "" $O/offline/none_s0
gen offline given $O/plans/PLAN_oracle.json $O/offline/given_s0
gen offline shuffle $O/plans/PLAN_oracle.json $O/offline/swapped_s0
gen offline given $O/plans/PLAN_retrieval.json $O/offline/retrieval_s0
gen psync given $O/plans/PLAN_oracle.json $O/psync/given_s0
gen psync given $O/plans/PLAN_oracle.json $O/psync/ttsync_s0 "--ctl-lookahead 0.5"
gen psync none "" $O/psync/none_s0
# none path vs the stored enc-s0 outputs (same protocol / seed)
$PY - $O/offline/none_s0 $ENC/offline/s0 $CIS > $O/NONE_BITWISE.json <<'PYEOF'
import sys, json, numpy as np
a, b = sys.argv[1], sys.argv[2]; cis = [int(x) for x in sys.argv[3].split(',')]
r = {ci: bool(np.array_equal(np.load(f'{a}/{ci}.npy'), np.load(f'{b}/{ci}.npy'))) for ci in cis}
d = {ci: float(np.abs(np.load(f'{a}/{ci}.npy') - np.load(f'{b}/{ci}.npy')).max()) for ci in cis}
print(json.dumps(dict(bitwise=r, all=all(r.values()), max_abs=d)))
PYEOF
log "NONE_CHECK $(cat $O/NONE_BITWISE.json | cut -c1-200)"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/sem_eval.py --sem $S --plans $O/plans --gt $O/offline --out $O/SEM_EVAL_offline.json \
  none=$O/offline/none_s0 given=$O/offline/given_s0 swapped=$O/offline/swapped_s0 retrieval=$O/offline/retrieval_s0 >> $O/eval.log 2>&1 && log "SEM_EVAL_offline"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/sem_eval.py --sem $S --plans $O/plans --gt $O/psync --out $O/SEM_EVAL_psync.json \
  none=$O/psync/none_s0 given=$O/psync/given_s0 ttsync=$O/psync/ttsync_s0 >> $O/eval.log 2>&1 && log "SEM_EVAL_psync"
CUDA_VISIBLE_DEVICES=$G $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json none=$O/offline/none_s0 given=$O/offline/given_s0 \
  retrieval=$O/offline/retrieval_s0 >> $O/eval.log 2>&1 && log "BENCH_EVAL"
chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
