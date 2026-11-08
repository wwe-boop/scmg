#!/bin/bash
# 11-09 P2 sem_cf full run + controls (exp/text-path docs/criteria/P2_SEMCF_1109.md "Full run + controls"): controls c1 plain / c2 onset / c3 late
# trained (frozen enc-s0, 2000 steps), then a lock-file GPU queue of generations (43 recordings x seeds 0-2), then the pooled metric,
# FGD v2 / liveliness and the none-path bitwise check.   ex002_p2_full.sh <out dir>   env PY, NG (default 8)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O/locks; NG=${NG:-8}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; S=$X/sem-1109
PY=${PY:-python}; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1
VAE=$B/trained/vae-r3/final.pt; ENC=$B/ex002-sfscope-1108/enc-s0; MAIN=$X/p2-quick-1109/train/final.pt; CIS=$(seq -s, 0 42)
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
# shifted token sets (c2 onset +0.2 s, c3 late +0.4 s) and plans
$PY - $S/tokens $O <<'PYEOF'
import json, sys, shutil
from pathlib import Path
src, O = Path(sys.argv[1]), Path(sys.argv[2])
for name, sh in (('onset', 0.2), ('late', 0.4)):
    d = O / f'tokens_{name}'; d.mkdir(parents=True, exist_ok=True)
    ctl = json.loads((src / 'CTL.json').read_text())
    for k, v in ctl.items():
        for t in v: t['t'] = t['t'] + sh
    (d / 'CTL.json').write_text(json.dumps(ctl)); shutil.copy(src / 'PREP.json', d / 'PREP.json')
PYEOF
$PY scripts/ex002/sem_plan.py --sem $S --cis $CIS --out $O/plans >> $O/lane.log 2>&1 && log "PLANS"
$PY - $O/plans <<'PYEOF'
import json, sys
from pathlib import Path
P = Path(sys.argv[1]); base = json.loads((P / 'PLAN_oracle.json').read_text())
for name, sh in (('onset', 0.2), ('late', 0.4)):
    ev = [dict(e, t=e['t'] + sh) for e in base['events']]; (P / f'PLAN_oracle_{name}.json').write_text(json.dumps(dict(base, events=ev)))
PYEOF
train() {  # gpu label tokens sc_extra
  [ -f $O/$2/DONE.json ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE \
    --init $ENC/train/final.pt --out $O/$2 --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers ${WORKERS:-5} --seed 0 --eval-every 1000 --snap-every 1000000 \
    --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $3 --ctl-focus 0.5 --train-only ctl_ \
    --cfg '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_desc": true}' \
    --sc "{\"p_stream\": 0.67, \"p_async\": 1.0, \"p_async_seg\": 0.0, \"w_int\": 0.3, \"sem_w\": 2.0, \"self_force_p\": 0, \"self_force_scope\": \"encoders\", \"ctl_w\": 1.0, \"ctl_field_drop\": 0.2, \"ctl_lr_scale\": 1.0, $4, \"sem_bank\": \"$S/SEM_BANK.json\"}" \
    > $O/$2.log 2>&1 && log "TRAINED $2" || log "TRAIN_FAILED $2"
}
T=$O/TASKS.txt; : > $T
for s in 0 1 2; do
  for x in "main|offline|none||none" "main|offline|given|PLAN_oracle|given" "main|offline|shuffle|PLAN_oracle|swapped" "main|offline|given|PLAN_retrieval|retrieval" \
           "main|psync|given|PLAN_oracle|psync_given" "main|psync|given|PLAN_oracle|tts_given" \
           "c1|offline|given|PLAN_oracle|c1_given" "c1|offline|shuffle|PLAN_oracle|swapped_c1" "c2|offline|given|PLAN_oracle_onset|c2_given" \
           "c2|offline|shuffle|PLAN_oracle_onset|swapped_c2" "c3|offline|given|PLAN_oracle_late|c3_given" "c3|offline|shuffle|PLAN_oracle_late|swapped_c3"; do
    echo "$x|$s" >> $T; done; done
gen() {  # gpu model proto mode plan label seed
  local ck=$MAIN; [ $2 != main ] && ck=$O/$2/final.pt; local out=$O/gen/$6/s$7; local la=""; [ $6 = tts_given ] && la="--ctl-lookahead 0.5"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/ex002/bench_ex002.py infer --ckpt $ck --protocol $3 --seed $7 --out $out --mode $4 ${5:+--plan $O/plans/$5.json} $la >> $O/infer_$1.log 2>&1 \
    && log "GEN $6 s$7" || log "GEN_FAILED $6 s$7"
}
worker() {
  local g=$1
  while :; do
    local left=0 did=0
    while IFS='|' read -r m p md pl lab s; do
      local key="$lab-$s"; [ -e $O/locks/$key.done ] && continue; left=1; [ -d $O/locks/$key ] && continue
      if [ $m != main ] && [ ! -f $O/$m/final.pt ]; then grep -q "TRAIN_FAILED $m" $O/lane.log && touch $O/locks/$key.done; continue; fi
      mkdir $O/locks/$key 2>/dev/null || continue
      gen $g $m $p $md "$pl" $lab $s; touch $O/locks/$key.done; did=1; break
    done < $T
    [ $left = 0 ] && break; [ $did = 0 ] && sleep 30
  done
}
( train 0 c1 $S/tokens '"sem_cf_p": 0.0, "w_sem_desc": 0.0' ) &
( train 1 c2 $O/tokens_onset '"sem_cf_p": 0.5, "w_sem_desc": 1.0' ) &
( train 2 c3 $O/tokens_late '"sem_cf_p": 0.5, "w_sem_desc": 1.0' ) &
for g in $(seq 3 $((NG - 1))); do worker $g & sleep 2; done
wait $(jobs -p)
for g in 0 1 2; do worker $g & sleep 2; done; wait $(jobs -p); log "GEN_ALL_DONE"
for d in $O/gen/*/s*; do p=offline; case $d in *psync_*|*tts_*) p=psync;; esac; s=${d##*/s}; for gf in $ENC/$p/gt_*.npy; do ln -sf $gf $O/gen/$(basename $(dirname $d))/; done; done
J=$(for l in none given swapped retrieval c1_given swapped_c1 c2_given swapped_c2 c3_given swapped_c3; do echo "$l=$O/gen/$l/s0,$O/gen/$l/s1,$O/gen/$l/s2"; done)
$PY scripts/ex002/sem_eval_full.py --sem $S --plans $O/plans --gt $ENC/offline --out $O/SEM_EVAL_offline.json $J >> $O/eval.log 2>&1 && log "SEM_EVAL_offline" &
$PY scripts/ex002/sem_eval_full.py --sem $S --plans $O/plans --gt $ENC/psync --out $O/SEM_EVAL_psync.json psync_none=$ENC/psync/s0,$ENC/psync/s1,$ENC/psync/s2 \
  psync_given=$O/gen/psync_given/s0,$O/gen/psync_given/s1,$O/gen/psync_given/s2 tts_given=$O/gen/tts_given/s0,$O/gen/tts_given/s1,$O/gen/tts_given/s2 >> $O/eval.log 2>&1 && log "SEM_EVAL_psync" &
$PY - $O/gen/none $ENC/offline > $O/NONE_BITWISE.json <<'PYEOF'
import sys, json, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2]); r = {}
for s in (0, 1, 2):
    for f in sorted((a / f's{s}').glob('[0-9]*.npy')):
        r[f's{s}/{f.stem}'] = bool(np.array_equal(np.load(f), np.load(b / f's{s}' / f.name)))
print(json.dumps(dict(n=len(r), all=all(r.values()), mismatches=[k for k, v in r.items() if not v])))
PYEOF
log "NONE_CHECK $(cut -c1-200 $O/NONE_BITWISE.json)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/ex002/bench_ex002.py eval --out $O/BENCH_EVAL.json $(for s in 0 1 2; do echo none_s$s=$ENC/offline/s$s given_s$s=$O/gen/given/s$s retrieval_s$s=$O/gen/retrieval/s$s; done) >> $O/eval.log 2>&1 && log "BENCH_EVAL" &
wait $(jobs -p); chmod -R a+rwX $O 2>/dev/null; log "LANE_DONE"
