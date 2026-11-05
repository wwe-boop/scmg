#!/bin/bash
# 11-05 G1 semantic head with EXPLICIT human semantic labels (BEAT2 official annotation, train_ids only; leak-checked):
# Semantic Gesture Head p(g_k | T, A, M<k) on the planner output + its g_k fed to the Local Flow (sem_head + sem_cond,
# zero-init -> starts bitwise as the warm-start model), g_k = 0 none / 1 deictic / 2 iconic / 3 metaphoric / 4 beat per
# 320 ms chunk (losses.py rule: majority class with >= 4 frames). Stream training + self_force_p 0 (the selected setting,
# = sf_g1 'sf0'), 2000 steps; each lane: official benchmark (offline + engine, inference seed 0), then the semantic
# recognition of its predictions (pair_win / det_auc / bal_top1, 3 recogniser seeds) and a summary vs ctrl
# (FGD ratio must stay <= 1.05).
#   SIDE=dev  (default): hos-base on the held-out-speaker view, benchmark on the 400-clip unseen-speaker dev (selection);
#                        recognisers rec-sem-hos-seed* (trained on the hos-view train split only, selected on a train slice)
#   SIDE=test (only after selection): fix-base on the full view, benchmark on val_common; recognisers rec-sem-v1*
# Lanes (LANES, default all four; label:gest(0/1/2):sc-fragment; 2 = gesture label track WITHOUT the semantic head, 11-06):
#   ctrl    : no semantic head (same as sf0)                semw03 : w_gest 0.3, class-weighted CE
#   semw1   : w_gest 1.0, class-weighted CE                 semw1tf: w_gest 1.0 + sem_tf 0.9 (Flow sees the true g more)
#   sem_g1.sh <out root>        env GPUS (one per lane), PY, BENCH_PY, EG/OFF/REF_DIR... as bench.sh (test side)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos
SB=${SEMREC:-$B/semantic-baseline-1105/g1}                       # recogniser data + checkpoints + floor controls
SIDE=${SIDE:-dev}
if [ $SIDE = dev ]; then
  INIT=${INIT:-$B/echog-g1-arch-1104/hos-base/best.pt}; GEST=${GEST:-$B/models/gest-beat2sem-echog-g1-v1-hos}; TV=$HV; LO=
  SEMDATA=$SB/data-sem-hos; RECS="$SB/rec-sem-hos-seed*"; FLOOR=$SB/outputs-dev; SPL=dev
  VENV=(G1_VIEW=$HV G1_BENCH_STEMS=${BENCH_STEMS:-$A/dev_stems.txt} REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt
        VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1)
else
  INIT=${INIT:-$B/echog-g1-arch-1104/fix-base/best.pt}; GEST=${GEST:-$B/models/gest-beat2sem-echog-g1-v1}; TV=$C/echog-g1-v1; LO=--labels-only
  SEMDATA=$SB/data-sem; RECS="$SB/rec-sem-v1*"; FLOOR=$SB/outputs-ctl; SPL=test
  VENV=(${BENCH_STEMS:+G1_BENCH_STEMS=$BENCH_STEMS})
fi
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0'
CW='"gest_cw": [1.0, 3.8, 3.0, 3.0, 0.7, 1.0, 1.0]'           # ~ inverse sqrt chunk frequency (beat 88 %, semantic 11 %)
SEMCFG='{"adsc": true, "sem_head": true, "sem_cond": true}'
GL=(${GPUS:-0 1 2 3}); i=0; R=$O/$SIDE
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
log() { echo "$(date -Is) [$SIDE] $*" >> $O/sem.log; }
# leak check of the label track before anything trains (exit 1 = refuse)
${PY:-python} scripts/semantic/check_gest_leak.py --gest $GEST --view $TV --src-view $C/echog-g1-v1 $LO \
  --dev-stems $A/dev_stems.txt $B/echog-g1-dev-assets/dev_stems.txt --heldout-speakers english_2_scott english_6_carla > $O/leak_check-$SIDE.txt 2>&1 \
  || { log "LEAK_CHECK_FAILED"; exit 1; }
log "LEAK_CHECK $(tail -1 $O/leak_check-$SIDE.txt)"
score() {  # gpu label=dir ...   (PAIRS from the environment)
  local g=$1; shift
  PY=${PY:-python} SCORE_EXTRA=${SCORE_EXTRA:-} bash scripts/semantic/sem_score_dirs.sh $g $SEMDATA "$RECS" $O/sem-$SIDE $SPL "$@"
}
lane() {   # gpu label cfg sc gest(0/1)
  local g=$1 lab=$2 cf=$3 sc=$4 ge=$5
  ${PY:-python} -c "import json,sys; json.loads(sys.argv[1]); json.loads(sys.argv[2])" "$cf" "$sc" || { log "BAD_JSON $lab"; return 2; }
  log "lane $lab gpu $g cfg $cf sc $sc gest $ge"
  env $( [ $ge != 0 ] && echo G1_GEST=$GEST ) INIT=$INIT VAE=$DEC "${VENV[@]}" \
    bash scripts/r4_lane.sh $g $R $lab "$cf" "$sc" ${STEPS:-2000} > $O/lane-$SIDE-$lab.log 2>&1
  log "LANE_END $lab"
  local specs=() p
  for p in offline engine; do [ -d $R/$lab/bench/$p/seed_000 ] && specs+=($lab-$p=$R/$lab/bench/$p/seed_000); done
  [ ${#specs[@]} -gt 0 ] && PAIRS= score $g "${specs[@]}" > $O/sem-$SIDE-$lab.log 2>&1 && log "SEM_DONE $lab" || log "SEM_FAILED $lab"
}
for spec in ${LANES:-ctrl:0: semw03:1:\"w_gest\":0.3 semw1:1:\"w_gest\":1.0 semw1tf:1:\"w_gest\":1.0,\"sem_tf\":0.9}; do
  lab=${spec%%:*}; rest=${spec#*:}; ge=${rest%%:*}; frag=${rest#*:}
  if [ $ge = 1 ]; then cf=$SEMCFG; sc="{$S, $CW, ${frag//,/, }}"
  else cf='{"adsc": true}'; sc="{$S${frag:+, ${frag//,/, }}}"; fi   # 11-06: ge 0 / 2 take a fragment too (empty = old "{$S}")
  g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1))
  lane $g $lab "$cf" "$sc" $ge &
done
wait
# ground truth + floors on the same events, then the table (pairs vs ctrl)
PAIRS=$(for l in $(ls $R); do [ $l = ctrl ] || printf "%s," "$l-offline:ctrl-offline,$l-engine:ctrl-engine"; done)
PAIRS=${PAIRS%,} score ${GL[0]} gt=- gt-mismatch=$FLOOR/gt-mismatch gt-shift2s=$FLOOR/gt-shift2s > $O/sem-$SIDE-final.log 2>&1
${PY:-python} - $O $SIDE <<'EOF' >> $O/sem.log 2>&1
import json, sys
from pathlib import Path
O, side = Path(sys.argv[1]), sys.argv[2]; rep = json.loads((O / f'sem-{side}' / 'REPORT.json').read_text())['table']
fgd = {}
for l in sorted(p.name for p in (O / side).iterdir() if p.is_dir()):
    for p in ('offline', 'engine'):
        f = O / side / l / 'bench' / p / 'benchmark_seed_000.json'
        if f.exists(): fgd[f'{l}-{p}'] = json.loads(f.read_text())['metrics']['FGD']
rows = {}
for k, v in fgd.items():
    p = k.rsplit('-', 1)[1]; t = rep.get(k, {})
    rows[k] = dict(FGD=v, FGD_ratio_vs_ctrl=v / fgd[f'ctrl-{p}'] if f'ctrl-{p}' in fgd else None,
                   **{m: t[m]['mean'] for m in ('pair_win', 'det_auc', 'bal_top1') if m in t})
for k in ('gt', 'gt-mismatch', 'gt-shift2s'):
    if k in rep: rows[k] = {m: rep[k][m]['mean'] for m in ('pair_win', 'det_auc', 'bal_top1')}
(O / f'SUMMARY-{side}.json').write_text(json.dumps(rows, indent=1)); print('SUMMARY', side, json.dumps(rows))
EOF
log "SEM_G1_DONE"
