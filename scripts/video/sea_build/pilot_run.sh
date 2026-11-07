#!/bin/bash
# Option-3 pilot (criteria docs/criteria/VIDEO_REGATE_1107.md "Option 3 pilot", pre-registered): 100 whole Seamless tracks ->
# frozen release07 retargeter -> (c)1 robot-side gate -> newdata-1103 utterance pipeline. CPU only.
#   pilot_run.sh <pilot dir with PILOT_TRACKS.jsonl> <workers>      (resumable: finished stages / items are skipped)
set -uo pipefail
P=$1; NW=${2:-80}; CODE=$(cd "$(dirname "$0")/../../.." && pwd); SB=$CODE/scripts/video/sea_build
R7=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/retarget-release07-headroom-20261026
PYR=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture/.venv-sim/bin/python
PYA=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/u3a-venv-asr-hf-20261023/bin/python
SNAPN=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/newdata-1103/SEAMLESS_SNAPSHOT.jsonl
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable SCMG_J=34
log() { echo "$(date -Is) $*" >> $P/pilot.log; }
mkdir -p $P/logs $P/retarget $P/regate
log "START code $(cat $CODE/GIT_COMMIT 2>/dev/null) workers $NW"
(cd $R7 && sha256sum -c --quiet FILES.sha256) || { log "RELEASE07_PIN_FAIL"; exit 3; }
if [ ! -f $P/CONVERT.done ]; then
  for i in $(seq 0 15); do python3 -I $SB/pilot_convert.py $P/PILOT_TRACKS.jsonl $P $i 16 > $P/logs/convert_$i.log 2>&1 & done; wait $(jobs -p)
  [ $(cat $P/logs/convert_*.log | grep -c CONVERT_DONE) -eq 16 ] && date > $P/CONVERT.done || { log "CONVERT_FAIL"; exit 4; }
fi
log "CONVERTED $(cat $P/ITEMS_*.jsonl | wc -l) windows"
cat $P/ITEMS_*.jsonl | python3 -I -c "
import json,sys,os
for l in sys.stdin:
    r=json.loads(l); p=sys.argv[1]+'/rows/'+r['id']+'.json'; os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(json.dumps(r)); print(r['id'])
" $P > $P/ids.txt
cat $P/ids.txt | xargs -P $NW -I{} sh -c "[ -e $P/retarget/{}/RESULT.json ] || [ -e $P/retarget/{}/REJECTED.json ] && exit 0; rm -rf $P/retarget/{}; cd $R7 && nice -n 10 $PYR -B $R7/retarget_adapter.py $P/rows/{}.json $P/retarget/{} > $P/logs/{}.log 2>&1 < /dev/null; exit 0"
log "RETARGETED"
python3 -I $SB/pilot_snapshot.py $P >> $P/pilot.log 2>&1
cd $R7
for i in $(seq 0 7); do SCMG_CODE=$CODE $PYR -B $CODE/scripts/video/regate_seamless.py $P/SNAPSHOT.jsonl $P/regate/regate_$i.jsonl $i 8 > $P/logs/regate_$i.log 2>&1 & done; wait $(jobs -p)
python3 -I $CODE/scripts/video/regate_agg.py $P/regate > $P/REGATE_AGG.txt 2>&1
for i in $(seq 0 7); do $PYR -B $SB/select_near.py $P/regate $P/SNAPSHOT.jsonl $P $i 8 > $P/logs/select_$i.log 2>&1 & done; wait $(jobs -p)
log "GATED_WINDOWS $(wc -l < $P/SELECT.jsonl)"
cd $P
for i in $(seq 0 15); do $PYA -u $SB/extract.py $P $i 16 > $P/logs/extract_$i.log 2>&1 & done; wait $(jobs -p)
$PYA -u $SB/features.py $P > $P/logs/features.log 2>&1
python3 -I $SB/gate.py $P >> $P/pilot.log 2>&1
log "PILOT_DONE"; exit 0
