#!/bin/bash
# 11-06 correctness of Streamer(cuda_graph=True) (scmg/cuda_graph.py) on the held-out-speaker dev clips:
#   1. offline / psync, eager vs graph on N dev stems (every 20th of echog-g1-hos-assets/dev_stems.txt): max |q50 diff|,
#      bitwise flag, then the official evaluator (bench.sh flags) on both -> FGD / BA / Div / jerk difference
#   2. scripts/tests/test_causal_stream.py as before and with CUDA_GRAPH=1 (graph path vs the eager base, future invariance)
#   cg_check.sh <out dir>        env GPU (default 0), PY (python with onnxruntime), N (20)
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; O=$1; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
L=$B/baselines-g1-1105/latency; DR=$B/baseline-repro-1104/echog/data_root; A=$B/echog-g1-hos-assets   # dev GT / audio
PY=${PY:-$L/envs/causal/bin/python}; EPY=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python
OFF=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official
CK=${CK:-$B/baselines-g1-1105/assets/czm6k/final.pt}; VAE=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt   # CK: causal-matched deployment model (runs/test/czm6k)
HOSV=$B/views/echog-g1-v1-hos; TEXT=$C/scmg/text-e5-echog-g1-v1; TEXTC=$B/models/text-e5-causal-echog-g1-v1
WORDS=$B/causal-train-1105/words-dev; PS=$CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
CAMPPLUS=${CAMPPLUS:-$L/snap/campplus}
export CUDA_VISIBLE_DEVICES=${GPU:-0} SCMG_J=39 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
$PY -c "
s=[l.strip() for l in open('$B/echog-g1-hos-assets/dev_stems.txt') if l.strip()][::20][:${N:-20}]; open('$O/stems.txt','w').write('\n'.join(s)+'\n')"
log() { echo "$(date -Is) $*" | tee -a $O/check.log; }
for proto in offline psync; do
  case $proto in offline) PA=(--protocol offline);;
    psync) PA=(--protocol psync --text-causal $TEXTC --wav-dir $A/audio --words-dir $WORDS --pstats $PS --campplus $CAMPPLUS --causal-feat ${CAUSAL_FEAT:-inc} --spk-refresh ${SPK_REFRESH:-grades});; esac
  for cg in off on; do
    F=(); [ $cg = on ] && F=(--cuda-graph)
    $PY scripts/echog_g1/infer_test.py --ckpt $CK --vae $VAE --stats $C/scmg/STATS_echog_g1_v1.json --view $HOSV --text-cache $TEXT \
      --out $O/${proto}_$cg --stems $O/stems.txt --seed 0 --steps 10 --cfg-w 2.5 --split val --save50 "${PA[@]}" "${F[@]}" > $O/infer_${proto}_$cg.log 2>&1 \
      && log "INFER_DONE $proto $cg" || log "INFER_FAILED $proto $cg"
    PYTHONPATH=$OFF/src $EPY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $O/${proto}_$cg --ref-dir $A/motion_39d_30fps --val-split $O/stems.txt \
      --wav-dir $A/audio --mmae-file $DR/eval_assets/mmae/g1_mmae_30body_30fps.npy --g1-ae-ckpt /mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/weights/evaluation/g1_fgd_encoder.pt \
      --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems --tag ${proto}_$cg --out $O/bench_${proto}_$cg.json > $O/eval_${proto}_$cg.log 2>&1 \
      && log "EVAL_DONE $proto $cg" || log "EVAL_FAILED $proto $cg"
  done
done
$PY - $O <<'PY' | tee -a $O/check.log
import json, sys, glob, os
import numpy as np
O = sys.argv[1]; res = {}
for proto in ('offline', 'psync'):
    d = []; eq = 0; n = 0
    for f in sorted(glob.glob(f'{O}/{proto}_off/*.q50.npy')):
        a = np.load(f); b = np.load(f.replace(f'{proto}_off/', f'{proto}_on/')); d.append(float(np.abs(a - b).max())); eq += int(np.array_equal(a, b)); n += 1
    m = {cg: json.load(open(f'{O}/bench_{proto}_{cg}.json'))['metrics'] for cg in ('off', 'on')}
    get = lambda x: {'FGD': x['FGD'], 'BA': x['BA']['mean'], 'Div': x['Div'], 'Jerk_lw_gap': x['Jerk_length_weighted_gap']}
    res[proto] = dict(clips=n, bitwise_equal_clips=eq, max_abs_q50_diff_rad=max(d) if d else None, off=get(m['off']), on=get(m['on']),
                      fgd_diff=abs(m['on']['FGD'] - m['off']['FGD']))
print(json.dumps(res, indent=1)); json.dump(res, open(f'{O}/CG_CHECK.json', 'w'), indent=1)
PY
mkdir -p $O/ds/annotations; ln -sfn $A/audio $O/ds/audio; ln -sfn $WORDS $O/ds/annotations/words   # dev audio + words
TCS=$(head -3 $O/stems.txt | paste -sd,)
for G in 0 1; do
  CUDA_GRAPH=$G CK=$B/echog-g1-sf-1105/dev/sf0/final.pt VAE=$VAE STATS=$C/scmg/STATS_echog_g1_v1.json HOSV=$HOSV TEXT=$TEXT TEXTC=$TEXTC \
    DS=$O/ds PSTATS=$PS CAMPPLUS=$CAMPPLUS BASE_FILE=${BASE_FILE:?} STREAM_CORE=$L/snap/echog_stream/echog_stream/stream_core.py \
    TC_DEV=$B/echog-g1-hos-assets/dev_stems.txt SP=$O $PY scripts/tests/test_causal_stream.py > $O/test_causal_stream_cg$G.log 2>&1 \
    && log "TEST_CAUSAL_STREAM cuda_graph=$G PASS" || log "TEST_CAUSAL_STREAM cuda_graph=$G FAIL (see log)"
done
log CG_CHECK_DONE
