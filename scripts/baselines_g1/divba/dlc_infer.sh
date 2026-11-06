#!/bin/bash
# dlc_infer.sh (generic copy of dlc_zt.sh): REL, OUT, CKS (checkpoint sha prefix) and CONFIGS ("name|steps|infer extra[|ckpt]" lines)
# from env; COMMON_EXTRA is appended to every config (11-06: "--seed-mode per_clip"); GPUS = bench.sh shard GPU list.
# zehan-scmg-g1-divba-zt-1106 / -sde-1106: zero-training Div/BA candidates on the D-spk DEV split only (400 clips; criteria fixed before the
# run in scripts/baselines_g1/divba/CRITERIA.md, commit ac2289c). Model czm6k dev (hos-base + causal-matched 6000 steps).
# Configs: ctrl (10 steps, noise 1.0) | st16 (16 ODE steps) | ns09 / ns08 (initial-noise scale 0.9 / 0.8);
# protocols offline + psync (incremental feat89, speaker refresh = grades, as the main table) x inference seeds 0 / 1.
# Official evaluator after each run (bench.sh). Test set never touched. Safety / diag_beats: local CPU afterwards.
set -uo pipefail
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/${DLC_JOB_ID:-manual}; mkdir -p $L
exec > >(tee -a $L/run.log) 2>&1
REL=${REL:?release dir}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
D=$B/baseline-repro-1104/echog/data_root; A=$B/echog-g1-hos-assets; CZ=$B/causal-train-1105; HV=$B/views/echog-g1-v1-hos
CK=${CK:-$CZ/runs-local/dev/czm6k/final.pt}; DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt
O=${OUT:?out dir}; mkdir -p $O
export PY=/usr/local/bin/python EVAL_PY=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python
export OFF=$W/official MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
# dev_env of causal_g1.sh (D-spk view, its 400 dev clips, dev words / audio)
export REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val \
  TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 WORDS_DIR=$CZ/words-dev CAUSAL_WAV=$A/audio
export CAMPPLUS=$CZ/campplus SPK_REFRESH=grades CAUSAL_FEAT=inc OMP_NUM_THREADS=2 USE_TF=0
unset SCMG_LOCAL_MIRROR
sha256sum $CK | grep -q ^${CKS:-6b4e5bce73fa52a9} || { echo "ckpt sha mismatch"; exit 3; }
cd $REL && echo "release commit $(cat GIT_COMMIT) host $(hostname) nproc $(nproc)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
( while true; do echo "$(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | paste -sd';')"; sleep 30; done ) > $L/MONITOR.log 2>&1 &
MON=$!
while IFS='|' read -r cfg ST EX CKC; do
  [ -z "$cfg" ] && continue
  for p in psync offline; do for sd in 0 1; do
    P=$O/$cfg/$p/seed_00$sd; [ -f $O/$cfg/$p/benchmark_seed_00$sd.json ] && continue
    echo "$(date -Is) START $cfg $p s$sd steps $ST extra '$EX ${COMMON_EXTRA:-}' ckpt ${CKC:-$CK}"
    STEPS=$ST INFER_EXTRA="$EX ${COMMON_EXTRA:-}" GPUS="${GPUS:-0 0 0 0 1 1 1 1}" CFGW=2.5 bash scripts/echog_g1/bench.sh scmg ${CKC:-$CK} $DEC $P $p $sd > $O/bench_${cfg}_${p}_s$sd.log 2>&1 \
      && echo "$(date -Is) DONE $cfg $p s$sd $(grep -h 'G1-emageFGD' $O/bench_${cfg}_${p}_s$sd.log | tail -1)" || echo "$(date -Is) FAILED $cfg $p s$sd"
  done; done
done <<< "${CONFIGS:?configs}"
kill $MON; echo "$(date -Is) JOB_DONE"
