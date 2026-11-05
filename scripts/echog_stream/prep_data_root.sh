#!/bin/bash
# Full ECHO-G G1 data root on CPFS for the streaming fine-tune (11-05). Run on a DSW host (speech_server), NOT on DLC GPUs:
#   bash scripts/echog_stream/prep_data_root.sh
# The 1104 data root (baseline-repro-1104/echog/data_root) holds the 3242 val_common clips only; training needs all 14987
# train_drop clips. Downloads the missing official condition shards (HF dataset gaopusen/ECHO-G, pinned revision, via
# hf-mirror), verifies every shard against the official ARCHIVE_SHA256SUMS, extracts condition_30fps + motion_39d_30fps
# (all 18229 clips) and links audit/splits/stats/eval_assets (official) and audio (val only; training reads no audio).
set -euo pipefail
D=${D:-/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105}; R=$D/data_root; H=$D/hf-data
REF=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/reference/official-data
OLD=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog
REV=a1280e7524817e74f06c928285698378ba5d85bc; URL=${HF_URL:-https://hf-mirror.com}/datasets/gaopusen/ECHO-G/resolve/$REV
mkdir -p $H $R
declare -A SHA=(
 [motion_39d_30fps-00000-of-00001.tar]=baa8b6b73d0e45b2c8bf5299eb3c0f5fc2626b990c65567329b38fbc9511e0c0
 [condition_30fps-00000-of-00006.tar]=4da1e2ccb28fa2dbd558f82b199d9947fbd2db87abe2ba5602c7074a4ab460ea
 [condition_30fps-00001-of-00006.tar]=8d885e11fca9f2131bca77ab713f071279bd329990c8773a0af8f65ba2c896e5
 [condition_30fps-00002-of-00006.tar]=42c5d45b96a9df7e028d58083ac5bb60a9e3a204634ce4672b1bbf32e2571dfb
 [condition_30fps-00003-of-00006.tar]=8a1443a718af57bb0c172e6437e436d7509482aaa74cf0a429f5bfddc0a5086c
 [condition_30fps-00004-of-00006.tar]=fe162492a8a06af6b98a435277be01939b7333412195d31262876e1a312e016c
 [condition_30fps-00005-of-00006.tar]=774e1c2efc543e429eb0846249f554fe2a262f1aead5c721e1bcdd3398c8bdba )
src() {  # existing copy or a fresh download into $H (verified below either way)
  for p in $REF/data/$1 $OLD/hf-data/data/$1 $H/$1; do [ -f $p ] && { echo $p; return; }; done
  curl -sSL --retry 5 --retry-delay 10 -o $H/$1.part "$URL/data/$1" && mv $H/$1.part $H/$1 && echo $H/$1
}
for f in "${!SHA[@]}"; do ( src $f > $H/$f.src ) & done; wait
declare -A P
for f in "${!SHA[@]}"; do p=$(cat $H/$f.src); P[$f]=$p
  echo "${SHA[$f]}  $p" | sha256sum -c - || { echo "SHA_MISMATCH $f ($p)"; exit 1; }
done
for f in "${!SHA[@]}"; do tar -xf ${P[$f]} -C $R; done
for l in audit splits stats eval_assets; do ln -sfn $REF/$l $R/$l; done
ln -sfn $OLD/data_root/audio $R/audio
nc=$(ls $R/condition_30fps | wc -l); nm=$(ls $R/motion_39d_30fps | wc -l); echo "conditions $nc motions $nm"
[ $nc = 18229 ] && [ $nm = 18229 ] && echo "PREP_DONE $(date -Is)" > $D/data_root.DONE && cat $D/data_root.DONE
