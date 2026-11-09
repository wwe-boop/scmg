#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/bagger_env.sh"
export CUDA_VISIBLE_DEVICES=${1:-1}
ROOT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/bagger-1103
mkdir -p "$ROOT/tests"
exec "$PY" scripts/test_bagger.py --orig /mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/bagger-1103/base --out "$ROOT/tests" >> "$ROOT/tests/test.log" 2>&1
