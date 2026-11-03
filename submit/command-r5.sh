#!/bin/bash
set -euo pipefail
: "${DLC_JOB_ID:?real DLC job required}"
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/$DLC_JOB_ID
mkdir -p "$L"
exec > >(tee -a "$L/$(hostname).log") 2>&1
P=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r5-20261102
PHASE=${1:-full}
printf '%s\n' '53af3bae8931e01e075a60cd07c15ffc3474f07017b13cdbf478b396a09ecc1b  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r5-20261102/FILES.sha256' 'f339d9a5ddfc89e37dcdd72c33af3d62fed7b3ccacc462a18426faacacd53e4f  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r5-20261102/scripts/dlc_entry.sh' | sha256sum -c -
test ! -L "$P"
exec /bin/bash "$P/scripts/dlc_entry.sh" "$PHASE"
