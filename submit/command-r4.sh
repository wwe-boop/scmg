#!/bin/bash
set -euo pipefail
: "${DLC_JOB_ID:?real DLC job required}"
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/$DLC_JOB_ID
mkdir -p "$L"
exec > >(tee -a "$L/$(hostname).log") 2>&1
P=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r4-20261102
PHASE=${1:-full}
printf '%s\n' 'cfd5b94a8515426a59dd09eb475233a1b04e3aff8be2e6677c060fdabca74ea0  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r4-20261102/FILES.sha256' 'f339d9a5ddfc89e37dcdd72c33af3d62fed7b3ccacc462a18426faacacd53e4f  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r4-20261102/scripts/dlc_entry.sh' | sha256sum -c -
test ! -L "$P"
exec /bin/bash "$P/scripts/dlc_entry.sh" "$PHASE"
