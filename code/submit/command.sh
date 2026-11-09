#!/bin/bash
set -euo pipefail
: "${DLC_JOB_ID:?real DLC job required}"
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/$DLC_JOB_ID
mkdir -p "$L"
exec > >(tee -a "$L/$(hostname).log") 2>&1
P=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r1-20261102
PHASE=${1:-full}
printf '%s\n' 'ede2be8913c5a67505ef95d43e8bc17fd63fac08e40a7405f889716575f366ac  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r1-20261102/FILES.sha256' '99b28637ffce7a1172f49479e8266b83085f6f6e7fa6dab7a09f715f866f9fbd  /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-r1-20261102/scripts/dlc_entry.sh' | sha256sum -c -
test ! -L "$P"
exec /bin/bash "$P/scripts/dlc_entry.sh" "$PHASE"
