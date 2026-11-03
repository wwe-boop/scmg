#!/bin/bash
# Evaluation / own-GPU code snapshot from a CLEAN git tree (11-03): scripts/make_release.sh [label]
#   -> /mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/<YYYYMMDD-HHMMSS>[-label]/ (git archive of HEAD) + GIT_COMMIT; tag rel-<dir>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git diff --quiet && git diff --cached --quiet || { echo "refusing: uncommitted changes (commit first)"; exit 1; }
[ -z "$(git ls-files --others --exclude-standard scmg scripts)" ] || { echo "refusing: untracked files in scmg/ scripts/"; exit 1; }
n=$(date +%Y%m%d-%H%M%S)${1:+-$1}; R=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/$n
mkdir -p "$R"; git archive HEAD scmg scripts | tar -x -C "$R"; git rev-parse HEAD > "$R/GIT_COMMIT"
git tag "rel-$n"; echo "$R  (commit $(git rev-parse --short HEAD), tag rel-$n)"
