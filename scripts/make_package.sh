#!/bin/bash
# DLC training package from a CLEAN git tree (11-03): scripts/make_package.sh <name> [phase]
#   -> /mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/scmg-train-<name>-<YYYYMMDD>/ (git archive of HEAD: scmg/ scripts/)
#      + RUN_NAME, ADSC_TARGET, VAE_SHA256, GIT_COMMIT, FILES.sha256, and <pkg>-submit/command.sh (sha-pinned); tag pkg-<name>
set -euo pipefail
name=${1:?name, e.g. r7i}; phase=${2:-full}
cd "$(git rev-parse --show-toplevel)"
git diff --quiet && git diff --cached --quiet || { echo "refusing: uncommitted changes (commit first)"; exit 1; }
[ -z "$(git ls-files --others --exclude-standard scmg scripts)" ] || { echo "refusing: untracked files in scmg/ scripts/"; exit 1; }
git rev-parse -q --verify "refs/tags/pkg-$name" >/dev/null && { echo "refusing: tag pkg-$name exists"; exit 1; }
D=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928; day=$(date +%Y%m%d); P=$D/scmg-train-$name-$day
[ -e "$P" ] && { echo "refusing: $P exists"; exit 1; }
VAE=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt
mkdir -p "$P"; git archive HEAD scmg scripts | tar -x -C "$P"
echo "scmg-train-$name-$(date +%m%d)" > "$P/RUN_NAME"; echo scmg-train-r6-1102 > "$P/ADSC_TARGET"
sha256sum "$VAE" | cut -c1-64 > "$P/VAE_SHA256"; git rev-parse HEAD > "$P/GIT_COMMIT"
(cd "$P" && find . -type f ! -name FILES.sha256 | sed 's|^\./||' | sort | xargs sha256sum > FILES.sha256)
F=$(sha256sum "$P/FILES.sha256" | cut -c1-64); E=$(sha256sum "$P/scripts/dlc_entry.sh" | cut -c1-64)
mkdir -p "$P-submit"; C="$P-submit/command.sh"
cat > "$C" <<CMD
#!/bin/bash
set -euo pipefail
: "\${DLC_JOB_ID:?real DLC job required}"
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/\$DLC_JOB_ID
mkdir -p "\$L"
exec > >(tee -a "\$L/\$(hostname).log") 2>&1
P=$P
PHASE=\${1:-$phase}
printf '%s\n' '$F  $P/FILES.sha256' '$E  $P/scripts/dlc_entry.sh' | sha256sum -c -
test ! -L "\$P"
exec /bin/bash "\$P/scripts/dlc_entry.sh" "\$PHASE"
CMD
chmod -R a-w "$P"
git tag -a "pkg-$name" -m "DLC package $P (FILES $F, command $(sha256sum "$C" | cut -c1-64))"
echo "package  $P"; echo "commit   $(git rev-parse --short HEAD)  tag pkg-$name"; echo "FILES    $F"; echo "command  $C  $(sha256sum "$C" | cut -c1-64)"
echo "user_command: bash $C $phase"
