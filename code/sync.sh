#!/bin/bash
# Push local code to CPFS as a NEW release directory (never overwrite files in place: other CPFS clients, e.g.
# zehan-server, keep stale handles to replaced files). Prints the release path; `latest` file records it.
set -e
cd "$(dirname "$0")"
REL=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/$(date +%Y%m%d-%H%M%S)
tar czf - --exclude=__pycache__ --exclude='*.pyc' scmg scripts | \
  ssh speech_server "mkdir -p $REL && tar xzf - -C $REL && chmod -R a+rX $REL && echo $REL > /mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/LATEST.$(date +%s)"
echo $REL
