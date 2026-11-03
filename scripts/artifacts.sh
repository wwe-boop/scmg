ARTIFACT_PY=${ARTIFACT_PY:-$PY}
# Shared by lane/final/SEMAUX. Every call propagates status explicitly (also safe inside && / functions).
valid_json() { "$ARTIFACT_PY" "$CODE/scripts/artifacts.py" valid "$1" "$2"; }
json_run() {
  local path=$1 kind=$2 mode=$3; shift 3
  valid_json "$path" "$kind" && return 0
  "$ARTIFACT_PY" "$CODE/scripts/artifacts.py" run "$path" "$kind" "$mode" "$@" || return $?
}
invalidate_marker() {
  if [ -e "$1" ]; then
    local archive
    archive=$(mktemp -d "$(dirname "$1")/.previous-marker-XXXXXX") || return 1
    mv "$1" "$archive/$(basename "$1")" || return 1
  fi
}
semantic_run() {  # output-root label checkpoint cfg-weight variants n
  local so=$1 label=$2 ck=$3 w=$4 variants=$5 n=$6 vv
  invalidate_marker "$so/DONE" || return 1
  if ! "$ARTIFACT_PY" "$CODE/scripts/artifacts.py" semantic "$SEMMAN/MANIFEST.json" "$so/$label" "$variants" "$n"; then
    $PY "$CODE/scripts/gen_semantic.py" "$SEMMAN" "$so" "$label=$ck@$w" --variants "$variants" --n "$n" || return 1
    # New generations require fresh scores even when old score JSON was syntactically valid.
    for vv in ${variants//,/ }; do invalidate_marker "$so/score-$vv.json" || return 1; done
  fi
  "$ARTIFACT_PY" "$CODE/scripts/artifacts.py" semantic "$SEMMAN/MANIFEST.json" "$so/$label" "$variants" "$n" || return 1
  for vv in ${variants//,/ }; do
    json_run "$so/score-$vv.json" score file $PY "$CODE/scripts/semantic_eval.py" score \
      --gen-dir "$so/$label/$vv" --manifest "$SEMMAN/MANIFEST.json" || return 1
  done
  for vv in ${variants//,/ }; do valid_json "$so/score-$vv.json" score || return 1; done
  touch "$so/DONE" || return 1
}
