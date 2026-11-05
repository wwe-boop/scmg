"""Speaker-disjoint halves for the semantic line (11-06 rule 3: retrieval bank speakers and recogniser-training speakers apart).
Train-split speakers with BEAT2 semantic events (data-sem-hos EVENTS.json) are split deterministically into
  R : recogniser training speakers        K : retrieval-bank / synthesis speakers
by greedy balancing of their deictic + iconic + metaphoric event counts (largest first, to the lighter half; ties -> R).
Test speakers (1_wayne, 3_solomon, 21_ayana) and dev speakers (2_scott, 6_carla) are in neither half (they never train
anything here). Writes SPLIT_RK.json and a recogniser data dir whose train events are R speakers only (K train events ->
split 'bank'; dev / test events unchanged; MOTION.npz symlinked).
    python scripts/semantic/g1_speaker_split.py --data <data-sem-hos> --out <dir>
"""
import argparse, json, os
from collections import Counter
from pathlib import Path

TEST_SPK = {'1_wayne', '3_solomon', '21_ayana'}; DEV_SPK = {'2_scott', '6_carla'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    d = json.loads((Path(a.data) / 'EVENTS.json').read_text())
    cnt = Counter(e['spk'] for e in d['events'] if e['split'] == 'train' and e['cls'] != 'beat')
    spk = sorted((s for s in cnt if s not in TEST_SPK | DEV_SPK), key=lambda s: (-cnt[s], s))
    R, K, nR, nK = [], [], 0, 0
    for s in spk:
        if nR <= nK: R.append(s); nR += cnt[s]
        else: K.append(s); nK += cnt[s]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = dict(R=sorted(R), K=sorted(K), events_R=nR, events_K=nK, excluded_test=sorted(TEST_SPK), excluded_dev=sorted(DEV_SPK),
                 rule='greedy balance of train semantic-event counts, largest first; test / dev speakers excluded', source=a.data)
    (out / 'SPLIT_RK.json').write_text(json.dumps(split, indent=1))
    rd = out / 'data-sem-hos-R'; rd.mkdir(exist_ok=True)
    ev = [dict(e, split='bank') if (e['split'] == 'train' and e['spk'] in set(K)) else e for e in d['events']]
    bad = [e for e in ev if e['split'] == 'train' and e['spk'] not in set(R)]
    if bad: raise SystemExit(f'{len(bad)} train events outside R (e.g. {bad[0]["spk"]})')
    (rd / 'EVENTS.json').write_text(json.dumps(dict(d, events=ev, split={k: ('bank' if (v == 'train' and k.split('_')[1] + '_' + k.split('_')[2] in set(K)) else v)
                                                                       for k, v in d['split'].items()})))
    mo = rd / 'MOTION.npz'
    if not mo.exists(): os.symlink(os.path.realpath(Path(a.data) / 'MOTION.npz'), mo)
    print(json.dumps(split, indent=1))


if __name__ == '__main__':
    main()
