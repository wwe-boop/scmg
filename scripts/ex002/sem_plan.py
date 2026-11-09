"""P2 sem_cf quick-lane plans (11-09; exp/text-path docs/criteria/P2_SEMCF_1109.md §2, §5), ex002_ctl_run plan format (kind 'sem').
  oracle    : the dev GT semantic events of the quick-lane recordings (DEV_EVENTS.json of sem_bank.py): token time t_tok
              (keyword onset - 0.2 s), hand + standardised descriptor of the event's GT window; 'shuffle' = another recording's
              event (derangement across recordings, seed 20261109).
  retrieval : the deployable rule from the TTS text: every word of the recording (data-v36_1 alignments, g1_lexicon) whose lexicon
              category maps to a class (scmg.ctl.LEX2SEM) -> token at onset - 0.2 s; greedy in time, tokens closer than 1.2 s to
              the previous one dropped; descriptor + hand from the bank: same keyword, else same lexicon category, else same class;
              entry sha1("<clip>:<t:.3f>:sem") mod n.
    SCMG_J=34 python scripts/ex002/sem_plan.py --sem SEM_DIR --cis 0,4,...,36 --out OUT_DIR"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg.ctl import LEX2SEM
from sem_bank import V, EV, recordings, rec_words


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sem', required=True); ap.add_argument('--cis', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); S = Path(a.sem); O = Path(a.out); O.mkdir(parents=True, exist_ok=True); cis = [int(x) for x in a.cis.split(',')]
    dev = [e for e in json.loads((S / 'DEV_EVENTS.json').read_text())['events'] if e['ci'] in cis]
    rng = np.random.default_rng(20261109); p = rng.permutation(len(dev))
    for _ in range(100):
        bad = [k for k in range(len(dev)) if dev[p[k]]['ci'] == dev[k]['ci']]
        if not bad: break
        for k in bad: j = int(rng.integers(len(dev))); p[k], p[j] = p[j], p[k]
    ev = [dict(ci=e['ci'], clip=e['stem'], kind='sem', t=e['t_tok'], ev_t=e['t'], hand=e['hand'], desc=e['desc'], cls=e['cls'], kw=e['kw'],
               shuffle=dict(hand=dev[p[k]]['hand'], desc=dev[p[k]]['desc'])) for k, e in enumerate(dev)]
    (O / 'PLAN_oracle.json').write_text(json.dumps(dict(events=ev, n_events=len(ev), cis=cis, rule='sem_plan.py oracle')))
    # deployable retrieval
    Bk = np.load(S / 'SEM_BANK.npz'); kw, cat, cls, Dd, hand = Bk['kw'], Bk['cat'], Bk['cls'], Bk['D'], Bk['hand']
    D = json.loads(Path(EV).read_text()); inv = {v: k for k, v in D['meta']['dev_ci'].items()}
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text())
    recs = recordings(items, set(sp['val_ids'])); out = []; how = {'kw': 0, 'cat': 0, 'cls': 0}
    for ci in cis:
        stem = inv[ci]; last = -1e9
        for w, c, on, off, iid, f0 in sorted(rec_words(recs[stem]), key=lambda x: x[2]):
            if c not in LEX2SEM: continue
            t = on - 0.2
            if t - last < 1.2: continue
            cand = np.nonzero(kw == w)[0]; src = 'kw'
            if not len(cand): cand = np.nonzero(cat == c)[0]; src = 'cat'
            if not len(cand): cand = np.nonzero(cls == LEX2SEM[c])[0]; src = 'cls'
            if not len(cand): continue
            j = int(cand[int(hashlib.sha1(f"{stem}:{t:.3f}:sem".encode()).hexdigest()[:12], 16) % len(cand)])
            out.append(dict(ci=ci, clip=stem, kind='sem', t=float(t), word=w, cat=c, src=src, bank=j, hand=int(hand[j]), desc=Dd[j].tolist(),
                            shuffle=dict(hand=int(hand[j]), desc=Dd[j].tolist())))
            how[src] += 1; last = t
    (O / 'PLAN_retrieval.json').write_text(json.dumps(dict(events=out, n_events=len(out), cis=cis, sources=how, rule='sem_plan.py retrieval')))
    print(json.dumps(dict(oracle=len(ev), retrieval=len(out), sources=how)))


if __name__ == '__main__':
    main()
