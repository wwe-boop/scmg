"""11-07 duplex: YIELD control tokens of the train-side post-speech tails whose GT returns to rest (kinds retract / low; hold
tails carry none): one token per tail item at the speech offset t (a = t, b = t + tail length), split 'train'.
    python make_yield_tokens.py --view <tail view (hos)> --hos-split <views/echog-g1-v1-hos/SPLIT.json> --out <dir>
Writes <dir>/CTL.json + PREP.json (split_sha256 of the view's SPLIT.json, as scmg.data.load_ctl_tokens checks)."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.ctl import YIELD
ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
v = Path(a.view); items = json.loads((v / 'train.json').read_text())
split = set(json.loads((v / 'SPLIT.json').read_text())['train_ids'])
out = {}; n = dict(retract=0, low=0, hold=0, outside_split=0)
for it in items:
    t = it.get('tail')
    if not t: continue
    if it['id'] not in split: n['outside_split'] += 1; continue
    n[t['kind']] += 1
    if t['kind'] == 'hold': continue
    t0 = t['speech_frames'] / 50.
    out[it['id']] = [dict(type=YIELD, cls='yield', t=t0, a=t0, b=t0 + t['frames'] / 50., split='train', kind=t['kind'])]
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
(o / 'CTL.json').write_text(json.dumps(out))
(o / 'PREP.json').write_text(json.dumps(dict(split_sha256=hashlib.sha256((v / 'SPLIT.json').read_bytes()).hexdigest(), view=str(v), counts=n,
                                             rule='make_yield_tokens.py: YIELD at the speech offset of retract / low tails')))
print(n, len(out))
