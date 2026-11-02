"""Soft-intent labels on the VoxCPM LM tokens (row-aligned with the token cache from prep_tokens.py).

For semantic-synthesis items (semsynth_r7p1) the executed intents of MOTION.json (key, char_start, anchor_text, votes)
are mapped onto the Qwen2.5 token that contains the anchor's first character (as BJ50 labels the anchor-start token);
every other token of a labelled item is class 0 (none); tokens of items without intent information are -1 (no loss).
Class ids follow the view's VOCAB.json (2..213; 0 none, 1 unknown) -- identical to BJ50's table.
Outputs (into --out): INTENT_LABEL.npy int16 [Ntok], INTENT_CONF.npy float16 [Ntok] (votes / 5), PREP.json.
    python -m scmg.prep_intent --tok-cache <dir> --out <dir>"""
import argparse, json, os
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .data import VIEW

MAN = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/unified-ex002-manifest-v3_6/semsynth_r7p1.jsonl'
REGATE = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/synth-sem-r1/fix-r7p1-20261030/REGATE_SPEED_0p70.jsonl'
LM = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/Qwen2.5-0.5B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--tok-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=LM)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((Path(a.view) / 'VOCAB.json').read_text())
    tc = Path(a.tok_cache); ids = np.load(tc / 'TOK_IDS.npy'); index = json.loads((tc / 'TOK_INDEX.json').read_text())
    lab = np.full(len(ids), -1, np.int16); conf = np.zeros(len(ids), np.float16)
    tok = AutoTokenizer.from_pretrained(a.model)
    motion = {}
    for l in open(REGATE):
        r = json.loads(l); motion[r['id']] = os.path.join(os.path.dirname(r['motion']), 'MOTION.json')
    man = {}
    for l in open(MAN):
        r = json.loads(l)
        man[r['clip_id']] = r
    items = json.loads((Path(a.view) / 'train.json').read_text())
    st = Counter()
    for it in items:
        if not str(it.get('data_source', '')).startswith('semsynth') or it['id'] not in index: continue
        cid = it.get('clip_id') or it['id'].split('#')[0]
        r = man.get(cid)
        if r is None: st['no_manifest'] += 1; continue
        pair = cid.replace('semsynth-r7p1-', '').split('__')[0]
        mp = motion.get(pair)
        if mp is None or not os.path.exists(mp): st['no_motion'] += 1; continue
        try:
            mo = json.loads(open(mp, encoding='utf-8', errors='replace').read())
        except Exception:
            st['bad_motion'] += 1; continue
        executed = Counter(e['key'] for e in r.get('events', []) if e.get('executed'))
        full = r['text']['plain']; text = it.get('text') or ''
        off = full.find(text) if text else -1
        if off < 0: st['text_mismatch'] += 1; continue
        r0, n = index[it['id']]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = enc['offset_mapping'][:n]
        lab[r0:r0 + n] = 0
        for e in mo.get('intent', []):
            k = e.get('key')
            if k not in vocab or executed[k] <= 0: st['intent_not_executed'] += 1; continue
            c = int(e['char_start']) - off
            if not (0 <= c < len(text)):
                continue                                               # anchor lies in another piece of the clip
            j = next((t for t, (s, f) in enumerate(spans) if s <= c < max(f, s + 1)), None)
            if j is None: st['anchor_not_tokenised'] += 1; continue
            lab[r0 + j] = vocab[k]; conf[r0 + j] = min(1., float(e.get('votes', 5)) / 5.); executed[k] -= 1; st['labelled'] += 1
        st['items'] += 1
    np.save(out / 'INTENT_LABEL.npy', lab); np.save(out / 'INTENT_CONF.npy', conf)
    pos = lab[lab > 1]
    (out / 'PREP.json').write_text(json.dumps(dict(stats=st, tokens=int(len(lab)), labelled_tokens=int((lab >= 0).sum()),
                                                   intent_tokens=int(len(pos)), classes_used=int(len(set(pos.tolist()))),
                                                   tok_cache=str(tc), n_classes=max(vocab.values()) + 1), indent=1, default=int))
    print(json.dumps(st), 'intent tokens', len(pos), 'classes', len(set(pos.tolist())))


if __name__ == '__main__':
    main()
