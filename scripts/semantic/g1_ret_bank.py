"""Causal retrieval of semantic gesture exemplars for the Local Flow (11-06 round 4, iconic route).
Bank  : BEAT2 annotated segments (default class iconic) of the K-half speakers (g1_speaker_split.py SPLIT_RK.json; no test / dev
        speaker), TRAIN items only. Entry = (key, class, latent snippet): key = mean PREFIX-CAUSAL e5 feature of the tokens with
        time in [t - 0.2, t + LOOK] (t = the annotated keyword onset; the look-ahead phrase a TTS text stream delivers), snippet =
        frozen-VAE latents of the ground-truth motion [t - PRE, t - PRE + 2.4 s) (30 latents), normalised by the model's lat stats.
Query : a token (class c, time t) of an item; key built the same way from THAT item's causal e5 (only words up to t + LOOK);
        top-1 cosine among bank entries of class c from ANOTHER recording (never the query's own recording).
Output dir: BANK.npz (key, cls, lat), BANK.json [{stem, spk, recording, t}], RET_<name>.json {stem: [{t, cls, bank, sim}]},
        RETLOG_<name>.json [{query, retrieved}] (for check_sem_leak.py --bank BANK.json --retrieval-log RETLOG).
    python scripts/semantic/g1_ret_bank.py --ctl <CTL dir> --split SPLIT_RK.json --view <hos view> --text-causal <e5 causal>
        --vae <vae> --ckpt <model with lat stats> --out <dir> [--queries train | dev-ann | dev-pred:<PRED.json> | dev-auto]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ctl as C

PRE, LEN, LOOK, BACK = 0.8, 30, 1.0, 0.2
rec_of = lambda stem: re.sub(r'_utt_\d+$', '', stem)
topic_of = lambda stem: '_'.join(stem.split('_')[4:6])          # BEAT2 English is partly scripted: same topic id across speakers


def key_of(X, T, ix, stem, t):
    k0, kn = ix[stem]; tt = np.asarray(T[k0:k0 + kn], np.float64)
    m = np.isfinite(tt) & (tt >= t - BACK) & (tt <= t + LOOK)
    if not m.any():
        ok = np.isfinite(tt) & (tt <= t + LOOK)
        if not ok.any(): return None
        m = np.zeros_like(ok); m[np.where(ok)[0][-1]] = True
    v = np.asarray(X[k0:k0 + kn], np.float32)[m].mean(0)
    return v / max(np.linalg.norm(v), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'split', 'view', 'text_causal', 'vae', 'ckpt', 'out'): ap.add_argument('--' + k.replace('_', '-'), required=True)
    ap.add_argument('--exclude-same-topic', action='store_true', help='also never retrieve from the same BEAT2 topic (another speaker '
                    'reading the same script); outputs get the suffix _xt')
    ap.add_argument('--classes', default='iconic'); ap.add_argument('--queries', nargs='+', default=['train', 'dev-ann'])
    a = ap.parse_args(); torch.set_num_threads(4)
    from scmg.vae import MotionVAE
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text()); S = json.loads(Path(a.split).read_text()); K = set(S['K'])
    v = Path(a.view); items = {it['id']: it for it in json.loads((v / 'train.json').read_text())}
    sp = json.loads((v / 'SPLIT.json').read_text()); train_ids = set(sp['train_ids'])
    stats = None
    q = np.load(v / 'train_q.npy', mmap_mode='r')
    tc = Path(a.text_causal); X = np.load(tc / 'TEXT_FEAT.npy', mmap_mode='r'); T = np.load(tc / 'TEXT_TIME.npy'); ix = json.loads((tc / 'TEXT_INDEX.json').read_text())
    vck = torch.load(a.vae, map_location='cpu'); vae = MotionVAE(**vck['cfg']); vae.load_state_dict(vck['ema']); vae.eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); lm, ls = ck['ema']['lat_mean'].float(), ck['ema']['lat_std'].float()
    st = json.loads(Path(ck.get('stats_path', '')).read_text()) if ck.get('stats_path') else None
    want = {C.TYPE_ID[c] for c in a.classes.split(',')}
    from scmg.common import P as Pm
    statf = Path(Pm('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_echog_g1_v1.json'))
    stj = json.loads(statf.read_text()); qm, qs = np.asarray(stj['q_mean'], np.float32), np.asarray(stj['q_std'], np.float32)
    keys, cls, lats, meta = [], [], [], []
    for s, ts in sorted(raw.items()):
        it = items.get(s)
        if it is None or s not in train_ids or it.get('speaker') not in K or s not in ix: continue
        for t in ts:
            if t['type'] not in want: continue
            kv = key_of(X, T, ix, s, t['t'])
            if kv is None: continue
            f0 = int(round((t['t'] - PRE) * 50)); n = it['frames']
            idx = np.clip(np.arange(f0, f0 + LEN * 4), 0, n - 1)
            mo = (np.asarray(q[it['offset'] + idx], np.float32) - qm) / qs
            with torch.no_grad(): mu, _ = vae.encode(torch.tensor(mo)[None])
            lat = ((mu[0] - lm) / ls).numpy().astype(np.float16)
            keys.append(kv); cls.append(t['type']); lats.append(lat)
            meta.append(dict(stem=s, spk=it['speaker'], recording=rec_of(s), t=t['t'], cls=C.TYPE_NAMES[t['type']]))
    keys, cls, lats = np.stack(keys), np.asarray(cls), np.stack(lats)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    np.savez(o / 'BANK.npz', key=keys, cls=cls, lat=lats); (o / 'BANK.json').write_text(json.dumps(meta))
    brec = np.array([m['recording'] for m in meta]); btop = np.array([topic_of(m['stem']) for m in meta])
    print('bank', len(meta), 'entries of', sorted({m['spk'] for m in meta}), flush=True)

    def retrieve(stem, t, c):
        kv = key_of(X, T, ix, stem, t)
        if kv is None: return None
        ok = (cls == c) & (brec != rec_of(stem))
        if a.exclude_same_topic: ok &= btop != topic_of(stem)
        if not ok.any(): return None
        sim = keys @ kv; sim[~ok] = -9
        j = int(sim.argmax()); return j, float(sim[j])

    for qname in a.queries:
        if qname == 'train':
            src = {s: [t for t in ts if t['type'] in want] for s, ts in raw.items() if ts and ts[0]['split'] == 'train'}
        elif qname == 'dev-ann':
            src = {s: [t for t in ts if t['type'] in want] for s, ts in raw.items() if ts and ts[0]['split'] == 'dev'}
        elif qname.startswith('dev-pred:'):
            Pd = json.loads(Path(qname.split(':', 1)[1]).read_text()); src = {s: [t for t in ts if t['type'] in want] for s, ts in Pd.items()}
            qname = 'dev-pred-' + Path(qname.split(':', 1)[1]).parent.name
        elif qname == 'dev-auto':
            src = {s: [dict(t=t['t'], type=C.TYPE_ID[C.LEX2SEM[t['cls'][4:]]]) for t in ts if t['cls'].startswith('lex_') and t['cls'][4:] in C.LEX2SEM
                       and C.TYPE_ID[C.LEX2SEM[t['cls'][4:]]] in want] for s, ts in raw.items() if ts and ts[0]['split'] == 'dev'}
        else: raise SystemExit(f'query set {qname}')
        out, log = {}, []
        for s, ts in sorted(src.items()):
            if s not in ix: continue
            r = []
            for t in ts:
                got = retrieve(s, t['t'], t['type'])
                if got is None: continue
                r.append(dict(t=round(float(t['t']), 3), cls=C.TYPE_NAMES[t['type']], bank=got[0], sim=round(got[1], 4)))
                log.append(dict(query=s, retrieved=meta[got[0]]['stem']))
            if r: out[s] = r
        sfx = '_xt' if a.exclude_same_topic else ''
        (o / f'RET_{qname}{sfx}.json').write_text(json.dumps(out)); (o / f'RETLOG_{qname}{sfx}.json').write_text(json.dumps(log))
        print(qname, sum(len(x) for x in out.values()), 'retrievals in', len(out), 'items', flush=True)
    (o / 'BANK_INFO.json').write_text(json.dumps(dict(pre=PRE, len_latents=LEN, look=LOOK, back=BACK, classes=a.classes, split=a.split,
                                                      text_causal=a.text_causal, vae=a.vae, ckpt=a.ckpt, n=len(meta)), indent=1))


if __name__ == '__main__':
    main()
