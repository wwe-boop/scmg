"""11-07 shared EX002 benchmark entry (coordinator: one entry for the frozen-protocol numbers of both agents).
  infer : the 43 dev recordings (380 utterances) of data-v36_1 (SPLIT val = the fixed v3.2 BEAT2 dev), protocols
          offline  final text, H = inf                                         (= ex002_ctl_run.py infer --mode none)
          engine   x2 early text streaming (15 tok/s, lag 0.15 s), prefix-causal e5, deploy timing, H = 160 ms
          psync    strict causal: a word is visible once it ended in the arrived audio (voiced), feat89 / CAM++ of the arrived
                   PCM every tick (scmg.causal_feat.CausalInputs: recording PCM = its items' audio16k concatenated in offset
                   order, words = the items' alignment units, fixed prosody statistics, CAM++ grades), H = 160 ms
          per_clip seeds: seed + crc32(recording) (the ex002_ctl_run rule); 10 steps, CFG 2.0 (EX002 defaults); generation of
          L // 16 chunks, output y = cmd[:L] (as ex002_ctl_run) -> <out>/<ci>.npy (50 Hz, 34 joints, rad) + <out>/../gt_<ci>.npy
  eval  : FKD = scmg.evaluate.kin_feats / frechet of whole recordings vs GT; FGD v2 = mean over the two frozen encoders
          (USERS/zehan/fgd-ex002-v2-20261102/code model.Embedder, config.WEIGHTS / human_seed{0,1} / best.pt) of
          frechet(GT features, generated features) -- the same calls in the same order as ex002_ctl_run.evaluate
    SCMG_J=34 python scripts/ex002/bench_ex002.py infer --ckpt C --protocol psync --seed 0 --out D [--shard i/n]
    SCMG_J=34 python scripts/ex002/bench_ex002.py eval --out res.json label=<dir> ..."""
import argparse, json, os, sys, time, zlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'
X = B + '/ex002-causal-1107'


def rec_inputs(view, c, CF, wdir):
    """recording PCM (int16), float wave and recording-relative words (items in offset order), item word offsets."""
    its = sorted(c['items'], key=lambda it: it['offset']); pcms, words, w0 = [], [], {}; t0 = 0
    for it in its:
        ma = it['manifest_audio'] if isinstance(it['manifest_audio'], dict) else eval(it['manifest_audio'])
        p, _ = CF.clip_audio(ma['path']); pcms.append(CF.pad_pcm(p, it['frames']))
        w0[it['id']] = len(words)                    # prep_causal_ex002.py words = the item's alignment.json units
        for w in CF.clip_words(Path(wdir) / f"{it['id']}.json", it['frames']):
            words.append(dict(w, start_time=w['start_time'] + t0 / 50., end_time=w['end_time'] + t0 / 50.))
        t0 += it['frames']
    pcm = np.concatenate(pcms)
    return pcm, pcm.astype(np.float32) / 32768., words, w0


def rec_tword(view, c, s, words, w0, tokz, CF, wdir):
    """tword [K, 2] of the window sample s (whole recording): the rows in sample()'s order (c['items'] list order, kn rows per
    item), word indices = the item's alignment units located by token_word_span, shifted to the recording word list."""
    K = len(s['tmask']); tw = np.full((K, 2), -1, np.int64); rows, tts = [], []
    for it in c['items']:
        if it.get('text_null'): continue
        e = view.text['index'].get(it['id'])
        if not e: continue
        k0, kn = e; i0 = it['offset'] - c['start']
        iw = CF.clip_words(Path(wdir) / f"{it['id']}.json", it['frames'])
        sp = CF.token_word_span(it['text'], iw, it['lang'], tokz, kn) if iw else np.full((kn, 2), -1, np.int64)
        sp = np.where(sp >= 0, sp + w0[it['id']], -1)
        rows.append(sp); tts.append(view.text['time'][k0:k0 + kn] + i0 / 50.)
    if not rows: return tw
    sp, tt = np.concatenate(rows), np.concatenate(tts)
    if len(sp) > K:                                   # sample()'s truncation: the K rows closest to the window centre
        T = c['frames'] // 16 * 16
        order = np.argsort(np.nan_to_num(np.abs(tt - T / 100.), nan=1e9))[:K]; order.sort(); sp = sp[order]
    tw[:len(sp)] = sp
    if int(np.asarray(s['tmask']).sum()) != len(sp): raise ValueError(f"{c['clip']}: {len(sp)} token rows != tmask {int(np.asarray(s['tmask']).sum())}")
    return tw


def infer(a):
    from scmg.common import CHUNK_FRAMES, U80
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg.infer import generate
    from scmg.model import INF
    if a.protocol in ('engine', 'psync'): os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = json.loads(Path(a.stats).read_text()); m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, 'val', a.text_cache, st, mmap=True, tok_cache=a.tok_cache)
    qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
    if a.protocol == 'psync':
        from scmg import causal_feat as CF
        tokz = CF.e5_tokenizer(); enc = CF.speaker_encoder(a.campplus); pst = CF.load_pstats(a.pstats)
    by = {}
    if a.plan:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from ex002_ctl_run import tokens_for; from scmg.ctl import token_value
        for e in json.loads(Path(a.plan).read_text())['events']: by.setdefault(e['ci'], []).append(e)
    si, sn = (int(x) for x in a.shard.split('/')); t0 = time.time(); log = []
    for ci in range(len(view.clips))[si::sn]:
        o = out / f'{ci}.npy'
        if o.exists(): continue
        c = view.clips[ci]; L = c['frames']; nch = L // CHUNK_FRAMES
        if nch < 3: continue
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
        seed = (a.seed + zlib.crc32(c['clip'].encode())) % (2 ** 63) if a.seed_mode == 'per_clip' else a.seed
        if a.plan and a.mode != 'none' and by.get(ci):       # 11-08 (default off): commanded-event tokens of an ex002_ctl_run plan
            toks = tokens_for(by[ci], a.mode); tv = [token_value(t) for t in toks]
            cond.update(ctl_t=np.array([t['t'] for t in toks], np.float32), ctl_type=np.array([x[0] for x in tv], np.int64),
                        ctl_v=np.stack([x[1] for x in tv]), ctl_vm=np.stack([x[2] for x in tv]))
        kw = dict(H=INF)
        if a.protocol == 'engine': kw = dict(H=2 * U80, text_mode='x2', x2_rate=15., x2_lag=0.15, deploy_timing=True)
        if a.protocol == 'psync':
            pcm, wave, words, w0 = rec_inputs(view, c, CF, a.words_dir)
            cond.update(wtime=CF.word_times(words).astype(np.float64), tword=rec_tword(view, c, s, words, w0, tokz, CF, a.words_dir))
            ci_ = CF.CausalInputs(pcm, wave, words, c['items'][0]['lang'], len(s['feat']), pst, view.fm, view.fs, enc, Tpad=len(s['feat']),   # the generated nch * 16 frames
                                  spk_refresh=a.spk_refresh, feat_mode=a.causal_feat)
            kw = dict(H=2 * U80, text_mode='voiced', x2_rate=15., x2_lag=0.15, deploy_timing=True, causal_inputs=ci_)
        if a.ctl_lookahead > 0 and 'ctl_type' in cond: kw['ctl_lookahead'] = a.ctl_lookahead
        cmd, _, tk = generate(m, vae, s['feat'], cond, st, nch, steps=a.steps, seed=seed, standby=qm, ctx_t=m.ctx_t, cfg_w=a.cfg_w, **kw)
        y = cmd[:L].astype(np.float32)
        np.save(o, y)
        g = out.parent / f'gt_{ci}.npy'
        if not g.exists(): np.save(g, (s['q'] * qs + qm)[:L].astype(np.float32))
        log.append(dict(ci=ci, clip=c['clip'], frames=L, seed=int(seed), tick_ms_p50=float(np.median(tk[1:]) * 1e3) if len(tk) > 1 else None))
        if ci % 8 == 0: print(f'{ci}/{len(view.clips)} {c["clip"]} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for r in log: f.write(json.dumps(r) + '\n')
    (out / 'PROTOCOL.json').write_text(json.dumps(dict(protocol=a.protocol, seed=a.seed, seed_mode=a.seed_mode, steps=a.steps, cfg_w=a.cfg_w,
                                                       ckpt=a.ckpt, vae=a.vae, view=a.view, spk_refresh=a.spk_refresh, causal_feat=a.causal_feat), indent=1))
    print('EX002_BENCH_INFER_DONE', a.protocol, a.seed, f'{time.time() - t0:.0f}s')


def evaluate(a):
    from scmg.evaluate import kin_feats, frechet
    sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code')
    from model import Embedder
    from config import WEIGHTS
    emb = [Embedder(WEIGHTS / f'human_seed{k}' / 'best.pt') for k in (0, 1)]       # as score_dirs.py / ex002_ctl_run (frozen FGD v2)
    res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); d = Path(d); gt_dir = d.parent
        Fg, Ft, gens, gts = [], [], [], []
        for f in sorted(d.glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); gt = np.load(gt_dir / f'gt_{ci}.npy'); n = min(len(y), len(gt)); y, gt = y[:n], gt[:n]
            if a.gt_as_gen: y = gt
            Fg += kin_feats(y); Ft += kin_feats(gt); gens.append(y); gts.append(gt)
        r = dict(n_clips=len(gens), frames=int(sum(len(g) for g in gens)), FKD=float(frechet(Fg, Ft)) if len(Fg) > 110 else None)
        r['FGD_v2_seeds'] = [float(frechet(np.concatenate([e_.features(g) for g in gts]), np.concatenate([e_.features(y) for y in gens]))) for e_ in emb]
        r['FGD_v2'] = float(np.mean(r['FGD_v2_seeds']))
        res[lab] = r; print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--ckpt', required=True); p.add_argument('--vae', default=B + '/trained/vae-r3/final.pt'); p.add_argument('--out', required=True)
    p.add_argument('--view', default=C + '/train-v1-20261101/views/data-v36_1'); p.add_argument('--stats', default=C + '/scmg/STATS_v36_1.json')
    p.add_argument('--text-cache', default=C + '/scmg/text-e5-v36_1'); p.add_argument('--tok-cache', default=C + '/scmg/tok-qwen25-v36_1')
    p.add_argument('--text-causal', default=B + '/models/text-e5-causal-v36_1')
    p.add_argument('--protocol', choices=['offline', 'engine', 'psync'], default='offline')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--seed-mode', choices=['run', 'per_clip'], default='per_clip')
    p.add_argument('--steps', type=int, default=10); p.add_argument('--cfg-w', type=float, default=2.0); p.add_argument('--shard', default='0/1')
    p.add_argument('--pstats', default=X + '/PROSODY_STATS_ex002_v36_1_train.json'); p.add_argument('--campplus', default=B + '/causal-train-1105/campplus')
    p.add_argument('--words-dir', default=X + '/words', help='prep_causal_ex002.py words (alignment.json units per item)')
    p.add_argument('--spk-refresh', default='grades', choices=['tick', 'grades']); p.add_argument('--causal-feat', default='inc', choices=['prefix', 'inc'])
    p.add_argument('--plan', default=None, help='11-08 (default off): ex002_ctl_run plan; with --mode given / shuffle its commands become control tokens')
    p.add_argument('--mode', default='none', choices=['none', 'given', 'shuffle'])
    p.add_argument('--ctl-lookahead', type=float, default=0., help='streaming: commands visible this many s before their time (0 = generate default)')
    p = sub.add_parser('eval'); p.add_argument('--out', required=True); p.add_argument('--gt-as-gen', action='store_true', help='sanity: GT scored as the generation (FGD v2 = 0)')
    p.add_argument('specs', nargs='+')
    a = ap.parse_args(); dict(infer=infer, eval=evaluate)[a.cmd](a)


if __name__ == '__main__':
    main()
