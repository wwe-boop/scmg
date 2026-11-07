"""11-07 EX002 interruption benchmark (3b; criteria docs/criteria/EX002_INTERRUPT_1107.md): dev interruptions + the in-stream
continuation of a model under the causal psync protocol with the deployment-faithful pause (audio after t_i = digital silence).
  plan  : the 43 dev recordings of data-v36_1; every utterance (item) whose speech span lasts >= 3 s gets interruptions at 30 / 50 /
          70 % of its speech span, kept when t_i + 2.2 s lies inside the generated part of the recording (L // 16 chunks)
          -> {interrupts: [{ci, clip, item, frac, t_i (recording s), f_i (50 Hz frame)}]}
  infer : per interruption the bench_ex002.py psync stream of the whole recording (same window sample, same CausalInputs,
          per_clip seed = seed + crc32(recording)) as infer_retract.RetractStreamer: chunk restart, audio silence after t_i
          (PostInterrupt: PCM zero from t_i, words ended by t_i, speaker of the speech heard), tokens whose words ended by t_i
          kept; frames up to f_i + 110 -> <out>/<ci>@<item>@<frac>.npy (50 Hz, 34 joints, rad). The frames before the
          interruption equal the uninterrupted bench psync output bitwise (same code path).
    SCMG_J=34 python scripts/ex002/ex002_interrupt.py plan --out I.json
    SCMG_J=34 python scripts/ex002/ex002_interrupt.py infer --ckpt C --interrupts I.json --out D [--shard i/n] [--yield-token]"""
import argparse, json, os, sys, time, zlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / 'interrupt'))
import bench_ex002 as BX
C, B, X = BX.C, BX.B, BX.X
TAIL = 110                                      # frames generated after f_i (2.2 s; the metric window is 107)


def plan(a):
    from scmg.common import CHUNK_FRAMES
    from scmg.data import View
    st = json.loads(Path(a.stats).read_text()); view = View(a.view, 'val', a.text_cache, st, mmap=True)
    rows = []
    for ci, c in enumerate(view.clips):
        Lg = c['frames'] // CHUNK_FRAMES * CHUNK_FRAMES
        for it in sorted(c['items'], key=lambda x: x['offset']):
            s0, s1 = (float(x) for x in it['speech']); i0 = (it['offset'] - c['start']) / 50.
            if s1 - s0 < 3.0: continue
            for fr in (0.3, 0.5, 0.7):
                t = i0 + s0 + fr * (s1 - s0); f = int(round(t * 50))
                if f + TAIL > Lg: continue
                rows.append(dict(ci=ci, clip=c['clip'], item=it['id'], frac=fr, t_i=f / 50., f_i=f))
    Path(a.out).write_text(json.dumps(dict(interrupts=rows, rule=__doc__.split('\n')[3].strip(), n_recordings=len({r['ci'] for r in rows}),
                                           n_items=len({r['item'] for r in rows}))))
    print(len(rows), 'interrupts', len({r['item'] for r in rows}), 'utterances', len({r['ci'] for r in rows}), 'recordings')


def infer(a):
    from scmg.common import CHUNK_FRAMES, CHUNK, U80
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg import causal_feat as CF
    from infer_retract import RetractStreamer
    os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = json.loads(Path(a.stats).read_text()); m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, 'val', a.text_cache, st, mmap=True, tok_cache=a.tok_cache)
    qm = np.asarray(st['q_mean'], np.float32)
    tokz = CF.e5_tokenizer(); enc = CF.speaker_encoder(a.campplus); pst = CF.load_pstats(a.pstats)
    rows = json.loads(Path(a.interrupts).read_text())['interrupts']
    si, sn = (int(x) for x in a.shard.split('/')); rows = rows[si::sn]; t0 = time.time(); log = []
    cache = {}
    for k, r in enumerate(rows):
        o = out / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
        if o.exists(): continue
        ci = r['ci']; c = view.clips[ci]; L = c['frames']; nch = L // CHUNK_FRAMES
        if ci not in cache:
            s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
            pcm, wave, words, w0 = BX.rec_inputs(view, c, CF, a.words_dir)
            tw = BX.rec_tword(view, c, s, words, w0, tokz, CF, a.words_dir)
            cache = {ci: (s, pcm, wave, words, tw)}
        s, pcm, wave, words, tw = cache[ci]
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        cond.update(wtime=CF.word_times(words).astype(np.float64), tword=tw)
        T = len(s['feat']); lang = c['items'][0]['lang']
        def mk(silence_from=None, T_=None, t_words=None):
            if silence_from is None:
                return CF.CausalInputs(pcm, wave, words, lang, T, pst, view.fm, view.fs, enc, Tpad=T, spk_refresh=a.spk_refresh, feat_mode=a.causal_feat)
            pz = np.array(pcm[:silence_from * CF.HOP], np.int16); pz = np.concatenate([pz, np.zeros(max(0, T_ * CF.HOP - len(pz)), np.int16)])
            wz = [w for w in words if CF.ended(w['end_time'], t_words)]
            return CF.CausalInputs(pz, wave, wz, lang, T_, pst, view.fm, view.fs, None, Tpad=T_, spk=False, feat_mode=a.causal_feat)
        t_i = r['t_i']; f_i = r['f_i']
        keep = CF.token_visible(CF.visible_words(cond['wtime'], t_i, 0., 'voiced', 15., 0.15, 0.5), tw) & np.asarray(s['tmask'], bool)
        n_run = -(-(f_i + TAIL) // CHUNK_FRAMES)
        seed = (a.seed + zlib.crc32(c['clip'].encode())) % (2 ** 63)
        stm = RetractStreamer(m, vae, s['feat'], cond, st, H=2 * U80, steps=a.steps, seed=seed, standby=qm, ctx_t=m.ctx_t, cfg_w=a.cfg_w,
                              text_mode='voiced', x2_rate=15., x2_lag=0.15, deploy_timing=True, causal_inputs=mk(), asr_lag=0.5,
                              t_int=t_i, keep_tok=keep, audio='silence', fm=view.fm, fs=view.fs, n_total=n_run * CHUNK_FRAMES,
                              restart='chunk', fresh_feat=lambda silence_from=None, T=None, t_words=None: mk(silence_from, T, t_words),
                              yield_token=a.yield_token)
        while (stm.o + stm.k * CHUNK) < n_run * CHUNK: stm.tick(None, None, None)
        stm.flush()
        y = torch.stack(stm.all_frames()).cpu().numpy().astype(np.float32)[:f_i + TAIL]
        np.save(o, y); log.append(dict(r, k_int=stm.k_int, frames=len(y), seed=int(seed)))
        if k % 20 == 0: print(f'{k + 1}/{len(rows)} {o.name} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for x in log: f.write(json.dumps(x) + '\n')
    (out / 'PROTOCOL.json').write_text(json.dumps(dict(ckpt=a.ckpt, protocol='psync', audio_after_t_i='silence', restart='chunk', seed=a.seed,
                                                       seed_mode='per_clip', steps=a.steps, cfg_w=a.cfg_w, yield_token=a.yield_token), indent=1))
    print('EX002_INTERRUPT_DONE', len(rows), f'{time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for n_ in ('plan', 'infer'):
        p = sub.add_parser(n_); p.add_argument('--out', required=True)
        p.add_argument('--view', default=C + '/train-v1-20261101/views/data-v36_1'); p.add_argument('--stats', default=C + '/scmg/STATS_v36_1.json')
        p.add_argument('--text-cache', default=C + '/scmg/text-e5-v36_1')
    p = sub.choices['infer']
    p.add_argument('--ckpt', required=True); p.add_argument('--vae', default=B + '/trained/vae-r3/final.pt'); p.add_argument('--interrupts', required=True)
    p.add_argument('--tok-cache', default=C + '/scmg/tok-qwen25-v36_1'); p.add_argument('--text-causal', default=B + '/models/text-e5-causal-v36_1')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--steps', type=int, default=10); p.add_argument('--cfg-w', type=float, default=2.0)
    p.add_argument('--shard', default='0/1'); p.add_argument('--pstats', default=X + '/PROSODY_STATS_ex002_v36_1_train.json')
    p.add_argument('--campplus', default=B + '/causal-train-1105/campplus'); p.add_argument('--words-dir', default=X + '/words')
    p.add_argument('--spk-refresh', default='grades'); p.add_argument('--causal-feat', default='inc'); p.add_argument('--yield-token', action='store_true')
    a = ap.parse_args(); dict(plan=plan, infer=infer)[a.cmd](a)


if __name__ == '__main__':
    main()
