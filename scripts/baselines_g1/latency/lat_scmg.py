"""SCMG (ours) latency on the ECHO-G G1 benchmark settings: sf0/final.pt + decft-w3d0m05 VAE, 10 flow steps, CFG 2.5,
fp32, batch 1, release 20261105-203114-g1-semscale.
  offline: final text, H = inf (benchmark offline protocol): compute starts once the whole utterance has arrived.
  engine : EX002 engine protocol (x2 text 15 tok/s lag 0.15 s, prefix-causal e5, deploy timing, H = 2 x 80 ms) -- its
           model inputs are slices of the whole-clip precomputed feat89 / CAM++ (reference row only).
  psync  : the strictly causal streaming protocol (design v5 section 3, exp/causal-stream scmg/causal_feat.py, code snapshot
           with the worktree state in SCMG_REL): every tick recomputes feat89 from the arrived PCM prefix (CausalFeat89,
           the G1 view extractor + fixed training prosody statistics) and CAM++ of the prefix (SpeakerPrefix), a word is
           visible once it has ended within the arrived audio (text_mode voiced), H = 160 ms; these costs are inside the
           tick and are split out below (causal_feat89, causal_campplus).
Stages (cuda-synchronised):
  audio_feat  raw wav -> 24 kHz -> the deployed numpy 89-d extractor (bj50 PrefixFeatures: log-mel + prosody + speech
              flag, same definition as the feat89 training features); offline = one call on the whole clip, engine = one
              call per tick on the arrived prefix (as scripts/e2e_latency.py)
  text_enc    multilingual-e5-base fp16: offline = one bidirectional forward of the transcript; engine = one prefix-causal
              forward per token that has arrived by the tick (as prep_text --causal), batched per tick
  generate    Streamer.tick minus VAE decode (planner + local flow, CFG)
  decode      VAE decode inside the tick (+ the final flush)
  to30        50 -> 30 FPS resample of the whole clip (benchmark format only; not on the robot path, excluded from e2e)
The model inputs are the benchmark's cached conditions (identical values to the evaluated run); the extractor / e5 calls
measure the cost of producing them online.  CAM++ speaker vector: one per speaker (enrolment), excluded; its one-off cost
is reported separately (campplus_ms) when the CAM++ code is importable.
  SCMG_J=39 python lat_scmg.py --protocol offline|engine --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402

REL = Path(os.environ.get('SCMG_REL', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/20261105-203114-g1-semscale'))
sys.path.insert(0, str(REL)); sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/edge-bj50')
CAMPPLUS = os.environ.get('SCMG_CAMPPLUS', '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baselines-g1-1105/latency/snap/campplus')
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
CKPT = f'{B}/echog-g1-sf-1105/test/sf0/final.pt'; VAE = f'{B}/echog-g1-decft-1105/decft-w3d0m05/final.pt'
VIEW_TEST = f'{C}/echog-g1-v1/test'; STATS = f'{C}/scmg/STATS_echog_g1_v1.json'
TEXT_TEST = f'{C}/scmg/text-e5-echog-g1-v1-test'; TEXT_CAUSAL = f'{B}/models/text-e5-causal-echog-g1-v1-test'
E5 = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base'
DR = Path(f'{B}/baseline-repro-1104/echog/data_root'); WORDS = Path(f'{B}/baseline-repro-1104/gmr-repro/data/echog/annotations/words')


def anchors_from_words(words, fs=24000):
    text, an = '', []
    for w in words:
        s = str(w['text']).strip()
        if not s: continue
        if text: text += ' '
        c0 = len(text); text += s
        a, b = float(w['start']) * fs, float(w['end']) * fs; step = (b - a) / len(s)
        for j in range(len(s)):
            an.append(dict(raw_codepoint_start=c0 + j, raw_codepoint_end=c0 + j + 1, output_sample_start=int(a + j * step),
                           output_sample_end=int(a + (j + 1) * step)))
    return text, an


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', required=True, choices=['offline', 'engine', 'psync'])
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args(); eng = a.protocol in ('engine', 'psync'); ps = a.protocol == 'psync'
    mon = LC.GPUMon(); host0 = LC.host_snapshot()
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if eng: os.environ['SCMG_TEXT_CAUSAL'] = TEXT_CAUSAL
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer, AutoModel
    from scmg.common import CHUNK_FRAMES, U80
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg.infer import Streamer
    from scmg.model import INF
    from scmg import g1
    from bj50 import prosody as P
    from bj50.fe_numpy import PrefixFeatures
    taps = np.load('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/edge-bj50/bj50/rp_taps.npy')
    dev = 'cuda'; clk = LC.Sync()
    stats = json.loads(Path(STATS).read_text()); qm = np.asarray(stats['q_mean'], np.float32)
    m, vae = load(CKPT, VAE, dev)
    view = View(VIEW_TEST, 'test', TEXT_TEST, stats)
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    tok = AutoTokenizer.from_pretrained(E5); e5 = AutoModel.from_pretrained(E5, torch_dtype=torch.float16).to(dev).eval()
    sep = tok.sep_token_id
    dec_t = [0.0]; vdec = vae.decode
    def timed_decode(*x, **k):
        t0 = clk.now(); y = vdec(*x, **k); dec_t[0] += clk.now() - t0; return y
    vae.decode = timed_decode
    H = 2 * U80 if eng else INF; CH = CHUNK_FRAMES / 50.; H_s = 0.16 if eng else None
    if ps:
        from scmg import causal_feat as CF
        tokz = CF.e5_tokenizer(); enc_spk = CF.speaker_encoder(CAMPPLUS)
        pst = CF.load_pstats(REL / 'scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json')
        WAVD = Path(os.environ.get('CAUSAL_WAV', str(DR / 'audio')))
    stems = LC.stems50()[:a.n or None]; rows = []
    for k, stem in enumerate(stems):
        ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]
        L = c['frames']; n30 = int(it['echog']['frames30']); nch = -(-L // CHUNK_FRAMES)
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        words = json.load(open(WORDS / f'{stem}.json'))['words']
        x, sr = sf.read(str(DR / 'audio' / f'{stem}.wav'), dtype='float32')
        x = x if x.ndim == 1 else x.mean(1); dur = len(x) / sr
        kw = dict(text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True) if eng else {}
        ct = dict(causal_feat89=0., causal_campplus=0.)
        if ps:                                   # = infer_test.py --protocol psync (exp/causal-stream)
            kw['text_mode'] = 'voiced'
            cw = CF.clip_words(WORDS / f'{stem}.json', L)
            kn = view.text['index'].get(stem, [0, 0])[1] if not it.get('text_null') else 0
            if int(np.asarray(s['tmask']).sum()) != kn: raise ValueError(f'{stem}: tmask rows != cache rows')
            cond.update(CF.text_extras(it, cw, tokz, kn, len(s['tmask'])))
            pcm16, wave = CF.clip_audio(WAVD / f'{stem}.wav')
            ci_ = CF.CausalInputs(pcm16, wave, cw, it['lang'], L, pst, view.fm, view.fs, enc_spk, Tpad=len(s['feat']), spk=True)
            f_feat, f_spk = ci_.feat, ci_.spk
            def t_feat(Tk, _f=f_feat):
                t0 = clk.now(); r = _f(Tk); ct['causal_feat89'] += clk.now() - t0; return r
            def t_spk(Tk, _f=f_spk):
                t0 = clk.now(); r = _f(Tk); ct['causal_campplus'] += clk.now() - t0; return r
            ci_.feat, ci_.spk = t_feat, t_spk
            kw.update(causal_inputs=ci_, asr_lag=0.5)
            wt = np.asarray(cond['wtime'], np.float64); tw = np.asarray(cond['tword'])
            tok_end = np.array([wt[tw[j, 1], 1] if tw[j, 1] >= 0 else np.inf for j in range(len(tw))])
        S = Streamer(m, vae, s['feat'], cond, stats, H=H, steps=10, seed=0, standby=qm, ctx_t=m.ctx_t, cfg_w=2.5, cfg_t=None,
                     noise_scale=1.0, **kw)
        # ---- text tokens (e5 tokenisation of the transcript, as prep_text) and their arrival under x2
        text, anchors = anchors_from_words(words)
        enc = tok([text], return_tensors='pt', truncation=True, max_length=160)
        ids = enc['input_ids'][0].numpy(); sel = [j for j in range(len(ids)) if ids[j] not in tok.all_special_ids]
        if eng:
            arr = S.arr_text.float().cpu().numpy() if S.arr_text is not None else np.zeros(len(sel))
            arr = np.resize(arr, len(sel)) if len(arr) != len(sel) else arr     # cache rows == e5 tokens (same tokeniser)
            if ps:                                # voiced: x2-committed AND its last word has ended within now + H
                te_ = np.resize(tok_end, len(sel)) if len(tok_end) != len(sel) else tok_end
                arr = np.maximum(arr, te_ - H_s)
        st = dict(audio_feat=0., text_enc=0., generate=0., decode=0., to30=0.)
        if ps: st.update(causal_feat89=0., causal_campplus=0.)
        sched = []; ticks = []; pf = PrefixFeatures(P, taps); done_tok = 0
        t0 = clk.now(); p24 = resample_poly(x, 24000, sr).astype(np.float32); t_rs = clk.now() - t0
        for kk in range(nch):
            avail = dur if not eng else min(dur, kk * CH + H_s)
            ta = clk.now()
            if ps or (not eng and kk > 0):
                fa = 0.                           # psync: the causal feat89 / CAM++ run inside the tick (timed there)
            else:
                n = max(int(avail * 24000), 960)
                pf.at(text, p24[:n], [q for q in anchors if q['output_sample_start'] < n], n >= len(p24), len(p24))
                fa = clk.now() - ta + (t_rs if kk == 0 else 0.)
            tb = clk.now(); te = 0.
            if not eng and kk == 0:
                with torch.no_grad(): e5(**{q: v.to(dev) for q, v in enc.items()})
                te = clk.now() - tb
            elif eng:
                new = [j for j in range(done_tok, len(sel)) if arr[j] <= avail]
                if new:
                    Lp = max(sel[j] for j in new) + 2
                    xx = np.full((len(new), Lp), tok.pad_token_id, np.int64); mk = np.zeros_like(xx)
                    for r, j in enumerate(new):
                        p = sel[j]; xx[r, :p + 1] = ids[:p + 1]; xx[r, p + 1] = sep; mk[r, :p + 2] = 1
                    with torch.no_grad(): e5(input_ids=torch.as_tensor(xx).to(dev), attention_mask=torch.as_tensor(mk).to(dev))
                    done_tok = max(new) + 1
                te = clk.now() - tb
            dec_t[0] = 0.; ct['causal_feat89'] = ct['causal_campplus'] = 0.; tc = clk.now()
            fr = S.tick(arrived_s=avail)
            if kk == nch - 1: fr = torch.cat([fr, S.flush()], 0)
            tg = clk.now() - tc; dd = dec_t[0]; cf, cs = ct['causal_feat89'], ct['causal_campplus']
            st['audio_feat'] += fa; st['text_enc'] += te; st['generate'] += tg - dd - cf - cs; st['decode'] += dd
            if ps: st['causal_feat89'] += cf; st['causal_campplus'] += cs
            e0 = (S.n_emitted - len(fr)) / 50.
            sched.append((avail, fa + te + tg, e0, int(len(fr)), 50.))
            ticks.append(dict(t=kk * CH, len=CH, audio_feat=fa, text_enc=te, generate=tg - dd - cf - cs, decode=dd, causal_feat89=cf, causal_campplus=cs))
        cmd = torch.stack(S.frames).cpu().numpy()[:L].astype(np.float64)
        t0 = clk.now(); g1.to30(cmd, n30); st['to30'] = clk.now() - t0
        comp = st['audio_feat'] + st['text_enc'] + st['generate'] + st['decode'] + st.get('causal_feat89', 0.) + st.get('causal_campplus', 0.)
        rows.append(dict(stem=stem, dur=dur, frames50=L, n_ticks=nch, stages=st, compute=comp, schedule=sched, ticks=ticks))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} {st}', flush=True)
    extra = dict(protocol=a.protocol, ckpt=CKPT, vae=VAE, steps=10, cfg_w=2.5, dtype='fp32', release=str(REL),
                 frame_fps=50, note_to30='to30 is reported but excluded from compute / e2e (benchmark format only)',
                 host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__)
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    LC.write(a.out, f'scmg_{a.protocol}', rows, extra)


if __name__ == '__main__':
    main()
