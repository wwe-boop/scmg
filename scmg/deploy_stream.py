"""Online deployment worker for the SCMG streaming candidate (speech_server, GPU1 only).

One persistent NDJSON process.  stdin:
  {"op":"speak","uid":..,"text":..[,"sr":24000|16000,"pcm":"f32"|"s16","est_timing":false,"spc":0.21]}
  {"op":"audio","uid":..,"start":..,"end":..,"pcm":<base64 f32le 24 kHz, or s16le at "sr" when speak said so>}
  {"op":"text","uid":..,"text":..}            late / replaced reply text (chain mode: tts_text snapshots replace)
  {"op":"anchor","uid":..,"anchor":{native_cursor_v3 fields}}
  {"op":"audio_done","uid":..,"samples":..}   {"op":"cancel","uid":..}   {"op":"ping"}
stdout:
  {"type":"ready"|"pong"|"meta"|"frames"|"done"|"error", ...}

Deployment rules (2026-11-04):
  * every 320 ms model tick k reads only the PCM prefix that has actually arrived (16k + H*STRIDE frames, H=2 latents
    = 160 ms look-ahead); the 89-d features are recomputed from that prefix with the pinned feat89 extractor and only
    units (zh chars / en words) whose offset is complete;
  * TTS is 24 kHz and is decimated to 16 kHz before feat89;
  * text is the whole sentence early, but token features are prefix-causal e5 (prep_text --causal) and visibility uses
    the legacy x2 commitment rule (15 tok/s, lag 0.15 s) plus deploy_timing (a token is timed only once voiced);
  * candidate = sbag-seed1-final.pt, CFG 2.5, 10 steps, H=2, trim_start + cfg_null_fix, as scripts/baseline_eval.sh.

Chain mode (2026-11-05, robot's own TTS through the speech-action protocol, gateway/chain_scmg.py):
  * checkpoint from $SCMG_DEPLOY_CKPT (default landing-l1-1104/sibag-coll-s1/final.pt, the 11-05 EX002 deploy pick);
  * audio is the robot TTS 16 kHz S16LE as published on /speech/tts_audio_stream (no 24 kHz round trip);
  * there are no TTS text_progress anchors: with est_timing the character times are estimated on every tick from
    the ARRIVED prefix only -- non-punctuation characters are laid on the voiced 20 ms frames at `spc` s/char (the
    BJ50 chain constant 0.21), re-scaled to the whole voiced time once the audio is complete; punctuation stays
    untimed (x2 cursor arrival).  Without any anchor every token is untimed and arrives on the x2 cursor;
  * empty text at speak (text not there yet): the Streamer is built on the first due tick with the text known by
    then, else the period text '。' (as the BJ50 chain); {"op":"text"} replaces the text features in place.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CPFS = Path('/mnt/cpfs/zbl-cpfs-new')
CACHE = Path(os.environ.get('SCMG_DEPLOY_CACHE', '/tmp/scmg-deploy-cache'))


def _cached(rel, fallback):
    p = CACHE / rel
    return p if p.exists() else fallback

B = CPFS / 'CKPT/zehan/gesture-scmg'
D = CPFS / 'dataset/zehan/gesture/scmg'
VIEW = CPFS / 'dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
_CK = Path(os.environ.get('SCMG_DEPLOY_CKPT', str(B / 'landing-l1-1104/sibag-coll-s1/final.pt')))
CKPT = _cached('ckpt/' + os.environ.get('SCMG_DEPLOY_CKPT_CACHE_NAME', _CK.parent.name + '-' + _CK.name), _CK)
VAE = _cached('ckpt/vae-r3-final.pt', B / 'trained/vae-r3/final.pt')
STATS = D / 'STATS_v36_1.json'
TEXT_MODEL = _cached('models/multilingual-e5-base', CPFS / 'CKPT/zehan/models/scmg/multilingual-e5-base')
FEAT89 = CPFS / 'USERS/zehan/unified-train-A-20261025'
SR = 24000
SR16 = 16000
FRAME24 = 480           # 24 kHz samples per 50 Hz frame
FRAME16 = 320           # 16 kHz samples per 50 Hz frame
CHUNK_FRAMES = 16       # 320 ms motion chunk
H_FRAMES = 8            # H=2 latents = 160 ms audio look-ahead
FPS = 50
NAMES34 = [
    'waist_yaw_joint', 'waist_pitch_joint', 'head_yaw_joint', 'head_pitch_joint',
    'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 'left_arm_joint4', 'left_arm_joint5',
    'left_arm_joint6', 'left_arm_joint7',
    'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5',
    'right_arm_joint6', 'right_arm_joint7',
    'left_1_1_joint', 'left_1_2_joint', 'left_1_3_joint', 'left_2_1_joint', 'left_2_2_joint',
    'left_2_3_joint', 'left_3_1_joint', 'left_3_2_joint',
    'right_1_1_joint', 'right_1_2_joint', 'right_1_3_joint', 'right_2_1_joint', 'right_2_2_joint',
    'right_2_3_joint', 'right_3_1_joint', 'right_3_2_joint',
]
SOURCE_ID = 5           # semsynth_r7p1 (synthetic speech), closest training source to TTS
PUNCT = set('。！？!?；;…，,、：:“”"\'‘’（）()《》<>【】[]—-·~～ \t\n')
FALLBACK_TEXT = '。'


def emit(row):
    row['rt'] = time.monotonic()
    sys.stdout.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    sys.stdout.flush()


def _np(x):
    return np.asarray(x, np.float32)


class E5:
    """multilingual-e5-base with prefix-causal token features (same tokenisation/selection as prep_text.py)."""

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(TEXT_MODEL), local_files_only=True)
        self.m = AutoModel.from_pretrained(str(TEXT_MODEL), torch_dtype=torch.float16,
                                           local_files_only=True).cuda().eval()
        self.cls = self.tok.cls_token_id if self.tok.cls_token_id is not None else self.tok.bos_token_id
        self.sep = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.tok.eos_token_id

    def encode(self, text):
        """(features [K,768] float32, offsets [(char_start,char_end)], ids list). One forward per token prefix."""
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(e['input_ids'])
        offs = [tuple(x) for x in e['offset_mapping']]
        if not ids:
            return np.zeros((0, 768), np.float32), offs, ids
        # all prefixes in ONE right-padded batch: row j = [CLS] ids[:j+1] [SEP] PAD..., so position j+1 is
        # identical to the unpadded prefix (causal attention; padding sits after the query).
        L = len(ids)
        x = self.torch.full((L, L + 2), self.tok.pad_token_id, dtype=self.torch.long, device='cuda')
        mk = self.torch.zeros((L, L + 2), dtype=self.torch.long, device='cuda')
        for j in range(L):
            x[j, :j + 3] = self.torch.tensor([self.cls] + ids[:j + 1] + [self.sep], device='cuda')
            mk[j, :j + 3] = 1
        with self.torch.inference_mode():
            h = self.m(input_ids=x, attention_mask=mk).last_hidden_state
            out = h[self.torch.arange(L, device='cuda'), self.torch.arange(1, L + 1, device='cuda')]
        return out.float().cpu().numpy(), offs, ids


class Model:
    def __init__(self):
        import torch
        from scmg.evaluate import load
        self.torch = torch
        self.stats = json.loads(STATS.read_text())
        self.qm = _np(self.stats['q_mean'])
        self.fm = _np(self.stats['f_mean'])
        self.fs = _np(self.stats['f_std'])
        t0 = time.time()
        self.m, self.vae = load(str(CKPT), str(VAE), 'cuda')
        self.load_s = time.time() - t0
        self.sha = self._sha(CKPT)
        self.cfg = dict(ckpt=str(CKPT), vae=str(VAE), stats=str(STATS), cfg_w=2.5, steps=10, H=2,
                        trim_start=True, cfg_null_fix=True, text_mode='x2', x2_rate=15.0, x2_lag=0.15,
                        source=SOURCE_ID, speaker='null', ckpt_sha256=self.sha)

    @staticmethod
    def _sha(p):
        import hashlib
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        return h.hexdigest()

    def warm(self):
        """One throw-away tick so the first real request does not pay CUDA/kernel setup."""
        import torch
        from scmg.infer import Streamer
        from scmg.common import U80
        T = 32
        cond = dict(text_c=torch.zeros(4, 768), ttime=torch.full((4,), float('nan')),
                    tmask=torch.ones(4, dtype=torch.bool), spk=torch.zeros(192), spk_valid=torch.tensor(0.),
                    source=torch.tensor(SOURCE_ID), ref=torch.zeros(128, 34), ref_valid=torch.tensor(0.))
        s = Streamer(self.m, self.vae, torch.zeros(T, 89), cond, self.stats, H=2 * U80, steps=10,
                     standby=self.qm, ctx_t=self.m.ctx_t, cfg_w=2.5, text_mode='x2', x2_rate=15., x2_lag=.15,
                     deploy_timing=True, trim_start=True, cfg_null_fix=True)
        s.tick()
        torch.cuda.synchronize()


class Utt:
    def __init__(self, uid, text, model, e5, sr=SR, pcm_fmt='f32', est_timing=False, spc=.21):
        self.uid, self.text, self.model, self.t0 = uid, text, model, time.monotonic()
        if (sr, pcm_fmt) not in ((SR, 'f32'), (SR16, 's16')):
            raise ValueError('audio must be 24 kHz f32 or 16 kHz s16')
        self.sr, self.pcm_fmt, self.est_timing, self.spc = int(sr), pcm_fmt, bool(est_timing), float(spc)
        self.frame = self.sr // FPS
        self.e5 = e5
        self.s = None
        self.pcm = []                      # float32 chunks at self.sr
        self.samples = 0
        self.done = False
        self.cancel = False
        self.anchors = {}
        self.tick_k = 0
        self.frames_sent = 0
        self.tick_ms = []
        self.first_frame_ms = None
        self.done_sent = False
        self.text_events = []
        emit(dict(uid=uid, type='meta', names=NAMES34, fps=FPS,
                  model=dict(sha256=model.sha, arch='scmg-landing-l1-sibag-coll', step=None), lead_s=H_FRAMES / FPS,
                  warmup_ticks=0, cfg=dict(model.cfg, chain=dict(sr=self.sr, pcm=pcm_fmt, est_timing=self.est_timing, spc=self.spc))))
        if self.text:
            self._build()

    def _text_cond(self, text):
        import torch
        feats, offs, ids = self.e5.encode(text)
        if not ids:
            raise ValueError('text has no tokens')
        return torch.as_tensor(feats, device='cuda').float(), offs, len(ids)

    def set_text(self, text):
        """Replace the reply text (newer snapshot).  Before the first tick it is simply the text to build with."""
        import torch
        text = str(text or '').strip()[:400]
        if not text or text == self.text:
            return
        self.text_events.append(dict(t_ms=round(1000 * (time.monotonic() - self.t0), 1), ticks=self.tick_k, n=len(text)))
        if self.s is None:
            self.text = text; return
        tc, offs, n = self._text_cond(text)
        self.text, self.offs, self.n_tok = text, offs, n
        self.ttime = np.full(n, np.nan, np.float32)
        self.s.cond['text_c'] = tc
        self.s.cond['tmask'] = torch.ones(n, dtype=torch.bool, device='cuda')
        self.anchors = {}
        self._refresh_text_time()

    def _build(self):
        import torch
        from scmg.infer import Streamer
        from scmg.common import U80
        model = self.model
        if not self.text:
            self.text = FALLBACK_TEXT
        # text: whole sentence known now; features are prefix-causal, visibility is the x2 arrival mask
        text_c, offs, n_tok = self._text_cond(self.text)
        self.offs = offs
        self.n_tok = n_tok
        self.ttime = np.full(self.n_tok, np.nan, np.float32)
        cond = dict(text_c=text_c,
                    ttime=torch.as_tensor(self.ttime, device='cuda').float(),
                    tmask=torch.ones(self.n_tok, dtype=torch.bool, device='cuda'),
                    spk=torch.zeros(192, device='cuda'), spk_valid=torch.tensor(0., device='cuda'),
                    source=torch.tensor(SOURCE_ID, device='cuda'),
                    ref=torch.zeros(128, 34, device='cuda'), ref_valid=torch.tensor(0., device='cuda'))
        self.s = Streamer(model.m, model.vae, torch.zeros(1, 89, device='cuda'), cond, model.stats,
                          H=2 * U80, steps=10, standby=model.qm, ctx_t=model.m.ctx_t, cfg_w=2.5,
                          text_mode='x2', x2_rate=15., x2_lag=.15, deploy_timing=True,
                          trim_start=True, cfg_null_fix=True)

    def add_audio(self, raw, start, end):
        b = base64.b64decode(raw)
        x = (np.frombuffer(b, '<f4').astype(np.float32) if self.pcm_fmt == 'f32'
             else np.frombuffer(b, '<i2').astype(np.float32) / 32768.)
        if start != self.samples:
            emit(dict(uid=self.uid, type='error', stage='audio', error=f'audio discontinuity {start} != {self.samples}'))
            self.cancel = True
            return
        self.pcm.append(x)
        self.samples += len(x)

    def add_anchor(self, a):
        try:
            rs, re = int(a['raw_codepoint_start']), int(a['raw_codepoint_end'])
            s0, s1 = int(a['output_sample_start']), int(a['output_sample_end'])
            rate = float(a.get('output_sample_rate') or SR)
        except Exception:
            return
        if re <= rs or s1 <= s0:
            return
        n = re - rs
        for i, c in enumerate(range(rs, re)):
            t0 = s0 / rate + (s1 - s0) / rate * (i / n)
            t1 = s0 / rate + (s1 - s0) / rate * ((i + 1) / n)
            self.anchors[c] = (t0, t1)

    def _estimate_anchors(self, T):
        """Chain mode: character times from the arrived prefix only (see module doc).  Rebuilt on every tick."""
        n = min(self.samples, T * self.frame)
        self.anchors = {}
        idx = [i for i, ch in enumerate(self.text) if ch not in PUNCT]
        if not idx or n < self.frame:
            return
        x = np.concatenate(self.pcm)[:n // self.frame * self.frame].reshape(-1, self.frame)
        rms = np.sqrt((x.astype(np.float64) ** 2).mean(1)) + 1e-9
        voiced = rms >= np.percentile(rms, 95) * 10 ** (-35. / 20)
        cum = np.r_[0., np.cumsum(voiced) * (1. / FPS)]                # voiced seconds before frame f
        total = cum[-1]
        if total <= 0:
            return
        spc = total / len(idx) if self.done else self.spc
        if self.done and self.samples > n:                              # not all frames visible yet: keep the constant
            spc = self.spc
        vf = np.flatnonzero(voiced)
        def wall(v):                                                    # voiced time -> wall time (s), v < total
            f = vf[min(int(v * FPS), len(vf) - 1)]
            return f / FPS + (v * FPS - int(v * FPS)) / FPS
        for j, c in enumerate(idx):
            v0, v1 = j * spc, (j + 1) * spc
            if v1 > total + 1e-6:
                break
            self.anchors[c] = (wall(v0), wall(min(v1, total) - 1e-6) + 1e-6)

    def _refresh_text_time(self):
        import torch
        for j, (cs, ce) in enumerate(self.offs):
            ts = [self.anchors[c][0] for c in range(cs, ce) if c in self.anchors]
            self.ttime[j] = min(ts) if ts else np.nan
        from scmg.model import x2_arrival
        tt = torch.as_tensor(self.ttime, device='cuda').float()
        self.s.cond['ttime'] = tt
        self.s.arr_text = x2_arrival(tt[None], self.s.cond['tmask'][None].bool(),
                                     torch.tensor([15.], device='cuda'), torch.tensor([.15], device='cuda'))[0]

    def _chars(self, T):
        """Only units complete by T frames (same rule as scripts/feat_causality.py)."""
        dur = T / FPS - 1e-3
        out = []
        for i, (cs, ce) in enumerate(self.offs):
            if cs not in self.anchors or ce - 1 not in self.anchors:
                continue
            on = min(self.anchors[c][0] for c in range(cs, ce) if c in self.anchors)
            off = max(self.anchors[c][1] for c in range(cs, ce) if c in self.anchors)
            if off <= T / FPS and on < dur:
                out.append(dict(index=int(cs), char=self.text[cs:ce], onset_s=float(on), offset_s=float(min(off, dur))))
        return out

    def _feat(self, T):
        import feat89 as F
        nsr = T * self.frame
        xs = np.concatenate(self.pcm)[:nsr] if self.pcm else np.zeros(0, np.float32)
        if len(xs) < nsr:
            xs = np.concatenate([xs, np.zeros(nsr - len(xs), np.float32)])
        if len(xs) == 0:
            return np.zeros((0, 89), np.float32)
        if self.sr == SR16:
            x16 = xs                                                   # robot TTS is already 16 kHz
        else:
            if nsr % 3:
                raise ValueError('prefix is not a multiple of 3 24 kHz samples')
            x16 = _resample_2_3(xs)
        x16 = np.clip(np.round(x16 * 32768.), -32768, 32767).astype(np.int16)
        chars = self._chars(T)
        try:
            rec = F.Recording.from_clip(x16, SR16, chars, self.text, 'zh', T)
            f = np.asarray(rec.segment(0, T)['feat'], np.float32)
        except (ValueError, IndexError):
            # no complete unit yet: mel only, prosody zero (the same fallback as scripts/feat_causality.py)
            from scipy.signal import resample_poly
            p24 = resample_poly(x16.astype(np.float32) / 32768., 3, 2)
            f = np.concatenate([F.gl_build.logmel(p24, T, F.mel()), np.zeros((T, 9), np.float32)], 1)
        return f.astype(np.float32)

    def _tick(self):
        import torch
        if self.cancel or self.done_sent:
            return False
        k = self.tick_k
        arrived = self.samples // self.frame
        if self.done:
            need = CHUNK_FRAMES * k < arrived
        else:
            need = CHUNK_FRAMES * k + H_FRAMES <= arrived
        if not need:
            return False
        T = min(arrived, CHUNK_FRAMES * k + H_FRAMES) if not self.done else arrived
        T = max(1, T)
        if self.s is None:
            self._build()
        if self.est_timing:
            self._estimate_anchors(T)
        f = self._feat(T)
        self._refresh_text_time()
        self.s.feat = torch.as_tensor((f - self.model.fm) / self.model.fs, device='cuda').float()
        t0 = time.perf_counter()
        fr = self.s.tick()
        dt = (time.perf_counter() - t0) * 1000
        self.tick_ms.append(dt)
        self._send_frames(fr, input_samples=self.samples, input_done=self.done)
        self.tick_k += 1
        return True

    def _send_frames(self, fr, input_samples, input_done):
        if fr is None or len(fr) == 0:
            return
        q = np.asarray(fr.detach().cpu().numpy(), np.float32)
        if not np.all(np.isfinite(q)):
            emit(dict(uid=self.uid, type='error', stage='scmg', error='non-finite frame'))
            self.cancel = True
            return
        start = self.frames_sent
        self.frames_sent += len(q)
        if self.first_frame_ms is None:
            self.first_frame_ms = 1000 * (time.monotonic() - self.t0)
        emit(dict(uid=self.uid, type='frames', start=start, q=q.tolist(), input_samples=input_samples,
                  input_done=bool(input_done)))

    def finish(self):
        if self.done_sent or self.cancel:
            return
        self.done = True
        # remaining chunks (including a partial last chunk)
        while CHUNK_FRAMES * self.tick_k < self.samples // self.frame:
            self._tick()
        if self.tick_k == 0:
            self.done_sent = True
            emit(dict(uid=self.uid, type='done', frames=0, ticks=0, tick_ms_p50=None, tick_ms_p95=None,
                      first_frame_ms=None, audio_s=self.samples / self.sr, text_tokens=int(getattr(self, 'n_tok', 0)), text=self.text))
            return
        if self.s is None:
            self._build()
        fr = self.s.flush()
        self._send_frames(fr, input_samples=self.samples, input_done=True)
        self.done_sent = True
        tk = np.asarray(self.tick_ms, np.float32)
        emit(dict(uid=self.uid, type='done', frames=self.frames_sent, ticks=int(len(tk)),
                  tick_ms_p50=float(np.percentile(tk, 50)) if len(tk) else None,
                  tick_ms_p95=float(np.percentile(tk, 95)) if len(tk) else None,
                  first_frame_ms=self.first_frame_ms, audio_s=self.samples / self.sr,
                  text_tokens=int(self.n_tok), text=self.text, text_events=self.text_events,
                  anchors=len(self.anchors), est_timing=self.est_timing))


def _resample_2_3(x24):
    """Exact-length 24 kHz -> 16 kHz polyphase decimation (3/2)."""
    from scipy.signal import resample_poly
    n = len(x24)
    if n == 0:
        return np.zeros(0, np.float32)
    y = resample_poly(x24, 2, 3)
    n16 = n * 2 // 3
    if len(y) < n16:
        y = np.concatenate([y, np.zeros(n16 - len(y), np.float32)])
    return y[:n16].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('SCMG_TORCH_THREADS', '4')))
    sys.path.insert(0, str(FEAT89))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    stats = json.loads(STATS.read_text())
    t0 = time.time()
    e5 = E5()
    model = Model()
    model.warm()
    torch.cuda.synchronize()
    emit(dict(type='ready', pid=os.getpid(), load_s=time.time() - t0, model_load_s=model.load_s,
              affinity=sorted(os.sched_getaffinity(0)), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
              device=torch.cuda.get_device_name(0), mem_gib=torch.cuda.memory_allocated() / 2 ** 30,
              sha256=model.sha, names=NAMES34, fps=FPS))

    utts = {}
    for line in sys.stdin:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        op, uid = row.get('op'), row.get('uid')
        try:
            if op == 'ping':
                emit(dict(type='pong', active=len(utts), pid=os.getpid())); continue
            if op == 'speak':
                text = str(row.get('text') or '').strip()
                if len(text) > 400:
                    text = text[:400]
                if not text and not row.get('allow_empty'):
                    raise ValueError('text length')
                if uid in utts:
                    utts[uid].cancel = True
                utts[uid] = Utt(uid, text, model, e5, sr=int(row.get('sr') or SR), pcm_fmt=str(row.get('pcm') or 'f32'),
                                est_timing=bool(row.get('est_timing')), spc=float(row.get('spc') or .21))
                emit(dict(uid=uid, type='accepted', t=time.monotonic()))
                continue
            u = utts.get(uid)
            if u is None:
                continue
            if op == 'audio':
                u.add_audio(row['pcm'], int(row['start']), int(row['end']))
                for _ in range(int(os.environ.get('SCMG_TICKS_PER_AUDIO', '3'))):   # catch up after bursts
                    if not u._tick():
                        break
            elif op == 'text':
                u.set_text(row.get('text'))
            elif op == 'anchor':
                u.add_anchor(row['anchor'])
                u._tick()
            elif op == 'audio_done':
                u.finish()
                utts.pop(uid, None)
            elif op == 'cancel':
                u.cancel = True
                utts.pop(uid, None)
        except Exception as e:
            emit(dict(uid=uid, type='error', stage='scmg', error=f'{type(e).__name__}: {str(e)[:240]}',
                      trace=traceback.format_exc()[-800:]))
    for u in utts.values():
        u.cancel = True


if __name__ == '__main__':
    try:
        main()
    except Exception:
        emit(dict(type='fatal', error=traceback.format_exc()[-1200:]))
        raise
