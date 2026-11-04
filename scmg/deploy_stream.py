"""Online deployment worker for the SCMG streaming candidate (speech_server, GPU1 only).

One persistent NDJSON process.  stdin:
  {"op":"speak","uid":..,"text":..}
  {"op":"audio","uid":..,"start":..,"end":..,"pcm":<base64 f32le 24 kHz>}
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
CKPT = _cached('ckpt/sbag-seed1-final.pt', B / 'r4-1104/local-snaps/sbag-seed1-final.pt')
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
                        source=SOURCE_ID, speaker='null')

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
    def __init__(self, uid, text, model, e5):
        import torch
        from scmg.infer import Streamer
        from scmg.common import U80
        self.uid, self.text, self.model, self.t0 = uid, text, model, time.monotonic()
        self.pcm = []                      # float32 24 kHz chunks
        self.samples = 0
        self.done = False
        self.cancel = False
        self.anchors = {}
        self.tick_k = 0
        self.frames_sent = 0
        self.tick_ms = []
        self.first_frame_ms = None
        self.done_sent = False
        # text: whole sentence known now; features are prefix-causal, visibility is the x2 arrival mask
        feats, offs, ids = e5.encode(text)
        self.offs = offs
        self.n_tok = len(ids)
        self.ttime = np.full(self.n_tok, np.nan, np.float32)
        cond = dict(text_c=torch.as_tensor(feats, device='cuda').float(),
                    ttime=torch.as_tensor(self.ttime, device='cuda').float(),
                    tmask=torch.ones(self.n_tok, dtype=torch.bool, device='cuda'),
                    spk=torch.zeros(192, device='cuda'), spk_valid=torch.tensor(0., device='cuda'),
                    source=torch.tensor(SOURCE_ID, device='cuda'),
                    ref=torch.zeros(128, 34, device='cuda'), ref_valid=torch.tensor(0., device='cuda'))
        self.s = Streamer(model.m, model.vae, torch.zeros(1, 89, device='cuda'), cond, model.stats,
                          H=2 * U80, steps=10, standby=model.qm, ctx_t=model.m.ctx_t, cfg_w=2.5,
                          text_mode='x2', x2_rate=15., x2_lag=.15, deploy_timing=True,
                          trim_start=True, cfg_null_fix=True)
        emit(dict(uid=uid, type='meta', names=NAMES34, fps=FPS,
                  model=dict(sha256=model.sha, arch='scmg-r4-1104', step=None), lead_s=H_FRAMES / FPS,
                  warmup_ticks=0, cfg=model.cfg))

    def add_audio(self, raw, start, end):
        x = np.frombuffer(base64.b64decode(raw), '<f4').astype(np.float32)
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
        n24 = T * FRAME24
        x24 = np.concatenate(self.pcm)[:n24] if self.pcm else np.zeros(0, np.float32)
        if len(x24) < n24:
            x24 = np.concatenate([x24, np.zeros(n24 - len(x24), np.float32)])
        if len(x24) == 0:
            return np.zeros((0, 89), np.float32)
        if n24 % 3:
            raise ValueError('prefix is not a multiple of 3 24 kHz samples')
        x16 = _resample_2_3(x24)
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
            return
        k = self.tick_k
        arrived = self.samples // FRAME24
        if self.done:
            need = CHUNK_FRAMES * k < arrived
        else:
            need = CHUNK_FRAMES * k + H_FRAMES <= arrived
        if not need:
            return
        T = min(arrived, CHUNK_FRAMES * k + H_FRAMES) if not self.done else arrived
        T = max(1, T)
        f = self._feat(T)
        self._refresh_text_time()
        self.s.feat = torch.as_tensor((f - self.model.fm) / self.model.fs, device='cuda').float()
        t0 = time.perf_counter()
        fr = self.s.tick()
        dt = (time.perf_counter() - t0) * 1000
        self.tick_ms.append(dt)
        self._send_frames(fr, input_samples=self.samples, input_done=self.done)
        self.tick_k += 1

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
        while CHUNK_FRAMES * self.tick_k < self.samples // FRAME24:
            self._tick()
        if self.tick_k == 0:
            self.done_sent = True
            emit(dict(uid=self.uid, type='done', frames=0, ticks=0, tick_ms_p50=None, tick_ms_p95=None,
                      first_frame_ms=None, audio_s=self.samples / SR, text_tokens=int(self.n_tok), text=self.text))
            return
        fr = self.s.flush()
        self._send_frames(fr, input_samples=self.samples, input_done=True)
        self.done_sent = True
        tk = np.asarray(self.tick_ms, np.float32)
        emit(dict(uid=self.uid, type='done', frames=self.frames_sent, ticks=int(len(tk)),
                  tick_ms_p50=float(np.percentile(tk, 50)) if len(tk) else None,
                  tick_ms_p95=float(np.percentile(tk, 95)) if len(tk) else None,
                  first_frame_ms=self.first_frame_ms, audio_s=self.samples / SR,
                  text_tokens=int(self.n_tok), text=self.text))


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
                text = str(row['text']).strip()
                if not text or len(text) > 400:
                    raise ValueError('text length')
                if uid in utts:
                    utts[uid].cancel = True
                utts[uid] = Utt(uid, text, model, e5)
                emit(dict(uid=uid, type='accepted', t=time.monotonic()))
                continue
            u = utts.get(uid)
            if u is None:
                continue
            if op == 'audio':
                u.add_audio(row['pcm'], int(row['start']), int(row['end']))
                u._tick()
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
