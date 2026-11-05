"""ECHO-G streamed (stream_prefix row: the stream-fine-tuned StreamSGDiT echog-stream-1105/g1-confirm/train/final.pt through
scmg-wt-echog-stream scripts/echog_stream/stream_infer.py --audio prefix --text voiced, single-clip path) latency, batch 1.
Block k (320 ms) is decided at now_k = k * 0.32 s with look-ahead 0.16 s: the official Wav2Vec2 extraction is re-run on the
waveform prefix [0, now + ahead) (echog_slide.PrefixAudio), a token is visible once its word has ended by now + ahead,
window [max(0, f0 - ctx), f1 + tail), Euler 8 steps with the committed frames known; only block k is committed.
The loop is stream_core.stream_clips for one clip with cuda-synchronised timers.  Stages: audio_feat = prefix Wav2Vec2
per block; text_enc = Qwen3.5-4B bf16 on the newly visible tokens with the KV cache (a causal LM: prefix features equal
the whole-utterance features the model reads from the released condition); generate = window conditions + Euler;
decode = de-normalisation."""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402

L = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baselines-g1-1105/latency')
ES = Path(os.environ.get('ECHOG_STREAM_CODE', str(L / 'snap/echog_stream')))
W = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official')
CK = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105/g1-confirm/train/final.pt')
DR = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105/data_root')
WORDS = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro/data/echog/annotations/words')
NOCUDNN = os.environ.get('ECHOG_W2V_NOCUDNN') == '1'   # variant: Wav2Vec2 prefix without cuDNN (no transient workspace)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    mon = LC.GPUMon(); host0 = LC.host_snapshot()
    sys.path.insert(0, str(W / 'official/src')); sys.path.insert(0, str(ES / 'echog_stream')); sys.path.insert(0, str(ES / 'echog_g1'))
    import librosa
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from echo_g.condition_provenance import load_condition_manifest
    from echo_g.data import decode_condition, load_aligned_lengths
    from echo_g.utils import stable_seed
    from stream_core import block_start, load_stream_model, sample_window_euler, window_conditions, visible_tokens
    from echog_slide import PrefixAudio
    dev = torch.device('cuda'); clk = LC.Sync()
    model, config, mean, std, sc, meta = load_stream_model(CK, dev)
    fps = float(config.data.fps); mean, std = mean.to(dev), std.to(dev); steps = config.sampling.integration_steps
    block_s = sc.block_ms / 1000.; ahead = sc.ahead_s; ctx_f = int(round(sc.ctx_s * fps)); tail_f = int(round(sc.infer_tail_ms / 1000. * fps))
    lengths = load_aligned_lengths(DR / config.data.aligned_lengths_file)
    man = load_condition_manifest(DR / config.data.condition_manifest, config.data.condition_manifest_sha256) if config.data.condition_manifest else None
    pa = PrefixAudio(str(W / 'models/wav2vec2-large-xlsr-53-english'), dev)
    tok = AutoTokenizer.from_pretrained(str(W / 'models/Qwen3.5-4B'), use_fast=True)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(str(W / 'models/Qwen3.5-4B'), torch_dtype=torch.bfloat16).to(dev).eval()
    stems = LC.stems50()[:a.n or None]; rows = []
    for kk, stem in enumerate(stems):
        cp = DR / 'condition_30fps' / f'{stem}.pt'; payload = torch.load(cp, map_location='cpu', weights_only=True)
        F = min(lengths[stem], config.data.max_frames)
        c = decode_condition(payload, stem, config.data, config.model, F, condition_path=cp, condition_manifest=man)
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, 0))
        noise = torch.cat([torch.randn(F, model.motion_dim, generator=g, device=dev), torch.randn(tail_f, model.motion_dim, generator=g, device=dev)])
        text, centres = c['text'].to(dev), c['token_centers'].to(dev); tt = torch.as_tensor(payload['token_times']).double().numpy()
        wav = librosa.load(DR / 'audio' / f'{stem}.wav', sr=16_000, mono=True)[0]; dur = len(wav) / 16000.
        ids = tok(json.load(open(WORDS / f'{stem}.json'))['canonical_transcript'], return_tensors='pt')['input_ids'].to(dev)
        st = dict(audio_feat=0., text_enc=0., generate=0., decode=0.); sched = []; ticks = []
        D = model.motion_dim; out = torch.zeros(F, D, device=dev); k = 0; pkv = None; nvis_done = 0
        with torch.no_grad():
            while True:
                f0 = block_start(k, block_s, fps)
                if f0 >= F: break
                now = k * block_s; f1 = min(F, block_start(k + 1, block_s, fps)); ws = max(0, f0 - ctx_f); fe = f1 + tail_f
                t0 = clk.now()
                with torch.backends.cudnn.flags(enabled=not NOCUDNN):      # see w2v_prefix_mem.py: cuDNN workspace spike at 0.16 s
                    feats = pa(wav, now + ahead, fps, None).to(dev)
                ta = clk.now() - t0
                t0 = clk.now()
                nv = int(visible_tokens(tt, now, ahead, 'voiced', sc.x2_rate, sc.x2_lag).sum()); nv = min(nv, ids.shape[1])
                if nv > nvis_done:
                    o = qwen(input_ids=ids[:, nvis_done:nv], past_key_values=pkv, use_cache=True, output_hidden_states=True)
                    pkv = o.past_key_values; o.hidden_states[-2].float().cpu(); nvis_done = nv
                tx = clk.now() - t0
                t0 = clk.now()
                cond = window_conditions(feats, text, tt, centres, ws, fe, now, ahead, fps, 'voiced', sc.x2_rate, sc.x2_lag)
                Wn = fe - ws
                bt = dict(audio=cond['audio'][None], text=cond['text'][None], text_padding_mask=cond['text_padding_mask'][None],
                          time_distance=cond['time_distance'][None], motion_mask=torch.ones(1, Wn, dtype=torch.bool, device=dev))
                known = torch.zeros(1, Wn, D, device=dev); km = torch.zeros(1, Wn, dtype=torch.bool, device=dev)
                known[0, :f0 - ws] = out[ws:f0]; km[0, :f0 - ws] = True
                y = sample_window_euler(model, bt, noise[ws:fe][None].clone(), known, km, steps, sc.known_xt)
                out[f0:f1] = y[0, f0 - ws:f1 - ws]; tg = clk.now() - t0
                t0 = clk.now(); (out[f0:f1].float() * std + mean).cpu(); td = clk.now() - t0
                st['audio_feat'] += ta; st['text_enc'] += tx; st['generate'] += tg; st['decode'] += td
                sched.append((min(dur, now + ahead), ta + tx + tg + td, f0 / fps, f1 - f0, fps))
                ticks.append(dict(t=now, len=block_s, audio_feat=ta, text_enc=tx, generate=tg, decode=td))
                k += 1
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, frames30=F, blocks=k, stages=st, compute=comp, schedule=sched, ticks=ticks))
        print(f'{kk + 1}/{len(stems)} {stem} dur={dur:.2f} blocks={k} comp={comp:.3f} {st}', flush=True)
    extra = dict(ckpt=str(CK), stream=dict(block_ms=sc.block_ms, ahead_s=ahead, ctx_frames=ctx_f, tail_frames=tail_f, steps=steps,
                 audio='prefix Wav2Vec2 per block', text='voiced'), frame_fps=fps, code=str(ES),
                 host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__)
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    extra['w2v_cudnn'] = not NOCUDNN
    LC.write(a.out, 'echog_stream_nocudnn' if NOCUDNN else 'echog_stream', rows, extra)


if __name__ == '__main__':
    main()
