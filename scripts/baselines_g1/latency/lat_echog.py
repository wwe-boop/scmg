"""ECHO-G official (audio+text best.pt, v0.2.0) latency, batch 1, benchmark settings (EMA, Euler 8, CFG 1, fp32 model).
Whole-utterance model: compute starts once the whole clip (audio + word-timed transcript) has arrived.
Stages: audio_feat = Wav2Vec2 (wav2vec2-large-xlsr-53-english, final hidden, fp32) on the 16 kHz clip + 30 FPS
interpolation (echo_g.conditions.extract_audio_conditions); text_enc = Qwen3.5-4B bf16 forward, hidden layer -2
(extract_text_conditions); generate = sample_euler on the frozen released conditions (identical values to the
benchmark run; the encoder calls above measure the cost of producing them); decode = de-normalisation to physical units.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402

W = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official')
DR = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root')
MANIFEST_SHA = '81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    mon = LC.GPUMon(); host0 = LC.host_snapshot()
    sys.path.insert(0, str(W / 'official/src'))
    import librosa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, Qwen3_5ForConditionalGeneration
    from echo_g.inference import load_model_and_config
    from echo_g.data import ConditionDataset, collate_motion, load_aligned_lengths
    from echo_g.flow import sample_euler
    from echo_g.utils import move_batch, stable_seed
    from echo_g.conditions import interpolate_features
    dev = torch.device('cuda'); clk = LC.Sync()
    model, config, mean, std, _ = load_model_and_config(W / 'weights/best.pt', W / 'official/configs/sgdit_audio_text.yaml', dev)
    mean, std = mean.to(dev), std.to(dev)
    proc = Wav2Vec2Processor.from_pretrained(str(W / 'models/wav2vec2-large-xlsr-53-english'))
    w2v = Wav2Vec2ForCTC.from_pretrained(str(W / 'models/wav2vec2-large-xlsr-53-english')).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(str(W / 'models/Qwen3.5-4B'), use_fast=True)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(str(W / 'models/Qwen3.5-4B'), torch_dtype=torch.bfloat16).to(dev).eval()
    stems = LC.stems50()[:a.n or None]
    lengths = load_aligned_lengths(DR / 'audit/condition/per_clip_condition_motion_audit.csv')
    ds = ConditionDataset(DR / 'condition_30fps', stems, config.data, config.model, lengths,
                          DR / 'audit/condition/frozen_conditions.json', MANIFEST_SHA)
    rows = []
    for k, stem in enumerate(stems):
        transcript = json.load(open(f'/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro/data/echog/annotations/words/{stem}.json'))['canonical_transcript']
        batch = move_batch(collate_motion([ds[k]]), dev); n = int(batch['motion_mask'].sum())
        st = {}
        t0 = clk.now()
        wav, _ = librosa.load(str(DR / 'audio' / f'{stem}.wav'), sr=16000, mono=True); dur = len(wav) / 16000.
        with torch.no_grad():
            h = w2v(proc(wav, sampling_rate=16000, return_tensors='pt').input_values.to(dev), output_hidden_states=True).hidden_states[-1].float()
            interpolate_features(h, int(round(dur * 30))).cpu()
        st['audio_feat'] = clk.now() - t0
        t0 = clk.now()
        ids = tok(transcript, return_tensors='pt')['input_ids'].to(dev)
        with torch.no_grad():
            qwen(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[-2][0].float().cpu()
        st['text_enc'] = clk.now() - t0
        t0 = clk.now()
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, 0))
        noise = torch.zeros(1, batch['audio'].shape[1], model.motion_dim, device=dev)
        noise[0, :n] = torch.randn(n, model.motion_dim, generator=g, device=dev)
        with torch.no_grad(): y = sample_euler(model, batch, 8, 1.0, initial_noise=noise)
        st['generate'] = clk.now() - t0
        t0 = clk.now(); (y[0, :n].float() * std + mean).cpu(); st['decode'] = clk.now() - t0
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, frames30=n, stages=st, compute=comp, schedule=[(dur, comp, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} {st}', flush=True)
    extra = dict(ckpt=str(W / 'weights/best.pt'), steps=8, cfg=1.0, encoders=dict(audio='wav2vec2-large-xlsr-53-english fp32',
                 text='Qwen3.5-4B bf16 layer -2'), frame_fps=30, host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__)
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    LC.write(a.out, 'echog', rows, extra)


if __name__ == '__main__':
    main()
