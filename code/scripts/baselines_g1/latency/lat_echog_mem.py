"""ECHO-G GPU memory by stage (torch.cuda.max_memory_allocated with reset_peak_memory_stats before every stage).
  weights: Qwen3.5-4B bf16 / Wav2Vec2 fp32 / StreamSGDiT (and the official DiT) resident after loading
  whole-utterance: Wav2Vec2 forward (output_hidden_states as the official extractor), Qwen forward with
                   output_hidden_states=True (official extractor) vs a forward hook on layer -2 only, DiT sample_euler
  stream:          per block: prefix Wav2Vec2, Qwen on the newly visible tokens (KV cache, all hidden states vs hook),
                   Qwen prefix recompute without cache (hook), window conditions + Euler; KV cache size after the clip
Reports peaks above the resident weights ('delta') and the absolute peak, per stage, max over clips.
  python lat_echog_mem.py --out DIR [--n N]"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402
from lat_echog_stream import ES, W, CK, DR, WORDS  # noqa: E402

MB = 2 ** 20


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(W / 'official/src')); sys.path.insert(0, str(ES / 'echog_stream')); sys.path.insert(0, str(ES / 'echog_g1'))
    import librosa
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from echo_g.condition_provenance import load_condition_manifest
    from echo_g.data import decode_condition, load_aligned_lengths
    from echo_g.utils import stable_seed
    from echo_g.inference import load_model_and_config
    from echo_g.flow import sample_euler
    from stream_core import block_start, load_stream_model, sample_window_euler, window_conditions, visible_tokens
    from echog_slide import PrefixAudio
    dev = torch.device('cuda'); res = dict(weights={}, whole={}, stream={})
    alloc = lambda: torch.cuda.memory_allocated() / MB
    def load(name, fn):
        b = alloc(); x = fn(); torch.cuda.synchronize(); res['weights'][name] = alloc() - b; return x
    tok = AutoTokenizer.from_pretrained(str(W / 'models/Qwen3.5-4B'), use_fast=True)
    qwen = load('qwen3.5-4b_bf16', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(str(W / 'models/Qwen3.5-4B'), torch_dtype=torch.bfloat16).to(dev).eval())
    pa = load('wav2vec2_fp32', lambda: PrefixAudio(str(W / 'models/wav2vec2-large-xlsr-53-english'), dev))
    sm, config, mean, std, sc, meta = load('stream_dit', lambda: load_stream_model(CK, dev))
    om, ocfg, _, _, _ = load('official_dit', lambda: load_model_and_config(W / 'weights/best.pt', W / 'official/configs/sgdit_audio_text.yaml', dev))
    resident = alloc(); res['resident_mib'] = resident
    layer = qwen.model.language_model.layers[-2] if hasattr(qwen.model, 'language_model') else qwen.model.layers[-2]
    cap = {}
    def hook(_m, _i, o): cap['h'] = (o[0] if isinstance(o, tuple) else o).float().cpu()
    fps = float(config.data.fps); steps = config.sampling.integration_steps
    block_s = sc.block_ms / 1000.; ahead = sc.ahead_s; ctx_f = int(round(sc.ctx_s * fps)); tail_f = int(round(sc.infer_tail_ms / 1000. * fps))
    lengths = load_aligned_lengths(DR / config.data.aligned_lengths_file)
    man = load_condition_manifest(DR / config.data.condition_manifest, config.data.condition_manifest_sha256) if config.data.condition_manifest else None
    stems = LC.stems50()[LC.N_WARM:]
    stems = sorted(stems, key=lambda s: -lengths[s])[:a.n] if a.n else stems          # longest first when limited
    def peak(key, d, fn):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b = alloc()
        r = fn(); torch.cuda.synchronize(); p = torch.cuda.max_memory_allocated() / MB
        e = d.setdefault(key, dict(delta_mib=0., abs_mib=0.)); e['delta_mib'] = max(e['delta_mib'], p - b); e['abs_mib'] = max(e['abs_mib'], p)
        return r
    for si, stem in enumerate(stems):
        cp = DR / 'condition_30fps' / f'{stem}.pt'; payload = torch.load(cp, map_location='cpu', weights_only=True)
        F = min(lengths[stem], config.data.max_frames)
        c = decode_condition(payload, stem, config.data, config.model, F, condition_path=cp, condition_manifest=man)
        text, centres = c['text'].to(dev), c['token_centers'].to(dev); tt = torch.as_tensor(payload['token_times']).double().numpy()
        wav = librosa.load(DR / 'audio' / f'{stem}.wav', sr=16_000, mono=True)[0]
        ids = tok(json.load(open(WORDS / f'{stem}.json'))['canonical_transcript'], return_tensors='pt')['input_ids'].to(dev)
        with torch.no_grad():
            # ---- whole utterance
            peak('wav2vec2_whole', res['whole'], lambda: pa(wav, math.inf, fps, None))
            peak('qwen_whole_all_hidden_states', res['whole'], lambda: qwen(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[-2][0].float().cpu())
            hh = layer.register_forward_hook(hook)
            peak('qwen_whole_layer-2_hook', res['whole'], lambda: qwen.model(input_ids=ids, use_cache=False) if hasattr(qwen, 'model') else None)
            hh.remove()
            g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, 0))
            cw = window_conditions(c['audio'].to(dev), text, tt, centres, 0, F, 0.0, math.inf, fps, 'final')   # = stream_infer --offline
            bt = dict(audio=cw['audio'][None], text=cw['text'][None], text_padding_mask=cw['text_padding_mask'][None],
                      time_distance=cw['time_distance'][None], motion_mask=torch.ones(1, F, dtype=torch.bool, device=dev))
            nz = torch.randn(1, F, om.motion_dim, generator=g, device=dev)
            peak('dit_sample_euler_whole', res['whole'], lambda: sample_euler(om, bt, ocfg.sampling.integration_steps, 1.0, initial_noise=nz))
            # ---- stream
            g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, 0))
            noise = torch.cat([torch.randn(F, sm.motion_dim, generator=g, device=dev), torch.randn(tail_f, sm.motion_dim, generator=g, device=dev)])
            D = sm.motion_dim; out = torch.zeros(F, D, device=dev); k = 0; pkv = None; nd = 0
            while True:
                f0 = block_start(k, block_s, fps)
                if f0 >= F: break
                now = k * block_s; f1 = min(F, block_start(k + 1, block_s, fps)); ws = max(0, f0 - ctx_f); fe = f1 + tail_f
                feats = peak('wav2vec2_prefix_per_block', res['stream'], lambda: pa(wav, now + ahead, fps, None).to(dev))
                nv = min(int(visible_tokens(tt, now, ahead, 'voiced', sc.x2_rate, sc.x2_lag).sum()), ids.shape[1])
                if nv > nd:
                    o = peak('qwen_incremental_kvcache_all_hidden_states', res['stream'],
                             lambda: qwen(input_ids=ids[:, nd:nv], past_key_values=pkv, use_cache=True, output_hidden_states=True))
                    pkv = o.past_key_values; o.hidden_states[-2].float().cpu(); del o
                    hh = layer.register_forward_hook(hook)
                    peak('qwen_prefix_recompute_nocache_layer-2_hook', res['stream'], lambda: qwen.model(input_ids=ids[:, :nv], use_cache=False))
                    hh.remove(); nd = nv
                def win():
                    cond = window_conditions(feats, text, tt, centres, ws, fe, now, ahead, fps, 'voiced', sc.x2_rate, sc.x2_lag)
                    Wn = fe - ws
                    b2 = dict(audio=cond['audio'][None], text=cond['text'][None], text_padding_mask=cond['text_padding_mask'][None],
                              time_distance=cond['time_distance'][None], motion_mask=torch.ones(1, Wn, dtype=torch.bool, device=dev))
                    kn = torch.zeros(1, Wn, D, device=dev); km = torch.zeros(1, Wn, dtype=torch.bool, device=dev)
                    kn[0, :f0 - ws] = out[ws:f0]; km[0, :f0 - ws] = True
                    return sample_window_euler(sm, b2, noise[ws:fe][None].clone(), kn, km, steps, sc.known_xt)
                y = peak('stream_dit_window_euler', res['stream'], win)
                out[f0:f1] = y[0, f0 - ws:f1 - ws]; k += 1
            kv = alloc() - resident; res['stream'].setdefault('kv_cache_and_leftovers_after_clip_mib', 0.)
            res['stream']['kv_cache_and_leftovers_after_clip_mib'] = max(res['stream']['kv_cache_and_leftovers_after_clip_mib'], kv)
            del pkv; torch.cuda.empty_cache()
        print(si, stem, F, json.dumps({k2: {k3: round(v3['delta_mib']) for k3, v3 in v2.items() if isinstance(v3, dict)} for k2, v2 in res.items() if k2 in ('whole', 'stream')}), flush=True)
    res.update(clips=len(stems), gpu=torch.cuda.get_device_name(0), max_reserved_mib=torch.cuda.max_memory_reserved() / MB)
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / 'echog_memory_by_stage.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
