"""CUDA leakage regression: real small chunk / Qwen2 / Qwen2+xattn+word-head models, no data downloads."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import torch
from scmg.common import J, FEAT, SPK, TEXT_DIM, CHUNK
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.infer import Streamer


def tiny_lm(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(dict(model_type='qwen2', vocab_size=128, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, use_cache=False)))
    os.environ['SCMG_LM_CONFIG'] = str(root)


def model(kind, **kw):
    torch.manual_seed(1103)
    extra = {} if kind == 'chunk' else dict(planner='voxcpm', vox_lm_init='random')
    if kind == 'vox_xattn': extra.update(vox_text_xattn=True, vox_word_aux=True)
    m = SCMG(Config(d=64, heads=4, audio_layers=2, planner_layers=2, flow_layers=2, **extra, **kw)).cuda().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith(('ada.weight', 'f_out.weight', '.g', 'x_out.weight')): p.normal_(0, .03)
    v = MotionVAE(z=m.cfg.z, c=32).cuda().eval().requires_grad_(False)
    return m, v


def fixture():
    torch.manual_seed(91); dev = 'cuda'; n = 24
    cond = dict(text=torch.randn(n, TEXT_DIM, device=dev), text_c=torch.randn(n, TEXT_DIM, device=dev),
                ttime=torch.arange(n, device=dev).float() * .22 + .1,
                tmask=torch.ones(n, dtype=torch.bool, device=dev), tids=torch.arange(n, device=dev) + 10,
                tidm=torch.ones(n, dtype=torch.bool, device=dev),
                spk=torch.randn(SPK, device=dev), spk_valid=torch.ones((), device=dev),
                source=torch.zeros((), dtype=torch.long, device=dev),
                ref=torch.randn(64, J, device=dev), ref_valid=torch.ones((), device=dev))
    cond['titime'] = cond['ttime'].clone()
    cond['tseg'] = cond['tiseg'] = torch.zeros(n, device=dev)
    cond['tiend'] = torch.full((n,), 8., device=dev)
    return torch.randn(400, FEAT, device=dev), cond, dict(q_mean=[0.] * J, q_std=[1.] * J)


def test():
    checks = []
    for kind in ('chunk', 'vox', 'vox_xattn'):
        m, v = model(kind, mtc=True, adsc=True)
        feat, cond, stats = fixture()
        def run(c, tick, trim):
            s = Streamer(m, v, feat, c, stats, H=2, steps=3, seed=8, text_mode='x2', x2_rate=4., x2_lag=.15,
                         deploy_timing=True, window_lm_text=True, trim_start=trim, cfg_null_fix=True, cfg_w=2.)
            plans = []; orig = m.plan
            def capture(*args, **kwargs):
                h = orig(*args, **kwargs); plans.append(h[:, s._W - 3].clone()); return h
            m.plan = capture
            try:
                for _ in range(tick + 1): s.tick()
            finally: m.plan = orig
            return torch.stack(plans[-2:]), torch.stack(s.lat[-CHUNK:]), s
        for trim in (False, True):
            for tick in (0, 2, 14):
                h0, z0, s = run(cond, tick, trim)
                future = s.arr_text > tick * .32
                future_lm = s.arr_lm > tick * .32
                assert future.any() and (~future).any(), 'fixture needs arrived AND unarrived words'
                for route in ('e5', 'lm', 'both'):
                    changed = {k: x.clone() for k, x in cond.items()}
                    if route in ('e5', 'both'):
                        changed['text_c'][future] = torch.randn_like(changed['text_c'][future]) * 100
                        changed['text'][future] = torch.randn_like(changed['text'][future]) * 100
                    if route in ('lm', 'both'): changed['tids'][future_lm] = (changed['tids'][future_lm] + 53) % 128
                    h1, z1, _ = run(changed, tick, trim)
                    assert torch.equal(h0, h1), (kind, tick, trim, route, 'planner leak', (h0-h1).abs().max())
                    assert torch.equal(z0, z1), (kind, tick, trim, route, 'latent leak', (z0-z1).abs().max())
                    checks.append(dict(kind=kind, tick=tick, trim=trim, route=route, h_bit_equal=True, latent_bit_equal=True))
                # Positive control: changing arrived text must affect the active route / generator.
                changed = {k: x.clone() for k, x in cond.items()}
                changed['text_c'][~future] += 5
                changed['tids'][~future_lm] = (changed['tids'][~future_lm] + 37) % 128
                hp, zp, _ = run(changed, tick, trim)
                assert not torch.equal(h0, hp) and not torch.equal(z0, zp), (kind, 'inactive text route')
        print(kind, 'future e5 / LM perturbations bit-identical; arrived-word controls sensitive', flush=True)
    return checks


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); tiny_lm(a.out / 'tiny-qwen')
    with torch.no_grad(): result = test()
    (a.out / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    print(f'ALL_OK {len(result)} future-word perturbations', flush=True)
