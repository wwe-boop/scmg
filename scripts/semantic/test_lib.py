"""Fixed-action library insertion (Streamer inpaint, 11-07 item 5) acceptance tests; helpers from test_ctl.py (same env).
 A default: inpaint=None and an all-zero weight vector give the default sampling bitwise (the default path vs HEAD is test_ctl B)
 B locality: with weights only from latent L0 on, every frame of chunks whose windows end before L0 is unchanged
 C exactness: committed latents at weight-1 positions equal the target latents exactly; frames change inside the insertion
Prints ALL_OK / SOME_FAIL."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def run(m, vae, v, st, ci, nch, inpaint=None):
    from scmg.common import CHUNK_FRAMES
    from scmg.data import WindowDataset
    from scmg.infer import Streamer
    from scmg.model import INF
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    keys = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')
    cond = {k: s[k] for k in keys if k in s}; kw = {} if inpaint is None else dict(inpaint=inpaint)
    with torch.no_grad():
        S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=4, seed=0, ctx_t=0.9, cfg_w=2.5, **kw)
        for i in range(nch): S.tick()
        S.flush()
    return torch.stack(S.frames).cpu().numpy(), torch.stack(S.lat).cpu().numpy()


def main():
    from scmg.common import CHUNK, CHUNK_FRAMES
    ok = {}
    st, v, vae = TC.setup(split='val'); m = TC.model({'adsc': True}).eval()
    ci = 0; nch = 10; nl = nch * CHUNK + 16
    y0, l0 = run(m, vae, v, st, ci, nch)
    z = l0.shape[1]
    yz, _ = run(m, vae, v, st, ci, nch, dict(lat=np.zeros((nl, z), np.float32), w=np.zeros(nl, np.float32), seed=0))
    ok['A_zero_weights_bitwise'] = np.array_equal(y0, yz)
    L0 = 5 * CHUNK; tgt = np.random.default_rng(0).standard_normal((nl, z)).astype(np.float32) * 0.5
    w = np.zeros(nl, np.float32); w[L0:L0 + 8] = 1.0
    y1, l1 = run(m, vae, v, st, ci, nch, dict(lat=tgt, w=w, seed=0))
    # chunk c's window reaches latents up to (c + 1 + look-ahead) * CHUNK; generation of chunk c (and its provisional chunks)
    # cannot see weights at >= L0 if its last window latent < L0; frames are emitted 1 latent late and decode ~7 latents back
    from scmg.infer import LOOK
    safe_chunk = L0 // CHUNK - 1 - LOOK
    safe_frames = max(0, safe_chunk * CHUNK_FRAMES - 4)
    d = np.abs(y0 - y1).max(-1); first = int(np.argmax(d > 0)) if (d > 0).any() else len(d)
    ok['B_locality'] = first >= safe_frames
    ok['C_exact_latents'] = bool(np.array_equal(l1[L0:L0 + 8], tgt[L0:L0 + 8]))
    ok['C_changed'] = bool((d > 0).any())
    print(dict(first_changed=first, safe_from=safe_frames, L0=L0))
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()
