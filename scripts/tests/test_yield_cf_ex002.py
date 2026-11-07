"""11-08 yield_cf_prepare on EX002 (34 joints): counterfactual YIELD rows on a data-v36_1 batch with the EX002 control tokens and the
causal precompute. Checks per prepared sample: one YIELD token at t_y; frames before t_y unchanged; waist + arms reach the EX002 human
rest pose (R_sil, 34 joints) by the window end (< 0.05 rad); head + fingers held at the frame before t_y; text withdrawn after t_y.
    SCMG_J=34 python scripts/tests/test_yield_cf_ex002.py      env HUMAN (human_ref_ex002 .json), SP"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]; sys.path.insert(0, str(CODE))
B_ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; D_ = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
os.environ.setdefault('SCMG_TEXT_CAUSAL', B_ + '/models/text-e5-causal-v36_1')
from scmg.common import STRIDE, U80
from scmg import causal_feat as CF
from scmg.ctl import YIELD
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, yield_cf_prepare
from scmg.evaluate import load
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


stats = json.loads(Path(D_ + '/scmg/STATS_v36_1.json').read_text())
v = View(D_ + '/train-v1-20261101/views/data-v36_1', 'train', D_ + '/scmg/text-e5-v36_1', stats, mmap=True, tok_cache=D_ + '/scmg/tok-qwen25-v36_1',
         ctl_tokens=B_ + '/control-token-1105/ex002/tokens')
v.load_causal(B_ + '/ex002-causal-1107/pre-train-slim2m', StageCfg().horizons, spk_atol=1e-3, check_spk=False)
it = iter(WindowDataset(v, 256, max_tokens=256, seed=7)); b = collate([next(it) for _ in range(12)])
m, vae = load(B_ + '/ex002-sweep-1107/czm-s12000-lr1e-4-w0.1/train/final.pt', B_ + '/trained/vae-r3/final.pt', 'cpu')
for k, val in {'adsc': True, 'ctl': True, 'ctl_flow': True}.items(): setattr(m.cfg, k, val)
m.add_modules()
st = dict(qm=torch.tensor(stats['q_mean']), qs=torch.tensor(stats['q_std']), fm_raw=stats['f_mean'], fs_raw=stats['f_std'])
B = b['q'].shape[0]; K = 256 // 16; torch.manual_seed(1)
stream = b['cz_ok'].bool(); H = torch.full((B,), 2 * U80)
with torch.no_grad():
    mu, _ = vae.encode(b['q']); x1 = (mu - m.lat_mean) / m.lat_std
sc = StageCfg(yield_cf_p=1.0, yield_human=os.environ['HUMAN'])
b0 = {k: x.clone() for k, x in b.items()}
b2, q2, x12, ycf = yield_cf_prepare(m, vae, b, sc, st, stream, H, K, x1, b['q'])
on = ycf['on'] if ycf else torch.zeros(B, dtype=torch.bool); ly = ycf['ly'] if ycf else None
ok('some samples prepared', bool(on.any()), f'{int(on.sum())} / {B} (cz_ok {int(stream.sum())})')
rest = torch.tensor(json.loads(Path(os.environ['HUMAN']).read_text())['rest']['R_sil']['joints'])
qm, qs = st['qm'], st['qs']; UP = [0, 1] + list(range(4, 18)); HOLD = [2, 3] + list(range(18, 34)); good = True; info = []
for i in on.nonzero()[:, 0].tolist():
    fy = int(ly[i]) * STRIDE; ty = fy / 50.
    slot = (b2['ctl_type'][i] == YIELD).nonzero()[:, 0]
    good &= len(slot) == 1 and abs(float(b2['ctl_t'][i, slot[0]]) - ty) < 1e-5
    good &= torch.equal(q2[i, :fy], b0['q'][i, :fy])
    qp = q2[i].float() * qs + qm; end = min(int(b0['mask'][i].sum()), 256) - 1; q0 = b0['q'][i, fy - 1].float() * qs + qm
    dr = float((qp[end, UP] - rest[UP]).abs().max()); dh = float((qp[fy:end, HOLD] - q0[HOLD][None]).abs().max())
    good &= dr < 0.05 and dh < 1e-4
    te = torch.nan_to_num(b0['cz_tend'][i], nan=1e9); good &= bool((b2['tmask'][i] == (b0['tmask'][i] & (te <= ty + CF.WORD_EPS))).all())
    info.append(f'ly {int(ly[i])} rest-err {dr:.3f} hold-err {dh:.1e}')
ok('token / motion (waist + arms -> rest, head + fingers held) / text rewrite', good, ' | '.join(info[:5]))
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)
