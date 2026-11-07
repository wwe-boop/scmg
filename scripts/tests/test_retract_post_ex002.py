"""11-08: the EX002 ports (scripts/ex002/retract_post_ex002.py) use the G1 code path bit for bit.
 a r0causal_core on G1 dev data (B3c continuation, prefix from no_int, G1 icommon.retract_after T 1.4) == the existing G1 R0causal
   outputs of blend_retract.py (dev/psync_czm6kws_sil/R0causal, frozen recipe) as float32, every channel, 12 interrupts
 b funnel_core == the verbatim G1 funnel_retract.py run here on the same 12 dev interrupts (Funnel_p recipe)
 c EX002 adapter smoke: r0causal / funnel on 5 EX002 dev interrupts: frames < f_i == the uninterrupted output, finite, the funnel
   output ends closer to the damped retraction than the raw continuation
    PYTHONPATH=<repo>:<ECHO-G official src>:<pydeps/safeproj> python scripts/tests/test_retract_post_ex002.py   env SP"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE / 'scripts/ex002'))
import retract_post_ex002 as RP
G1S = Path('/home/xr-25b7/projects/gesture/scmg-wt-intcausal/scripts/interrupt'); sys.path.insert(0, str(G1S))
import icommon as G
SP = Path(os.environ['SP']); SP.mkdir(parents=True, exist_ok=True); FAIL = []
W = Path('/home/xr-25b7/bulk/interrupt-1105'); O = W / 'dev/psync_czm6kws_sil'


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


ints = json.load(open(W / 'interrupts_dev.json'))['interrupts']
rows = [r for r in ints if (O / 'R0causal' / f"{r['stem']}@{r['frac']:.1f}.pt").exists()][::75][:12]
H = json.load(open(W / 'human_ref_v2_train_drop.json')); rest = np.array(H['rest']['R_sil']['joints'])
ch = 10 + G.UPPER; bad = 0
for r in rows:
    k = f"{r['stem']}@{r['frac']:.1f}"; f = r['f_i']
    c = np.asarray(torch.load(O / 'B3c' / f'{k}.pt', weights_only=False)['robot_repr'], np.float64)
    u = np.asarray(torch.load(O / 'no_int' / f"{r['stem']}.pt", weights_only=False)['robot_repr'], np.float64); c = c.copy(); c[:f] = u[:f]
    rr = G.retract_after(c, f, rest, 1.4); c, rr = RP.align(c, rr)
    y = RP.r0causal_core(c, f, rr, G.FPS, ch).astype(np.float32)
    ref = torch.load(O / 'R0causal' / f'{k}.pt', weights_only=False)['robot_repr'].numpy()
    bad += int(not np.array_equal(y, ref))
ok('a r0causal_core == G1 blend_retract outputs bitwise', bad == 0, f'{len(rows) - bad} / {len(rows)} interrupts')
sub = SP / 'ints12.json'; sub.write_text(json.dumps(dict(interrupts=rows)))
r = subprocess.run([sys.executable, str(G1S / 'funnel_retract.py'), '--icommon', str(G1S), '--in-dir', str(O / 'B3c'), '--prefix-from', str(O / 'no_int'),
                    '--out-dir', str(SP / 'funnel_g1'), '--interrupts', str(sub), '--human', str(W / 'human_ref_v2_train_drop.json'),
                    '--tr', '1.4', '--tf', '2.2', '--m0', '1.0', '--smooth', '0.1', '--eps', '0.02', '--alpha', '0.6'], capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:])
bad = 0
for r_ in rows:
    k = f"{r_['stem']}@{r_['frac']:.1f}"; f = r_['f_i']
    c = np.asarray(torch.load(O / 'B3c' / f'{k}.pt', weights_only=False)['robot_repr'], np.float64).copy()
    u = np.asarray(torch.load(O / 'no_int' / f"{r_['stem']}.pt", weights_only=False)['robot_repr'], np.float64); c[:f] = u[:f]
    rr = G.retract_after(c, f, rest, 1.4); c, rr = RP.align(c, rr)
    y, _ = RP.funnel_core(c, f, rr, G.FPS, ch); y = y.astype(np.float32)
    ref = torch.load(SP / 'funnel_g1' / f'{k}.pt', weights_only=False)['robot_repr'].numpy(); bad += int(not np.array_equal(y, ref))
ok('b funnel_core == G1 funnel_retract.py bitwise', r.returncode == 0 and bad == 0, f'{len(rows) - bad} / {len(rows)} interrupts')
# c EX002 smoke
B = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'); SEL = B / 'ex002-sweep-1107/czm-s12000-lr1e-4-w0.1'
E = json.load(open(B / 'ex002-interrupt-1107/interrupts_dev_ex002.json'))['interrupts'][::97][:5]
(SP / 'ex5.json').write_text(json.dumps(dict(interrupts=E))); hum = os.environ.get('HUMAN', str(B / 'qual-1108/ex002int/human_ref_partial458'))
good = True; closer = []
import icommon_ex002 as I
restE = np.array(json.load(open(hum + '.json'))['rest']['R_sil']['joints'])
for mode in ('r0causal', 'funnel'):
    r = subprocess.run([sys.executable, str(CODE / 'scripts/ex002/retract_post_ex002.py'), mode, '--in-dir', str(SEL / 'int_psync_s0'), '--prefix-from', str(SEL / 'psync/s0'),
                        '--out-dir', str(SP / mode), '--interrupts', str(SP / 'ex5.json'), '--human', hum], capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); good = False; continue
    for e in E:
        nm = f"{e['ci']}@{e['item']}@{e['frac']:.1f}.npy"; y = np.load(SP / mode / nm); u = np.load(SEL / 'psync/s0' / f"{e['ci']}.npy"); f = e['f_i']
        good &= bool(np.array_equal(y[:f], u[:f].astype(np.float32)) and np.isfinite(y).all())
        if mode == 'funnel':
            c = np.load(SEL / 'int_psync_s0' / nm).astype(np.float64); c[:f] = u[:f]; rr = I.retract_after(c, f, restE, 1.4)
            closer.append(np.abs(y[-1, I.UPPER] - rr[len(y) - 1, I.UPPER]).mean() <= np.abs(c[len(y) - 1, I.UPPER] - rr[len(y) - 1, I.UPPER]).mean() + 1e-6)
ok('c EX002 adapter: executed prefix bitwise, finite; funnel ends closer to the retraction', good and all(closer), f'{sum(closer)} / {len(closer)}')
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)
