import json, numpy as np, sys
sys.path.insert(0,'/home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt'); import icommon as I
W='/home/xr-25b7/bulk/interrupt-1105/'
hw=np.load(W+'human_ref_v2_train_drop_windows.npz'); Hk=np.stack([I.kin_features(I.fk(w),30) for w in hw['rr_ctx'].astype(np.float64)])[hw['retract'].astype(bool)]
def rows(f,m): return {(r['stem'],r['frac']):r for r in map(json.loads,open(W+f)) if r['method']==m}
fa,ma,fb,mb=sys.argv[1:5]; A=rows(fa,ma); B=rows(fb,mb)
keys=sorted(set(A)&set(B)); stems=sorted({k[0] for k in keys}); by={s:[k for k in keys if k[0]==s] for s in stems}
rk=lambda R,ks: I.kin_distance(np.array([R[k]['kin'] for k in ks]),Hk)[0]
rng=np.random.default_rng(0); d=[]
for _ in range(500):
    ks=[k for s in rng.choice(stems,len(stems)) for k in by[s]]; d.append(rk(A,ks)-rk(B,ks))
print(f'{ma} - {mb}: RKD diff {rk(A,keys)-rk(B,keys):+.3f} 95% [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] n={len(keys)}')
