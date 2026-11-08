import numpy as np, torch, json, sys, os
from multiprocessing import Pool
D='/mnt/bulk/zehan/echog-dataset/ds/motion_39d_30fps'
stems=[s.strip() for s in open('/mnt/bulk/zehan/echog-dataset/official-meta/splits/train_drop.txt') if s.strip()]
stems=[s for s in stems if not (s.startswith('english_2_scott') or s.startswith('english_6_carla'))]
def f(ss):
    V=[];A=[];J=[]
    for s in ss:
        x=torch.load(f'{D}/{s}.pt',map_location='cpu',weights_only=False)
        q=np.asarray(x['robot_repr'] if isinstance(x,dict) else x,np.float64)[:,10:]
        if q.shape[0]<5: continue
        d1=np.diff(q,axis=0)*30; d2=np.diff(q,2,axis=0)*900; d3=np.diff(q,3,axis=0)*27000
        V.append(np.abs(d1).astype(np.float32));A.append(np.abs(d2).astype(np.float32));J.append(np.abs(d3).astype(np.float32))
    return np.concatenate(V),np.concatenate(A),np.concatenate(J)
if __name__=='__main__':
    with Pool(12) as p: parts=p.map(f,[stems[i::48] for i in range(48)])
    V=np.concatenate([x[0] for x in parts]);A=np.concatenate([x[1] for x in parts]);J=np.concatenate([x[2] for x in parts])
    out=dict(n_clips=len(stems),n_frames=int(V.shape[0]))
    for nm,X in (('vel',V),('acc',A),('jerk',J)):
        for qq in (99,99.9,99.99,100):
            out[f'{nm}_p{qq}']=np.percentile(X,qq,axis=0).round(3).tolist()
    json.dump(out,open(sys.argv[1],'w'),indent=0)
    from echo_g.evaluation.robot_repr import G1_JOINT_NAMES as N
    for j,n in enumerate(N): print(f'{n:22s} v99.9 {out["vel_p99.9"][j]:6.2f} a99 {out["acc_p99"][j]:7.1f} a99.9 {out["acc_p99.9"][j]:7.1f} a99.99 {out["acc_p99.99"][j]:7.1f} amax {out["acc_p100"][j]:8.1f} | j99 {out["jerk_p99"][j]:8.0f} j99.9 {out["jerk_p99.9"][j]:8.0f} j99.99 {out["jerk_p99.99"][j]:8.0f}')
    print(out['n_clips'],out['n_frames'])
