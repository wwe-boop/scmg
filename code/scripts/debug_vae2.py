import json, numpy as np, torch, torch.nn as nn
from scmg.data import View, WindowDataset, collate
from scmg import vae as V
D='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st=json.load(open(D+'/STATS_v36_1.json'))
v=View(split='val', stats=st, mmap=True)
b=next(iter(torch.utils.data.DataLoader(WindowDataset(v,256),32,collate_fn=collate))); q=b['q'].cuda()
def run(name, make, beta=1e-3, det=False, steps=200):
    torch.manual_seed(0); m=make().cuda(); opt=torch.optim.AdamW(m.parameters(),3e-4)
    for s in range(steps):
        if det:
            mu,_=m.encode(q); y=m.decode(mu); l=V.recon_losses(y,q); loss=l['rec']+l['vel']
        else:
            y,l=m(q,beta=beta); loss=l['loss']
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu,_=m.encode(q); print(name,'rec',round(float((m.decode(mu)-q).abs().mean()),4),'mu tstd',round(float(mu.std(1).mean()),4),flush=True)
run('default', V.MotionVAE)
run('beta0', V.MotionVAE, beta=0.)
run('deterministic', V.MotionVAE, det=True)
class NoNorm(V.MotionVAE):
    def __init__(s):
        super().__init__()
        for mod in s.modules():
            if isinstance(mod, V.Res): mod.n=nn.Identity()
run('nonorm-det', NoNorm, det=True)
# input sensitivity at init
m=V.MotionVAE().cuda(); a,_=m.encode(q); b2,_=m.encode(q*0); print('init sensitivity', float((a-b2).abs().mean()), float(a.abs().mean()))
