import json, torch
from scmg.data import View, WindowDataset, collate
from scmg import vae as V
st=json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'))
v=View(split='val', stats=st, mmap=True)
b=next(iter(torch.utils.data.DataLoader(WindowDataset(v,256),32,collate_fn=collate))); q=b['q'].cuda()
for lr in (1e-3,):
    torch.manual_seed(0); m=V.MotionVAE().cuda(); opt=torch.optim.AdamW(m.parameters(),lr)
    for i in range(500):
        with torch.autocast('cuda',torch.bfloat16): y,l=m(q, deriv=min(1., i/300))
        opt.zero_grad(); l['loss'].backward(); opt.step()
        if i%100==0 or i==499: print(lr,i,{k:round(float(x),3) for k,x in l.items()},flush=True)
    mu,_=m.encode(q); print('det rec',(m.decode(mu)-q).abs().mean().item(),'mu tstd',mu.std(1).mean().item())
