import json, numpy as np, torch
from scmg.data import View, WindowDataset, collate
from scmg.vae import MotionVAE
D='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st=json.load(open(D+'/STATS_v36_1.json'))
v=View(split='val', stats=st, mmap=True)
ds=WindowDataset(v,256)
it=iter(torch.utils.data.DataLoader(ds,32,collate_fn=collate))
b=next(it); q=b['q'].cuda(); m=b['mask'].cuda()
print('q stats', q.mean().item(), q.std().item(), q.abs().mean().item(), 'mask', m.mean().item())
for name, kw in [('fp32',dict()),]:
    torch.manual_seed(0); vae=MotionVAE().cuda(); opt=torch.optim.AdamW(vae.parameters(),3e-4)
    for s in range(301):
        y,l=vae(q,m); opt.zero_grad(); l['loss'].backward(); opt.step()
        if s%50==0: print(name,s,{k:round(float(x),4) for k,x in l.items()}, 'yabs',y.abs().mean().item())
    mu,lv=vae.encode(q); print('mu std',mu.std().item(),'lv mean',lv.mean().item())
    yd=vae.decode(mu); print('det rec', (yd-q).abs().mean().item())
with torch.no_grad():
    e=(yd-q).abs()
    print('per joint', e.mean((0,1)).cpu().numpy().round(2))
    print('per frame mod4', [round(e[:,i::4].mean().item(),3) for i in range(4)])
    print('per time decile', [round(e[:,i*25:(i+1)*25].mean().item(),3) for i in range(10)])
    print('q std per joint', q.std((0,1)).cpu().numpy().round(2))
    print('window-mean baseline', (q-q.mean(1,keepdim=True)).abs().mean().item())
    x=q.transpose(1,2); h=x
    for i,l in enumerate(vae.enc): 
        h=l(h); print('enc',i,type(l).__name__,tuple(h.shape), round(h.std(-1).mean().item(),4))
    z=vae.encode(q)[0]; print('mu time-std', z.std(1).mean().item(), 'mu batch-std', z.std(0).mean().item())
