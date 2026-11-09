import json, torch, torch.nn as nn
from scmg.data import View, WindowDataset, collate
from scmg import vae as V
D='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st=json.load(open(D+'/STATS_v36_1.json'))
v=View(split='val', stats=st, mmap=True)
b=next(iter(torch.utils.data.DataLoader(WindowDataset(v,256),32,collate_fn=collate))); q=b['q'].cuda()
print('smooth', (q[:,1:]-q[:,:-1]).abs().mean().item(), 'abs', q.abs().mean().item())
class Simple(nn.Module):
    def __init__(s):
        super().__init__()
        s.e=nn.Sequential(nn.Conv1d(34,256,3,padding=1),nn.SiLU(),nn.Conv1d(256,256,4,stride=4),nn.SiLU(),nn.Conv1d(256,16,1))
        s.d=nn.Sequential(nn.ConvTranspose1d(16,256,4,stride=4),nn.SiLU(),nn.Conv1d(256,256,3,padding=1),nn.SiLU(),nn.Conv1d(256,34,3,padding=1))
    def forward(s,x): return s.d(s.e(x.transpose(1,2))).transpose(1,2)
torch.manual_seed(0); m=Simple().cuda(); opt=torch.optim.AdamW(m.parameters(),1e-3)
for i in range(300):
    y=m(q); l=(y-q).abs().mean(); opt.zero_grad(); l.backward(); opt.step()
    if i%100==0 or i==299: print('simple',i,l.item())
# MotionVAE pieces: does decode(encode) depend on the input at all after training? check grad wrt input
m2=V.MotionVAE().cuda(); qq=q.clone().requires_grad_(True); mu,_=m2.encode(qq); y=m2.decode(mu); y.sum().backward()
print('grad wrt input', qq.grad.abs().mean().item())
x=q.transpose(1,2); h=m2.enc[0](x); print('first conv out', tuple(h.shape))
c=V.CConv(34,8,3); print('lp rp', c.lp, c.rp, 'padding attr', c.padding)
