import sys
sys.path.insert(0,'/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES; print(len(G1_JOINT_NAMES), G1_JOINT_NAMES)
sys.path.insert(0,'/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/code')
from retarget_g1_ex002 import Retargeter
rt=Retargeter()
print('g1 njnt', rt.g.njnt, [rt.g.joint(i).name for i in range(rt.g.njnt)])
print('ex names', rt.names[:20])
for s in rt.sides: print(s['side'], 'upper', round(s['upper'],3), 'fore', round(s['fore'],3))
