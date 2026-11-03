import json, os, sys, numpy as np, torch
sys.path.insert(0, "/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src"); sys.path.insert(0, "/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/code")
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
import mujoco
from retarget_g1_ex002 import kinetic_model
from robogesture_cfg import COLLISION_XML_PATH
G="/mnt/cpfs/zbl-cpfs-new/USERS/zehan/g1-cross-20261103"; E=G+"/g1eval"; keep=json.load(open(G+"/subset/KEEP_CI.json"))
for d in ("ref","wav","pred/gt","pred/echog-native","pred/echog-upper","pred/robogesture-upper"): os.makedirs(E+"/"+d, exist_ok=True)
gm = kinetic_model(COLLISION_XML_PATH)
UP = [10 + G1_JOINT_NAMES.index(n) for n in G1_JOINT_NAMES if n.startswith(("waist", "left_sh", "left_el", "left_wr", "right_sh", "right_el", "right_wr"))]
def save(p, x): torch.save(dict(robot_repr=torch.from_numpy(np.ascontiguousarray(x, np.float32)), fps=30.0, real_num_frames=len(x)), p)
with open(E+"/split.txt","w") as f: f.write("\n".join(f"dev_{c}" for c in keep)+"\n")
for c in keep:
    ref = np.load(f"{G}/floor/g1_{c}.npz")["robot_repr"]; N = len(ref); s=f"dev_{c}"
    save(f"{E}/ref/{s}.pt", ref); save(f"{E}/pred/gt/{s}.pt", ref)
    w=f"{E}/wav/{s}.wav"; 
    if not os.path.exists(w): os.symlink(f"/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/audio/clip_{c}.wav", w)
    eg = np.load(f"/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/eval-dev/official/g1_{c}.npz")["robot_repr"][:N]; save(f"{E}/pred/echog-native/{s}.pt", eg)
    u = ref.copy(); u[:len(eg), UP] = eg[:, UP]; save(f"{E}/pred/echog-upper/{s}.pt", u)
    rq = np.load(f"/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/eval-dev/raw/g1_{c}.npz")["qpos"]; r = ref.copy(); n = min(N, len(rq))
    for j, name in enumerate(G1_JOINT_NAMES):
        if 10 + j in UP: r[:n, 10 + j] = rq[:n, 7 + int(gm.joint(name + "_joint").qposadr[0])]
    save(f"{E}/pred/robogesture-upper/{s}.pt", r)
print("built", len(keep))
