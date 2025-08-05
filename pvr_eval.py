# pvr_eval.py (OpenPose-15)
import torch
import torch.nn.functional as F
from loss_phys_indep import BONES, BONE_L0, JOINT_LIMIT

def compute_pvr(seq, bone_pairs=BONES, bone_l0=BONE_L0, joint_limit=JOINT_LIMIT, acc_thres=0.01):
    B,T,J,_ = seq.shape
    device = seq.device

    pj, qi = zip(*bone_pairs)
    v = seq[:,:,pj] - seq[:,:,qi]          # [B,T,B,3]
    bone_len = v.norm(dim=-1)              # [B,T,B]
    delta = (bone_len - bone_l0.to(device)).abs() / (bone_l0.to(device) + 1e-4)
    v_bone = (delta > 0.3).float().mean().item()

    hip, knee, ankle = 12,13,14
    v1 = seq[:,:,hip] - seq[:,:,knee]
    v2 = seq[:,:,ankle] - seq[:,:,knee]
    cos = F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999, 0.9999)
    angle = torch.acos(cos) * 180 / 3.1416
    low, high = joint_limit['knee_l']
    v_joint = ((angle < low) | (angle > high)).float().mean().item()

    vel = seq[:,1:] - seq[:,:-1]
    acc = vel[:,1:] - vel[:,:-1]           # [B,T-2,J,3]
    acc_mag = acc.norm(dim=-1)             # [B,T-2,J]
    v_acc = (acc_mag > acc_thres).float().mean().item()
    print("elbow_r angle (deg):", angle[0].detach().cpu().numpy())
    print("bone delta:", delta[0].detach().cpu().numpy())
    print("acc_mag:", acc_mag[0].detach().cpu().numpy())
    return {
        'PVR_bone': v_bone,
        'PVR_joint': v_joint,
        'PVR_acc': v_acc
    }
