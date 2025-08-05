# pvr_eval.py (NTU-25)
import torch
import torch.nn.functional as F
from loss_phys_indep import BONES, BONE_L0, JOINT_LIMIT

def compute_pvr(seq, bone_pairs=BONES, bone_l0=BONE_L0, joint_limit=JOINT_LIMIT, acc_thres=2.0):
    # seq: [B,T,J,3]
    B,T,J,_ = seq.shape
    device = seq.device

    pj, qi = zip(*bone_pairs)
    v = seq[:,:,pj] - seq[:,:,qi]          # [B,T,B,3]
    bone_len = v.norm(dim=-1)              # [B,T,B]
    delta = (bone_len - bone_l0.to(device)).abs() / (bone_l0.to(device) + 1e-4)
    v_bone = (delta > 0.1).float().mean().item()

    hip, knee, ankle = 17, 19, 21
    v1 = seq[:,:,hip] - seq[:,:,knee]
    v2 = seq[:,:,ankle] - seq[:,:,knee]
    cos = F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999, 0.9999)
    angle = torch.acos(cos) * 180 / 3.1416
    low, high = joint_limit['knee_r']
    mask_joint = (angle < low) | (angle > high)
    v_joint = mask_joint.float().mean().item()

    vel = seq[:,1:] - seq[:,:-1]
    acc = vel[:,1:] - vel[:,:-1]           # [B,T-2,J,3]
    acc_mag = acc.norm(dim=-1)             # [B,T-2,J]
    v_acc = (acc_mag > acc_thres).float().mean().item()

    return {
        'PVR_bone': v_bone,
        'PVR_joint': v_joint,
        'PVR_acc': v_acc
    }

