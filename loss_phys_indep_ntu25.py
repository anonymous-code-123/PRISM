# loss_phys_indep.py
import torch
import torch.nn.functional as F

BONES = [
    (0,1), (1,20), (20,4), (4,5),      # spine to right arm
    (20,8), (8,9),                      # spine to left arm
    (0,16), (16,18),                    # root to right leg
    (0,12), (12,14),                    # root to left leg
    (9,10), (5,6),                      # lower arms
    (14,15), (18,19)                    # lower legs
]


BONE_L0 = torch.tensor([
    0.30, 0.25, 0.27, 0.24, 0.24, 0.23, 0.45, 0.42, 0.45, 0.24, 0.23, 0.43, 0.43
])  # shape=[len(BONES)]

JOINT_LIMIT = {
    'knee_r': (0, 145),
    'elbow_r': (0, 150)
}

def bone_length_loss(seq):
    # seq: [B,T,J,3]
    pj, qi = zip(*BONES)
    v = seq[:,:,pj] - seq[:,:,qi]             # [B,T,B,3]
    bone_len = v.norm(dim=-1)                 # [B,T,B]
    diff = bone_len - BONE_L0.to(seq.device)
    return (diff**2).mean()

def joint_limit_penalty(seq):
    # 以右膝为例（17:hip, 19:knee, 21:ankle）
    hip, knee, ankle = 17, 19, 21
    v1 = seq[:,:,hip] - seq[:,:,knee]
    v2 = seq[:,:,ankle] - seq[:,:,knee]
    cos = F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999, 0.9999)
    angle = torch.acos(cos) * 180 / 3.1416
    low, high = JOINT_LIMIT['knee_r']
    hinge = F.relu(angle - high) + F.relu(low - angle)
    return hinge.mean()

def momentum_smooth(seq):
    vel = seq[:,1:] - seq[:,:-1]           # [B,T-1,J,3]
    acc = vel[:,1:] - vel[:,:-1]           # [B,T-2,J,3]
    return acc.abs().mean()

def total_correlation(coef):
    # coef: [B,T,K] → flatten → [N,K]
    x = coef.reshape(-1, coef.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / (x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag**2).mean()

def sparsity(coef):
    return coef.abs().mean()

