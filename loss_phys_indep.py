# loss_phys_indep.py (OpenPose-15)
import torch
import torch.nn.functional as F

# OpenPose 15 keypoints
# {'Nose':0, 'Neck':1, 'Rsho':2, 'Relb':3, 'Rwri':4, 'Lsho':5, 'Lelb':6, 'Lwri':7,
#  'Mhip':8, 'Rhip':9, 'Rkne':10, 'Rank':11, 'Lhip':12, 'Lkne':13, 'Lank':14}

BONES = [
    (1, 2), (2, 3), (3, 4),       # right arm
    (1, 5), (5, 6), (6, 7),       # left arm
    (1, 8),                       # neck to mid-hip
    (8, 9), (9,10), (10,11),      # right leg
    (8,12), (12,13), (13,14)      # left leg
]

BONE_L0 = torch.tensor([
    0.15, 0.25, 0.23,    # right arm
    0.15, 0.25, 0.23,    # left arm
    0.4,                # torso
    0.2, 0.42, 0.45,     # right leg
    0.2, 0.42, 0.45      # left leg
])

JOINT_LIMIT = {
    'knee_l': (0, 145),   # 12-13-14
    'knee_r': (0, 145),   # 9-10-11
    'elbow_l': (0, 150),  # 5-6-7
    'elbow_r': (0, 150),  # 2-3-4
}

def bone_length_loss(seq):
    pj, qi = zip(*BONES)
    v = seq[:,:,pj] - seq[:,:,qi]             # [B,T,B,3]
    bone_len = v.norm(dim=-1)                 # [B,T,B]
    diff = bone_len - BONE_L0.to(seq.device)
    return (diff**2).mean()

def joint_limit_penalty(seq):
    penalties = []

    hip, knee, ankle = 12,13,14
    v1 = seq[:,:,hip] - seq[:,:,knee]
    v2 = seq[:,:,ankle] - seq[:,:,knee]
    angle = torch.acos(F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999,0.9999)) * 180/3.1416
    l, h = JOINT_LIMIT['knee_l']
    penalties.append(F.relu(angle - h) + F.relu(l - angle))

    hip, knee, ankle = 9,10,11
    v1 = seq[:,:,hip] - seq[:,:,knee]
    v2 = seq[:,:,ankle] - seq[:,:,knee]
    angle = torch.acos(F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999,0.9999)) * 180/3.1416
    l, h = JOINT_LIMIT['knee_r']
    penalties.append(F.relu(angle - h) + F.relu(l - angle))

    a, b, c = 5,6,7
    v1 = seq[:,:,a] - seq[:,:,b]
    v2 = seq[:,:,c] - seq[:,:,b]
    angle = torch.acos(F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999,0.9999)) * 180/3.1416
    l, h = JOINT_LIMIT['elbow_l']
    penalties.append(F.relu(angle - h) + F.relu(l - angle))

    a, b, c = 2,3,4
    v1 = seq[:,:,a] - seq[:,:,b]
    v2 = seq[:,:,c] - seq[:,:,b]
    angle = torch.acos(F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999,0.9999)) * 180/3.1416
    l, h = JOINT_LIMIT['elbow_r']
    penalties.append(F.relu(angle - h) + F.relu(l - angle))

    return torch.stack(penalties).mean()

def momentum_smooth(seq):
    vel = seq[:,1:] - seq[:,:-1]           # [B,T-1,J,3]
    acc = vel[:,1:] - vel[:,:-1]           # [B,T-2,J,3]
    return acc.abs().mean()

def total_correlation(coef):
    x = coef.reshape(-1, coef.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / (x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag**2).mean()

def sparsity(coef):
    return coef.abs().mean()
