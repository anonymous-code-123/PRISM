from agent.base_agent import BaseAgent
from functional.motion import get_foot_vel
import torch
from loss_phys_indep import (
    bone_length_loss,
    joint_limit_penalty,
    momentum_smooth,
    total_correlation,
    sparsity,
    BONES,
    BONE_L0,
    JOINT_LIMIT
)
from pvr_eval import compute_pvr


class Agent2x(BaseAgent):
    def __init__(self, config, net):
        super(Agent2x, self).__init__(config, net)
        self.inputs_name = ['input1', 'input2', 'input12', 'input21']
        self.targets_name = ['target1', 'target2', 'target12', 'target21']

    def forward(self, data):
        inputs = [data[name].to(self.device) for name in self.inputs_name]
        targets = [data[name].to(self.device) for name in self.targets_name]

        # update loss metric
        losses = {}

        if self.use_triplet:
            outputs, motionvecs, staticvecs = self.net.cross_with_triplet(*inputs)
            losses['m_tpl1'] = self.triplet_weight * self.tripletloss(motionvecs[2], motionvecs[0], motionvecs[1])
            losses['m_tpl2'] = self.triplet_weight * self.tripletloss(motionvecs[3], motionvecs[1], motionvecs[0])
            losses['b_tpl1'] = self.triplet_weight * self.tripletloss(staticvecs[2], staticvecs[0], staticvecs[1])
            losses['b_tpl2'] = self.triplet_weight * self.tripletloss(staticvecs[3], staticvecs[1], staticvecs[0])
        else:
            outputs = self.net.cross(inputs[0], inputs[1])

        for i, target in enumerate(targets):
            losses['rec' + self.targets_name[i][6:]] = self.mse(outputs[i], target)

        if self.use_footvel_loss:
            losses['foot_vel'] = 0
            for i, target in enumerate(targets):
                losses['foot_vel'] += self.footvel_loss_weight * self.mse(get_foot_vel(outputs[i], self.foot_idx),
                                                                          get_foot_vel(target, self.foot_idx))

        
#        print(outputs[0].shape) [B: 64, J: 34, T: 64]
        B,JC,T = outputs[0].shape
        seq_hat = outputs[0].permute(0, 2, 1).reshape(B, T, -1, 2)
        coef = self.net.mot_encoder.last_coef  # [B,T,K]

        λ1, λ2, λ3, λ4 = 0.001, 0.1, 100.0, 1e-1
        losses['phys_bone']  = λ1 * bone_length_loss(seq_hat)
        losses['phys_joint'] = λ2 * joint_limit_penalty(seq_hat)
        losses['phys_mom']   = λ2 * momentum_smooth(seq_hat)
        losses['indep_tc']   = λ3 * total_correlation(coef)
        losses['indep_sparse'] = λ4 * sparsity(coef)
        
        print({
            "L_rec1": losses["rec1"].item(),
            "L_rec12": losses["rec12"].item(),
            "L_bone": losses['phys_bone'].item(),
            "L_joint": losses['phys_joint'].item(),
            "L_mom":  losses['phys_mom'].item(),
            "L_tc":  losses['indep_tc'].item(),
            "L_sp": losses['indep_sparse'].item()
        })        
        
        outputs_dict = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output12": outputs[2],
            "output21": outputs[3],
            "recon_seq": seq_hat.detach(),
        }

        return outputs_dict, losses

