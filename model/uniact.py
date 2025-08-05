import torch
import torch.nn as nn
from .backbone_unik import *
from .networks import AutoEncoder2x
from .pdan import *
import torch.nn.functional as F
from common import config

#config.initialize()

def get_autoencoder(config):
    assert config.name is not None
    if config.name == 'skeleton':
        return AutoEncoder2x(config.mot_en_channels, config.body_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)
    elif config.name == 'view':
        return AutoEncoder2x(config.mot_en_channels, config.view_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)   #FIXME: max/avg
    else:
        return AutoEncoder3x(config.mot_en_channels, config.body_en_channels,
                             config.view_en_channels, config.de_channels)


class Action(nn.Module):
    def __init__(self, num_class=60, num_joints=25, num_person=2, tau=1, num_heads=3, in_channels=2, weights = 'no'):
        super(Action, self).__init__()

        self.unik = UNIK(num_joints, num_person, tau, num_heads, in_channels)

        self.pdan = Dilated_TCN(inter_dim=1024, input_dim=512, num_classes=num_class)
        self.projector = nn.Linear(320, 512)


    def forward(self, x, A, mask_new=1):
        N, C, T, V, M = x.size()
        max_len = 0

        out = self.unik(x)# N MC512 T A: n mc320 t
#        out = torch.cat([out, A.permute(0, 1, 2)], dim=1)
        out = out + self.projector(A.permute(0, 2, 1)).permute(0, 2, 1)
  #     out = self.projection2(out.permute(0,2,1)).permute(0,2,1)
    
        out = self.pdan(out, mask_new)

        return out

class UniAction(nn.Module):
    def __init__(self, num_class=60, num_joints=25, num_person=2, tau=1, num_heads=3, in_channels=2, drop_out=0, backbone_fixed=False, weights='no', task='classification' ):
        super(UniAction, self).__init__()
        self.backbone_fixed = backbone_fixed
        self.task = task
        #load model
        config.initializecom()
        self.decomposer = get_autoencoder(config)
        self.decomposer.load_state_dict(torch.load('train_log_lac/exp_view/model/model_epoch300.pth'))
        self.decomposer.eval()
        print('decomposer weights loaded')
        for l, module in self.decomposer._modules.items():
            print('fixed layers:', l)
            for p in module.parameters():
                p.requires_grad=False

        self.action = Action(num_class, num_joints, num_person, tau, num_heads, in_channels=in_channels)
        
      #  self.fc = nn.Linear(512, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        
        if weights!='no':
            print('pre-training: ', weights)
            weights = torch.load(weights)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                v.cuda(0)] for k, v in weights.items()])

            keys = list(weights.keys())
            try:
                self.action.load_state_dict(weights)
            except:
                state = self.action.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.action.load_state_dict(state)

        # freeze the backbone for Linear evaluation
        if self.backbone_fixed:
            for l, module in self.action._modules.items():
                if l !='fc':
                    print('fixed layers:', l)
                    for p in module.parameters():
                        p.requires_grad=False

    def get_pad(self):
        return int(self.pad)


    @torch.no_grad()
    def get_primitives(self, x):
        N, C, T, V, M = x.size()
        #N, C, T, V, M -> N, VC, T
        x_lac = x[:,:,:,[0,2,6,8,10,5,7,9,4,12,14,16,11,13,15],:] # 17->15
        x_lac = x_lac.permute(0, 3, 1, 2, 4).contiguous().view(N*M, 15*C, T)
        m, b, alp_m, alp_b1 = self.decomposer.mot_encoder(x_lac)
        out = F.upsample(m.permute(0,1,2), T, mode='nearest') #* mask[:, 0:1, :] # N C T
        return out.view(N, -1, T) #N C256 T

    def forward(self, x, y=None, mask=1):
        N, C, T, V, M = x.size()
        if self.task == 'retargeting':
            return self.decomposer.transfer(x, y)
        if self.task == 'composition':
            return self.decomposer.compose(x, y)
        if self.task == 'classification':
            A = self.get_primitives(x)
            x = self.action(x, A, mask)# n, cls, t
            x = x.mean(2)
            return x
        if self.task == 'detection':
            A = self.get_primitives(x)
            x = self.action(x, A, mask)
            return x
