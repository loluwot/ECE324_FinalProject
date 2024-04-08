import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, vgg19
        
import pydantic
from pydantic import BaseModel

from typing import List, Union
from datamodule import unnormalize

import lpips
from einops.layers.torch import Rearrange

def make_layers(cfg, input_shape=(0,0,0), shapes=None, batch_norm: bool = True, invert=False, running=True):
    
    cur_shape = input_shape
    layers = []
    shapes = shapes or []
    pool_l = nn.MaxPool2d(kernel_size=2, stride=2) if not invert else nn.UpsamplingNearest2d(scale_factor=2)
    cfg = cfg[::1 - 2*invert]

    # in_channels = int(cfg[0])
    for i, v in enumerate(cfg[1:]):
        if v == "M":
            layers += [pool_l]
            cur_shape = (cur_shape[0], *[x*2 if invert else x//2 for x in cur_shape[1:]])
        elif str(v).startswith('L'):
            if len(cur_shape) > 1:
                layers += [nn.Flatten()]
                shapes += [cur_shape]
                cur_shape = (reduce(lambda x, y: x*y, cur_shape),)
            layers += [nn.Linear(cur_shape[0], int(v[1:])), nn.ReLU(inplace=True)]
            cur_shape = (int(v[1:]),)
        else:
            if len(cur_shape) == 1:
                next_shape = shapes.pop(-1)
                layers += [nn.Linear(cur_shape[0], reduce(lambda x, y: x*y, next_shape)), nn.Unflatten(-1, next_shape)]
                cur_shape = next_shape
            v = int(v)
            conv2d = nn.Conv2d(cur_shape[0], v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, track_running_stats=running), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] 
            cur_shape = (v, *cur_shape[1:])
    if invert:
        layers = layers[:-1] #remove last relu
    return nn.Sequential(*layers), cur_shape, shapes

class Autoencoder(nn.Module):
    def __init__(self, cfg, full_cfg):
        super().__init__()
        self.encoder, out_shape, shapes = make_layers(cfg.autoenc, running=cfg.fast_gradient, input_shape=(3, full_cfg['im_size'], full_cfg['im_size']))
        self.decoder, in_shape, _ = make_layers(cfg.autoenc, invert=True, shapes=shapes, running=cfg.fast_gradient, input_shape=out_shape)
        assert in_shape == (3, full_cfg['im_size'], full_cfg['im_size'])
    def encoder_alpha(self, x, y, alpha=0.):
        if x.ndim == 5:
            return alpha * self.encoder(x.squeeze(dim=1))[:, None] + (1 - alpha) * self.encoder(y.squeeze(dim=1))[:, None]
        res = alpha * self.encoder(x) + (1 - alpha) * self.encoder(y)
        print(x.shape, self.encoder(x).shape)
        return res
    def forward(self, x, y, alpha=0.):
        return self.decoder(self.encoder_alpha(x, y, alpha))
        # return self.decoder(self.encoder(y))
    
class Critic(nn.Module):
    def __init__(self, cfg, full_cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            make_layers(cfg.critic)[0],
            nn.AdaptiveAvgPool2d((cfg.critic_psize, cfg.critic_psize)),
            nn.Flatten(),
            nn.Linear(int(cfg.critic[-1]) * cfg.critic_psize ** 2, cfg.critic_hidden),
            nn.ReLU(True),
            # nn.Dropout(p=cfg.critic_dropout),
            # nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
            # nn.ReLU(True),
            nn.Dropout(p=cfg.critic_dropout),
            nn.Linear(cfg.critic_hidden, 1),
        )
        self.act = nn.Identity() if cfg.critic_act is False else nn.Sigmoid()
    
    def forward(self, x):
        return self.act(self.classifier(x))#, 0., 0.5)

loss_dict = {'mse': nn.MSELoss(), 'l1': nn.L1Loss(), 'alex': lpips.LPIPS(net='alex')}
loss_norm = {'mse': False, 'l1': False, 'alex': True}

class GenericAAI(nn.Module):
    def __init__(self, cfg, full_cfg):
        super().__init__()
        self.autoenc = Autoencoder(cfg, full_cfg)
        self.critic = Critic(cfg, full_cfg)
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.critic_criterion = loss_dict.get(self.cfg.critic_loss, nn.MSELoss())
        self.autoenc_criterion = loss_dict.get(self.cfg.autoenc_loss, nn.MSELoss())
        self.autoenc_unnorm = (lambda x: x) if not loss_norm[self.cfg.autoenc_loss] else (lambda x: unnormalize(x, tensor=True) * 2 - 1)

    def forward_autoenc(self, x, y):
        raise NotImplementedError
    
    def forward_critic(self, *args):
        raise NotImplementedError

class ACAI(GenericAAI):
    def __init__(self, cfg, full_cfg):
        super().__init__(cfg, full_cfg)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        x_z, y_z = self.autoenc.encoder(x), self.autoenc.encoder(y)
        alpha = 0.5*torch.rand(bs, dtype=x.dtype, device=x.device)[(slice(None, None),) + (None,)*(x_z.ndim - 1)]
        autoenc_y = self.autoenc.decoder(y_z)
        loss = self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in [autoenc_y, y]]).mean()
        res = alpha * x_z + (1 - alpha) * y_z
        loss += self.cfg.autoenc_lambda * self.critic(res).square().mean()
        # return loss, alpha, res, autoenc_y
        return (loss, dict()), y, res, alpha, autoenc_y

    def forward_critic(self, y, res, alpha, autoenc_y):
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha.squeeze())
        res = self.critic(self.cfg.critic_gamma * y + (1 - self.cfg.critic_gamma) * autoenc_y.detach())#.abs().mean()
        loss += self.critic_criterion(res, torch.zeros_like(res))
        return loss
   
from functools import reduce
from einops import rearrange

class AEAI(GenericAAI):
    def __init__(self, cfg, full_cfg):
        super().__init__(cfg, full_cfg)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        # M = self.cfg.M
        # alpha = ((torch.arange(M + 1) / M) * torch.ones((bs, 1)))[(Ellipsis,) + (None,)*3].to(x) # B M 1 1 1
        alpha = torch.rand(bs, 1, 1, 1).to(x) # B 1 1 1

        # RECON LOSS
        autoenc_y = self.autoenc(x, y, torch.zeros_like(alpha))
        autoenc_x = self.autoenc(y, x, torch.zeros_like(alpha))
        recon_loss = reduce(lambda x, y: x + y, [self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in tup]).mean() for tup in zip([autoenc_y, autoenc_x], [y, x])])        
        # SMOOTHNESS
        # if self.cfg.fast_gradient:
        #     res_z = self.autoenc.encoder_alpha(x, y, alpha) # B C H W
        #     # res_z_merged = rearrange(res_z, 'b m c h w -> (b m) c h w')
        #     res = self.autoenc.decoder(res_z)
        #     loss += self.cfg.smooth_lambda * torch.gradient(rearrange(res, '(b m) c h w -> b m c h w', m=M+1), spacing=(alpha[0].squeeze(),), dim=1)[0].square().mean()
        # else:
            ## SLOW BUT ACCURATE GRADIENT CALC ###
        def function(alpha, x, y):
            res_z = self.autoenc.encoder_alpha(x[None], y[None], alpha[None])#.squeeze()
            res = self.autoenc.decoder(res_z)            
            return res.flatten(), (res.squeeze(), res_z.squeeze())
        
        # x_exp, y_exp = [xx.repeat_interleave((M + 1), dim=0) for xx in (x, y)]
        jacobian, (res, res_z) = torch.func.vmap(torch.func.jacfwd(function, has_aux=True))(alpha, x, y)
        # print(torch.gradient(rearrange(res_merged, '(b m) c h w -> b m c h w', m=M+1), spacing=(alpha[0].squeeze(),), dim=1)[0].shape)
        # print(jacobian - torch.gradient(rearrange(res_merged, '(b m) c h w -> b m c h w', m=M+1), spacing=(alpha[0].squeeze(),), dim=1)[0])
        smoothness_loss = (self.cfg.smooth_lambda * jacobian.square()).mean()

        # ADVERSARIAL LOSS
        adv_loss = -self.cfg.autoenc_lambda * F.logsigmoid(self.critic(res)).mean()
        
        # CYCLE CONSISTENCY
        cycle_loss = self.cfg.cycle_lambda * (self.autoenc.encoder(self.autoenc.decoder(res_z)) - res_z).square().mean()
        loss = recon_loss + smoothness_loss + adv_loss + cycle_loss
        return (loss, {'recon':recon_loss, 'smoothness':smoothness_loss, 'adv':adv_loss, 'cycle':cycle_loss}), res, alpha, x, y

    def forward_critic(self, res, alpha, x, y):
        # alpha_mod = (0.5 - alpha).abs() * 2
        # loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha_mod.flatten())
        # t = torch.ones_like(alpha)
        # t[:, 1:-1] = 0.
        loss = F.binary_cross_entropy_with_logits((pred := self.critic(res.detach()).squeeze()), torch.zeros_like(pred))
        loss += F.binary_cross_entropy_with_logits((pred := self.critic(torch.cat([x, y], axis=0))), torch.ones_like(pred))
        return loss


