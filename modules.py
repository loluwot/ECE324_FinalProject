import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, vgg19
        
import pydantic
from pydantic import BaseModel

from typing import List, Union
from datamodule import unnormalize

import lpips

def make_layers(cfg, batch_norm: bool = True, invert=False):
    layers = []

    pool_l = lambda : nn.MaxPool2d(kernel_size=2, stride=2) if not invert else nn.UpsamplingNearest2d(scale_factor=2)
    cfg = cfg[::1 - 2*invert]

    in_channels = int(cfg[0])
    for i, v in enumerate(cfg[1:]):
        if v == "M":
            layers += [pool_l()]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] 
            in_channels = v

    if invert:
        layers = layers[:-1]
    return nn.Sequential(*layers)

class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = make_layers(cfg.autoenc)
        self.decoder = make_layers(cfg.autoenc, invert=True)

    def encoder_alpha(self, x, y, alpha=0.):
        if x.ndim == 5:
            return alpha * self.encoder(x.squeeze(dim=1))[:, None] + (1 - alpha) * self.encoder(y.squeeze(dim=1))[:, None]
        return alpha * self.encoder(x) + (1 - alpha) * self.encoder(y)
        
    def forward(self, x, y, alpha=0.):
        return self.decoder(self.encoder_alpha(x, y, alpha))
        # return self.decoder(self.encoder(y))
    
class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            make_layers(cfg.critic),
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
    def __init__(self, cfg):
        super().__init__()
        self.autoenc = Autoencoder(cfg)
        self.critic = Critic(cfg)
        self.cfg = cfg

    def forward_autoenc(self, x, y):
        raise NotImplementedError
    
    def forward_critic(self, *args):
        raise NotImplementedError

class ACAI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.autoenc = Autoencoder(cfg)
        self.critic = Critic(cfg)
        self.cfg = cfg
        self.critic_criterion = loss_dict.get(self.cfg.critic_loss, nn.MSELoss())
        self.autoenc_criterion = loss_dict.get(self.cfg.autoenc_loss, nn.MSELoss())
        self.autoenc_unnorm = (lambda x: x) if not loss_norm[self.cfg.autoenc_loss] else (lambda x: unnormalize(x, tensor=True) * 2 - 1)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        alpha = 0.5*torch.rand(bs, dtype=x.dtype, device=x.device)[(slice(None, None),) + (None,)*(x.ndim - 1)]
        autoenc_y = self.autoenc(x, y, torch.zeros_like(alpha))
        loss = self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in [autoenc_y, y]]).mean()
        res = self.autoenc(x, y, alpha)
        loss += self.cfg.autoenc_lambda * self.critic(res).square().mean()
        # return loss, alpha, res, autoenc_y
        return loss, y, res, alpha, autoenc_y

    def forward_critic(self, y, res, alpha, autoenc_y):
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha.squeeze())
        res = self.critic(self.cfg.critic_gamma * y + (1 - self.cfg.critic_gamma) * autoenc_y.detach())#.abs().mean()
        loss += self.critic_criterion(res, torch.zeros_like(res))
        return loss
   
from functools import reduce
from einops import rearrange

class AEAI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.autoenc = Autoencoder(cfg)
        self.critic = Critic(cfg)
        self.cfg = cfg
        self.critic_criterion = loss_dict.get(self.cfg.critic_loss, nn.MSELoss())
        self.autoenc_criterion = loss_dict.get(self.cfg.autoenc_loss, nn.MSELoss())
        self.autoenc_unnorm = (lambda x: x) if not loss_norm[self.cfg.autoenc_loss] else (lambda x: unnormalize(x, tensor=True) * 2 - 1)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        alpha = ((torch.arange(self.cfg.M + 1) / self.cfg.M) * torch.ones((bs, 1)))[(Ellipsis,) + (None,)*3].to(x) # B M 1 1 1

        # RECON LOSS
        autoenc_y = self.autoenc(x, y, alpha[:, 0])
        autoenc_x = self.autoenc(y, x, alpha[:, 0])
        loss = reduce(lambda x, y: x + y, [self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in tup]).mean() for tup in zip([autoenc_y, autoenc_x], [y, x])])
        # print('RECON LOSS', loss)
        # ADVERSARIAL LOSS
        res_z = self.autoenc.encoder_alpha(x[:, None], y[:, None], alpha) # B M C H W
        res_z_merged = rearrange(res_z, 'b m c h w -> (b m) c h w')
        res_merged = self.autoenc.decoder(res_z_merged)
        # res = rearrange(res_merged, '(b m) c h w -> b m c h w', m=self.cfg.M + 1)
        # print('ADV LOSS', self.cfg.autoenc_lambda * (self.critic(res_merged).abs()+1e-5).log().mean())
        loss -= self.cfg.autoenc_lambda * (self.critic(res_merged).abs() + 1e-5).log().mean()
        
        # CYCLE CONSISTENCY
        # print('CYCLE CONSIST', self.cfg.cycle_lambda * (self.autoenc.encoder(self.autoenc.decoder(res_z_merged)) - res_z_merged).square().mean())
        loss += self.cfg.cycle_lambda * (self.autoenc.encoder(self.autoenc.decoder(res_z_merged)) - res_z_merged).square().mean()

        # SMOOTHNESS
        # print('SMOOTHNESS', self.cfg.smooth_lambda * torch.gradient(res_merged, spacing=(alpha[0].squeeze(),), dim=1)[0].square().mean())
        loss += self.cfg.smooth_lambda * torch.gradient(res_merged, spacing=(alpha[0].squeeze(),), dim=1)[0].square().mean()

        return loss, res_merged, alpha, autoenc_y

    def forward_critic(self, res, alpha, autoenc_y):
        alpha_mod = (0.5 - alpha).abs() * 2
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha_mod.flatten())
        return loss


