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

    def forward(self, x, y, alpha=0.):
        return self.decoder(alpha * self.encoder(x) + (1 - alpha)*self.encoder(y))
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
        loss = self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in [autoenc_y, y]]) 
        res = self.autoenc(x, y, alpha)
        loss += self.cfg.autoenc_lambda * self.critic(res).square().mean()
        return loss, alpha, res, autoenc_y

    def forward_critic(self, x, y, res, alpha, autoenc_y):
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha.squeeze())
        res = self.critic(self.cfg.critic_gamma * y + (1 - self.cfg.critic_gamma) * autoenc_y.detach())#.abs().mean()
        loss += self.critic_criterion(res, torch.zeros_like(res))
        return loss
   


    