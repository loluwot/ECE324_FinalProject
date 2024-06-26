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

norm_dict = {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d}

def make_layers(cfg, input_shape=(0,0,0), shapes=None, batch_norm: bool = True, invert=False, running=True, config=None):
    
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
                layers += [conv2d, norm_dict[config.norm_type](v, track_running_stats=running), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] 
            cur_shape = (v, *cur_shape[1:])
    if invert:
        layers = layers[:-1] #remove last relu
    return nn.Sequential(*layers), cur_shape, shapes

class Autoencoder(nn.Module):
    def __init__(self, cfg, full_cfg):
        super().__init__()
        self.encoder, out_shape, shapes = make_layers(cfg.autoenc, running=cfg.fast_gradient, input_shape=(3, full_cfg['im_size'], full_cfg['im_size']), config=cfg)
        self.decoder, in_shape, _ = make_layers(cfg.autoenc, invert=True, shapes=shapes, running=cfg.fast_gradient, input_shape=out_shape, config=cfg)
        assert in_shape == (3, full_cfg['im_size'], full_cfg['im_size'])
    def encoder_alpha(self, x, y, alpha=0.):
        if x.ndim == 5:
            return alpha * self.encoder(x.squeeze(dim=1))[:, None] + (1 - alpha) * self.encoder(y.squeeze(dim=1))[:, None]
        x_z, y_z = self.encoder(x), self.encoder(y)
        alpha = alpha[(slice(None, None),) + (None,)*(x_z.ndim - 1)]
        res = alpha * x_z + (1 - alpha) * y_z
        return res
    def forward(self, x, y, alpha=0.):
        return self.decoder(self.encoder_alpha(x, y, alpha))
        # return self.decoder(self.encoder(y))
    
class Critic(nn.Module):
    def __init__(self, cfg, full_cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            make_layers(cfg.critic, input_shape=(3, full_cfg['im_size'], full_cfg['im_size']), config=cfg)[0],
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
        res = self.autoenc.decoder(alpha * x_z + (1 - alpha) * y_z)
        loss += self.cfg.autoenc_lambda * self.critic(res).square().mean()
        # return loss, alpha, res, autoenc_y
        return (loss, dict()), y, res, alpha, autoenc_y

    def forward_critic(self, y, res, alpha, autoenc_y):
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha.squeeze())
        res = self.critic(self.cfg.critic_gamma * y + (1 - self.cfg.critic_gamma) * autoenc_y.detach())#.abs().mean()
        loss += self.critic_criterion(res, torch.zeros_like(res))
        return (loss, dict())


class ACAIMod(GenericAAI):
    def __init__(self, cfg, full_cfg):
        super().__init__(cfg, full_cfg)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        x_z, y_z = self.autoenc.encoder(x), self.autoenc.encoder(y)
        alpha = 0.5*torch.rand(bs, dtype=x.dtype, device=x.device)[(slice(None, None),) + (None,)*(x_z.ndim - 1)]
        autoenc_y = self.autoenc.decoder(y_z)
        recon_loss = self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in [autoenc_y, y]]).mean()
        
        def function(alpha, x_z, y_z):
            res = self.autoenc.decoder((alpha * x_z + (1 - alpha) * y_z)[None])            
            return res.flatten(), (res.squeeze(),)
        
        jacobian, (res,) = torch.func.vmap(torch.func.jacfwd(function, has_aux=True))(alpha, x_z, y_z)
        smoothness_loss = (self.cfg.smooth_lambda * jacobian.square()).mean()
        # smoothness_loss = 0.

        adv_loss = self.cfg.autoenc_lambda * self.critic(res).square().mean()        
        loss = recon_loss + smoothness_loss + adv_loss

        return (loss, dict({'recon': recon_loss, 'smoothness': smoothness_loss, 'adv': adv_loss})), y, res, alpha, autoenc_y

    def forward_critic(self, y, res, alpha, autoenc_y):
        loss = self.critic_criterion(self.critic(res.detach()).squeeze(), alpha.squeeze())
        res = self.critic(self.cfg.critic_gamma * y + (1 - self.cfg.critic_gamma) * autoenc_y.detach())#.abs().mean()
        loss += self.critic_criterion(res, torch.zeros_like(res))
        return (loss, dict())
   
from functools import reduce
from einops import rearrange

class AEAI(GenericAAI):
    def __init__(self, cfg, full_cfg):
        super().__init__(cfg, full_cfg)

    def forward_autoenc(self, x, y):
        bs = x.shape[0]
        alpha = torch.rand(bs).to(x) # B

        # RECON LOSS
        autoenc_y = self.autoenc.decoder(self.autoenc.encoder(y))
        autoenc_x = self.autoenc.decoder(self.autoenc.encoder(x))
        recon_loss = reduce(lambda x, y: x + y, [self.autoenc_criterion(*[self.autoenc_unnorm(z) for z in tup]).mean() for tup in zip([autoenc_y, autoenc_x], [y, x])])        
        res_z = self.autoenc.encoder_alpha(x, y, alpha) # B C H W
        res = self.autoenc.decoder(res_z)

        # ADVERSARIAL LOSS
        if self.cfg.discrim_loss == 'bce':
            adv_loss = -self.cfg.autoenc_lambda * F.logsigmoid(self.critic(res)).mean()
        if self.cfg.discrim_loss == 'wasserstein':
            adv_loss = (self.cfg.autoenc_lambda * self.critic(res)).mean()

        # CYCLE CONSISTENCY
        cycle_loss = self.cfg.cycle_lambda * (self.autoenc.encoder(res) - res_z).square().mean()
        loss = recon_loss + adv_loss + cycle_loss
        return (loss, {'recon':recon_loss, 'adv':adv_loss, 'cycle':cycle_loss}), res, alpha, x, y

    def forward_critic(self, res, alpha, x, y):
        res = res.detach()
        res.requires_grad_()
        negative_samples = self.critic(res).squeeze()
        positive_samples = self.critic(torch.cat([x, y], axis=0)).squeeze()

        loss_components = dict()

        if self.cfg.discrim_loss == 'bce':
            predictions = torch.cat([negative_samples, positive_samples], axis=0)
            targets = torch.cat([torch.zeros_like(negative_samples), torch.ones_like(positive_samples)], axis=0)
            loss = F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=torch.tensor([0.5]).to(x))
        if self.cfg.discrim_loss == 'wasserstein':
            loss = positive_samples.mean() - negative_samples.mean()
            if self.cfg.wlambda is not None:
                gradients = torch.autograd.grad(outputs=negative_samples, 
                                                inputs=res, 
                                                grad_outputs=torch.ones_like(negative_samples), 
                                                create_graph=True,
                                                retain_graph=True)[0]
                gp = self.cfg.wlambda * (gradients.view(res.shape[0], -1).norm(2, dim=1) - 1).square().mean()
                loss_components['gradient_penalty'] = gp
                loss += gp
        return loss, loss_components


