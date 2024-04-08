import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import pydantic
from pydantic import BaseModel

from modules import ACAI, AEAI

from typing import List, Union
from torchvision.utils import make_grid
from datamodule import unnormalize

import wandb

model_dict = {'acai': ACAI, 'aeai': AEAI}
optimizer_dict = {'adam': torch.optim.Adam, 'radam': torch.optim.RAdam}

class LitModelCfg(BaseModel):
    
    ##### TRAINING PARAMS

    optimizer : str = "adam"
    lr_autoenc : float = 1e-4
    lr_critic : float = 1e-4

    ##### ARCHITECTURE
    
    architecture : str = 'acai'
    autoenc: List[Union[str, int]] = [3, 64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512]
    critic: List[Union[str, int]] = [3, 64, "M", 128, "M", 256, 256, "M", 256, "M", 256]
    critic_dropout : float = 0.
    critic_act : bool = False
    critic_psize : int = 3
    critic_hidden : int = 1024

    M : int = 5
    autoenc_lambda : float = 0.5
    cycle_lambda : float = 0.5
    smooth_lambda : float = 0.5

    critic_gamma : float = 0.2

    critic_loss : str = 'l1'
    autoenc_loss : str = 'alex'
    
    fast_gradient : bool = True

    ##### VIS #######
    visualize_n_batches : int = 1
    check_samples_every_n_epochs : int = 10
    visualize_n_samples : int = 5

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.full_config = config
        self.config = LitModelCfg.parse_obj(config)
        # self.model = ACAI(self.config)
        self.model = model_dict[self.config.architecture](self.config)

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        cfg = self.config
        cuda = torch.cuda.is_available()
        autoenc_opt = optimizer_dict[cfg.optimizer](self.model.autoenc.parameters(), lr=cfg.lr_autoenc)#, fused=cuda)
        critic_opt = optimizer_dict[cfg.optimizer](self.model.critic.parameters(), lr=cfg.lr_critic)#, fused=cuda)
        return autoenc_opt, critic_opt
    
    def training_step(self, batch, batch_idx):

        # torch.cuda.empty_cache()

        a_opt, c_opt = self.optimizers()
        x_b, y_b = torch.tensor_split(batch, 2)

        loss, *args = self.model.forward_autoenc(x_b, y_b)
        a_opt.zero_grad()
        self.manual_backward(loss)
        a_opt.step()

        c_loss = self.model.forward_critic(*args)
        c_opt.zero_grad()
        self.manual_backward(c_loss)
        c_opt.step()

        self.log_dict({"autoenc_loss": loss, "critic_loss": c_loss}, prog_bar=True)
        if ((self.current_epoch + 1) % self.config.check_samples_every_n_epochs == 0) and (batch_idx < self.config.visualize_n_batches):
            self._visualize_results(x_b, y_b)

    @torch.no_grad()
    def _visualize_results(self, x_b, y_b):
        N = self.config.visualize_n_samples
        samples = x_b[[0]], y_b[[0]]
        true_alpha = torch.linspace(1, 0, N).to(x_b)
        results = self.model.autoenc(*[x.repeat(N, 1, 1, 1) for x in samples], true_alpha[(slice(None, None),) + (None,)*(x_b.ndim - 1)])
        
        wandb.log(
            {
                "alpha": wandb.Table(
                    data = torch.stack([0.5 - torch.abs(0.5 - true_alpha), self.model.critic(results).squeeze()], axis=-1).cpu().numpy(), 
                    columns = ['target', 'prediction']
                )
            }
        )
        image_grid = make_grid([unnormalize(x, tensor=True) for x in [samples[0][0]] + list(results.unbind(dim=0)) + [samples[1][0]]], nrow=N + 2)
        wandb.log({'result':wandb.Image(image_grid)})