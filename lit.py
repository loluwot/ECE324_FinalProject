import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import pydantic
from pydantic import BaseModel

from modules import ACAI

from typing import List, Union
from torchvision.utils import make_grid
from datamodule import unnormalize

import wandb

class LitModelCfg(BaseModel):
    
    ##### TRAINING PARAMS

    lr : float = 1e-4

    ##### ARCHITECTURE
    
    autoenc: List[Union[str, int]] = [3, 64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512]
    critic: List[Union[str, int]] = [3, 64, "M", 128, "M", 256, 256, "M", 256, "M", 256]
    critic_dropout : float = 0.
    critic_act : bool = False
    critic_psize : int = 3
    critic_hidden : int = 1024

    autoenc_lambda : float = 0.5
    critic_gamma : float = 0.2

    critic_loss : str = 'l1'
    
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
        self.model = ACAI(self.config)

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        cfg = self.config
        cuda = torch.cuda.is_available()
        autoenc_opt = torch.optim.Adam(self.model.autoenc.parameters(), lr=cfg.lr)#, fused=cuda)
        critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=cfg.lr)#, fused=cuda)
        return autoenc_opt, critic_opt
    
    def training_step(self, batch, batch_idx):

        # torch.cuda.empty_cache()

        a_opt, c_opt = self.optimizers()
        x_b, y_b = torch.tensor_split(batch.to(torch.float32), 2)

        loss, alpha, res, autoenc_y = self.model.forward_autoenc(x_b, y_b)
        a_opt.zero_grad()
        self.manual_backward(loss)
        a_opt.step()

        c_loss = self.model.forward_critic(x_b, y_b, res, alpha, autoenc_y)
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