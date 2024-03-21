import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import pydantic
from pydantic import BaseModel

from modules import ACAI

from typing import List, Union

class LitModelCfg(BaseModel):
    
    ##### TRAINING PARAMS

    lr : float = 1e-4

    ##### ARCHITECTURE
    
    autoenc: List[Union[str, int]] = [3, 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
    critic: List[Union[str, int]] = [3, 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
    critic_dropout : float = 0.1
    critic_act : bool = False
    critic_psize : int = 7
    critic_hidden : int = 4096

    autoenc_lambda : float = 0.
    critic_gamma : float = 1.
    

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model = ACAI(config)

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        cfg = self.config
        cuda = torch.cuda.is_available()
        autoenc_opt = torch.optim.Adam(self.model.autoenc.parameters(), lr=cfg.lr, fused=cuda)
        critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=cfg.lr, fused=cuda)
        return autoenc_opt, critic_opt
    
    def training_step(self, batch, batch_idx):
        a_opt, c_opt = self.optimizers()
        x_b, y_b = torch.split(batch, 2)

        loss, alpha, res, autoenc_y = self.model.forward_autoenc(x_b, y_b)
        
        a_opt.zero_grad()
        self.manual_backward(loss)
        a_opt.step()

        c_loss = self.model.forward_critic(x_b, y_b, res, alpha, autoenc_y)
        c_opt.zero_grad()
        self.manual_backward(c_loss)
        c_opt.step()

        self.log_dict({"autoenc_loss": loss, "critic_loss": c_loss}, prog_bar=True)

