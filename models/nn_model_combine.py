import torch
import torch.nn as nn
import models.nn_model
import models.nn_model_adaptive
import models.diffusion_processes as diff_proc
import models.hyperparams as hyperparams


# TODO: Продолжить проверку и написание
class MyCombineModel(nn.Module):
    def __init__(self, adapt_config, unet_config):
        super().__init__()
        self.adaptive_block = models.nn_model_adaptive.MyAdaptUNet(adapt_config)
        self.unet_block = models.nn_model.MyUNet(unet_config)
        # self.sheduler = sheduler

    def forward(self, noise, x0, text_emb, time_emb, attn_mask, t, mu, sheduler):
        e_adapt = self.adaptive_block(noise, text_emb, None, attn_mask)
        e_adapt = e_adapt - mu
        xt, e_adapt_added = diff_proc.forward_diffusion(x0, t, sheduler, e_adapt)
        e_adapt_pred = self.unet_block(xt, text_emb, time_emb, attn_mask)
        return e_adapt_added, e_adapt_pred