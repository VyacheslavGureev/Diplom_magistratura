import torch
import torch.nn as nn
import models.nn_model
import models.nn_model_adaptive
import models.diffusion_processes as diff_proc
import models.hyperparams as hyperparams


class MyCombineModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        adapt_config = model_config["ADAPT"]
        unet_config = model_config["DDPM"]
        self.adaptive_block = models.nn_model_adaptive.MyAdaptUNet(adapt_config)
        self.unet_block = models.nn_model.MyUNet(unet_config)
        self.act_logD = nn.Tanh()
        self.act_mu = nn.Tanh()

    def forward(self, x0, text_emb, attn_mask, sheduler):
        device = x0.device
        log_D, mu = self.adaptive_block(text_emb, attn_mask)
        log_D = self.act_logD(log_D)
        std = torch.exp(0.5 * log_D)
        e_adapt = std * torch.randn_like(std, device=device)
        t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)
        time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)
        xt, e_adapt_added = diff_proc.forward_diffusion(x0, t, sheduler, e_adapt)
        e_adapt_pred = self.unet_block(xt, text_emb, time_emb, attn_mask, log_D)
        return e_adapt_added, e_adapt_pred
