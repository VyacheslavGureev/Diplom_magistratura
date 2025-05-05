import torch
import torch.nn as nn
import models.nn_model
import models.nn_model_adaptive
import models.diffusion_processes as diff_proc
import models.hyperparams as hyperparams


# TODO: Предварительно всё правильно

class MyCombineModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        adapt_config = model_config["ADAPT"]
        unet_config = model_config["DDPM"]
        self.adaptive_block = models.nn_model_adaptive.MyAdaptUNet(adapt_config)
        # for param in self.adaptive_block.parameters():
        #     param.requires_grad = False
        self.unet_block = models.nn_model.MyUNet(unet_config)
        # self.net_log = nn.Sequential(
        #     nn.Conv2d(hyperparams.CHANNELS, hyperparams.CHANNELS, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=hyperparams.CHANNELS, num_channels=hyperparams.CHANNELS, affine=True),
        #     nn.SiLU(),
        #     nn.Conv2d(hyperparams.CHANNELS, hyperparams.CHANNELS, kernel_size=3, padding=1),
        # )

    def forward(self, x0, text_emb, attn_mask, sheduler):
        device = x0.device
        # self.adaptive_block.eval()
        log_D, mu = self.adaptive_block(text_emb, attn_mask)
        # log_D = torch.zeros_like(x0, device = device)
        # log_D = torch.ones_like(x0, device = device)
        # mu = torch.zeros_like(x0, device = device)
        # log_D_proj = self.net_log(log_D)
        # log_D = torch.zeros_like(x0)
        # mu = torch.zeros_like(x0)
        std = torch.exp(0.5 * log_D)
        e_adapt = std * torch.randn_like(std, device=device) + mu
        t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
        time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)
        xt, e_adapt_added = diff_proc.forward_diffusion(x0, t, sheduler, e_adapt)

        e_adapt_pred = self.unet_block(xt, text_emb, time_emb, attn_mask, log_D)
        return e_adapt_added, e_adapt_pred

        # e_adapt = self.adaptive_block(noise, text_emb, None, attn_mask)
        # e_adapt = e_adapt - mu  # центрируем адаптивный шум, чтобы модель учила полезные паттерны и не отвлекалась на среднее
        # t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=noise.device)  # случайные шаги t
        # time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)
        # xt, e_adapt_added = diff_proc.forward_diffusion(x0, t, sheduler, e_adapt)
        # e_adapt_pred = self.unet_block(xt, text_emb, time_emb, attn_mask)
        # return e_adapt_added, e_adapt_pred
