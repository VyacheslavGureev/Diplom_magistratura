import torch
import torch.nn as nn
import models.nn_model
import models.nn_model_adaptive
import models.diffusion_processes as diff_proc
import models.hyperparams as hyperparams


# TODO: Предварительно всё правильно

class MyCombineModel(nn.Module):
    def __init__(self, adapt_config, unet_config):
        super().__init__()
        self.adaptive_block = models.nn_model_adaptive.MyAdaptUNet(adapt_config)
        self.unet_block = models.nn_model.MyUNet(unet_config)

    def forward(self, noise, x0, text_emb, attn_mask, mu, sheduler):
        e_adapt = self.adaptive_block(noise, text_emb, None, attn_mask)
        e_adapt = e_adapt - mu  # центрируем адаптивный шум, чтобы модель учила полезные паттерны и не отвлекалась на среднее
        t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=noise.device)  # случайные шаги t
        time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)
        xt, e_adapt_added = diff_proc.forward_diffusion(x0, t, sheduler, e_adapt)
        e_adapt_pred = self.unet_block(xt, text_emb, time_emb, attn_mask)
        return e_adapt_added, e_adapt_pred
