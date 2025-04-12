import torch
import torchvision.utils as vutils
import numpy as np
import imageio
import os
from tqdm import tqdm

import models.hyperparams as hyperparams


class NoiseSheduler():
    def __init__(self, T, type, device):
        self.device = device
        if type == 'linear':
            # Максимально стандартное линейное расписание
            beta_start = 1e-4
            beta_end = 0.02
            self.create_diff_sheduler_linear(T, beta_start, beta_end)
        elif type == 'cosine':
            s = 0.008
            self.create_diff_scheduler_cosine(T, s)

    def create_diff_sheduler_linear(self, num_time_steps, beta_start, beta_end):
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
        self.a = 1 - self.b  # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.a_bar = self.a_bar.to(self.device)
        # self.show_shedule()

    def create_diff_scheduler_cosine(self, T, s):
        t = torch.linspace(0, T, steps=T + 1)  # Временные шаги 0...T
        f_t = torch.cos((t / T + s) / (1 + s) * (np.pi / 2)) ** 2  # Косинусная формула
        alpha_bar = f_t / f_t[0]  # Нормируем, чтобы α̅_T = 1
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]  # Вычисляем b_t
        self.b = torch.clip(beta, 0.0001, 0.999)  # Ограничиваем b_t
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.a_bar = self.a_bar.to(self.device)
        # self.show_shedule()

    def show_shedule(self):
        print(f"Min beta: {self.b.min().item()}")
        print(f"Max beta: {self.b.max().item()}")
        import matplotlib.pyplot as plt
        plt.plot(self.b.cpu().numpy(), label="Beta")
        plt.xlabel("Timestep")
        plt.ylabel("Beta")
        plt.title("Cosine Schedule for Beta")
        plt.legend()
        plt.show()
        plt.pause(3600)


def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    """
    Transform a scalar time-step into a vector representation of size t_emb_dim.

    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)

    :return tensor of size -> (B, t_emb_dim)
    """
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."

    factor = 2 * torch.arange(start=0,
                              end=t_emb_dim // 2,
                              dtype=torch.float32,
                              device=time_steps.device
                              ) / (t_emb_dim)

    factor = 10000 ** factor

    t_emb = time_steps[:, None]  # B -> (B, 1)
    t_emb = t_emb / factor  # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B , t_emb_dim)

    return t_emb


# --- Определение форвардного процесса (зашумление) ---
def forward_diffusion(x0: torch.Tensor, t: torch.Tensor, sheduler: NoiseSheduler, noise=None):
    """ Добавляет стандартный гауссовский шум к изображению """
    # В общем случае noise может быть не только гауссовским
    if noise is None:
        noise = torch.randn_like(x0, requires_grad=False, device=x0.device)
    at = sheduler.a_bar[t][:, None, None, None]
    xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
    return xt, noise


# Функция для reverse diffusion
def reverse_diffusion(model, text_embedding, attn_mask, sheduler: NoiseSheduler):
    # Инициализация случайного шума (начало процесса)
    orig_channels = 1
    x_t = torch.randn(hyperparams.BATCH_SIZE, orig_channels, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE,
                      device=next(model.parameters()).device)  # (B, C, H, W)
    t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int, device=next(model.parameters()).device)
    t_tensor = t_tensor.unsqueeze(1)
    t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
    output_dir = "trained/denoising/"
    # Удаляем все файлы в папке
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Запускаем процесс reverse diffusion
    model.eval()
    with torch.no_grad():
        i = 0
        for step in tqdm(range(hyperparams.T - 1, -1, -1), colour='white'):
            time_embedding = get_time_embedding(t_tensor[step], hyperparams.TIME_EMB_DIM)
            predicted_noise = model(x_t, text_embedding, time_embedding, attn_mask)
            # guidance_scale = 0.5  # Усиление текстового сигнала
            # predicted_noise_uncond = model(x_t, None, t_i, None) # Безусловное предсказание
            # predicted_noise_cond = model(x_t, text_embedding, t_i, attn_mask)  # Условное предсказание
            # predicted_noise = guidance_scale * predicted_noise_cond + (1 - guidance_scale) * predicted_noise_uncond
            x_t = (1 / torch.sqrt(sheduler.a[step])) * (
                    x_t - ((1 - sheduler.a[step]) / (torch.sqrt(1 - sheduler.a_bar[step]))) * predicted_noise)
            # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
            # Например, можно добавить немного шума обратно с каждым шагом:
            # if step > 0:  # Добавляем случайный шум на всех шагах, кроме последнего
            #     noise = torch.randn_like(x_t, device=next(model.parameters()).device) * (1 - sheduler.a[step]).sqrt() * 0.1
            #     x_t += noise
            if i % 20 == 0:
                # hyperparams.VIZ_STEP = True
                vutils.save_image(x_t, f"trained/denoising/step_{step}.png", normalize=True)
            i += 1
    images = []
    for step in sorted(os.listdir("trained/denoising"), key=lambda x: int(x.split("_")[1].split(".")[0]),
                       reverse=True):
        images.append(imageio.imread(os.path.join("trained/denoising", step)))
    imageio.mimsave("trained/denoising/denoising_process.gif", images, duration=0.3)  # 0.3 секунды на кадр
    return x_t
