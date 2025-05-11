import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import torchvision.utils as vutils
import numpy as np
import imageio
import os
from tqdm import tqdm

import models.hyperparams as hyperparams


class NoiseSheduler():
    def __init__(self, T, type, device):
        # super().__init__()
        # Обычно bt возрастает, at убывает, at_bar убывает быстро
        self.device = device
        self.T = T
        if type == 'linear':
            # Максимально стандартное линейное расписание
            beta_start = 1e-4
            beta_end = 0.02
            # beta_start = 0.1
            # beta_end = 0.2
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


class NoiseShedulerAdapt(NoiseSheduler):
    def __init__(self, T, type, device):
        super().__init__(T, type, device)

    # D это скаляр
    # Пока что такой пересчёт кэфов справедлив только для линейного расписания,
    # в будущем возможно добавлю пересчёт и для для косинусового расписания
    def update_coeffs(self, D):
        # D = torch.tensor(0.7)
        D = D.to(self.device)
        b_max = torch.max(self.b)
        b_min = torch.min(self.b)

        C = 0.000040358
        eps = C ** (1 / self.T)
        opora = (1 / D) * (1 - eps)

        if b_max >= opora:
            b_max_new = opora * 0.99
            s = b_max / b_min
            b_min_new = b_max_new / s
            # b_min_new = b_max_new * 0.1
            self.b = torch.linspace(b_min_new, b_max_new, self.T).to(self.device)
        self.a = 1 - self.b * D
        self.a_bar = torch.cumprod(self.a, dim=0)


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
def forward_diffusion(x0: torch.Tensor, t: torch.Tensor, sheduler, noise=None):
    """ Добавляет стандартный гауссовский шум к изображению """
    # В общем случае noise может быть не только гауссовским
    if noise is None:
        noise = torch.randn_like(x0, requires_grad=False, device=x0.device)
    at_bar = sheduler.a_bar[t][:, None, None, None]
    # если кэфы правильно пересчитаны с дисперсией шума D (любого), то замкнутая формула не меняется
    xt = torch.sqrt(at_bar) * x0 + torch.sqrt(1 - at_bar) * noise
    return xt, noise
