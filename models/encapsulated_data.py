import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import models.hyperparams as hyperparams
import models.nn_model as nn_model


# class CustomLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, predicted_noise, target_noise):
#         # Например, взвешенный MSE
#         loss = (predicted_noise - target_noise) ** 2
#         weights = torch.exp(-torch.abs(target_noise))  # Больше весов для сложных примеров
#         weighted_loss = (weights * loss).mean()
#         return weighted_loss

# def custom_loss(predict, target, base_loss=torch.nn.MSELoss()):
#     delta = torch.abs(predict - target)  # Ошибка предсказания
#     weights = torch.exp(delta)  # Больший вес для сложных примеров
#     return (weights * base_loss(predict, target)).mean()

# def entropy(image):
#     """
#     Вычисляет энтропию Шеннона для изображения.
#     Ожидает вход размерности (C, H, W) и вычисляет энтропию по всем пикселям.
#     """
#     image = image.flatten()  # Преобразуем в 1D для гистограммы
#     hist = torch.histc(image, bins=256, min=0, max=1)  # Гистограмма значений пикселей
#     prob = hist / hist.sum()  # Нормируем до вероятностей
#     entropy_value = -torch.sum(prob * torch.log(prob + 1e-8))  # Энтропия
#     return entropy_value


# def custom_loss(predicted_noise, target_noise, images, base_loss=torch.nn.MSELoss(reduction="none")):
#     """
#     Кастомная функция ошибки, учитывающая энтропию входных изображений.
#
#     predicted_noise: (B, C, H, W) - предсказанный шум
#     target_noise: (B, C, H, W) - целевой шум
#     images: (B, C, H, W) - исходные изображения
#     base_loss: функция ошибки (по умолчанию MSELoss)
#
#     Возвращает усредненный взвешенный лосс.
#     """
#     batch_size = images.shape[0]
#
#     # Вычисляем энтропию для каждого изображения в батче
#     weights = torch.tensor([entropy(images[i]) for i in range(batch_size)],
#                            dtype=torch.float32,
#                            device=predicted_noise.device)
#
#     # Вычисляем MSE Loss между предсказанным и целевым шумом
#     loss = base_loss(predicted_noise, target_noise)  # (B, C, H, W)
#
#     # Усредняем по пространственным измерениям (C, H, W) -> (B,)
#     loss = loss.mean(dim=(1, 2, 3))  # Теперь shape (B,)
#
#     # Домножаем на вес энтропии и усредняем по батчу
#     return (weights * loss).mean()


# def custom_loss(predict, target, base_loss=torch.nn.MSELoss()):
#     delta = torch.abs(predict - target)  # Ошибка предсказания
#     weights = torch.exp(delta)  # Больший вес для сложных примеров
#     return (weights * base_loss(predict, target)).mean()


class EncapsulatedModel:
    def __init__(self):
        # Создание модели с нуля
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM, 1, 2, hyperparams.BATCH_SIZE)
        self.model.to(self.device)

        self.ema = EMA(self.model, self.device)

        cross_attn_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "cross_attn" in name:  # Указываем название слоев
                cross_attn_params.append(param)  # Отдельный список для Cross-Attention
            else:
                other_params.append(param)  # Остальные параметры
        self.optimizer = optim.AdamW([
            {"params": other_params, "lr": hyperparams.LR},  # Обычный LR
            {"params": cross_attn_params, "lr": hyperparams.LR * 0.3}  # Уменьшенный LR для Cross-Attention
        ], weight_decay=1e-4)

        self.criterion = nn.MSELoss()
        # self.criterion = custom_loss
        self.history = {0: {'train_loss': math.inf, 'val_loss': math.inf}}


class EMA:
    def __init__(self, model, device):  # для mnist хорошо 0,99
        self.device = device
        decay = 0.99
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model)  # Создаём копию модели
        self.ema_model.to(self.device)
        self.ema_model.eval()  # В режиме валидации (не обучаем) (ema модель всегда должна быть в режиме eval!!!)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)  # Отключаем градиенты

    def update(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)  # EMA обновление


class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        # Данные про датасет в одном месте
        self.train = train
        self.val = val
        self.test = test
