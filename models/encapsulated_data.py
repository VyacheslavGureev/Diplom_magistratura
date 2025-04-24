import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import models.hyperparams as hyperparams
import models.nn_model as nn_model
import models.nn_model_adaptive as nn_model_adapt
import models.utils as utils


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

# Данные про модель в одном месте
class EncapsulatedModel:
    def __init__(self, unet_config_file, device):
        # Создание модели с нуля
        self.device = device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.unet_config = utils.load_json(unet_config_file)

        # self.unet_config = {'TEXT_EMB_DIM' : hyperparams.TEXT_EMB_DIM, 'TIME_EMB_DIM' : hyperparams.TIME_EMB_DIM,
        #                'BATCH_SIZE' : hyperparams.BATCH_SIZE, 'ORIG_C' : 1,
        #                'DOWN':
        #                    [{'in_C': 16, 'out_C': 32, 'SA': False},
        #                     {'in_C': 32, 'out_C': 64, 'SA': True},
        #                     {'in_C': 64, 'out_C': 128, 'SA': False}],
        #                'BOTTLENECK': [{'in_C': 128, 'out_C': 128}],
        #                'UP': [{'in_C': 128, 'out_C': 64, 'sc_C': 64, 'SA': False, 'CA': False},
        #                       {'in_C': 64 + 64, 'out_C': 32, 'sc_C': 32, 'SA': True, 'CA': True}]}

        self.model = nn_model.MyUNet(self.unet_config)
        self.model.to(self.device)

        self.ema = EMA(self.model, self.device)  # ??? (пока не работаю с ema, в будущем доработаю)

        cross_attn_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "cross_attn" in name:  # Указываем название слоев
                cross_attn_params.append(param)  # Отдельный список для Cross-Attention
            else:
                other_params.append(param)  # Остальные параметры
        self.optimizer = optim.AdamW([
            {"params": other_params, "lr": hyperparams.LR},  # Обычный LR
            {"params": cross_attn_params, "lr": hyperparams.LR}  # Уменьшенный LR для Cross-Attention
        ], weight_decay=1e-4)

        self.criterion = nn.MSELoss()
        # можно ещё добавлять KL-loss (как в оригинале ddpm, но её можно опустить) или VLB-loss
        # self.criterion = custom_loss
        self.history = {0: {'train_loss': math.inf, 'val_loss': math.inf}}


def kl_divergence(mu, logvar):
    return (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).mean()


def adapt_loss(e, e_a, e_a_pred, mu, D, mse=torch.nn.MSELoss()):
    lam_1 = 0.5
    lam_2 = 0.5
    lam_3 = 1
    lam_4 = 0.1
    logvar = torch.log(D.clamp(min=1e-8))
    L = lam_1 * mse(e, e_a) + lam_2 * (
            mse(torch.fft.fft2(e).real, torch.fft.fft2(e_a).real) + mse(torch.fft.fft2(e).imag,
                                                                        torch.fft.fft2(e_a).imag)) + lam_3 * mse(
        e_a_pred,
        e_a) + lam_4 * kl_divergence(
        mu, logvar)
    return L


class EncapsulatedModelAdaptive(EncapsulatedModel):
    def __init__(self, unet_config_file, adaptive_config_file, device):
        super().__init__(unet_config_file, device)
        self.adaptive_config = utils.load_json(adaptive_config_file)

        self.adapt_model = nn_model_adapt.MyAdaptUNet(self.adaptive_config)
        self.adapt_model.to(self.device)

        cross_attn_params = []
        other_params = []
        for name, param in self.adapt_model.named_parameters():
            if "cross_attn" in name:  # Указываем название слоев
                cross_attn_params.append(param)  # Отдельный список для Cross-Attention
            else:
                other_params.append(param)  # Остальные параметры
        self.optimizer.add_param_group({
            "params": other_params,
            "lr": hyperparams.LR  # обычный LR
        })
        self.optimizer.add_param_group({
            "params": cross_attn_params,
            "lr": hyperparams.LR  # можно уменьшить, если надо
        })
        self.criterion_adapt = adapt_loss

        self.adapt_model.apply(self.init_weights_unit_var)

    def init_weights_unit_var(self, module):
        try:
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        except:
            pass


# Данные про датасет в одном месте
class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


class EncapsulatedDataloadersTextDescr(EncapsulatedDataloaders):
    def __init__(self, train, val, test, text_descr):
        super().__init__(train, val, test)
        self.text_descr = text_descr


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
