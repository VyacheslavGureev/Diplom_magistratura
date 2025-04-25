import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import models.hyperparams as hyperparams
import models.nn_model as nn_model
import models.nn_model_adaptive as nn_model_adapt
import models.nn_model_combine as nn_model_combine
import models.utils as utils

# TODO: Предварительно всё правильно

# Данные про модель в одном месте. Решение в стиле "тяжёлого" ООП, через setup
# Базовый класс
class ModelInOnePlace:
    # Общий метод инициализации
    def __init__(self, device):
        self.device = device
        print(self.device)
        self.history = {0: {'train_loss': math.inf, 'val_loss': math.inf}}

    # Кастомный метод инициализации (переопределяем в классах-наследниках)
    # Вызываем после создания объекта и передаём аргументы, предназначенные именно для конкретного объекта
    def setup(self, *args, **kwargs):
        pass


class EncapsulatedModel(ModelInOnePlace):
    def setup(self, unet_config_file):
        unet_config = utils.load_json(unet_config_file)
        self.model = nn_model.MyUNet(unet_config)
        self.model.to(self.device)
        # self.ema = EMA(self.model, self.device)  # ??? (пока не работаю с ema, в будущем доработаю)
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
        # можно ещё добавлять KL-loss (как в оригинале ddpm, но её можно опустить) или VLB-loss
        self.criterion = nn.MSELoss()


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


class EncapsulatedModelAdaptive(ModelInOnePlace):
    def setup(self, unet_config_file, adaptive_config_file, sheduler):
        adaptive_config = utils.load_json(adaptive_config_file)
        unet_config = utils.load_json(unet_config_file)
        self.model = nn_model_combine.MyCombineModel(adaptive_config, unet_config, sheduler)
        self.model.to(self.device)
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
        self.criterion = adapt_loss
        self.model.apply(self.init_weights_unit_var)

    def init_weights_unit_var(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


# Данные про датасет в одном месте. Решение в стиле простенького ООП
class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


class EncapsulatedDataloadersTextDescr(EncapsulatedDataloaders):
    def __init__(self, train, val, test, text_descr):
        super().__init__(train, val, test)
        self.text_descr = text_descr

# class EMA:
#     def __init__(self, model, device):  # для mnist хорошо 0,99
#         self.device = device
#         decay = 0.99
#         self.model = model
#         self.decay = decay
#         self.ema_model = copy.deepcopy(model)  # Создаём копию модели
#         self.ema_model.to(self.device)
#         self.ema_model.eval()  # В режиме валидации (не обучаем) (ema модель всегда должна быть в режиме eval!!!)
#         for param in self.ema_model.parameters():
#             param.requires_grad_(False)  # Отключаем градиенты
#
#     def update(self):
#         with torch.no_grad():
#             for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
#                 ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)  # EMA обновление
