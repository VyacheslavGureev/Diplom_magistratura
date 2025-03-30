import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import models.hyperparams as hyperparams
import models.nn_model as nn_model


class EncapsulatedModel:
    def __init__(self):
        # Создание модели с нуля
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM, 1, 2, hyperparams.BATCH_SIZE)
        self.model.to(self.device)

        self.ema = EMA(self.model, self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparams.LR, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
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
