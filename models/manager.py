import math
import time

import os
import torchvision.utils as vutils
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
from torchviz import make_dot
import hiddenlayer as hl

import models.hyperparams as hyperparams
import models.dataset_creator as dc
import models.model_adaptive as encapsulated_data
import models.nn_model as nn_model
import models.diffusion_processes as diff_proc
import models.utils as utils


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Сбрасываем patience
        else:
            self.counter += 1  # Увеличиваем patience

        return self.counter >= self.patience  # True = остановка


# TODO: Продолжить проверку и написание

# Универсальный класс для главных взаимодействий с моделью
class ModelManager():

    def __init__(self):
        pass

    def train_model(self, e_model,
                    e_loader, epochs, sheduler):
        plateau_scheduler = lr_scheduler.ReduceLROnPlateau(e_model.optimizer, mode='min', factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=4)
        for epoch in range(epochs):
            running_train_loss = e_model.training_model(e_loader, sheduler)
            # running_val_loss = e_model.validating_model(e_loader, sheduler)
            running_val_loss = 0

            avg_loss_train = running_train_loss / len(e_loader.train)
            avg_loss_val = running_val_loss / len(e_loader.val)
            print(
                f"Epoch {epoch + 1}, Train Loss: {avg_loss_train}, Val Loss: {avg_loss_val}")
            hist = e_model.history
            last_epoch = max(hist.keys())
            last_epoch += 1
            hist[last_epoch] = {}
            hist[last_epoch]['train_loss'] = running_train_loss / len(e_loader.train)
            hist[last_epoch]['val_loss'] = running_val_loss / len(e_loader.val)
            e_model.history = hist
            if early_stopping(avg_loss_val):
                print("Ранняя остановка! Обучение завершено.")
                break  # Прерываем обучение
            plateau_scheduler.step(avg_loss_val)  # Дополнительно уменьшает, если застряли

    def viz_my_model(self, e_model):
        pass
        # Предположим, у тебя есть модель
        # model = e_model.model
        # model.eval()
        # device = next(model.parameters()).device
        #
        # # Создаём dummy-входы (примеры — адаптируй под себя!)
        # dummy_x = torch.randn(16, 1, 32, 32, requires_grad=True, device=next(model.parameters()).device)
        # dummy_txt_emb = torch.randn(16, 50, 512, requires_grad=True, device=next(model.parameters()).device)
        # dummy_time_emb = torch.randn(16, 256, requires_grad=True, device=next(model.parameters()).device)
        # dummy_attn_mask = torch.randn(16, 50, requires_grad=True, device=next(model.parameters()).device)
        #
        # # Предположим твоя модель
        # model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM, 1, 1, hyperparams.BATCH_SIZE,
        #                              hyperparams.UNET_CONFIG)
        # model = model.to(device)
        # model.eval()
        # # Заворачиваем модель
        # wrapped = nn_model.WrappedModel(model, dummy_txt_emb, dummy_time_emb, dummy_attn_mask)
        # wrapped = wrapped.to(device)
        # # Строим граф
        # graph = hl.build_graph(wrapped, dummy_x)
        # graph.save("unet_graph", format="png")

        # Torchviz не заработал, у меня нет времени разбираться с ним!
        # import onnx
        # print(onnx.__version__)
        # # Экспорт модели
        # torch.onnx.export(
        #     model,
        #     (dummy_x, dummy_txt_emb, dummy_time_emb, dummy_attn_mask),  # <- кортеж входов
        #     "multi_input_model.onnx",
        #     input_names=["images", "text_embs", "t", "attn_mask"],
        #     output_names=["output"],
        #     opset_version=13,
        #     dynamic_axes={
        #         "images": {0: "batch_size"},
        #         "text_embs": {0: "batch_size"},
        #         "t": {0: "batch_size"},
        #         "attn_mask": {0: "batch_size"},
        #         "output": {0: "batch_size"},
        #     }
        # )

        # Torchviz не заработал, у меня нет времени разбираться с ним!
        # # Прогон через модель
        # output = model(dummy_x, dummy_txt_emb, dummy_time_emb, dummy_attn_mask)
        # # Создаём граф
        # dot = make_dot(output, params=dict(model.named_parameters()))
        # # Сохраняем как PDF или PNG
        # dot.format = 'pdf'  # можно 'png'
        # dot.render('unet_graph')
