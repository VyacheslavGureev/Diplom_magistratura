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
import models.encapsulated_data as encapsulated_data
import models.nn_model as nn_model
import models.diffusion_processes as diff_proc


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


# class DiffusionReverseProcess:
#     r"""
#
#     Reverse Process class as described in the
#     paper "Denoising Diffusion Probabilistic Models"
#
#     """
#
#     def __init__(self,
#                  num_time_steps=1000,
#                  beta_start=1e-4,
#                  beta_end=0.02
#                  ):
#
#         # Precomputing beta, alpha, and alpha_bar for all t's.
#         self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
#         self.a = 1 - self.b  # a -> alpha
#         self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar
#
#     def sample_prev_timestep(self, xt, noise_pred, t):
#
#         r""" Sample x_(t-1) given x_t and noise predicted
#              by model.
#
#              :param xt: Image tensor at timestep t of shape -> B x C x H x W
#              :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
#              :param t: Current time step
#
#         """
#
#         # Original Image Prediction at timestep t
#         x0 = xt - (torch.sqrt(1 - self.a_bar.to(xt.device)[t]) * noise_pred)
#         x0 = x0 / torch.sqrt(self.a_bar.to(xt.device)[t])
#         x0 = torch.clamp(x0, -1., 1.)
#
#         # mean of x_(t-1)
#         mean = (xt - ((1 - self.a.to(xt.device)[t]) * noise_pred) / (torch.sqrt(1 - self.a_bar.to(xt.device)[t])))
#         mean = mean / (torch.sqrt(self.a.to(xt.device)[t]))
#
#         # only return mean
#         if t == 0:
#             return mean, x0
#
#         else:
#             variance = (1 - self.a_bar.to(xt.device)[t - 1]) / (1 - self.a_bar.to(xt.device)[t])
#             variance = variance * self.b.to(xt.device)[t]
#             sigma = variance ** 0.5
#             z = torch.randn(xt.shape).to(xt.device)
#
#             return mean + sigma * z, x0

# Универсальный класс для главных взаимодействий с моделью
class ModelManager():

    def __init__(self):
        pass

    # --- Создание модели ---
    def create_model(self, device):
        return encapsulated_data.EncapsulatedModel(device)

    def create_dataloaders(self, dataset, train_size_percent, val_size_percent):
        # Разделяем датасеты
        train_size = int(train_size_percent * len(dataset))
        val_size = int(val_size_percent * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                                [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
                                  collate_fn=dc.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
                                collate_fn=dc.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
                                 collate_fn=dc.collate_fn)  # Тестовый датасет можно не перемешивать
        e_loader = encapsulated_data.EncapsulatedDataloaders(train_loader, val_loader, test_loader)
        return e_loader

    def train_model(self, e_model: encapsulated_data.EncapsulatedModel,
                    e_loader: encapsulated_data.EncapsulatedDataloaders, epochs, sheduler):
        plateau_scheduler = lr_scheduler.ReduceLROnPlateau(e_model.optimizer, mode='min', factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(epochs):
            running_train_loss = self.training_model(e_model, e_loader, sheduler)
            running_val_loss = self.validating_model(e_model, e_loader, sheduler)
            # running_val_loss = 0

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

    def training_model(self, e_model: encapsulated_data.EncapsulatedModel,
                       e_loader: encapsulated_data.EncapsulatedDataloaders, sheduler):
        print("Тренировка")
        model = e_model.model
        ema = e_model.ema
        device = e_model.device
        optimizer = e_model.optimizer
        criterion = e_model.criterion
        train_loader = e_loader.train

        ema.ema_model.eval()
        model.train()  # Включаем режим обучения

        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей

        i = 0
        scaler = torch.cuda.amp.GradScaler()
        start_time_ep = time.time()
        for images, text_embs, attention_mask in train_loader:
            if hyperparams.OGRANICHITEL:
                if i == hyperparams.N_OGRANICHITEL:
                    print('Трен. заверш.')
                    break
            start_time = time.time()

            images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)

            optimizer.zero_grad()

            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
            time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

            xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

            with torch.cuda.amp.autocast():  # Включаем AMP
                # guidance_prob = 0.15  # 15% примеров будут безусловными
                # if random.random() < guidance_prob:
                #     predicted_noise = model(xt, None, time_emb, None)  # Безусловное предсказание
                # else:
                #     predicted_noise = model(xt, text_embs, time_emb, attention_mask)  # Условное предсказание
                predicted_noise = model(xt, text_embs, time_emb, attention_mask)
                loss_train = criterion(predicted_noise, added_noise)  # сравниваем с добавленным шумом
            running_loss += loss_train.item()

            scaler.scale(loss_train).backward()  # Масштабируем градиенты
            scaler.step(optimizer)  # Делаем шаг оптимизатора
            scaler.update()  # Обновляем скейлер

            # loss_train.backward()

            # optimizer.step()

            ema.update()

            i += 1
            end_time = time.time()
            print(f"Процентов {(i / len(train_loader)) * 100}, {end_time - start_time}, loss: {loss_train.item():.4f}")

            if i % log_interval == 0:
                print(f"Batch: {i}, Current Train Loss: {loss_train.item():.4f}")

        end_time_ep = time.time()
        print(f'Трен. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def validating_model(self, e_model: encapsulated_data.EncapsulatedModel,
                         e_loader: encapsulated_data.EncapsulatedDataloaders, sheduler):
        print("Валидация")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        val_loader = e_loader.val

        model.eval()  # Переключаем в режим валидации

        # Оценка на валидационном датасете
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей

        i = 0
        with torch.no_grad():
            start_time_ep = time.time()
            for images, text_embs, attention_mask in val_loader:
                if hyperparams.OGRANICHITEL:
                    if i == hyperparams.N_OGRANICHITEL:
                        print('Вал. заверш.')
                        break
                start_time = time.time()

                images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)

                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
                time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

                xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

                with torch.cuda.amp.autocast():  # Включаем AMP
                    predicted_noise = model(xt, text_embs, time_emb, attention_mask)
                    loss_val = criterion(predicted_noise, added_noise)

                running_loss += loss_val.item()

                i += 1
                end_time = time.time()
                print(f"Процентов {(i / len(val_loader)) * 100}, {end_time - start_time}")

                if i % log_interval == 0:
                    print(f"Batch: {i}, Current Val Loss: {loss_val.item():.4f}")

            end_time_ep = time.time()
            print(f'Вал. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def test_model(self, e_model: encapsulated_data.EncapsulatedModel,
                   e_loader: encapsulated_data.EncapsulatedDataloaders, sheduler):
        print("Тестирование")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        test_loader = e_loader.test

        model.eval()

        test_loss = 0.0
        i = 0
        with torch.no_grad():
            start_time_ep = time.time()
            for images, text_embs, attention_mask in test_loader:
                if hyperparams.OGRANICHITEL:
                    if i == hyperparams.N_OGRANICHITEL:
                        print('Тест. заверш.')
                        break
                start_time = time.time()

                images, text_embs, attention_mask = images.to(device), text_embs.to(
                    device), attention_mask.to(device)

                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
                time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

                xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

                with torch.cuda.amp.autocast():  # Включаем AMP
                    predicted_noise = model(xt, text_embs, time_emb, attention_mask)
                    loss_test = criterion(predicted_noise, added_noise)
                test_loss += loss_test.item()

                i += 1
                end_time = time.time()
                print(
                    f"Процентов {(i / len(test_loader)) * 100}, test loss: {loss_test.item()}, {end_time - start_time}")
            end_time_ep = time.time()
        print(f'Тест. заверш. {end_time_ep - start_time_ep}')
        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.4f}')

    def save_my_model_in_middle_train(self, e_model: encapsulated_data.EncapsulatedModel, model_dir, model_file):
        # Сохранение
        model_filepath = model_dir + model_file
        model = e_model.model
        optimizer = e_model.optimizer
        history = e_model.history

        model.cpu()
        ema = e_model.ema
        ema_model = ema.ema_model
        ema_model.cpu()
        torch.save({
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema': ema_model.state_dict(),  # EMA-веса
            'decay': ema.decay
        }, model_filepath)

    def load_my_model_in_middle_train(self, model_dir, model_file, device):
        # Загрузка
        model_filepath = model_dir + model_file
        checkpoint = torch.load(model_filepath)
        e_model = encapsulated_data.EncapsulatedModel(device)

        model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM, 1, 1, hyperparams.BATCH_SIZE,
                                hyperparams.UNET_CONFIG).to(
            device)
        model.load_state_dict(checkpoint['model_state_dict'])
        cross_attn_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "cross_attn" in name:  # Указываем название слоев
                cross_attn_params.append(param)  # Отдельный список для Cross-Attention
            else:
                other_params.append(param)  # Остальные параметры
        optimizer = optim.AdamW([
            {"params": other_params, "lr": hyperparams.LR},  # Обычный LR
            {"params": cross_attn_params, "lr": hyperparams.LR}  # Уменьшенный LR для Cross-Attention
        ], weight_decay=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=hyperparams.LR, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint.get('history', {0: {'train_loss': math.inf,
                                                 'val_loss': math.inf}})  # Если модель была обучена, но во время её обучения ещё не был реализован функционал сохранения истории обучения
        # e_model.device = device
        e_model.model = model
        e_model.optimizer = optimizer
        e_model.history = history

        ema = encapsulated_data.EMA(model, device)
        ema.decay = checkpoint['decay']
        ema.ema_model.load_state_dict(checkpoint['ema'])  # Загружаем EMA-веса
        return e_model

    def get_img_from_text(self, e_model: encapsulated_data.EncapsulatedModel, text, sheduler):
        text_embs, masks = dc.get_text_emb(text)
        model = e_model.model
        # ema = e_model.ema
        # Повторяем тензоры, чтобы размерность по батчам совпадала
        text_emb_batch = text_embs.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1, -1)  # (B, tokens, text_emb_dim)
        mask_batch = masks.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1)  # (B, tokens)

        mask_batch = mask_batch.to(next(model.parameters()).device)
        text_emb_batch = text_emb_batch.to(next(model.parameters()).device)
        img = diff_proc.reverse_diffusion(model, text_emb_batch, mask_batch, sheduler)
        return img

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
