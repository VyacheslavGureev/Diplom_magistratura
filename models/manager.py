import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import models.hyperparams as hyperparams
import models.dataset_creator as dc
import models.encapsulated_data as encapsulated_data
import models.nn_model as nn_model


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


class ModelManager():

    def __init__(self, device):
        self.device = device

        # beta_start = 1e-4
        # beta_end = 0.02
        # self.create_diff_sheduler_linear(hyperparams.T, beta_start, beta_end)

        s = 0.008
        self.create_diff_scheduler_cosine(hyperparams.T, s)

    def create_diff_sheduler_linear(self, num_time_steps, beta_start, beta_end):
        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
        self.a = 1 - self.b  # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.a_bar = self.a_bar.to(self.device)

    def create_diff_scheduler_cosine(self, T, s):
        """Генерирует b_t на основе косинусного расписания"""
        """Генерирует b_t на основе косинусного расписания"""
        t = torch.linspace(0, T, steps=T + 1)  # Временные шаги 0...T
        f_t = torch.cos((t / T + s) / (1 + s) * (np.pi / 2)) ** 2  # Косинусная формула
        alpha_bar = f_t / f_t[0]  # Нормируем, чтобы α̅_T = 1
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]  # Вычисляем b_t
        self.b = torch.clip(beta, 0.0001, 0.1)  # Ограничиваем b_t
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.a_bar = self.a_bar.to(self.device)

        # print(f"Min beta: {self.b.min().item()}")
        # print(f"Max beta: {self.b.max().item()}")
        # import matplotlib.pyplot as plt
        # T = 1000
        # s = 0.008
        # plt.plot(self.b.cpu().numpy(), label="Beta")
        # plt.xlabel("Timestep")
        # plt.ylabel("Beta")
        # plt.title("Cosine Schedule for Beta")
        # plt.legend()
        # plt.show()
        # plt.pause(3600)

    # --- Создание модели ---
    def create_model(self):
        return encapsulated_data.EncapsulatedModel()

    def create_dataloaders(self, dataset, train_size_percent, val_size_percent):
        # Разделяем датасеты
        train_size = int(train_size_percent * len(dataset))
        val_size = int(val_size_percent * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                                [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
                                  collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
                                collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
                                 collate_fn=self.collate_fn)  # Тестовый датасет можно не перемешивать
        e_loader = encapsulated_data.EncapsulatedDataloaders(train_loader, val_loader, test_loader)
        return e_loader

    def collate_fn(self, batch):
        if len(batch) % hyperparams.BATCH_SIZE != 0:
            additional_batch = random.choices(batch, k=hyperparams.BATCH_SIZE - (len(batch) % hyperparams.BATCH_SIZE))
            batch = batch + additional_batch
        images, text_embs, masks = zip(*batch)  # Разбираем батч по частям
        images = torch.stack(images)  # Объединяем картинки (B, C, H, W)
        text_embs = torch.stack(text_embs)  # Объединяем текстовые эмбеддинги (B, max_length, txt_emb_dim)
        masks = torch.stack(masks)  # Объединяем маски внимания (B, max_length)
        return images, text_embs, masks

    # --- Определение форвардного процесса (зашумление) ---
    def forward_diffusion(self, x0, t, noise=None):
        """ Добавляет стандартный гауссовский шум к изображению """
        if noise is None:
            noise = torch.randn_like(x0, requires_grad=False)
        at = self.a_bar[t][:, None, None, None]
        xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
        return xt

    def get_time_embedding(self,
                           time_steps: torch.Tensor,
                           t_emb_dim: int
                           ) -> torch.Tensor:

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

    def train_model(self, e_model: encapsulated_data.EncapsulatedModel,
                    e_loader: encapsulated_data.EncapsulatedDataloaders, epochs):

        optimizer = e_model.optimizer

        # step_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # MNIST очень большой, поэтому каждый 5 эпох уменьшаем в 2 раза
        plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

        early_stopping = EarlyStopping(patience=3)

        for epoch in range(epochs):
            running_train_loss = self.training_model(e_model, e_loader)
            running_val_loss = self.validating_model(e_model, e_loader)

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

            # step_scheduler.step()  # Уменьшает каждые N эпох
            plateau_scheduler.step(avg_loss_val)  # Дополнительно уменьшает, если застряли



    def training_model(self, e_model: encapsulated_data.EncapsulatedModel,
                       e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Тренировка")
        model = e_model.model
        ema = e_model.ema
        device = e_model.device
        optimizer = e_model.optimizer
        criterion = e_model.criterion
        train_loader = e_loader.train

        ema.ema_model.eval()
        model.train()  # Включаем режим обучения

        loss = None
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей

        i = 0
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
            time_emb = self.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

            xt = self.forward_diffusion(images, t).to(device)  # добавляем шум
            predicted_noise = model(xt, text_embs, time_emb, attention_mask)

            loss_train = criterion(predicted_noise, torch.randn_like(xt))  # сравниваем с реальным шумом
            loss = loss_train
            running_loss += loss_train.item()
            loss_train.backward()

            optimizer.step()

            ema.update()

            i += 1
            end_time = time.time()
            print(f"Процентов {(i / len(train_loader)) * 100}, {end_time - start_time}")

            if i % log_interval == 0:
                print(f"Batch: {i}, Current Loss: {loss.item():.4f}")

        end_time_ep = time.time()
        print(f'Трен. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def validating_model(self, e_model: encapsulated_data.EncapsulatedModel,
                         e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Валидация")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        val_loader = e_loader.val

        model.eval()  # Переключаем в режим валидации

        # Оценка на валидационном датасете
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей
        loss = None

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
                time_emb = self.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

                xt = self.forward_diffusion(images, t).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, time_emb, attention_mask)

                loss_val = criterion(predicted_noise, torch.randn_like(xt))
                loss = loss_val
                running_loss += loss_val.item()

                i += 1
                end_time = time.time()
                print(f"Процентов {(i / len(val_loader)) * 100}, {end_time - start_time}")

                if i % log_interval == 0:
                    print(f"Batch: {i}, Current Loss: {loss.item():.4f}")

            end_time_ep = time.time()
            print(f'Вал. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def test_model(self, e_model: encapsulated_data.EncapsulatedModel,
                   e_loader: encapsulated_data.EncapsulatedDataloaders):
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
                time_emb = self.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

                xt = self.forward_diffusion(images, t).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, time_emb, attention_mask)

                loss_test = criterion(predicted_noise, torch.randn_like(xt))
                test_loss += loss_test.item()

                i += 1
                end_time = time.time()
                print(
                    f"Процентов {(i / len(test_loader)) * 100}, test loss: {loss_test.item()}, {end_time - start_time}")
            end_time_ep = time.time()
        print(f'Тест. заверш. {end_time_ep - start_time_ep}')
        # accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        # print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}')
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
        e_model = encapsulated_data.EncapsulatedModel()

        model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.LR, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint.get('history', {0: {'train_loss': math.inf,
                                                 'val_loss': math.inf}})  # Если модель была обучена, но во время её обучения ещё не был реализован функционал сохранения истории обучения
        e_model.device = device
        e_model.model = model
        e_model.optimizer = optimizer
        e_model.history = history

        ema = encapsulated_data.EMA(model, device)
        ema.decay = checkpoint['decay']
        ema.ema_model.load_state_dict(checkpoint['ema'])  # Загружаем EMA-веса
        return e_model

    # Функция для reverse diffusion
    def reverse_diffusion(self, model, ema, text_embedding, attn_mask, device):
        # Инициализация случайного шума (начало процесса)
        orig_channels = 1
        x_t = torch.randn(hyperparams.BATCH_SIZE, orig_channels, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE).to(
            device)  # (B, C, H, W)
        # self.show_image(x_t[5])
        t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int)
        t_tensor = t_tensor.unsqueeze(1)
        t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
        t_tensor = t_tensor.to(device)
        # Запускаем процесс reverse diffusion
        ema.ema_model.eval()
        model.eval()
        with torch.no_grad():
            i = 0
            for step in tqdm(range(hyperparams.T - 1, -1, -1), colour='white'):
                t_i = self.get_time_embedding(t_tensor[step], hyperparams.TIME_EMB_DIM)
                t_i = t_i.to(device)
                predicted_noise = model(x_t, text_embedding, t_i, attn_mask)
                # predicted_noise = ema.ema_model(x_t, text_embedding, t_i, attn_mask)
                # if i == 500:
                # self.show_image(predicted_noise[5])
                x_t = (1 / torch.sqrt(self.a[step])) * (
                        x_t - ((1 - self.a[step]) / (torch.sqrt(1 - self.a_bar[step]))) * predicted_noise)
                # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
                # Например, можно добавить немного шума обратно с каждым шагом:
                if step > 0:  # Добавляем случайный шум на всех шагах, кроме последнего
                    noise = torch.randn_like(x_t).to(device) * (1 - self.a[step]).sqrt()
                    x_t += noise
                i += 1
        # Вернем восстановленное изображение
        return x_t

    def get_img_from_text(self, e_model: encapsulated_data.EncapsulatedModel, text, device):
        text_embs, masks = dc.get_text_emb(text)
        model = e_model.model
        ema = e_model.ema

        # Повторяем тензоры, чтобы размерность по батчам совпадала
        text_emb_batch = text_embs.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1, -1)  # (B, tokens, text_emb_dim)
        mask_batch = masks.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1)  # (B, tokens)

        mask_batch = mask_batch.to(device)
        text_emb_batch = text_emb_batch.to(device)
        img = self.reverse_diffusion(model, ema, text_emb_batch, mask_batch, device)
        return img

    def show_image(self, tensor_img):
        """ Визуализация тензора изображения """
        # img = tensor_img.cpu().detach().numpy()
        img = tensor_img.cpu().detach().numpy().transpose(1, 2, 0)  # Приводим к (H, W, C)
        img = (img - img.min()) / (
                img.max() - img.min())  # Нормализация к [0,1] (matplotlib ждёт данные в формате [0, 1], другие не примет)
        plt.imshow(img)
        plt.axis("off")  # Убираем оси
        plt.show()
        plt.pause(3600)
