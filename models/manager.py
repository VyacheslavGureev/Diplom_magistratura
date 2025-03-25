import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import models.hyperparams as hyperparams
import models.dataset_creator as dc
import models.encapsulated_data as encapsulated_data
import models.nn_model as nn_model


class DiffusionReverseProcess:
    r"""

    Reverse Process class as described in the
    paper "Denoising Diffusion Probabilistic Models"

    """

    def __init__(self,
                 num_time_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02
                 ):

        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
        self.a = 1 - self.b  # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar

    def sample_prev_timestep(self, xt, noise_pred, t):

        r""" Sample x_(t-1) given x_t and noise predicted
             by model.

             :param xt: Image tensor at timestep t of shape -> B x C x H x W
             :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
             :param t: Current time step

        """

        # Original Image Prediction at timestep t
        x0 = xt - (torch.sqrt(1 - self.a_bar.to(xt.device)[t]) * noise_pred)
        x0 = x0 / torch.sqrt(self.a_bar.to(xt.device)[t])
        x0 = torch.clamp(x0, -1., 1.)

        # mean of x_(t-1)
        mean = (xt - ((1 - self.a.to(xt.device)[t]) * noise_pred) / (torch.sqrt(1 - self.a_bar.to(xt.device)[t])))
        mean = mean / (torch.sqrt(self.a.to(xt.device)[t]))

        # only return mean
        if t == 0:
            return mean, x0

        else:
            variance = (1 - self.a_bar.to(xt.device)[t - 1]) / (1 - self.a_bar.to(xt.device)[t])
            variance = variance * self.b.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0


class ModelManager():

    def __init__(self, device):
        self.device = device

        beta_start = 1e-4
        beta_end = 0.02
        self.create_diff_sheduler(hyperparams.T, beta_start, beta_end)

    def create_diff_sheduler(self, num_time_steps, beta_start, beta_end):
        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)  # b -> beta
        self.a = 1 - self.b  # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0)  # a_bar = alpha_bar
        self.b = self.b.to(self.device)
        self.a = self.a.to(self.device)
        self.a_bar = self.a_bar.to(self.device)

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
            noise = torch.randn_like(x0)
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

    # def get_time_emd(self, t, embed_dim, device):
    #     """t — это тензор со значениями [0, T], размерность (B,)"""
    #     half_dim = embed_dim // 2
    #     # freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim))
    #     freqs = torch.exp(
    #         -torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half_dim)).to(
    #         device)
    #     angles = t[:, None] * freqs[None, :]
    #     time_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    #     return time_embedding  # (B, embed_dim)

    # def get_coeffs(self, device):
    #     beta = torch.linspace(0.0001, 0.008, hyperparams.T)  # Линейно возрастающие b_t
    #     alpha = 1 - beta  # a_t
    #     alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)
    #     alpha, beta, alphas_bar = alpha.to(device), beta.to(device), alphas_bar.to(device)
    #     return alpha, beta, alphas_bar

    def train_model(self, e_model: encapsulated_data.EncapsulatedModel, e_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.training_model(e_model, e_loader)
            val_loss = self.validating_model(e_model, e_loader)

            hist = e_model.history
            last_epoch = max(hist.keys())
            last_epoch += 1

            hist[last_epoch] = {}
            hist[last_epoch]['train_loss'] = train_loss.item()
            hist[last_epoch]['val_loss'] = val_loss.item()
            e_model.history = hist

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

    def training_model(self, e_model: encapsulated_data.EncapsulatedModel,
                       e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Тренировка")
        model = e_model.model
        device = e_model.device
        optimizer = e_model.optimizer
        criterion = e_model.criterion
        train_loader = e_loader.train

        model.train()  # Включаем режим обучения

        loss = None
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
            loss_train.backward()

            optimizer.step()

            i += 1
            end_time = time.time()
            print(f"Процентов {(i / len(train_loader)) * 100}, {end_time - start_time}")
        end_time_ep = time.time()
        print(f'Трен. заверш. {end_time_ep - start_time_ep}')
        return loss

    def validating_model(self, e_model: encapsulated_data.EncapsulatedModel,
                         e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Валидация")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        val_loader = e_loader.val

        model.eval()  # Переключаем в режим валидации

        # Оценка на валидационном датасете
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

                i += 1
                end_time = time.time()
                print(f"Процентов {(i / len(val_loader)) * 100}, {end_time - start_time}")
            end_time_ep = time.time()
            print(f'Вал. заверш. {end_time_ep - start_time_ep}')
        return loss

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
        torch.save({
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filepath)

    def load_my_model_in_middle_train(self, model_dir, model_file, device):
        # Загрузка
        model_filepath = model_dir + model_file
        checkpoint = torch.load(model_filepath)
        e_model = encapsulated_data.EncapsulatedModel()

        model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM, hyperparams.TIME_EMB_DIM, device).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.LR)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint.get('history', {0: {'train_loss': math.inf,
                                                 'val_loss': math.inf}})  # Если модель была обучена, но во время её обучения ещё не был реализован функционал сохранения истории обучения
        e_model.device = device
        e_model.model = model
        e_model.optimizer = optimizer
        e_model.history = history

        return e_model

    # def save_my_model(self, model, model_dir, model_file):
    #     # Сохраняем только state_dict модели
    #     model_filepath = model_dir + model_file
    #     model.cpu()
    #     torch.save(model.state_dict(), model_filepath)
    #
    # def load_my_model(self, model_dir, model_file, device):
    #     # Загружаем модель
    #     model_filepath = model_dir + model_file
    #     model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM_REDUCED, device).to(
    #         device)  # Нужно заново создать архитектуру модели
    #     model.load_state_dict(torch.load(model_filepath))
    #     model.eval()  # Устанавливаем модель в режим оценки (для тестирования)
    #     return model

    # Функция для reverse diffusion
    def reverse_diffusion(self, model, text_embedding, attn_mask, device):
        # Инициализация случайного шума (начало процесса)
        x_t = torch.randn(hyperparams.BATCH_SIZE, 3, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE).to(
            device)  # (B, C, H, W)

        t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int)
        t_tensor = t_tensor.unsqueeze(1)
        t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
        t_tensor = t_tensor.to(device)
        # t_tensor = torch.full((hyperparams.BATCH_SIZE,), step).to(device)  # (B, )
        # Запускаем процесс reverse diffusion

        model.eval()
        with torch.no_grad():

            i = 0
            for step in tqdm(range(hyperparams.T - 1, -1, -1), colour='white'):
                # t_tensor = torch.full((hyperparams.BATCH_SIZE,), step).to(device) # (B, )
                # Получаем предсказание шума на текущем шаге
                # print(t_tensor[step])

                t_i = self.get_time_embedding(t_tensor[step], hyperparams.TIME_EMB_DIM)
                t_i = t_i.to(device)

                predicted_noise = model(x_t, text_embedding, t_i, attn_mask)

                # if i == 500:
                    # self.show_image(predicted_noise[5])

                x_t = (1 / torch.sqrt(self.a[step])) * (
                            x_t - ((1 - self.a[step]) / (torch.sqrt(1 - self.a_bar[step]))) * predicted_noise)

                # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
                # Например, можно добавить немного шума обратно с каждым шагом:
                # if step > 0:
                #     noise = torch.randn_like(x_t).to(device) * (1 - alpha_t).sqrt()
                #     x_t += noise
                i += 1
        # Вернем восстановленное изображение
        return x_t

    def get_img_from_text(self, e_model: encapsulated_data.EncapsulatedModel, text, device):
        text_embs, masks = dc.get_text_emb(text)
        model = e_model.model

        # Повторяем тензоры, чтобы размерность по батчам совпадала
        text_emb_batch = text_embs.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1, -1)  # (B, tokens, text_emb_dim)
        mask_batch = masks.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1)  # (B, tokens)

        mask_batch = mask_batch.to(device)
        text_emb_batch = text_emb_batch.to(device)
        img = self.reverse_diffusion(model, text_emb_batch, mask_batch, device)
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
