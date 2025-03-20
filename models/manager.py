import math

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


class ModelManager():

    def __init__(self):
        pass

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
    def forward_diffusion(self, x0, t, alphas_bar, noise=None):
        """ Добавляет стандартный гауссовский шум к изображению """
        if noise is None:
            noise = torch.randn_like(x0)
        at = alphas_bar[t][:, None, None, None]
        xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
        return xt

    # Функция для получения порядка данных
    def get_data_order(self, loader):
        order = []
        for batch in loader:
            order.extend(batch[1].tolist())  # Сохраняем метки
        return order


    def train_model(self, e_model: encapsulated_data.EncapsulatedModel, e_loader, epochs):
        for epoch in range(epochs):

            order = self.get_data_order(e_loader.train)
            print("Порядок данных:", order)

            train_loss = self.training_model(e_model, e_loader)
            order = self.get_data_order(e_loader.val)
            print("Порядок данных:", order)

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

        beta = torch.linspace(0.0001, 0.02, hyperparams.T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        loss = None
        i = 0
        for images, text_embs, attention_mask in train_loader:
            if i == 2:
                break

            optimizer.zero_grad()

            images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)
            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t

            xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
            predicted_noise = model(xt, text_embs, t, attention_mask)

            loss_train = criterion(predicted_noise, torch.randn_like(xt))  # сравниваем с реальным шумом
            loss = loss_train
            loss_train.backward()

            optimizer.step()

            i += 1
            print(f"Процентов {(i / len(train_loader)) * 100}")
        return loss

    def validating_model(self, e_model: encapsulated_data.EncapsulatedModel,
                         e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Валидация")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        val_loader = e_loader.val

        model.eval()  # Переключаем в режим валидации

        beta = torch.linspace(0.0001, 0.02, hyperparams.T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        # Оценка на валидационном датасете
        loss = None
        i = 0
        with torch.no_grad():
            for images, text_embs, attention_mask in val_loader:
                if i == 2:
                    break

                images, text_embs, attention_mask = images.to(device), text_embs.to(device), attention_mask.to(device)
                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
                xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, t, attention_mask)
                loss_val = criterion(predicted_noise, torch.randn_like(xt))
                loss = loss_val

                i += 1
                print(f"Процентов {(i / len(val_loader)) * 100}")
        return loss

    def test_model(self, e_model: encapsulated_data.EncapsulatedModel,
                   e_loader: encapsulated_data.EncapsulatedDataloaders):
        print("Тестирование")
        model = e_model.model
        device = e_model.device
        criterion = e_model.criterion
        test_loader = e_loader.test

        model.eval()

        beta = torch.linspace(0.0001, 0.02, hyperparams.T)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

        test_loss = 0.0
        i = 0
        with torch.no_grad():
            for images, text_embs, attention_mask in test_loader:
                images, text_embs, attention_mask = images.to(device), text_embs.to(
                    device), attention_mask.to(device)
                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
                xt = self.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум
                predicted_noise = model(xt, text_embs, t, attention_mask)
                loss_test = criterion(predicted_noise, torch.randn_like(xt))
                test_loss += loss_test.item()

                i += 1
                print(f"Процентов {(i / len(test_loader)) * 100}, test loss: {loss_test.item()}")
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

        model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM_REDUCED, device).to(device)
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

    def save_my_model(self, model, model_dir, model_file):
        # Сохраняем только state_dict модели
        model_filepath = model_dir + model_file
        model.cpu()
        torch.save(model.state_dict(), model_filepath)

    def load_my_model(self, model_dir, model_file, device):
        # Загружаем модель
        model_filepath = model_dir + model_file
        model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM_REDUCED, device).to(device)  # Нужно заново создать архитектуру модели
        model.load_state_dict(torch.load(model_filepath))
        model.eval()  # Устанавливаем модель в режим оценки (для тестирования)
        return model

    # Функция для reverse diffusion
    def reverse_diffusion(self, model, text_embedding, attn_mask, device):
        # Инициализация случайного шума (начало процесса)
        x_t = torch.randn(hyperparams.BATCH_SIZE, 3, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE).to(
            device)  # (B, C, H, W)
        # t = torch.linspace(0, hyperparams.T - 1, hyperparams.T).to(device) # (T, )

        beta = (torch.linspace(0.0001, 0.02, hyperparams.T)).to(device)  # Линейно возрастающие b_t
        alpha = 1 - beta  # a_t
        alpha = alpha.to(device)
        # alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)
        t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int)
        t_tensor = t_tensor.unsqueeze(1)
        t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
        t_tensor = t_tensor.to(device)
        # t_tensor = torch.full((hyperparams.BATCH_SIZE,), step).to(device)  # (B, )
        # Запускаем процесс reverse diffusion

        model.eval()
        with torch.no_grad():
            for step in tqdm(range(hyperparams.T - 1, -1, -1), colour='white'):
                # t_tensor = torch.full((hyperparams.BATCH_SIZE,), step).to(device) # (B, )
                # Получаем предсказание шума на текущем шаге
                predicted_noise = model(x_t, text_embedding, t_tensor[step], attn_mask)

                # Обновляем изображение, используя predicted_noise и шаг диффузии
                # beta = torch.linspace(0.0001, 0.02, hyperparams.T)  # Линейно возрастающие b_t
                # alpha = 1 - beta  # a_t
                # alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

                # alpha_t = get_alpha_t(step)  # Получить коэффициент для текущего шага (или из таблицы)
                # beta_t = get_beta_t(step)  # Получить шумовой коэффициент

                # Reverse update (шаг обратной диффузии)
                # x_t = (x_t - beta_t * predicted_noise) / alpha_t.sqrt()

                x_t = (1 / alpha[step]) * (x_t - (torch.sqrt(beta[step]) * predicted_noise))

                # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
                # Например, можно добавить немного шума обратно с каждым шагом:
                # if step > 0:
                #     noise = torch.randn_like(x_t).to(device) * (1 - alpha_t).sqrt()
                #     x_t += noise

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

    # Пример функции для получения alpha_t и beta_t на основе предварительно рассчитанных значений
    # def get_alpha_t(step):
    #     # Это может быть что-то как простая функция зависимости alpha от шага t
    #     alpha = 1 - (step / T)
    #     return alpha
    #
    # def get_beta_t(step):
    #     # Тоже может быть функция для получения шума
    #     beta = (step / T) * 0.02  # Для примера
    #     return beta
