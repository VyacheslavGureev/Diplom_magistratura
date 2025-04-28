import math
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
import imageio

import models.hyperparams as hyperparams
import models.nn_model as nn_model
import models.nn_model_adaptive as nn_model_adapt
import models.nn_model_combine as nn_model_combine
import models.utils as utils
import models.diffusion_processes as diff_proc
import models.dataset_creator as dc


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

    def training_model(self, e_loader, sheduler):
        raise NotImplementedError()

    def validating_model(self, e_loader, sheduler):
        raise NotImplementedError()

    def testing_model(self, e_loader, sheduler):
        raise NotImplementedError()

    def save_my_model_in_middle_train(self,
                                      model_dir,
                                      model_file):
        raise NotImplementedError()

    def load_my_model_in_middle_train(self,
                                      model_dir,
                                      model_file):
        raise NotImplementedError()

    def get_img_from_text(self, text, sheduler, **kwargs):
        raise NotImplementedError()


class EncapsulatedModel(ModelInOnePlace):
    def setup(self, unet_config_file):
        unet_config = utils.load_json(unet_config_file)
        self.unet_config = unet_config
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
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        else:
            try:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            except:
                pass
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def training_model(self, e_loader, sheduler):
        print("Тренировка")
        train_loader = e_loader.train

        self.model.train()  # Включаем режим обучения

        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей

        i = 0
        # scaler = torch.cuda.amp.GradScaler()
        start_time_ep = time.time()
        for images, text_embs, attention_mask in train_loader:
            if hyperparams.OGRANICHITEL:
                if i == hyperparams.N_OGRANICHITEL:
                    print('Трен. заверш.')
                    break
            start_time = time.time()

            images, text_embs, attention_mask = images.to(self.device), text_embs.to(self.device), attention_mask.to(
                self.device)

            self.optimizer.zero_grad()

            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=self.device)  # случайные шаги t
            time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

            xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

            # with torch.cuda.amp.autocast():  # Включаем AMP
            predicted_noise = self.model(xt, text_embs, time_emb, attention_mask)
            loss_train = self.criterion(predicted_noise, added_noise)  # сравниваем с добавленным шумом
            running_loss += loss_train.item()

            # scaler.scale(loss_train).backward()  # Масштабируем градиенты
            # scaler.step(optimizer)  # Делаем шаг оптимизатора
            # scaler.update()  # Обновляем скейлер

            loss_train.backward()
            self.optimizer.step()

            i += 1
            end_time = time.time()
            print(f"Процентов {(i / len(train_loader)) * 100}, {end_time - start_time}, loss: {loss_train.item():.4f}")

            if i % log_interval == 0:
                print(f"Batch: {i}, Current Train Loss: {loss_train.item():.4f}")

        end_time_ep = time.time()
        print(f'Трен. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def validating_model(self, e_loader, sheduler):
        print("Валидация")
        val_loader = e_loader.val

        self.model.eval()  # Переключаем в режим валидации

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
                images, text_embs, attention_mask = images.to(self.device), text_embs.to(
                    self.device), attention_mask.to(self.device)

                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=self.device)  # случайные шаги t
                time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)
                xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

                # with torch.cuda.amp.autocast():  # Включаем AMP
                predicted_noise = self.model(xt, text_embs, time_emb, attention_mask)
                loss_val = self.criterion(predicted_noise, added_noise)

                running_loss += loss_val.item()

                i += 1
                end_time = time.time()
                print(f"Процентов {(i / len(val_loader)) * 100}, {end_time - start_time}")

                if i % log_interval == 0:
                    print(f"Batch: {i}, Current Val Loss: {loss_val.item():.4f}")

            end_time_ep = time.time()
            print(f'Вал. заверш. {end_time_ep - start_time_ep}')
        return running_loss

    def testing_model(self, e_loader, sheduler):
        print("Тестирование")
        test_loader = e_loader.test

        self.model.eval()

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

                images, text_embs, attention_mask = images.to(self.device), text_embs.to(
                    self.device), attention_mask.to(self.device)

                t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=self.device)  # случайные шаги t
                time_emb = diff_proc.get_time_embedding(t, hyperparams.TIME_EMB_DIM)

                xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)

                # with torch.cuda.amp.autocast():  # Включаем AMP
                predicted_noise = self.model(xt, text_embs, time_emb, attention_mask)
                loss_test = self.criterion(predicted_noise, added_noise)
                test_loss += loss_test.item()

                i += 1
                end_time = time.time()
                print(
                    f"Процентов {(i / len(test_loader)) * 100}, test loss: {loss_test.item()}, {end_time - start_time}")
            end_time_ep = time.time()
        print(f'Тест. заверш. {end_time_ep - start_time_ep}')
        avg_test_loss = test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss:.4f}')

    def save_my_model_in_middle_train(self,
                                      model_dir,
                                      model_file):
        model_filepath = model_dir + model_file
        self.model.cpu()
        unet_config = self.unet_config
        torch.save({
            'history': self.history,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            # 'ema': ema_model.state_dict(),  # EMA-веса
            # 'decay': ema.decay
        }, model_filepath)
        utils.save_json(unet_config, model_dir + hyperparams.CURRENT_MODEL_CONFIG)

    def load_my_model_in_middle_train(self, model_dir, model_file):
        model_filepath = model_dir + model_file
        checkpoint = torch.load(model_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {0: {'train_loss': math.inf,
                                                      'val_loss': math.inf}})  # Если модель была обучена, но во время её обучения ещё не был реализован функционал сохранения истории обучения

    def get_img_from_text(self, text, sheduler, **kwargs):
        text_embs, masks = dc.get_text_emb(text)
        # Повторяем тензоры, чтобы размерность по батчам совпадала
        text_emb_batch = text_embs.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1, -1)  # (B, tokens, text_emb_dim)
        mask_batch = masks.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1)  # (B, tokens)
        mask_batch = mask_batch.to(self.device)
        text_emb_batch = text_emb_batch.to(self.device)
        img = self.reverse_diffusion(text_emb_batch, mask_batch, sheduler)
        return img

    def reverse_diffusion(self, text_embedding, attn_mask, sheduler):
        # Инициализация случайного шума (начало процесса)
        x_t = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE,
                          device=self.device)  # (B, C, H, W)
        t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int, device=self.device)
        t_tensor = t_tensor.unsqueeze(1)
        t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
        output_dir = "trained/denoising/"
        # Удаляем все файлы в папке
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Запускаем процесс reverse diffusion
        self.model.eval()
        with torch.no_grad():
            i = 0
            for step in tqdm(range(hyperparams.T - 1, -1, -1), colour='white'):
                time_embedding = diff_proc.get_time_embedding(t_tensor[step], hyperparams.TIME_EMB_DIM)
                predicted_noise = self.model(x_t, text_embedding, time_embedding, attn_mask)
                x_t = (1 / torch.sqrt(sheduler.a[step])) * (
                        x_t - ((1 - sheduler.a[step]) / (torch.sqrt(1 - sheduler.a_bar[step]))) * predicted_noise)
                # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
                # Например, можно добавить немного шума обратно с каждым шагом:
                # if step > 0:  # Добавляем случайный шум на всех шагах, кроме последнего
                #     noise = torch.randn_like(x_t, device=next(model.parameters()).device) * (1 - sheduler.a[step]).sqrt() * 0.1
                #     x_t += noise
                if i % 20 == 0:
                    # hyperparams.VIZ_STEP = True
                    vutils.save_image(x_t, f"trained/denoising/step_{step}.png", normalize=True)
                i += 1
        images = []
        for step in sorted(os.listdir("trained/denoising"), key=lambda x: int(x.split("_")[1].split(".")[0]),
                           reverse=True):
            images.append(imageio.imread(os.path.join("trained/denoising", step)))
        imageio.mimsave("trained/denoising/denoising_process.gif", images, duration=0.3)  # 0.3 секунды на кадр
        return x_t

        # TODO добиться правильной работы и хороших результатов от обычной ddpm