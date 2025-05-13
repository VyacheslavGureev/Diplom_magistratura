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

from PyQt5.QtCore import QObject, pyqtSignal

import models.hyperparams as hyperparams
import models.nn_model as nn_model
import models.nn_model_adaptive as nn_model_adapt
import models.nn_model_combine as nn_model_combine
import models.utils as utils
import models.diffusion_processes as diff_proc
import models.dataset_creator as dc
import models.model_ddpm as model_ddpm


def kl_divergence(mu, logvar):
    return (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).mean()


def adapt_loss(e, e_a, e_a_pred, mu, D, mse=torch.nn.MSELoss()):
    lam_1 = 0.5
    lam_2 = 0.5
    lam_3 = 1
    lam_4 = 0.1
    logvar = torch.log(D.clamp(min=1e-8))
    fft_adapt = torch.fft.fft2(e_a)
    fft_target = torch.fft.fft2(e)
    amp_adapt = torch.sqrt(fft_adapt.real ** 2 + fft_adapt.imag ** 2 + 1e-8)
    amp_target = torch.sqrt(fft_target.real ** 2 + fft_target.imag ** 2 + 1e-8)
    amp_adapt = torch.log1p(amp_adapt)
    amp_target = torch.log1p(amp_target)
    L_fft = mse(amp_adapt, amp_target)
    L = lam_1 * mse(e, e_a) + lam_2 * L_fft + lam_3 * mse(
        e_a_pred,
        e_a) + lam_4 * kl_divergence(
        mu, logvar)
    return L


class EncapsulatedModelAdaptive(model_ddpm.ModelInOnePlace):

    @classmethod
    def from_config(cls, config):
        return cls(device=config["device"])

    def setup_from_config(self, config):
        return self.setup(model_config_file=config["model_config_file"])

    def setup(self, model_config_file):
        model_config = utils.load_json(model_config_file)
        # unet_config = utils.load_json(unet_config_file)
        self.model_config = model_config
        # self.unet_config = unet_config
        self.model = nn_model_combine.MyCombineModel(model_config)
        self.model.to(self.device)

        # for param in self.model.adaptive_block.parameters():
        #     param.requires_grad = False
        # adaptive_params = set(p for p in self.model.adaptive_block.parameters())
        # other_params = [p for n, p in self.model.named_parameters()
        #                 if p.requires_grad and "cross_attn" not in n and p not in adaptive_params]
        # cross_attn_params = [p for n, p in self.model.named_parameters()
        #                      if p.requires_grad and "cross_attn" in n and p not in adaptive_params]

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
        # self.criterion = adapt_loss
        self.criterion = nn.MSELoss()
        # self.model.apply(self.init_weights)
        self.tokenizer = utils.load_data_from_file('datas/embedders/tokenizer.pkl')
        self.text_encoder = utils.load_data_from_file('datas/embedders/text_encoder.pkl')

    # def init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #     else:
    #         try:
    #             if hasattr(m, 'weight') and m.weight is not None:
    #                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         except:
    #             pass
    #     if hasattr(m, 'bias') and m.bias is not None:
    #         nn.init.constant_(m.bias, 0.0)
    #     if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
    #         if m.weight is not None:
    #             nn.init.constant_(m.weight, 1.0)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0.0)

    # def init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
    #     else:
    #         try:
    #             if hasattr(m, 'weight') and m.weight is not None:
    #                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
    #         except:
    #             pass
    #     if hasattr(m, 'bias') and m.bias is not None:
    #         nn.init.constant_(m.bias, 0.0)
    #     if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
    #         if m.weight is not None:
    #             nn.init.constant_(m.weight, 1.0)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0.0)

    # Модель обучается на данных, нормализованных к [-1, 1]
    # Пример делигирования функции тренировки объекту модели
    # Это Visitor-like подход, корректный с точки зрения ООП
    def training_model(self, e_loader, sheduler):
        print("Тренировка адапт")
        train_loader = e_loader.train
        # text_descr_loader = e_loader.text_descr
        # for i in range(10):
        # mu, D = self.model.adaptive_block.get_current_variance(train_loader, text_descr_loader, self.device)
        # sheduler.update_coeffs(D)
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей
        # var_calc_interval = 100  # Пересчитываем дисперсию каждые 100 батчей
        i = 0
        self.model.train()
        # scaler = torch.cuda.amp.GradScaler()
        start_time_epoch = time.time()
        for images, text_embs, attention_mask in train_loader:
            if hyperparams.OGRANICHITEL:
                if i == hyperparams.N_OGRANICHITEL:
                    print('Трен. заверш. адапт')
                    break
            start_time_batch = time.time()
            images, text_embs, attention_mask = images.to(self.device), text_embs.to(self.device), attention_mask.to(
                self.device)
            self.optimizer.zero_grad()
            # with torch.cuda.amp.autocast():  # Включаем AMP
            e_adapt_added, e_adapt_pred = self.model(images, text_embs, attention_mask, sheduler)
            loss_train = self.criterion(e_adapt_pred, e_adapt_added)
            # loss_train = self.criterion(noise, e_adapt_added, e_adapt_pred, mu - mu,
            #                             D)
            running_loss += loss_train.item()

            loss_train.backward()
            self.optimizer.step()

            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2-норма
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm:.4f}")

            # scaler.scale(loss_train).backward()  # Масштабируем градиенты
            # scaler.step(self.optimizer)  # Делаем шаг оптимизатора
            # scaler.update()  # Обновляем скейлер

            i += 1
            print(
                f"Процентов {(i / len(train_loader)) * 100}, {time.time() - start_time_batch}, loss: {loss_train.item():.4f}")

            if i % log_interval == 0:
                print(f"Batch: {i}, Current Train Loss: {loss_train.item():.4f}")
            # if i % var_calc_interval == 0:
            #     mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
            #     sheduler.update_coeffs(D)
        print(f'Трен. заверш. {time.time() - start_time_epoch}')
        return running_loss

    def validating_model(self, e_loader, sheduler):
        print("Валидация адапт")
        val_loader = e_loader.val
        text_descr_loader = e_loader.text_descr
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей
        # var_calc_interval = 100  # Пересчитываем дисперсию каждые 100 батчей
        i = 0
        # mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
        # sheduler.update_coeffs(D)
        self.model.eval()  # Включаем режим валидации
        start_time_epoch = time.time()
        with torch.no_grad():
            for images, text_embs, attention_mask in val_loader:
                if hyperparams.OGRANICHITEL:
                    if i == hyperparams.N_OGRANICHITEL:
                        print('Вал. заверш. адапт')
                        break
                start_time_batch = time.time()
                images, text_embs, attention_mask = images.to(self.device), text_embs.to(
                    self.device), attention_mask.to(
                    self.device)
                self.optimizer.zero_grad()
                # noise = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
                #                     hyperparams.IMG_SIZE).to(self.device)
                # with torch.cuda.amp.autocast():  # Включаем AMP
                e_adapt_added, e_adapt_pred = self.model(images, text_embs, attention_mask, sheduler)
                loss_val = self.criterion(e_adapt_pred, e_adapt_added)
                # loss_val = self.criterion(noise, e_adapt_added, e_adapt_pred, mu - mu,
                #                           D)
                running_loss += loss_val.item()

                i += 1
                print(
                    f"Процентов {(i / len(val_loader)) * 100}, {time.time() - start_time_batch}, loss: {loss_val.item():.4f}")

                if i % log_interval == 0:
                    print(f"Batch: {i}, Current Val Loss: {loss_val.item():.4f}")
                # if i % var_calc_interval == 0:
                #     mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
                #     sheduler.update_coeffs(D)

        print(f'Вал. заверш. {time.time() - start_time_epoch}')
        return running_loss

    def testing_model(self, e_loader, sheduler):
        print("Тестирование адапт")
        test_loader = e_loader.test
        # text_descr_loader = e_loader.text_descr
        running_loss = 0.0
        log_interval = 50  # Выводим лосс каждые 50 батчей
        # var_calc_interval = 100  # Пересчитываем дисперсию каждые 100 батчей
        i = 0
        # mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
        # sheduler.update_coeffs(D)
        self.model.eval()  # Включаем режим валидации
        start_time_epoch = time.time()
        with torch.no_grad():
            for images, text_embs, attention_mask in test_loader:
                if hyperparams.OGRANICHITEL:
                    if i == hyperparams.N_OGRANICHITEL:
                        print('Тест. заверш. адапт')
                        break
                start_time_batch = time.time()
                images, text_embs, attention_mask = images.to(self.device), text_embs.to(
                    self.device), attention_mask.to(
                    self.device)
                self.optimizer.zero_grad()
                # noise = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE,
                #                     hyperparams.IMG_SIZE).to(self.device)
                # with torch.cuda.amp.autocast():  # Включаем AMP
                e_adapt_added, e_adapt_pred = self.model(images, text_embs, attention_mask, sheduler)
                loss_test = self.criterion(e_adapt_pred, e_adapt_added)

                # e_adapt_added, e_adapt_pred = self.model(noise, images, text_embs, attention_mask, mu,
                #                                          sheduler)
                # так как отнимали от адаптивного шума mu, то в лоссе также используем mu = 0
                # loss_test = self.criterion(noise, e_adapt_added, e_adapt_pred, mu - mu,
                #                            D)
                running_loss += loss_test.item()

                i += 1
                print(
                    f"Процентов {(i / len(test_loader)) * 100}, {time.time() - start_time_batch}, loss: {loss_test.item():.4f}")

                if i % log_interval == 0:
                    print(f"Batch: {i}, Current Test Loss: {loss_test.item():.4f}")
                # if i % var_calc_interval == 0:
                #     mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
                #     sheduler.update_coeffs(D)

        print(f'Тест. заверш. {time.time() - start_time_epoch}')
        print(f'Test Loss avg: {loss_test / len(test_loader):.4f}')
        # return running_loss

    # Сохранение
    def save_my_model_in_middle_train(self,
                                      model_dir,
                                      model_file):
        model_filepath = model_dir + model_file
        self.model.cpu()
        model_config = self.model_config
        # unet_config = self.unet_config
        # adaptive_config = self.adaptive_config
        torch.save({
            'history': self.history,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            # 'ema': ema_model.state_dict(),  # EMA-веса
            # 'decay': ema.decay
        }, model_filepath)
        utils.save_json(model_config, hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_ADAPT)
        print('Веса и данные ddpm adaptive сохранены!')

    # Загрузка
    def load_my_model_in_middle_train(self, model_dir, model_file):
        model_filepath = model_dir + model_file
        checkpoint = torch.load(model_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {0: {'train_loss': math.inf,
                                                      'val_loss': math.inf}})  # Если модель была обучена, но во время её обучения ещё не был реализован функционал сохранения истории обучения
        print('Веса и данные ddpm adaptive загружены!')


    # Функция для reverse diffusion
    def reverse_diffusion(self, text_embedding, attn_mask, sheduler, text_descr_loader):
        # Инициализация случайного шума (начало процесса)
        # mu, D = self.model.adaptive_block.get_current_variance(text_descr_loader, self.device)
        # sheduler.update_coeffs(D)
        x_t = torch.randn(hyperparams.BATCH_SIZE, hyperparams.CHANNELS, hyperparams.IMG_SIZE, hyperparams.IMG_SIZE,
                          device=self.device)  # (B, C, H, W)
        t_tensor = torch.arange(0, hyperparams.T, 1, dtype=torch.int, device=self.device)
        t_tensor = t_tensor.unsqueeze(1)
        t_tensor = t_tensor.expand(hyperparams.T, hyperparams.BATCH_SIZE)
        output_dir = "trained/denoising_adapt/"
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
                self.signal_progress.emit(int((i / hyperparams.T) * 100))
                # log_D = torch.ones_like(x_t)
                # log_D = torch.zeros_like(x_t)
                log_D, mu = self.model.adaptive_block(text_embedding, attn_mask)
                log_D = self.model.act_logD(log_D)
                # mu = self.model.act_mu(mu)
                # log_D_proj = self.model.net_log(log_D)
                time_embedding = diff_proc.get_time_embedding(t_tensor[step], hyperparams.TIME_EMB_DIM)
                predicted_noise = self.model.unet_block(x_t, text_embedding, time_embedding, attn_mask, log_D)
                x_t = (1 / torch.sqrt(sheduler.a[step])) * (
                        x_t - ((1 - sheduler.a[step]) / (torch.sqrt(1 - sheduler.a_bar[step]))) * predicted_noise)
                # Можно добавить дополнительные шаги, такие как коррекция или уменьшение шума
                # Например, можно добавить немного шума обратно с каждым шагом:
                # if step > 0:  # Добавляем случайный шум на всех шагах, кроме последнего
                #     noise = torch.randn_like(x_t, device=next(model.parameters()).device) * (1 - sheduler.a[step]).sqrt() * 0.1
                #     x_t += noise
                if i % 20 == 0:
                    # hyperparams.VIZ_STEP = True
                    # этот модуль выполняет нормализацию по максимуму!
                    vutils.save_image(x_t, f"trained/denoising_adapt/step_{step}.png", normalize=True)
                i += 1
        images = []
        for step in sorted(os.listdir("trained/denoising_adapt"), key=lambda x: int(x.split("_")[1].split(".")[0]),
                           reverse=True):
            images.append(imageio.imread(os.path.join("trained/denoising_adapt", step)))
        imageio.mimsave("trained/denoising_adapt/denoising_process.gif", images, duration=0.3)  # 0.3 секунды на кадр
        self.signal_progress.emit(0)
        vutils.save_image(x_t, f"trained/denoising_adapt/denoised_result.png", normalize=True)
        x_t = self.custom_normalize(x_t) # перевод в исходный диапазон - [0, 1]
        return x_t, "trained/denoising_adapt/denoised_result.png"

    def custom_normalize(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    def get_img_from_text(self, text, sheduler, **kwargs):
        ed = kwargs['ed']
        text_descr_loader = ed.text_descr
        text_embs, masks = dc.get_text_emb(text, self.tokenizer, self.text_encoder)
        # Повторяем тензоры, чтобы размерность по батчам совпадала
        text_emb_batch = text_embs.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1, -1)  # (B, tokens, text_emb_dim)
        mask_batch = masks.unsqueeze(0).expand(hyperparams.BATCH_SIZE, -1)  # (B, tokens)
        text_emb_batch, mask_batch = text_emb_batch.to(self.device), mask_batch.to(self.device)
        img, filepath = self.reverse_diffusion(text_emb_batch, mask_batch, sheduler, text_descr_loader)
        return img, filepath
