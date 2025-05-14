import sys
import threading

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from services.navigation_service import CommonObject

from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

import pickle
import os
import torch
import lpips
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
import random
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import models.nn_model as neural
import models.dataset_creator as dc
import models.hyperparams as hyperparams
import models.manager as manager
import models.diffusion_processes as diff_proc
import models.utils as utils
import models.model_adaptive as model_adaptive
import models.model_ddpm as model_ddpm


def main():
    app = QtWidgets.QApplication(sys.argv)
    common_obj = CommonObject()
    # Отображаем главное окно
    common_obj.main_view.show()
    sys.exit(app.exec_())





def common_pipeline():
    utils.set_seed(42)  # Чтобы модели и процессы были стабильными и предсказуемыми, а эксперименты воспроизводимыми
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Реализация через регистрацию моделей и общий конфиг (с класс-методом внутри каждого класса модели).
    # Подход максимального ООП - отсутствие if-ов
    dataset_registry = {
        "mnist": dc.DatasetMNIST,
        # "images": dc.DatasetImages,
        "mnist_descr": dc.DatasetMNISTDescr,
    }
    model_registry = \
        {
            "ddpm": model_ddpm.EncapsulatedModel,
            "adaptive": model_adaptive.EncapsulatedModelAdaptive,
        }
    # Конфиг эксперимента
    config_ddpm = {
        "model_type": "ddpm",
        # "model_type": "adaptive",
        "dataset_type": "mnist",
        # "dataset_type": "mnist_descr",
        "model_file": hyperparams.MODEL_NAME_DDPM,
        # "model_file": hyperparams.MODEL_NAME_ADAPT,
        "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_DDPM,
        # "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_ADAPT,
        "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_DDPM,
        # "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_ADAPT,

        # "need_create": True,
        "need_create": False,
        # "need_save": True,
        "need_save": False,

        "model_dir": hyperparams.MODELS_DIR,
        "device": device,
    }
    # Конфиг эксперимента
    config_adapt = {
        # "model_type": "ddpm",
        "model_type": "adaptive",
        # "dataset_type": "mnist",
        "dataset_type": "mnist_descr",
        # "model_file": hyperparams.MODEL_NAME_DDPM,
        "model_file": hyperparams.MODEL_NAME_ADAPT,
        # "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_DDPM,
        "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_ADAPT,
        # "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_DDPM,
        "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_ADAPT,

        # "need_create": True,
        "need_create": False,
        # "need_save": True,
        "need_save": False,

        "model_dir": hyperparams.MODELS_DIR,
        "device": device,
    }
    model_manager = manager.ModelManager()
    config_all = {'ddpm': config_ddpm, 'adapt': config_adapt}
    models = {}
    for key in config_all.keys():
        config = config_all[key]
        model_cls = model_registry[config["model_type"]]  # Получаем тип модели для текущего эксперимента
        # Каждая модель сама знает, какие поля конфига ей нужно взять для своей инициализации (делегирование)
        model = model_cls.from_config(config)  # Создание объекта класса с абстракцией от конкретной сигнатуры инициализации
        model.setup_from_config(config)
        model.load_my_model_in_middle_train(config["model_dir"], config["model_file"])
        sheduler = diff_proc.NoiseShedulerAdapt(hyperparams.T, 'linear',
                                                device)  # Этот класс более универсальный, поэтому можно его использовать для всех моделей
        ds_cls = dataset_registry[config["dataset_type"]]
        ed = ds_cls().load_or_create(config)
        data_dict = {'model': model, 'sheduler': sheduler, 'ed': ed}
        models[key] = data_dict

    shutdown_flag = False
    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    # mode = 'load_gen'  #
    # mode = 'load_train_test_save'  #
    # mode = 'debug'  #

    # mode = 'lpips'
    # mode = 'fid'
    mode = 'psnr_ssim'
    # mode = 'ssim'

    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    TEXT_DESCRIPTIONS = {
        0: ["Это цифра ноль", "0", "ноль", "нуль"],
        1: ["Изображена единица", "1", "единица", "один"],
        2: ["Нарисована цифра два", "2", "два", "двойка"],
        3: ["На картинке цифра три", "3", "три", "тройка"],
        4: ["Четыре, написанное от руки", "4", "четыре", "четвёрка"],
        5: ["Это пятерка", "5", "пять", "пятёрка"],
        6: ["Цифра шесть, нарисованная от руки", "6", "шесть", "шестёрка"],
        7: ["На изображении семерка", "7", "семь", "семёрка"],
        8: ["Нарисована цифра восемь", "8", "восемь", "восьмёрка"],
        9: ["Рукописная девятка", "9", "девять", "девятка"]
    }
    if mode == 'psnr_ssim':
        # Метрики
        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # Тестовый датасет
        dataset = datasets.MNIST(root='./datas', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(32),
                                     transforms.ToTensor()
                                 ]))
        real_scores = []
        ddpm_scores = []
        adapt_scores = []
        n = 100
        for i in range(n):
            img, label = dataset[i]
            cap = TEXT_DESCRIPTIONS[label][1]
            real = img.to(device).unsqueeze(0)  # [1,1,32,32]
            # Получение изображения от обычной ddpm
            gen_ddpm, _ = models['ddpm']['model'].get_img_from_text(cap,
                                                                    models['ddpm']['sheduler'],
                                                                    ed=models['ddpm']['ed'])
            gen_ddpm = gen_ddpm.clamp(0, 1)  # на всякий случай (хотя значение уже нормализовано в диап. [0, 1])
            gen_ddpm = gen_ddpm[0:1]
            # Получение изображения от адаптивной
            gen_adapt, _ = models['adapt']['model'].get_img_from_text(cap,
                                                                      models['adapt']['sheduler'],
                                                                      ed=models['adapt']['ed'])
            gen_adapt = gen_adapt.clamp(0, 1)
            gen_adapt = gen_adapt[0:1]

            # PSNR и SSIM
            psnr_ddpm = psnr(gen_ddpm, real).item()
            ssim_ddpm = ssim(gen_ddpm, real).item()

            psnr_adapt = psnr(gen_adapt, real).item()
            ssim_adapt = ssim(gen_adapt, real).item()

            ddpm_scores.append((psnr_ddpm, ssim_ddpm))
            adapt_scores.append((psnr_adapt, ssim_adapt))

            print(((i + 1)/n)*100)

        # Отдельно по метрикам
        psnr_ddpm_all, ssim_ddpm_all = zip(*ddpm_scores)
        psnr_adapt_all, ssim_adapt_all = zip(*adapt_scores)
        print("\n--- Средние значения ---")
        print(
            f"DDPM: PSNR={sum(psnr_ddpm_all) / len(psnr_ddpm_all):.4f}, SSIM={sum(ssim_ddpm_all) / len(ssim_ddpm_all):.4f}")
        print(
            f"Adapt: PSNR={sum(psnr_adapt_all) / len(psnr_adapt_all):.4f}, SSIM={sum(ssim_adapt_all) / len(ssim_adapt_all):.4f}")

        x = list(range(n))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, psnr_ddpm_all, label='DDPM PSNR', color='blue')
        plt.plot(x, psnr_adapt_all, label='Adapt PSNR', color='green')
        plt.xlabel("Изображение #")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(x, ssim_ddpm_all, label='DDPM SSIM', color='blue')
        plt.plot(x, ssim_adapt_all, label='Adapt SSIM', color='green')
        plt.xlabel("Изображение #")
        plt.ylabel("SSIM")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("psnr_ssim_comparison.png", dpi=300)
        plt.show()

    elif mode == 'lpips':
        real_dataset = datasets.MNIST(root='./datas', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((hyperparams.IMG_SIZE, hyperparams.IMG_SIZE)),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
                                          transforms.Normalize(mean=[0.5], std=[0.5])
                                      ]))
        # Загружаем LPIPS-модель (например, на основе AlexNet)
        lpips_model = lpips.LPIPS(net='alex')  # или 'vgg', 'squeeze'
        lpips_model.to(device)
        lpips_model.eval()
        # Примерные тензоры изображений (должны быть [B, 3, H, W], значения в [-1, 1])
        # Преобразуем grayscale в RGB, если MNIST или fMNIST
        real = []
        ddpm = []
        ddpm_ad = []
        i = 0
        n = 100
        for i in range(n):
            img, label = real_dataset[i]
            cap = TEXT_DESCRIPTIONS[label][1]
            img = img.unsqueeze(0)
            img = img.to(device)
            real.append(img)

            model = models['ddpm']['model']
            sheduler = models['ddpm']['sheduler']
            ed = models['ddpm']['ed']
            img, _ = model.get_img_from_text(cap, sheduler, ed=ed)
            img = img[0]
            img = img.repeat(3, 1, 1)
            img = img.unsqueeze(0) * 2 - 1
            ddpm.append(img)

            model = models['adapt']['model']
            sheduler = models['adapt']['sheduler']
            ed = models['adapt']['ed']
            img, _ = model.get_img_from_text(cap, sheduler, ed=ed)
            img = img[0]
            img = img.repeat(3, 1, 1)
            img = img.unsqueeze(0) * 2 - 1
            ddpm_ad.append(img)

            i += 1
            print((i / n) * 100)
        i = 0
        print('генерация завершена')

        # def to_rgb(img):
        #     return img.repeat(3, 1, 1) if img.shape[0] == 1 else img

        dist_real_base = []
        dist_real_adapt = []
        dist_models = []

        # Вычислить LPIPS
        with torch.no_grad():
            for i in range(n):
                dist = lpips_model(real[i], ddpm[i])
                dist = dist.squeeze()
                dist = dist.cpu().detach().numpy()
                dist = float(dist)
                dist_real_base.append(dist)
                print(f"LPIPS r-b: {dist:.4f}")

                dist = lpips_model(real[i], ddpm_ad[i])
                dist = dist.squeeze()
                dist = dist.cpu().detach().numpy()
                dist = float(dist)
                dist_real_adapt.append(dist)
                print(f"LPIPS r-a: {dist:.4f}")

                dist = lpips_model(ddpm[i], ddpm_ad[i])
                dist = dist.squeeze()
                dist = dist.cpu().detach().numpy()
                dist = float(dist)
                dist_models.append(dist)
                print(f"LPIPS m: {dist:.4f}")
        print('сравнение завершено')
        # График
        plt.figure(figsize=(12, 6))
        plt.plot(dist_real_base, label='LPIPS(Real, DDPM)', color='blue')
        plt.plot(dist_real_adapt, label='LPIPS(Real, Adapt)', color='green')
        plt.plot(dist_models, label='LPIPS(DDPM, Adapt)', color='red')
        plt.xlabel('Image Index')
        plt.ylabel('LPIPS Distance')
        plt.title('Сравнение генераций по LPIPS')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("lpips_comparison.png", dpi=300)
        plt.show()
        print('график построен')

    elif mode == 'load_gen':
        pass
    elif mode == 'fid':
        # Подготовка метрики
        fid = FrechetInceptionDistance(feature=2048).to(device)  # или .to(device)
        # Добавляем настоящие изображения
        real_dataset = datasets.MNIST(root='./datas', train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(299),  # нужно для InceptionV3
                                                 transforms.ToTensor(),
                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # grayscale -> RGB
                                             ]))
        n = 100
        for i in range(n):  # или сколько нужно
            img, label = real_dataset[i]
            cap = TEXT_DESCRIPTIONS[label][1]
            img = img.unsqueeze(0)
            img = img.clamp(0, 1)  # на всякий случай
            img = (img * 255.0).round()  # теперь в [0, 255] с округлением
            img = img.to(torch.uint8)  # перевод в uint8
            fid.update(img.to(device), real=True)

            # model = models['ddpm']['model']
            # sheduler = models['ddpm']['sheduler']
            # ed = models['ddpm']['ed']
            model = models['adapt']['model']
            sheduler = models['adapt']['sheduler']
            ed = models['adapt']['ed']
            img, _ = model.get_img_from_text(cap, sheduler, ed=ed)
            img = img.repeat(1, 3, 1, 1)
            img = img[0]
            img = transforms.Resize(299)(img)
            # img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
            img = img.unsqueeze(0)
            img = img.clamp(0, 1)  # на всякий случай
            img = (img * 255.0).round()  # теперь в [0, 255] с округлением
            img = img.to(torch.uint8)  # перевод в uint8
            fid.update(img.to(device), real=False)
            print((i/n)*100)
        print('генерация завершена')
        # Считаем FID
        print("FID:", fid.compute().item())







        # i = model.get_img_from_text(text, sheduler, ed=ed)
    elif mode == 1:
        pass
    # if mode == 'load_test':
    #     em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
    #                                                      hyperparams.CURRENT_MODEL_NAME, device)
    #     print('Загрузка завершена!')
    #     model_manager.test_model(em, ed)
    #     print('Тестирование завершено!')
    elif mode == 'img':
        i, _, _ = next(iter(ed.train))
        # i, _, _ = next(iter(ed.val))
        # i, _, _ = next(iter(ed.test))
        utils.show_image(i[6])
        # t, m = next(iter(ed.text_descr))
        # print(t)
        # print(m)
    # elif mode == 'create_train_save':
    #     em = model_manager.create_model()
    #     model_manager.train_model(em, ed, hyperparams.EPOCHS)
    #     model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
    #                                                 hyperparams.CURRENT_MODEL_NAME)
    #     print('Готово!')
    elif mode == 'create_train_test_save':
        model_manager.train_model(model, ed, hyperparams.EPOCHS, sheduler)
        # model.testing_model(ed, sheduler)
        print('Тестирование завершено!')
        model.save_my_model_in_middle_train(config["model_dir"], config["model_file"])
        print('Готово! create_train_test_save')
    elif mode == 'load_train_test_save':
        model.load_my_model_in_middle_train(config["model_dir"], config["model_file"])
        print('Загрузка завершена!')
        model_manager.train_model(model, ed, hyperparams.EPOCHS, sheduler)
        model.testing_model(ed, sheduler)
        print('Тестирование завершено!')
        model.save_my_model_in_middle_train(config["model_dir"], config["model_file"])
        print('Продолжение тренировки завершено! load_train_test_save')
    # elif mode == 'debug':
    #     # em = model_manager.create_model(device)
    #     # model_manager.viz_my_model(em)
    #
    #     images, text_embs, attention_mask = next(iter(ed.test))
    #     images = images.to(device)
    #     t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
    #     t = torch.tensor([199], device=device)
    #     t = t.expand(hyperparams.BATCH_SIZE)
    #     xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)
    #     utils.show_image(xt[0])
    if shutdown_flag:
        os.system("shutdown /s /t 60")  # выключение через 60 секунд


if __name__ == '__main__':
    # main()
    common_pipeline()
