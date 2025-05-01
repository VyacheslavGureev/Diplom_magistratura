import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication

from services.navigation_service import CommonObject

from torch.utils.data import random_split, DataLoader

import pickle
import os
import torch
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


# TODO: Продолжить проверку и написание


def main():
    pass
    # app = QtWidgets.QApplication(sys.argv)
    # common_obj = CommonObject()
    # # Отображаем главное окно
    # common_obj.main_view.show()
    # sys.exit(app.exec_())


def vanile_ddpm():
    # unet_config = {'TEXT_EMB_DIM' : hyperparams.TEXT_EMB_DIM, 'TIME_EMB_DIM' : hyperparams.TIME_EMB_DIM,
    #                'BATCH_SIZE' : hyperparams.BATCH_SIZE, 'ORIG_C' : 1,
    #                'DOWN':
    #                    [{'in_C': 16, 'out_C': 32, 'SA': False},
    #                     {'in_C': 32, 'out_C': 64, 'SA': True},
    #                     {'in_C': 64, 'out_C': 128, 'SA': False}],
    #                'BOTTLENECK': [{'in_C': 128, 'out_C': 128}],
    #                'UP': [{'in_C': 128, 'out_C': 64, 'sc_C': 64, 'SA': False, 'CA': False},
    #                       {'in_C': 64 + 64, 'out_C': 32, 'sc_C': 32, 'SA': True, 'CA': True}]}
    # utils.save_json(unet_config, hyperparams.CURRENT_MODEL_DIR + hyperparams.CURRENT_MODEL_CONFIG)
    # print('saved')

    # # Загружаем CLIP
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    #
    # dc.save_data_to_file(tokenizer, 'datas/embedders/tokenizer.pkl')
    # dc.save_data_to_file(text_encoder, 'datas/embedders/text_encoder.pkl')

    # dataset_full = dc.create_dataset_mnist("./datas",
    #                                        True)  # весь датасет mnist (тренировка + валидация (без теста)) (потому что True)
    # train_size = int(0.84 * len(dataset_full))
    # val_size = len(dataset_full) - train_size
    # # Тестовый датасет отдельно
    # tst_dataset = dc.create_dataset_mnist("./datas", False)
    # train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
    #                           collate_fn=dc.collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                         collate_fn=dc.collate_fn)
    # test_loader = DataLoader(tst_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                          collate_fn=dc.collate_fn)
    # e_loader = encapsulated_data.EncapsulatedDataloaders(train_loader, val_loader, test_loader)
    # utils.save_data_to_file(e_loader, 'trained/e_loader.pkl')
    # print('test')

    shutdown_flag = False
    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    mode = 'load_gen'  #
    # mode = 'debug'  #

    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_manager = manager.ModelManager()
    sheduler = diff_proc.NoiseSheduler(hyperparams.T, 'linear', device)
    if mode == 'load_gen':
        em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
                                                         hyperparams.CURRENT_MODEL_NAME,
                                                         hyperparams.CURRENT_MODEL_DIR +
                                                         hyperparams.CURRENT_MODEL_CONFIG,
                                                         device)
        print('Загрузка завершена!')
        # text = "Это цифра ноль"
        # text = "Изображена единица"
        # text = "Нарисована цифра два"
        # text = "На картинке цифра три"
        # text = "Четыре, написанное от руки"
        text = "5"
        # text = "Цифра шесть, нарисованная от руки"
        # text = "На изображении семерка"
        # text = "Нарисована цифра восемь"
        # text = "Рукописная девятка"
        i = model_manager.get_img_from_text(em, text, sheduler)
        # utils.show_image(i[1])
    else:
        # dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        # ed = model_manager.create_dataloaders(dataset, 0.75, 0.14)
        # Сохраняем объект в файл
        # with open("trained/e_loader.pkl", "wb") as f:
        #     pickle.dump(ed, f)
        # Загружаем объект из файла
        ed = utils.load_data_from_file("trained/e_loader.pkl")
        if mode == 1:
            pass
        # if mode == 'load_test':
        #     em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
        #                                                      hyperparams.CURRENT_MODEL_NAME, device)
        #     print('Загрузка завершена!')
        #     model_manager.test_model(em, ed)
        #     print('Тестирование завершено!')
        elif mode == 'img':
            # i, _, _ = next(iter(ed.train))
            # i, _, _ = next(iter(ed.val))
            i, _, _ = next(iter(ed.test))
            utils.show_image(i[6])
        # elif mode == 'create_train_save':
        #     em = model_manager.create_model()
        #     model_manager.train_model(em, ed, hyperparams.EPOCHS)
        #     model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
        #                                                 hyperparams.CURRENT_MODEL_NAME)
        #     print('Готово!')
        elif mode == 'create_train_test_save':
            em = model_manager.create_model(hyperparams.CURRENT_MODEL_DIR +
                                            hyperparams.CURRENT_MODEL_CONFIG,
                                            device)
            model_manager.train_model(em, ed, hyperparams.EPOCHS, sheduler)
            # model_manager.test_model(em, ed, sheduler)
            print('Тестирование завершено!')
            model_manager.save_my_model_in_middle_train(em,
                                                        hyperparams.CURRENT_MODEL_DIR,
                                                        hyperparams.CURRENT_MODEL_NAME,
                                                        hyperparams.CURRENT_MODEL_DIR +
                                                        hyperparams.CURRENT_MODEL_CONFIG)
            print('Готово! create_train_test_save')
        elif mode == 'load_train_test_save':
            em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
                                                             hyperparams.CURRENT_MODEL_NAME,
                                                             hyperparams.CURRENT_MODEL_DIR +
                                                             hyperparams.CURRENT_MODEL_CONFIG,
                                                             device)
            print('Загрузка завершена!')
            model_manager.train_model(em, ed, hyperparams.EPOCHS, sheduler)
            model_manager.test_model(em, ed, sheduler)
            print('Тестирование завершено!')
            model_manager.save_my_model_in_middle_train(em,
                                                        hyperparams.CURRENT_MODEL_DIR,
                                                        hyperparams.CURRENT_MODEL_NAME,
                                                        hyperparams.CURRENT_MODEL_DIR +
                                                        hyperparams.CURRENT_MODEL_CONFIG)
            print('Продолжение тренировки завершено!')
        elif mode == 'debug':
            # em = model_manager.create_model(device)
            # model_manager.viz_my_model(em)

            images, text_embs, attention_mask = next(iter(ed.test))
            images = images.to(device)
            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
            t = torch.tensor([199], device=device)
            t = t.expand(hyperparams.BATCH_SIZE)
            xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)
            utils.show_image(xt[0])
    if shutdown_flag:
        os.system("shutdown /s /t 60")  # выключение через 60 секунд


def adaptive_ddpm():
    # # Загружаем CLIP
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    #
    # dc.save_data_to_file(tokenizer, 'datas/embedders/tokenizer.pkl')
    # dc.save_data_to_file(text_encoder, 'datas/embedders/text_encoder.pkl')

    # dataset_full = dc.create_dataset_mnist("./datas",
    #                                        True)  # весь датасет mnist (тренировка + валидация (без теста)) (потому что True)
    # train_size = int(0.84 * len(dataset_full))
    # val_size = len(dataset_full) - train_size
    # # Тестовый датасет отдельно
    # tst_dataset = dc.create_dataset_mnist("./datas", False)
    # train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
    #                           collate_fn=dc.collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                         collate_fn=dc.collate_fn)
    # test_loader = DataLoader(tst_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                          collate_fn=dc.collate_fn)
    #
    # dataset_text_descr = dc.create_dataset_mnist_text_descr()
    # text_descr_loader = DataLoader(dataset_text_descr, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
    #                           collate_fn=dc.collate_fn_text_dataset)
    # e_loader = encapsulated_data.EncapsulatedDataloadersTextDescr(train_loader, val_loader, test_loader, text_descr_loader)
    # utils.save_data_to_file(e_loader, 'trained/e_loader_adapt.pkl')
    # print('test')

    shutdown_flag = False
    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    mode = 'load_gen'  #
    # mode = 'debug'  #

    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(42)  # Чтобы модели были стабильными и предсказуемыми, а эксперименты воспроизводимыми
    model_manager = manager.ModelManager()
    sheduler = diff_proc.NoiseShedulerAdapt(hyperparams.T, 'linear', device)
    ed = utils.load_data_from_file("trained/e_loader_adapt.pkl")
    if mode == 'load_gen':
        em = model_manager.create_model_adapt(
            hyperparams.CURRENT_MODEL_DIR +
            hyperparams.CURRENT_MODEL_CONFIG,
            hyperparams.CURRENT_MODEL_DIR +
            hyperparams.CURRENT_MODEL_CONFIG_ADAPT,
            device)
        em.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME)
        print('Загрузка завершена!')
        # text = "Это цифра ноль"
        # text = "Изображена единица"
        # text = "Нарисована цифра два"
        # text = "На картинке цифра три"
        # text = "Четыре, написанное от руки"
        text = "5"
        # text = "Цифра шесть, нарисованная от руки"
        # text = "На изображении семерка"
        # text = "Нарисована цифра восемь"
        # text = "Рукописная девятка"
        i = em.get_img_from_text(text, sheduler, ed=ed)

        # i = model_manager.get_img_from_text(em, text, sheduler)
        # utils.show_image(i[1])
    else:
        # dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        # ed = model_manager.create_dataloaders(dataset, 0.75, 0.14)
        # Сохраняем объект в файл
        # with open("trained/e_loader.pkl", "wb") as f:
        #     pickle.dump(ed, f)
        # Загружаем объект из файла
        # ed = utils.load_data_from_file("trained/e_loader_adapt.pkl")
        if mode == 1:
            pass
        # if mode == 'load_test':
        #     em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
        #                                                      hyperparams.CURRENT_MODEL_NAME, device)
        #     print('Загрузка завершена!')
        #     model_manager.test_model(em, ed)
        #     print('Тестирование завершено!')
        elif mode == 'img':
            # i, _, _ = next(iter(ed.train))
            # i, _, _ = next(iter(ed.val))
            i, _, _ = next(iter(ed.test))
            utils.show_image(i[6])
        # elif mode == 'create_train_save':
        #     em = model_manager.create_model()
        #     model_manager.train_model(em, ed, hyperparams.EPOCHS)
        #     model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
        #                                                 hyperparams.CURRENT_MODEL_NAME)
        #     print('Готово!')
        elif mode == 'create_train_test_save':
            em = model_manager.create_model_adapt(
                hyperparams.CURRENT_MODEL_DIR +
                hyperparams.CURRENT_MODEL_CONFIG,
                hyperparams.CURRENT_MODEL_DIR +
                hyperparams.CURRENT_MODEL_CONFIG_ADAPT,
                device)
            model_manager.train_model(em, ed, hyperparams.EPOCHS, sheduler)
            em.testing_model(ed, sheduler)
            print('Тестирование завершено!')
            em.save_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME)
            print('Готово! create_train_test_save')
        elif mode == 'load_train_test_save':
            em = model_manager.create_model_adapt(
                hyperparams.CURRENT_MODEL_DIR +
                hyperparams.CURRENT_MODEL_CONFIG,
                hyperparams.CURRENT_MODEL_DIR +
                hyperparams.CURRENT_MODEL_CONFIG_ADAPT,
                device)
            em.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME)
            print('Загрузка завершена!')
            model_manager.train_model(em, ed, hyperparams.EPOCHS, sheduler)
            em.testing_model(ed, sheduler)
            print('Тестирование завершено!')
            em.save_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME)
            print('Продолжение тренировки завершено!')
        elif mode == 'debug':
            # em = model_manager.create_model(device)
            # model_manager.viz_my_model(em)

            images, text_embs, attention_mask = next(iter(ed.test))
            images = images.to(device)
            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
            t = torch.tensor([199], device=device)
            t = t.expand(hyperparams.BATCH_SIZE)
            xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)
            utils.show_image(xt[0])
    if shutdown_flag:
        os.system("shutdown /s /t 60")  # выключение через 60 секунд


def common_pipeline():
    utils.set_seed(42)  # Чтобы модели и процессы были стабильными и предсказуемыми, а эксперименты воспроизводимыми
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Реализация через регистрацию моделей и общий конфиг (с класс-методом внутри каждого класса модели).
    # Подход максимального ООП - отсутствие if-ов
    model_registry = \
        {
            "ddpm": model_ddpm.EncapsulatedModel,
            "adaptive": model_adaptive.EncapsulatedModelAdaptive,
        }
    # Конфиг эксперимента
    config = {"model_type": "ddpm",
              # "model_type": "adaptive",
              "model_file": hyperparams.CURRENT_MODEL_NAME,
              "e_loader": "trained/e_loader_adapt.pkl",
              # "e_loader": "trained/e_loader.pkl",
              "need_create": True,
              # "need_create": False,
              "need_save": True,
              # "need_create": True,

              "model_dir": hyperparams.CURRENT_MODEL_DIR,
              "device": device,
              "unet_config_file": hyperparams.CURRENT_MODEL_DIR +
                                  hyperparams.CURRENT_MODEL_CONFIG,
              "adaptive_config_file": hyperparams.CURRENT_MODEL_DIR +
                                      hyperparams.CURRENT_MODEL_CONFIG_ADAPT
              }
    model_cls = model_registry[config["model_type"]]  # Получаем тип модели для текущего эксперимента
    # Каждая модель сама знает, какие поля конфига ей нужно взять для своей инициализации (делегирование)
    model = model_cls.from_config(config)  # Создание объекта класса с абстракцией от конкретной сигнатуры инициализации
    model.setup_from_config(config)  # Создание объекта класса с абстракцией от конкретной сигнатуры инициализации
    model_manager = manager.ModelManager()
    sheduler = diff_proc.NoiseShedulerAdapt(hyperparams.T, 'linear',
                                            device)  # Этот класс более универсальный, поэтому можно его использовать для всех моделей
    dataset = dc.DatasetMNIST()
    ed = dataset.load_or_create(config)
    # ed = utils.load_data_from_file(config["e_loader"])

    shutdown_flag = False
    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    mode = 'load_gen'  #
    # mode = 'debug'  #

    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #
    if mode == 'load_gen':
        model.load_my_model_in_middle_train(config["model_dir"], config["model_file"])
        print('Загрузка завершена!')
        # text = "Это цифра ноль"
        # text = "Изображена единица"
        # text = "Нарисована цифра два"
        # text = "На картинке цифра три"
        # text = "Четыре, написанное от руки"
        text = "5"
        # text = "Цифра шесть, нарисованная от руки"
        # text = "На изображении семерка"
        # text = "Нарисована цифра восемь"
        # text = "Рукописная девятка"
        i = model.get_img_from_text(text, sheduler, ed=ed)
    elif mode == 1:
        pass
    # if mode == 'load_test':
    #     em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
    #                                                      hyperparams.CURRENT_MODEL_NAME, device)
    #     print('Загрузка завершена!')
    #     model_manager.test_model(em, ed)
    #     print('Тестирование завершено!')
    elif mode == 'img':
        # i, _, _ = next(iter(ed.train))
        # i, _, _ = next(iter(ed.val))
        i, _, _ = next(iter(ed.test))
        utils.show_image(i[6])
    # elif mode == 'create_train_save':
    #     em = model_manager.create_model()
    #     model_manager.train_model(em, ed, hyperparams.EPOCHS)
    #     model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
    #                                                 hyperparams.CURRENT_MODEL_NAME)
    #     print('Готово!')
    elif mode == 'create_train_test_save':
        model_manager.train_model(model, ed, hyperparams.EPOCHS, sheduler)
        model.testing_model(ed, sheduler)
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
    elif mode == 'debug':
        # em = model_manager.create_model(device)
        # model_manager.viz_my_model(em)

        images, text_embs, attention_mask = next(iter(ed.test))
        images = images.to(device)
        t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
        t = torch.tensor([199], device=device)
        t = t.expand(hyperparams.BATCH_SIZE)
        xt, added_noise = diff_proc.forward_diffusion(images, t, sheduler)
        utils.show_image(xt[0])
    if shutdown_flag:
        os.system("shutdown /s /t 60")  # выключение через 60 секунд


if __name__ == '__main__':
    # main()

    # vanile_ddpm()
    # adaptive_ddpm()

    common_pipeline()
