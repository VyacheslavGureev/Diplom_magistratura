import sys
import threading

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    common_obj = CommonObject()
    # Отображаем главное окно
    common_obj.main_view.show()
    sys.exit(app.exec_())
    # thread_func()


# def thread_func():
#     print("Поток запущен")
#     device = torch.device("cuda:0")
#     torch.tensor([1.0], device=device)
#     print("Успешно создали тензор на CUDA")
#
# t = threading.Thread(target=thread_func)
# t.start()
# t.join()



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
    config = {
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
    model_cls = model_registry[config["model_type"]]  # Получаем тип модели для текущего эксперимента
    # Каждая модель сама знает, какие поля конфига ей нужно взять для своей инициализации (делегирование)
    model = model_cls.from_config(config)  # Создание объекта класса с абстракцией от конкретной сигнатуры инициализации
    model.setup_from_config(config)
    model_manager = manager.ModelManager()
    sheduler = diff_proc.NoiseShedulerAdapt(hyperparams.T, 'linear',
                                            device)  # Этот класс более универсальный, поэтому можно его использовать для всех моделей
    ds_cls = dataset_registry[config["dataset_type"]]
    ed = ds_cls().load_or_create(config)

    shutdown_flag = False
    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    mode = 'load_gen'  #
    # mode = 'load_train_test_save'  #
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
        text = "8"
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
    main()
    # common_pipeline()
