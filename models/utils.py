import torch
import random
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Детерминированные алгоритмы CuDNN
    torch.backends.cudnn.benchmark = False  # Отключаем авто-тюнинг ядер


def show_image(tensor_img):
    """ Визуализация тензора изображения """
    img = tensor_img.cpu().detach().numpy().transpose(1, 2, 0)  # Приводим к (H, W, C)
    img = (img - img.min()) / (
            img.max() - img.min())  # Нормализация к [0,1] (matplotlib ждёт данные в формате [0, 1], другие не примет)
    plt.imshow(img)
    plt.axis("off")  # Убираем оси
    plt.show()
    plt.pause(3600)





def load_data_from_file(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f'Объект {type(obj)} успешно загружен из {filepath}')
    return obj


def save_data_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f'Объект {type(obj)} успешно сохранён в {filepath}')


def load_json(filepath):
    with open(filepath, "r") as f:
        json_obj = json.load(f)
    print(f'Объект json {type(json_obj)} успешно загружен из {filepath}')
    return json_obj


def save_json(json_obj, filepath):
    with open(filepath, "w") as f:
        json.dump(json_obj, f, indent=4)
    print(f'Объект json {type(json_obj)} успешно сохранён в {filepath}')
