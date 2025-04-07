import pickle
import matplotlib.pyplot as plt


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
