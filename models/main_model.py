from PyQt5.QtCore import QObject, pyqtSignal

import torch

import models.dataset_creator as dc
import models.hyperparams as hyperparams
import models.manager as manager
import models.diffusion_processes as diff_proc
import models.utils as utils
import models.model_adaptive as model_adaptive
import models.model_ddpm as model_ddpm


# import models.nn_model as neural
# import models.dataset_creator as ld


class MainModel(QObject):
    signal_txt_save_status = pyqtSignal(str, str)
    signal_img_save_status = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        # self.init_nn_models()
        self.curr_used_nn_model = -1
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Реализация через регистрацию моделей и общий конфиг (с класс-методом внутри каждого класса модели).
        # Подход максимального ООП - отсутствие if-ов
        self.ready_models = {}
        self.dataset_registry = {
            "mnist": dc.DatasetMNIST,
            # "images": dc.DatasetImages,
            "mnist_descr": dc.DatasetMNISTDescr,
        }
        self.model_registry = \
            {
                "ddpm": model_ddpm.EncapsulatedModel,
                "adaptive": model_adaptive.EncapsulatedModelAdaptive,
            }
        self.config = {0: {
            "model_type": "ddpm",
            "dataset_type": "mnist",
            "model_file": hyperparams.MODEL_NAME_DDPM,
            "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_DDPM,
            "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_DDPM,
            "need_create": False,
            "need_save": False,
            "model_dir": hyperparams.MODELS_DIR,
            "device": None,
        },
            1: {
                "model_type": "adaptive",
                "dataset_type": "mnist_descr",
                "model_file": hyperparams.MODEL_NAME_ADAPT,
                "e_loader": hyperparams.E_LOADERS_DIR + hyperparams.E_LOADER_ADAPT,
                "model_config_file": hyperparams.CONFIGS_DIR + hyperparams.MODEL_CONFIG_ADAPT,
                "need_create": False,
                "need_save": False,
                "model_dir": hyperparams.MODELS_DIR,
                "device": None,
            }}

    def save_text_to_file(self, filepath, text):
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(text)
            self.signal_txt_save_status.emit("Успех!", f"Файл успешно сохранен: {filepath}")
        except Exception as e:
            self.signal_txt_save_status.emit("Ошибка!", f"Ошибка при сохранении: {str(e)}")

    def save_img_to_file(self, file_path, file_format, pixmap_data):
        success = pixmap_data.save(file_path, file_format)
        if not success:
            self.signal_img_save_status.emit("Ошибка!", f"Не удалось сохранить изображение!")
        else:
            self.signal_img_save_status.emit("Успех!", f"Изображение успешно сохранено!")

    # def gen_img(self, txt):
    #
    #     print(txt)

        # print('gen', txt)

        # device, model, optimizer, criterion = neural.create_model()
        # dataset = ld.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        # neural.train_ddpm(model, device, optimizer, criterion, dataset, 10)