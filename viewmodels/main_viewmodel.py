from PyQt5.QtCore import QObject, pyqtSignal, QThread
# from services.event_bus import EventBus
from models.main_model import MainModel

import torch
import time
import models.dataset_creator as dc
import models.hyperparams as hyperparams
import models.manager as manager
import models.diffusion_processes as diff_proc
import models.utils as utils
import models.model_adaptive as model_adaptive
import models.model_ddpm as model_ddpm


class MainViewModel(QObject):
    signal_button_status = pyqtSignal(bool, bool)
    signal_open_save_txt_dial = pyqtSignal()
    signal_open_save_img_dial = pyqtSignal()
    signal_open_load_img_dial = pyqtSignal(str, bool)
    signal_txt_save_status = pyqtSignal(str, str)
    signal_img_save_status = pyqtSignal(str, str)
    signal_progress = pyqtSignal(int)
    signal_mute_radio_btn = pyqtSignal(bool)

    def __init__(self, model: MainModel):
        super().__init__()

        self.model = model
        self.init_nn_models()
        self.subscribe_to_model()

        self.text_data = None
        self.pixmap_data = None
        self.threads = {}

    def subscribe_to_model(self):
        self.model.signal_txt_save_status.connect(self.return_txt_save_status)
        self.model.signal_img_save_status.connect(self.return_img_save_status)

    def check_buttons(self, txt):
        btn_status_txt = True
        btn_status_gen = True
        if txt == "" or txt.isspace():
            btn_status_txt = False
            btn_status_gen = False
        # else:
        #     self.text_data = txt
        self.signal_button_status.emit(btn_status_txt, btn_status_gen)

    def set_text_data(self, txt):
        if not (txt == "" or txt.isspace()):
            self.text_data = txt

    def set_pixmap_data(self, pixmap):
        self.pixmap_data = pixmap

    def request_save_file(self):
        self.signal_open_save_txt_dial.emit()

    def request_save_img(self):
        self.signal_open_save_img_dial.emit()

    def save_text_to_file(self, filepath):
        self.model.save_text_to_file(filepath, self.text_data)

    def save_img_to_file(self, file_path, file_format):
        self.model.save_img_to_file(file_path, file_format, self.pixmap_data)

    def return_txt_save_status(self, status_txt, msg):
        self.signal_txt_save_status.emit(status_txt, msg)

    def return_img_save_status(self, status_txt, msg):
        self.signal_img_save_status.emit(status_txt, msg)

    def select_nn_model(self, id):
        self.model.curr_used_nn_model = id

    def init_nn_models(self):
        utils.set_seed(42)  # Чтобы модели и процессы были стабильными и предсказуемыми, а эксперименты воспроизводимыми
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for id in range(len(self.model.config)):
            self.model.config[id]['device'] = device

    # функция-прокладка во viewmodel для запроса ещё более низкоуровневой информации
    def request_gen_img(self):
        config = self.model.config[self.model.curr_used_nn_model]
        if self.model.ready_models.get(self.model.curr_used_nn_model, None) == None:
            model_cls = self.model.model_registry[config["model_type"]]  # Получаем тип модели для текущего эксперимента
            ds_cls = self.model.dataset_registry[config["dataset_type"]]
            thread = QThread()
            model_builder = model_ddpm.ModelBuilder(model_cls, config, (hyperparams.T, 'linear',
                                                                        config[
                                                                            'device']), ds_cls)
            model_builder.moveToThread(thread)
            model_builder.data_ready.connect(self.new_model_created)
            thread.started.connect(model_builder.build_model)
            self.threads[self.model.curr_used_nn_model] = [thread,
                                                           model_builder]  # Сохраняем ссылки на потоки и билдеры, чтобы поток был не уничтожен сборщиком
            thread.start()
        else:
            model = self.model.ready_models[self.model.curr_used_nn_model]['model']
            sheduler = self.model.ready_models[self.model.curr_used_nn_model]['sheduler']
            ed = self.model.ready_models[self.model.curr_used_nn_model]['ed']
            self.gen_image(model, sheduler, ed)

    def on_result_ready(self, img, filepath):
        self.signal_open_load_img_dial.emit(filepath, False)
        self.signal_button_status.emit(True, True)
        self.signal_mute_radio_btn.emit(False)

    # Регистрируем новую нейронную модель в нашей модели приложения
    def new_model_created(self, model, sheduler, ed):
        model.signal_progress.connect(self.signal_progress)
        model.task_done.connect(self.on_result_ready)  # Обработка результата
        self.model.ready_models[self.model.curr_used_nn_model] = {'model': model, 'sheduler': sheduler, 'ed': ed}
        self.gen_image(model, sheduler, ed)

    def gen_image(self, model, sheduler, ed):
        model.start_task.emit(self.text_data, sheduler, ed)
        self.signal_button_status.emit(True,
                                       False)  # блокируем кнопку, чтобы избежать повторного нажатия во время выполнения задачи
        self.signal_mute_radio_btn.emit(True)
