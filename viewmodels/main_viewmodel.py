from PyQt5.QtCore import QObject, pyqtSignal
# from services.event_bus import EventBus
from models.main_model import MainModel


class MainViewModel(QObject):
    signal_button_status = pyqtSignal(bool, bool)
    signal_open_save_txt_dial = pyqtSignal()
    signal_open_save_img_dial = pyqtSignal()
    signal_open_load_img_dial = pyqtSignal()
    signal_txt_save_status = pyqtSignal(str, str)
    signal_img_save_status = pyqtSignal(str, str)

    def __init__(self, model: MainModel):
        super().__init__()

        self.model = model
        self.subscribe_to_model()

        self.text_data = None
        self.pixmap_data = None

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

    def request_gen_img(self):
        # self.signal_open_load_img_dial.emit()

        self.model.gen_img(self.text_data)

    def return_txt_save_status(self, status_txt, msg):
        self.signal_txt_save_status.emit(status_txt, msg)

    def return_img_save_status(self, status_txt, msg):
        self.signal_img_save_status.emit(status_txt, msg)
