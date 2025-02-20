from PyQt5.QtCore import QObject, pyqtSignal
# from services.event_bus import EventBus
from models.main_model import MainModel


class MainViewModel(QObject):
    # request_close = pyqtSignal()  # Сигнал для закрытия окна

    signal_button_status = pyqtSignal(bool, bool)
    signal_open_save_txt_dial = pyqtSignal()
    signal_txt_save_status = pyqtSignal(str, str)

    def __init__(self, model: MainModel):
        super().__init__()

        self.model = model

        self.text_data = None

    def check_buttons(self, txt):
        btn_status_txt = True
        btn_status_gen = True
        if txt == "" or txt.isspace():
            btn_status_txt = False
            btn_status_gen = False
        self.signal_button_status.emit(btn_status_txt, btn_status_gen)

    def set_text_data(self, txt):
        self.text_data = txt

    def request_save_file(self):
        self.signal_open_save_txt_dial.emit()

    def save_text_to_file(self, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(self.text_data)
            self.signal_txt_save_status.emit("Успех!", f"Файл успешно сохранен: {filepath}")
        except Exception as e:
            self.signal_txt_save_status.emit("Ошибка!", f"Ошибка при сохранении: {str(e)}")
