from PyQt5.QtCore import QObject, pyqtSignal

import models.nn_model as neural
import models.load_datas as ld


class MainModel(QObject):
    signal_txt_save_status = pyqtSignal(str, str)
    signal_img_save_status = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

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

    def gen_img(self, txt):
        print('gen', txt)
        device, model, optimizer, criterion = neural.create_model()
        dataset = ld.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        neural.train_ddpm(model, optimizer, dataset, 100)






















