from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem, QButtonGroup

from guipy.ui import Ui_MainWindow

from viewmodels.main_viewmodel import MainViewModel


class MainView(QMainWindow):

    def __init__(self, viewmodel: MainViewModel):
        super(MainView, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.viewmodel = viewmodel

        self.view_init()

    def view_init(self):
        self.ui.progressBarGen.setValue(0)
        self.ui.pushButtonTxtSave.setEnabled(False)
        self.ui.pushButtonGen.setEnabled(False)
        self.ui.pushButtonImgSave.setEnabled(False)
        self.scene = QGraphicsScene()
        self.ui.graphicsViewImg.setScene(self.scene)
        self.pixmap_item = None
        self.radio_button_group = QButtonGroup(self)
        self.radio_button_group.addButton(self.ui.radioButtonDDPM, id=0)  # id для идентификации
        self.radio_button_group.addButton(self.ui.radioButtonDDPMAdapt, id=1)

        self.connects()
        self.subscribe_to_viewmodel()

        # Команды после 2-х предыдущих строк эмулируют нажатия человека после запуска приложения
        self.ui.radioButtonDDPM.setChecked(True)
        self.ui.progressBarGen.setValue(0)


        # self.ui.textEditTxtRequest.setText('1')
        # self.ui.pushButtonGen.released.emit()

    def connects(self):
        self.ui.textEditTxtRequest.textChanged.connect(self.txt_check)
        self.ui.pushButtonTxtSave.released.connect(self.save_txt)
        self.ui.pushButtonGen.released.connect(self.gen_img)  # испускаем сигнал
        self.ui.pushButtonImgSave.released.connect(self.save_img)
        self.ui.radioButtonDDPM.toggled.connect(lambda: self.on_radio_toggled(0))
        self.ui.radioButtonDDPMAdapt.toggled.connect(lambda: self.on_radio_toggled(1))

    def subscribe_to_viewmodel(self):
        self.viewmodel.signal_button_status.connect(self.buttons_status_show)
        self.viewmodel.signal_open_save_txt_dial.connect(self.open_txt_save_dial)
        self.viewmodel.signal_open_save_img_dial.connect(self.open_img_save_dial)
        self.viewmodel.signal_txt_save_status.connect(self.show_save_status)
        self.viewmodel.signal_img_save_status.connect(self.show_save_status)
        self.viewmodel.signal_open_load_img_dial.connect(self.open_load_img_dial)
        self.viewmodel.signal_progress.connect(self.display_progress)

    def buttons_status_show(self, btn_txt, btn_gen):
        self.ui.pushButtonTxtSave.setEnabled(btn_txt)
        self.ui.pushButtonGen.setEnabled(btn_gen)

    def txt_check(self):
        text = self.ui.textEditTxtRequest.toPlainText()
        self.viewmodel.check_buttons(text)
        self.viewmodel.set_text_data(text)

    def save_txt(self):
        self.viewmodel.request_save_file()

    def save_img(self):
        self.viewmodel.request_save_img()

    # функция-прокладка во view
    def gen_img(self):
        self.viewmodel.request_gen_img()  # обращаемся в viewmodel

    def open_txt_save_dial(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл как...",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if file_path:
            self.viewmodel.save_text_to_file(file_path)

    def open_img_save_dial(self):
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)"
        )
        if file_path:
            # Определяем формат из фильтра
            if "PNG" in selected_filter:
                file_format = "PNG"
            elif "JPEG" in selected_filter:
                file_format = "JPEG"
            elif "BMP" in selected_filter:
                file_format = "BMP"
            else:
                file_format = "PNG"  # По умолчанию
            # Сохраняем изображение в выбранном формате
            self.viewmodel.save_img_to_file(file_path, file_format)

    def open_load_img_dial(self, file_path, need_filedial):
        if need_filedial:
            file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.scene.clear()
                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
                self.ui.graphicsViewImg.fitInView(self.pixmap_item, mode=1)
                self.viewmodel.set_pixmap_data(pixmap)
                self.ui.pushButtonImgSave.setEnabled(True)

    def show_save_status(self, status_txt, status_msg):
        self.msg = QMessageBox()
        self.msg.setWindowTitle(status_txt)
        self.msg.setText(status_msg)
        self.msg.setAttribute(70)
        self.msg.show()

    def on_radio_toggled(self, id):
        if self.sender().isChecked():  # Проверяем, что кнопка активирована
            self.viewmodel.select_nn_model(id)

    def display_progress(self, progress):
        self.ui.progressBarGen.setValue(progress)
