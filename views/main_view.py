from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

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
        self.connects()
        self.subscribe_to_viewmodel()

    def connects(self):
        self.ui.textEditTxtRequest.textChanged.connect(
            lambda: self.viewmodel.check_buttons(self.ui.textEditTxtRequest.toPlainText()))
        self.ui.pushButtonTxtSave.released.connect(self.save_txt)

    def subscribe_to_viewmodel(self):
        self.viewmodel.signal_button_status.connect(self.buttons_status_show)
        self.viewmodel.signal_open_save_txt_dial.connect(self.open_txt_save_dial)
        self.viewmodel.signal_txt_save_status.connect(self.show_txt_save_status)

    def buttons_status_show(self, btn_txt, btn_gen):
        self.ui.pushButtonTxtSave.setEnabled(btn_txt)
        self.ui.pushButtonGen.setEnabled(btn_gen)

    def save_txt(self):
        text = self.ui.textEditTxtRequest.toPlainText()
        self.viewmodel.set_text_data(text)
        self.viewmodel.request_save_file()

    def open_txt_save_dial(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл как...",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        if file_path:
            self.viewmodel.save_text_to_file(file_path)

    def show_txt_save_status(self, status_txt, status_msg):
        self.msg = QMessageBox()
        self.msg.setWindowTitle(status_txt)
        self.msg.setText(status_msg)
        self.msg.setAttribute(70)
        self.msg.show()
