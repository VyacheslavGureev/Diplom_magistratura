from PyQt5.QtCore import QObject, pyqtSignal


class MainModel(QObject):
    # signal_button_status = pyqtSignal(bool, bool)

    def __init__(self):
        super().__init__()
        # self.pushButtonTxtSave_status = False
        # self.pushButtonGen_status = False
        # self.pushButtonImgSave_status = False

    # def check_buttons(self, txt):
    #     if txt == "" or txt.isspace():
    #         self.pushButtonTxtSave_status = False
    #         self.pushButtonGen_status = False
    #     else:
    #         self.pushButtonTxtSave_status = True
    #         self.pushButtonGen_status = True
    #     self.signal_button_status.emit(self.pushButtonTxtSave_status, self.pushButtonGen_status)
