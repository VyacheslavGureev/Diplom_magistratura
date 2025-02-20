import sys

from PyQt5 import QtWidgets

from services.navigation_service import CommonObject


def main():
    app = QtWidgets.QApplication(sys.argv)
    common_obj = CommonObject()
    # Отображаем главное окно
    common_obj.main_view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
