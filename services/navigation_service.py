from models.main_model import MainModel
from viewmodels.main_viewmodel import MainViewModel
from views.main_view import MainView


class CommonObject():
    def __init__(self):
        # Создаем модели
        self.main_model = MainModel()

        # Создаем viewmodels
        self.main_viewmodel = MainViewModel(self.main_model)

        # Создаем views
        self.main_view = MainView(self.main_viewmodel)
