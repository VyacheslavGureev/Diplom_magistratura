import sys

from PyQt5 import QtWidgets

from services.navigation_service import CommonObject

import pickle
import torch
import models.nn_model as neural
import models.dataset_creator as dc
import models.hyperparams as hyperparams
import models.manager as manager


def main():
    pass
    # app = QtWidgets.QApplication(sys.argv)
    # common_obj = CommonObject()
    # # Отображаем главное окно
    # common_obj.main_view.show()
    # sys.exit(app.exec_())


def neural_func():

    # min_size = 20
    # print(min_size)
    # for i in range(3):
    #     min_size_p4 = min_size + 4
    #     print(min_size_p4)
    #     min_size_x2 = min_size_p4 * 2
    #     print(min_size_x2)
    #     min_size = min_size_x2
    # print(min_size + 4)







    # mode = 'img'  #
    # mode = 'create_train_save'  #
    mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    model_manager = manager.ModelManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'load_gen':
        em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME,
                                                         device)
        print('Загрузка завершена!')
        text = 'Cat .'
        i = model_manager.get_img_from_text(em, text, device)
        neural.show_image(i[1])
    else:
        # dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        # ed = model_manager.create_dataloaders(dataset, 0.7, 0.2)
        # Сохраняем объект в файл
        # with open("trained/e_loader.pkl", "wb") as f:
        #     pickle.dump(ed, f)
        # Загружаем объект из файла
        with open("trained/e_loader.pkl", "rb") as f:
            ed = pickle.load(f)

        if mode == 'load_test':
            em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
                                                             hyperparams.CURRENT_MODEL_NAME, device)
            print('Загрузка завершена!')
            model_manager.test_model(em, ed)
            print('Тестирование завершено!')
        elif mode == 'img':
            i, _, _ = next(iter(ed.test))
            neural.show_image(i[0])
        elif mode == 'create_train_save':
            em = model_manager.create_model()
            model_manager.train_model(em, ed, hyperparams.EPOCHS)
            model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
                                                        hyperparams.CURRENT_MODEL_NAME)
            print('Готово!')
        elif mode == 'create_train_test_save':
            em = model_manager.create_model()
            model_manager.train_model(em, ed, hyperparams.EPOCHS)
            model_manager.test_model(em, ed)
            print('Тестирование завершено!')
            model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
                                                        hyperparams.CURRENT_MODEL_NAME)
            print('Готово! create_train_test_save')
        elif mode == 'load_train_test_save':
            em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR,
                                                             hyperparams.CURRENT_MODEL_NAME, device)
            print('Загрузка завершена!')
            model_manager.train_model(em, ed, hyperparams.EPOCHS)
            model_manager.test_model(em, ed)
            print('Тестирование завершено!')
            model_manager.save_my_model_in_middle_train(em, hyperparams.CURRENT_MODEL_DIR,
                                                        hyperparams.CURRENT_MODEL_NAME)
            print('Продолжение тренировки завершено!')


if __name__ == '__main__':
    print('abc')

    # main()

    neural_func()

# Функция для получения порядка данных
def get_data_order(loader):
    order = []
    for batch in loader:
        order.extend(batch[1].tolist())  # Сохраняем метки
    return order