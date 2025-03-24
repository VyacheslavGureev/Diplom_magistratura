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

    # with open('datas/cifar-10-batches-py/data_batch_1', 'rb') as fo:
    #     dict = pickle.load(fo, encoding='bytes')





    # mode = 'img'  #
    mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    # mode = 'load_gen'  #
    # mode = 'debug'  #



    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_manager = manager.ModelManager(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'load_gen':
        em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME,
                                                         device)
        print('Загрузка завершена!')
        text = 'Girl .'
        i = model_manager.get_img_from_text(em, text, device)
        neural.show_image(i[5])
    else:
        # dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
        # ed = model_manager.create_dataloaders(dataset, 0.75, 0.14)
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
            model_manager.show_image(i[5])
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
        elif mode == 'debug':

            images, text_embs, attention_mask = next(iter(ed.test))

            images = images.to(device)

            beta = torch.linspace(0.0001, 0.008, hyperparams.T)  # Линейно возрастающие b_t
            alpha = 1 - beta  # a_t
            alphas_bar = torch.cumprod(alpha, dim=0).to(device)  # Накапливаемый коэффициент a_t (T,)

            # t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t

            t = torch.tensor([0])
            t = t.expand(hyperparams.BATCH_SIZE)
            t = t.to(device)
            xt = model_manager.forward_diffusion(images, t, alphas_bar).to(device)  # добавляем шум

            neural.show_image(xt[0])


if __name__ == '__main__':
    print('abc')

    # main()

    neural_func()
