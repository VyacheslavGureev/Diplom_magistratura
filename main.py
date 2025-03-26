import sys

from PyQt5 import QtWidgets

from services.navigation_service import CommonObject

from torch.utils.data import random_split, DataLoader

import pickle
import torch
import models.nn_model as neural
import models.dataset_creator as dc
import models.hyperparams as hyperparams
import models.manager as manager
import models.encapsulated_data as encapsulated_data


def main():
    pass
    # app = QtWidgets.QApplication(sys.argv)
    # common_obj = CommonObject()
    # # Отображаем главное окно
    # common_obj.main_view.show()
    # sys.exit(app.exec_())


def neural_func():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_manager = manager.ModelManager(device)
    # dataset_full = dc.create_dataset_mnist("./datas",
    #                                        True)  # весь датасет mnist (тренировка + валидация (без теста)) (потому что True)
    # train_size = int(0.83 * len(dataset_full))
    # val_size = len(dataset_full) - train_size
    # # Тестовый датасет отдельно
    # tst_dataset = dc.create_dataset_mnist("./datas", False)
    # # tst_size =  len(tst_dataset)
    # # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])
    # # test_dataset = random_split(tst_dataset, [tst_size])
    # train_loader = DataLoader(train_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=True,
    #                           collate_fn=model_manager.collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                         collate_fn=model_manager.collate_fn)
    # test_loader = DataLoader(tst_dataset, batch_size=hyperparams.BATCH_SIZE, shuffle=False,
    #                          collate_fn=model_manager.collate_fn)
    # e_loader = encapsulated_data.EncapsulatedDataloaders(train_loader, val_loader, test_loader)
    # with open("trained/e_loader.pkl", "wb") as f:
    #     pickle.dump(e_loader, f)
    # print('test')

    # mode = 'img'  #
    # mode = 'create_train_test_save'  #
    # mode = 'load_train_test_save'  #
    mode = 'load_gen'  #
    # mode = 'debug'  #

    # mode = 'create_train_save'  #
    # mode = 'load_test'  #
    # mode = 'load_gen'  #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_manager = manager.ModelManager(device)
    if mode == 'load_gen':
        em = model_manager.load_my_model_in_middle_train(hyperparams.CURRENT_MODEL_DIR, hyperparams.CURRENT_MODEL_NAME,
                                                         device)
        print('Загрузка завершена!')
        text = 'Пять'
        i = model_manager.get_img_from_text(em, text, device)
        model_manager.show_image(i[1])
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
            i, _, _ = next(iter(ed.train))
            model_manager.show_image(i[7])
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
            t = torch.randint(0, hyperparams.T, (hyperparams.BATCH_SIZE,), device=device)  # случайные шаги t
            t = torch.tensor([800])
            t = t.expand(hyperparams.BATCH_SIZE)
            t = t.to(device)
            xt = model_manager.forward_diffusion(images, t).to(device)  # добавляем шум
            model_manager.show_image(xt[0])


if __name__ == '__main__':
    print('abc')

    # main()

    neural_func()
