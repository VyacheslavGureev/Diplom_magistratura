import sys

from PyQt5 import QtWidgets

from services.navigation_service import CommonObject

import models.nn_model as neural
import models.dataset_creator as dc
import torch



def main():
    pass
    # app = QtWidgets.QApplication(sys.argv)
    # common_obj = CommonObject()
    # # Отображаем главное окно
    # common_obj.main_view.show()
    # sys.exit(app.exec_())


def n_func():
    dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
    model_manager = neural.ModelManager()
    ed = model_manager.create_dataloaders(dataset, 0.7, 0.2)

    # i, _, _ = next(iter(ed.test))
    # neural.show_image(i[0])
    # return

    em = model_manager.create_model()
    # ed = model_manager.create_dataloaders(dataset, 0.7, 0.2)
    model_manager.train_model(em, ed, 5)
    # model_manager.test_model(em, ed)
    # model_manager.save_my_model(em.model, "trained/", "128p.pth")
    model_manager.save_my_model_in_middle_train(em, "trained/", "128p.pth")
    print('Готово!')

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # m = model_manager.load_my_model("trained/", "first.pth", device)


if __name__ == '__main__':
    # main()

    n_func()








