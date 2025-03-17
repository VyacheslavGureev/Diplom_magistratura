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


if __name__ == '__main__':
    # main()



    dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")

    model_manager = neural.ModelManager()
    em = model_manager.create_model()
    ed = model_manager.create_dataloaders(dataset, 0.7, 0.2)
    model_manager.train_model(em, ed, 10)
    model_manager.test_model(em, ed)
    model_manager.save_my_model(em.model, "trained/", "first.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = model_manager.load_my_model("trained/", "first.pth", device)


    # device, model, optimizer, criterion = neural.create_model()
    # dataset = dc.create_dataset("datas/Flickr8k/Images/", "datas/Flickr8k/captions/captions.txt")
    # neural.train_ddpm(model, device, optimizer, criterion, dataset, 10)


