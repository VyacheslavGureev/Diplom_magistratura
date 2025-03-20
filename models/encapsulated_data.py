import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import models.hyperparams as hyperparams
import models.dataset_creator as dc
import models.nn_model as nn_model


class EncapsulatedModel:
    def __init__(self):
        # Создание модели с нуля
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = nn_model.MyUNet(hyperparams.TEXT_EMB_DIM_REDUCED, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparams.LR)
        self.criterion = nn.MSELoss()
        self.history = {0: {'train_loss': math.inf, 'val_loss': math.inf}}


class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test
