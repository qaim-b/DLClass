import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from ClassesML.EarlyStopper import EarlyStopper

class ScopeClassifier:

    def __init__(self, model, hyperparameters):

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=hyperparameters["learning_rate"])

        if "patience_lr" in hyperparameters:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode='max',
                                               patience=hyperparameters["patience_lr"], factor=0.1)
        else:
            self.scheduler = None

        if "early_stopping" in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None


class ScopeGAN:

    def __init__(self, generator, discriminator, hyper_model):

        self.criterion = nn.MSELoss()
        self.criterion_discriminator = nn.BCELoss()

        self.generator = generator
        self.discriminator = discriminator
        self.hyperparameters = hyper_model



        self.optimizer_generator = optim.Adam(generator.parameters(),
                                      lr=hyper_model["learning_rate"])
        self.optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                      lr=hyper_model["learning_rate"])