import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.CNN import CNN
from Utilities.Utilities import Utilities
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from sklearn.model_selection import ParameterSampler
from tabulate import tabulate   # ✅ needed for final table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, 'Dataset', 'FASHION', 'FASHION')

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

images = train_dataset[0][0].to(device)
labels = train_dataset[1][0].to(device)

Utilities.images_as_canvas(images=images)

hyperparameter_best = dict(
    input_dim=input_dim,
    output_dim=n_classes,
    hidden_layers_size=[128, 64],
    activation="relu",
    kernel_size=(5, 5),
    filters=[16, 32, 64],
    batch_normalization=False,
    dropout_rate=0.10,
    learning_rate=0.001,
    patience_lr=5,
    max_epoch=10
)

hyperparameter_choices = {}
for k in hyperparameter_best.keys():
    hyperparameter_choices[k] = [hyperparameter_best[k]]

hyperparameter_choices["learning_rate"] = [0.01, 0.005, 0.001, .0005, 0.0001]
hyperparameter_choices["activation"] = ["relu", "sigmoid", "tanh"]
hyperparameter_choices["filters"] = [[8, 16, 32], [16, 32, 64]]

hyperparameter_try = list(ParameterSampler(hyperparameter_choices, n_iter=5))

# --------------------------------------------------
# ✅ ADDITION 1: container for experiment metrics
# --------------------------------------------------
metric_list = []

model = CNN(hyperparameter_best).to(device)
scope = ScopeClassifier(model, hyperparameter_best)

input_size = (
    128,
    hyperparameter_best["input_dim"][0],
    hyperparameter_best["input_dim"][1],
    hyperparameter_best["input_dim"][2]
)
input_data = torch.rand(size=input_size, device=device)
print(summary(model, input_data=input_data, depth=5))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]

trainer = TrainerClassifier(hyperparameter=hyperparameter_best)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid
)

train_accuracy_list, valid_accuracy_list = trainer.run()

# --------------------------------------------------
# ✅ ADDITION 2: summarize AFTER epochs are done
# --------------------------------------------------
best_val_accuracy = max(valid_accuracy_list)
final_val_accuracy = valid_accuracy_list[-1]
final_train_accuracy = train_accuracy_list[-1]
epochs_ran = len(train_accuracy_list)

metric_list.append(best_val_accuracy)

# --------------------------------------------------
# ✅ ADDITION 3: attach metrics to hyperparameters
# --------------------------------------------------
hyperparameter_best_logged = hyperparameter_best.copy()
hyperparameter_best_logged["best_val_accuracy"] = best_val_accuracy
hyperparameter_best_logged["final_val_accuracy"] = final_val_accuracy
hyperparameter_best_logged["final_train_accuracy"] = final_train_accuracy
hyperparameter_best_logged["epochs_ran"] = epochs_ran

# --------------------------------------------------
# Final table (your existing code, now meaningful)
# --------------------------------------------------
idx = np.argsort(metric_list)[::-1]   # best first
hyperparameter_sorted = [hyperparameter_best_logged]

df = pd.DataFrame.from_dict(hyperparameter_sorted)
print(tabulate(df, headers='keys', tablefmt='psql'))
