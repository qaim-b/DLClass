import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from ClassesML.AutoEncoder import AutoEncoder
from ClassesData.DatasetLoader import DatasetLoader
from Utilities.Utilities import Utilities
from ClassesML.Scope import ScopeAutoEncoder
matplotlib.use("TkAgg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, 'Dataset', 'FASHION')

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

images = train_dataset[0][0].to(device)
labels = train_dataset[1][0].to(device)

#Utilities.images_as_canvas(images=images)

hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      hidden_layers_size=[128, 64, 32],
                      output_activation="sigmoid",
                      latent_dim=[2],
                      activation="relu",
                      batch_normalization=False,
                      dropout_rate=0.01,
                      learning_rate=0.001,
                      max_epochs=10,)

model = AutoEncoder(hyperparameter).to(device)
scope = ScopeAutoEncoder(model, hyperparameter)

input_size = (128, hyperparameter['input_dim'][0], hyperparameter['input_dim'][1], hyperparameter['input_dim'][2])
input_data = torch.rand(size=input_size, device=device)
print(summary(model=model, input_data=input_data, depth=5))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid=val_dataset[0]
y_valid= val_dataset[1]

# Train the AutoEncoder
train_loss_dict = dict()
valid_loss_dict = dict()

for epoch in range(hyperparameter['max_epochs']):

    model.train()
    total_loss = 0.0
    n_batch = len(x_train)

    for n in range(n_batch):

        x=x_train[n].to(device)

        # Forward pass
        x_hat = model(x)
        loss = scope.criterion(x_hat, x)

        # Backward pass and optimization
        scope.optimizer.zero_grad()
        loss.backward()
        scope.optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / n_batch

    print("Epoch:" + str(epoch+1) + "/" + str(hyperparameter['max_epochs']))
    print("Training Loss: " + str(train_loss))

    total_loss = 0.0
    n_batch = len(x_valid)

    model.eval()

    for n in range(n_batch):
        x = x_valid[n].to(device)

        # Forward pass
        x_hat = model(x)
        loss = scope.criterion(x_hat, x)
        total_loss += loss.item()

    valid_loss = total_loss / n_batch
    print("Validation Loss: " + str(valid_loss))

    train_loss_dict[epoch] = train_loss
    valid_loss_dict[epoch] = valid_loss

x_fit = torch.cat(train_dataset[0], dim=0).to(device)
y_fit = torch.cat(train_dataset[1], dim=0)

x_transform = torch.cat(val_dataset[0], dim=0).to(device)
y_transform = torch.cat(val_dataset[1], dim=0)

z_fit = model.encode(x_fit).detach().cpu().numpy()
z_transform = model.encode(x_transform).detach().cpu().numpy()

Utilities.plot_latent_space(z_fit, y_fit)
Utilities.plot_latent_space(z_transform, y_transform)

# Generate new samples from the latent space
n_sample = 100
latent_dim = 2
x_values = torch.linspace(-10, 0, n_sample, device=device)
y_values = torch.linspace(0, 10, n_sample, device=device)
z = torch.stack((x_values, y_values), dim=1)

generated_images = model.sample(z=z)
Utilities.images_as_canvas(generated_images)

# Plot original vs reconstructed images
sources_images=train_dataset[0][0].to(device)
reconstructed_images=model(sources_images)
Utilities.images_as_canvas(sources_images)
Utilities.images_as_canvas(reconstructed_images)