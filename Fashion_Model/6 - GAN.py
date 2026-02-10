import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from ClassesML.GAN import Generator, Discriminator
from ClassesML.Scope import ScopeGAN
from ClassesData.DatasetLoader import DatasetLoader
from Utilities.Utilities import Utilities


matplotlib.use("TkAgg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, 'Dataset', 'FASHION', 'FASHION')

dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

images = train_dataset[0][0].to(device)
labels = train_dataset[1][0].to(device)

plt.rcParams['figure.figsize'] = [10, 8]
Utilities.images_as_canvas(images=images)

hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      n_classes=n_classes,
                      activation="relu",
                      discriminator_activation="relu",
                      filters=(128, 64, 32, 16),
                      discriminator_filters=(16, 32, 64),
                      kernel_size=(5, 5),
                      embedding_dim=32,
                      latent_dim=64,
                      hidden_layers_size=[128, 64, 32],
                      batch_normalization=False,
                      dropout_rate=0.2,
                      learning_rate=0.00001,
                      max_epochs=10)

generator = Generator(hyperparameter).to(device)

noise = torch.randn(128, hyperparameter['latent_dim'], device=device)
labels = torch.randint(0, 10, (128,), device=device)

# Print generator summary
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(summary(generator, input_data=[noise, labels], depth=5))

discrimator = Discriminator(hyperparameter).to(device)
scope = ScopeGAN(generator, discrimator, hyperparameter)

# Keep the plot window open until user closes it
plt.show()

# Test Eembedding
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=64)
optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=0.001)
print(embedding_layer)

pred = embedding_layer(input_tensor)

print(pred)
print(pred.shape)

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid=val_dataset[0]
y_valid=val_dataset[1]

train_loss_dict = {}
val_loss_dict = {}

max_epoch = hyperparameter['max_epochs']

for epoch in range(max_epoch):

    # Train
    generator.train()
    discriminator.train()

    total_disc_loss = 0.0
    total_gen_loss = 0.0

    n_batch = len(x_train)

    for n in range(n_batch):

        real_images = x_train[n].to(device)
        y = y_train[n].to(device)
        batch_size = real_images.size()[0]