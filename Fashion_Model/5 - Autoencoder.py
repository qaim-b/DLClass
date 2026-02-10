import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock, FlattenDenseBlock, UnFlattenDenseBlock
from Utilities.Utilities import Utilities

class AutoEncoder(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.hidden_layers_size = hyperparameters['hidden_layers_size']
        self.activation = hyperparameters['activation']

        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']

        self.batch_normalization = hyperparameters['batch_normalization']
        self.dropout_rate = hyperparameters['dropout_rate']

        self.latent_dim = hyperparameters['latent_dim']
        self.output_activation = hyperparameters['output_activation']

        self.n_dense_layers = len(self.hidden_layers_size)

        # Create Encoder
        self.encoder_layers = nn.ModuleList()

        layer = FlattenDenseBlock(in_size=self.input_dim, out_size=self.hidden_layers_size[0], activation=Utilities.get_activation(self.activation),
                                  batch_normalization=self.batch_normalization, dropout_rate=self.dropout_rate)
        self.encoder_layers.append(layer)

        for i in range(self.n_dense_layers - 1):
            layer = DenseBlock(in_size=self.hidden_layers_size[i], out_size=self.hidden_layers_size[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_normalization, dropout_rate=self.dropout_rate)
            self.encoder_layers.append(layer)

        # Latent space
        layer = nn.Linear(self.hidden_layers_size[-1], self.latent_dim[0])
        self.encoder_layers.append(layer)

        # Create Decoder
        self.decoder_layers = nn.ModuleList()

        units = self.latent_dim + list(self.hidden_layers_size[::-1]) # Use it to flip a list in Python

        for i in range(0, len(units) - 1):
            layer = DenseBlock(in_size=units[i], out_size=units[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_normalization, dropout_rate=self.dropout_rate)
            self.decoder_layers.append(layer)

        layer = UnFlattenDenseBlock(in_size=units[-1], out_size=self.input_dim,
                                    activation=Utilities.get_activation(self.output_activation),
                                    batch_normalization=self.batch_normalization, dropout_rate=self.dropout_rate)
        self.decoder_layers.append(layer)

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def sample(self, z):
        with torch.no_grad():
            x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat