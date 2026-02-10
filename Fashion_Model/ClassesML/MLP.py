import torch.nn as nn

from Utilities.Utilities import Utilities
from ClassesML.Blocks import DenseBlock, FlattenDenseBlock

class MLP(nn.Module):

    def __init__(self, hyperparameter):
        nn.Module.__init__(self)

        self.hidden_layers_size = hyperparameter["hidden_layers_size"]
        self.activation = hyperparameter["activation"]

        self.batch_norm = hyperparameter["batch_normalization"]
        self.dropout = hyperparameter["dropout_rate"]

        self.input_dim = hyperparameter["input_dim"]
        self.output_dim = hyperparameter["output_dim"]

        self.n_dense_layer = len(hyperparameter["hidden_layers_size"])

        self.layers = nn.ModuleList()

        layer = FlattenDenseBlock(in_size=self.input_dim, out_size=self.hidden_layers_size[0],
                                   activation=Utilities.get_activation(self.activation),
                                   batch_normalization=self.batch_norm,
                                   dropout_rate=self.dropout)
        self.layers.append(layer)

        for i in range(self.n_dense_layer - 1):
            layer = DenseBlock(in_size=self.hidden_layers_size[i], out_size=self.hidden_layers_size[i + 1],
                               activation=Utilities.get_activation(self.activation),
                               batch_normalization=self.batch_norm,
                               dropout_rate=self.dropout)
            self.layers.append(layer)

        # Define output layer
        layer = nn.Linear(in_features=self.hidden_layers_size[-1], out_features=self.output_dim)
        self.layers.append(layer)

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):

        x_hat = self.classifier(x)

        return x_hat