import torch.nn as nn
from ClassesML.Blocks import DenseBlock, Conv2DBlock, BasicResNetBlock
from Utilities.Utilities import Utilities


class CNN(nn.Module):

    def __init__(self, hyperparameters):
        super(CNN, self).__init__()

        self.hidden_layers_size = hyperparameters['hidden_layers_size']
        self.activation = hyperparameters['activation']
        self.batch_normalization = hyperparameters['batch_normalization']
        self.dropout_rate = hyperparameters['dropout_rate']
        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.kernel_size = hyperparameters['kernel_size']
        self.filters = hyperparameters['filters']

        self.n_dense_layer = len(self.hidden_layers_size)
        self.n_conv_layer = len(self.filters)

        self.layers = nn.ModuleList()

        # -----------------------------
        # Convolutional layers
        # -----------------------------
        for i in range(self.n_conv_layer - 1):
            layer = Conv2DBlock(
                in_channels=1 if i == 0 else self.filters[i],
                out_channels=self.filters[i + 1],
                kernel_size=self.kernel_size,
                activation=Utilities.get_activation(self.activation),
                batch_normalization=self.batch_normalization,
                dropout_rate=self.dropout_rate
            )
            self.layers.append(layer)

            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        # -----------------------------
        # Dense layers
        # -----------------------------
        self.layers.append(nn.Flatten())

        self.layers.append(nn.LazyLinear(out_features=self.hidden_layers_size[0]))
        self.layers.append(Utilities.get_activation(self.activation))

        for i in range(len(self.hidden_layers_size) - 1):
            self.layers.append(
                DenseBlock(
                    in_size=self.hidden_layers_size[i],
                    out_size=self.hidden_layers_size[i + 1],
                    activation=Utilities.get_activation(self.activation),
                    batch_normalization=self.batch_normalization,
                    dropout_rate=self.dropout_rate
                )
            )

        self.layers.append(
            nn.Linear(
                in_features=self.hidden_layers_size[-1],
                out_features=self.output_dim
            )
        )

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.classifier(x)


class ResNet(nn.Module):

    def __init__(self, hyperparameters):
        super(ResNet, self).__init__()

        self.hidden_layers_size = hyperparameters['hidden_layers_size']
        self.activation = hyperparameters['activation']
        self.batch_normalization = hyperparameters['batch_normalization']
        self.dropout_rate = hyperparameters['dropout_rate']
        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.kernel_size = hyperparameters['kernel_size']
        self.filters = hyperparameters['filters']

        self.layers = nn.ModuleList()

        # -----------------------------
        # Residual blocks
        # -----------------------------
        for i in range(len(self.filters) - 1):
            layer = BasicResNetBlock(
                in_channels=1 if i == 0 else self.filters[i],
                out_channels=self.filters[i + 1],
                kernel_size=self.kernel_size,
                activation=Utilities.get_activation(self.activation),
                batch_normalization=self.batch_normalization,
                dropout_rate=self.dropout_rate
            )
            self.layers.append(layer)
            self.layers.append(nn.AvgPool2d(kernel_size=(2, 2)))

        # -----------------------------
        # Dense layers
        # -----------------------------
        self.layers.append(nn.Flatten())

        self.layers.append(nn.LazyLinear(out_features=self.hidden_layers_size[0]))
        self.layers.append(Utilities.get_activation(self.activation))

        for i in range(len(self.hidden_layers_size) - 1):
            self.layers.append(
                DenseBlock(
                    in_size=self.hidden_layers_size[i],
                    out_size=self.hidden_layers_size[i + 1],
                    activation=Utilities.get_activation(self.activation),
                    batch_normalization=self.batch_normalization,
                    dropout_rate=self.dropout_rate
                )
            )

        self.layers.append(
            nn.Linear(
                in_features=self.hidden_layers_size[-1],
                out_features=self.output_dim
            )
        )

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.classifier(x)
