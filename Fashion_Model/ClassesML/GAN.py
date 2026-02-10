import torch
import torch.nn as nn
from ClassesML.Blocks import DenseBlock, FlattenDenseBlock, Conv2DBlock, UnFlattenDenseBlock
from Utilities.Utilities import Utilities

class Generator(nn.Module):
    def __init__(self, hyperparameter):
        
        nn.Module.__init__(self)

        self.hyperparameter = hyperparameter

        self.activation = self.hyperparameter['activation']
        self.input_dim = self.hyperparameter['input_dim']
        self.n_classes = self.hyperparameter['n_classes']
        self.filters = self.hyperparameter['filters']
        self.kernel_size = self.hyperparameter['kernel_size']
        self.embedding_dim = self.hyperparameter['embedding_dim']
        self.latent_dim = self.hyperparameter['latent_dim']

        self.batch_normalization = self.hyperparameter['batch_normalization']
        self.dropout_rate = self.hyperparameter['dropout_rate']

        self.embedding_dim = self.hyperparameter['embedding_dim']
        self.latent_dim = self.hyperparameter['latent_dim']

        self.n_conv_discriminator_layers = len(self.hyperparameter['discriminator_filters'])

        self.n_conv_layers = len(self.filters)

        #Embedding
        self.embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.layers_generator = nn.ModuleList()

        size = self.input_dim[1]
        for i in range(0, self.n_conv_layers-1):
            size /= 2
        self.size_reshape_noise = (self.filters[0],int(size),int(size))

        layer = UnFlattenDenseBlock(self.latent_dim + self.embedding_dim,
                                    out_size=self.size_reshape_noise,
                                    batch_normalization=self.batch_normalization,
                                    activation=Utilities.get_activation(self.activation),
                                    dropout_rate=0.0)

        self.layers_generator.append(layer)

        for i in range (0, self.n_conv_layers - 1):
            layer = Conv2DBlock(in_channels=self.filters[i],
                                out_channels=self.filters[i+1],
                                kernel_size=self.kernel_size,
                                batch_normalization=self.batch_normalization,
                                activation=Utilities.get_activation(self.activation),
                                dropout_rate=self.dropout_rate)
            self.layers_generator.append(layer)
            layer = nn.Upsample(scale_factor=2, mode='nearest')
            self.layers_generator.append(layer)

        layer = Conv2DBlock(in_channels=self.filters[-1],
                            out_channels=1,
                            kernel_size=self.kernel_size,
                            batch_normalization=False,
                            activation=Utilities.get_activation("sigmoid"),
                            dropout_rate=0.0)
                            
        self.layers_generator.append(layer)

        self.generator = nn.Sequential(*self.layers_generator)

    def forward(self, noise, labels):
        embedded_labels = self.embedding(labels)
        gen_input = torch.cat((noise, embedded_labels), dim=1)
        img = self.generator(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, hyperparameter):

        nn.Module.__init__(self)

        self.hyperparameter = hyperparameter

        self.activation = self.hyperparameter['discriminator_activation']
        self.input_dim = self.hyperparameter['input_dim']
        self.n_classes = self.hyperparameter['n_classes']
        self.discriminator_filters = self.hyperparameter['discriminator_filters']
        self.kernel_size = self.hyperparameter['kernel_size']

        self.batch_normalization = self.hyperparameter['batch_normalization']
        self.dropout_rate = self.hyperparameter['dropout_rate']

        self.embedding_dim = self.hyperparameter['embedding_dim']

        self.n_conv_discriminator_layers = len(self.discriminator_filters)

        #Embedding for labels
        self.embedding = nn.Embedding(self.n_classes, self.input_dim[1] * self.input_dim[2])
        self.layers_discriminator = nn.ModuleList()

        # First conv layer takes image + embedded label as input
        layer = Conv2DBlock(in_channels=self.input_dim[0] + 1,
                            out_channels=self.discriminator_filters[0],
                            kernel_size=self.kernel_size,
                            batch_normalization=False,
                            activation=Utilities.get_activation(self.activation),
                            dropout_rate=self.dropout_rate)
        self.layers_discriminator.append(layer)

        # Downsampling
        layer = nn.AvgPool2d(2)
        self.layers_discriminator.append(layer)

        # Additional conv layers
        for i in range(0, self.n_conv_discriminator_layers - 1):
            layer = Conv2DBlock(in_channels=self.discriminator_filters[i],
                                out_channels=self.discriminator_filters[i+1],
                                kernel_size=self.kernel_size,
                                batch_normalization=self.batch_normalization,
                                activation=Utilities.get_activation(self.activation),
                                dropout_rate=self.dropout_rate)
            self.layers_discriminator.append(layer)

            # Downsampling
            layer = nn.AvgPool2d(2)
            self.layers_discriminator.append(layer)

        # Flatten and output layer
        self.layers_discriminator.append(nn.Flatten())

        # Calculate flattened size based on input dimensions and pooling
        size = self.input_dim[1]
        for i in range(self.n_conv_discriminator_layers):
            size = size // 2
        flattened_size = self.discriminator_filters[-1] * size * size

        layer = DenseBlock(in_size=flattened_size,
                          out_size=1,
                          batch_normalization=False,
                          activation=Utilities.get_activation("sigmoid"),
                          dropout_rate=0.0)
        self.layers_discriminator.append(layer)

        self.discriminator = nn.Sequential(*self.layers_discriminator)

    def forward(self, images, labels):
        # Embed labels and reshape to match image dimensions
        embedded_labels = self.embedding(labels)
        embedded_labels = embedded_labels.view(-1, 1, self.input_dim[1], self.input_dim[2])

        # Concatenate image with embedded labels
        disc_input = torch.cat((images, embedded_labels), dim=1)

        # Pass through discriminator
        validity = self.discriminator(disc_input)
        return validity