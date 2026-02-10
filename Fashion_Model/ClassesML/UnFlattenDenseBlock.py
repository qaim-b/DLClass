class UnFlattenDenseBlock(nn.Module):

    def __init__(self, in_size, out_size, activation=nn.ReLU(), batch_normalization=False, dropout_rate=0.1):

        super(UnFlattenDenseBlock, self).__init__()

        self.dense_layer = DenseBlock(in_size=in_size,
                                      out_size=np.prod(out_size),
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      dropout_rate=dropout_rate)
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=out_size)

    def forward(self, x):

        x = self.dense_layer(x)
        x = self.unflatten_layer(x)

        return x