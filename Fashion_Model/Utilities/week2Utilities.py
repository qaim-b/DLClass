import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

class Utilities:

    @staticmethod
    def images_as_canvas(images, title:str =""):

        if activation_str ==  'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == 'linear':
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy()*255
        canvas = canvas.astype("uint8")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(canvas)
        ax.axis('off')
        ax.set_title(title)
        plt.show()







