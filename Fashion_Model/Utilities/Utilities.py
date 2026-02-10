import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Utilities:

    @staticmethod
    def get_activation(activation_name):
        """Get activation function by name"""
        if activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            return nn.Tanh()
        elif activation_name.lower() == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()

    @staticmethod
    def compute_accuracy(y, y_hat):
        """Compute accuracy from logits"""
        _, predicted = torch.max(y_hat, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    @staticmethod
    def plot_confusion_matrix_fashion(y, y_hat):
        """Plot confusion matrix for Fashion MNIST"""
        # Convert logits to predictions if needed
        if len(y_hat.shape) > 1:
            y_hat_pred = np.argmax(y_hat, axis=1)
        else:
            y_hat_pred = y_hat

        accuracy = np.mean(y == y_hat_pred) * 100

        cm = confusion_matrix(y, y_hat_pred)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

    @staticmethod
    def images_as_canvas(images, n_cols=8):
        """Display images in a grid"""
        n_images = min(images.shape[0], 64)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten() if n_images > 1 else [axes]

        for i in range(n_images):
            img = images[i].cpu().numpy()
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show(block=False)