import os

os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

import numpy
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

class DatasetLoader:

    def __init__(self, root): #Init is called constructor.

        self.root = root

    def load_images_labels_data(self):

        train_data_batches = torch.load(os.path.join(self.root, 'train_data_batches.pt'))
        train_label_batches = torch.load(os.path.join(self.root, 'train_label_batches.pt'))

        val_data_batches = torch.load(os.path.join(self.root, 'val_data_batches.pt'))
        val_label_batches = torch.load(os.path.join(self.root, 'val_label_batches.pt'))

        train_dataset = [train_data_batches, train_label_batches]
        val_dataset = [val_data_batches, val_label_batches]

        input_dim = train_dataset[0][0].numpy().shape[1:]
        n_classes = 10

        return train_dataset, val_dataset, input_dim, n_classes

        input_dim = train_dataset







