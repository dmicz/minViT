import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import ViT

dataset = 'cifar-10'
block_size = 1

data_dir = os.path.join('data', dataset)
train_images = np.memmap(os.path.join(data_dir, 'train-images.bin'), dtype=np.float32, mode='r')
train_labels = np.memmap(os.path.join(data_dir, 'train-labels.bin'), dtype=np.int64, mode='r')
test_images = np.memmap(os.path.join(data_dir, 'test-images.bin'), dtype=np.float32, mode='r')
test_labels = np.memmap(os.path.join(data_dir, 'test-labels.bin'), dtype=np.int64, mode='r')

def get_batch(images, labels, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size], labels[i:i+batch_size]