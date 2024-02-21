import os
import numpy as np
import torch

from model import ViTConfig, ViT

dataset = 'cifar-10'
block_size = 1
device = 'cpu'
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = os.path.join('data', dataset)
train_images = np.memmap(os.path.join(data_dir, 'train-images.bin'), dtype=np.float32, mode='r')
train_labels = np.memmap(os.path.join(data_dir, 'train-labels.bin'), dtype=np.int64, mode='r')
test_images = np.memmap(os.path.join(data_dir, 'test-images.bin'), dtype=np.float32, mode='r')
test_labels = np.memmap(os.path.join(data_dir, 'test-labels.bin'), dtype=np.int64, mode='r')

def get_batch(split, batch_size):
    image_data, label_data = (train_images, train_labels) if split == 'train' else (test_images, test_labels)
    ix = torch.randint(len(image_data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy(image_data[i:i+block_size].astype(np.float32)) for i in ix])
    y = torch.tensor([torch.from_numpy(label_data[i:i+block_size].astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)