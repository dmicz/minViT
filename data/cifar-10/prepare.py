import os
import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform)

train_images = np.array([train_dataset.data[i].numpy() for i in range(len(train_dataset))])
train_labels = np.array([train_dataset.targets[i] for i in range(len(train_dataset))], dtype=np.int64)
test_images = np.array([test_dataset.data[i].numpy() for i in range(len(test_dataset))])
test_labels = np.array([test_dataset.targets[i] for i in range(len(test_dataset))], dtype=np.int64)


train_images.tofile(os.path.join(os.path.dirname(__file__), 'train-images.bin'))
train_labels.tofile(os.path.join(os.path.dirname(__file__), 'train-labels.bin'))
test_images.tofile(os.path.join(os.path.dirname(__file__), 'test-images.bin'))
test_labels.tofile(os.path.join(os.path.dirname(__file__), 'test-labels.bin'))

print(f"Train data saved with shape {train_images.shape} and {train_labels.shape}")
print(f"Test data saved with shape {test_images.shape} and {test_labels.shape}")