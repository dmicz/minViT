import os
import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform)

train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))], dtype=torch.uint8)
train_images = train_images.numpy().astype(np.float32)
train_labels = train_labels.numpy()

train_data = np.column_stack((train_labels.reshape(-1, 1), train_images.reshape(len(train_images), -1)))
train_data.tofile('train.bin')

test_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))], dtype=torch.uint8)
test_images = test_images.numpy().astype(np.float32)
test_labels = test_labels.numpy()

test_data = np.column_stack((test_labels.reshape(-1, 1), test_images.reshape(len(test_images), -1)))
test_data.tofile('test.bin')

print(f"Train data saved with shape {train_data.shape}")
print(f"Test data saved with shape {test_data.shape}")