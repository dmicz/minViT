import os
import numpy as np
import torch

from model import ViTConfig, ViT

dataset = 'cifar-10'
block_size = 32 * 32 * 3
device = 'cpu'
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

data_dir = os.path.join('data', dataset)
train_images = np.memmap(os.path.join(data_dir, 'train-images.bin'), dtype=np.float32, mode='r')
train_labels = np.memmap(os.path.join(data_dir, 'train-labels.bin'), dtype=np.int64, mode='r')
test_images = np.memmap(os.path.join(data_dir, 'test-images.bin'), dtype=np.float32, mode='r')
test_labels = np.memmap(os.path.join(data_dir, 'test-labels.bin'), dtype=np.int64, mode='r')

def get_batch(split, batch_size):
    image_data, label_data = (train_images, train_labels) if split == 'train' else (test_images, test_labels)
    ix = torch.randint(len(image_data) // block_size, (batch_size,))

    x = torch.stack([torch.from_numpy(image_data[i*block_size:(i+1)*block_size].reshape(3, 32, 32)) for i in ix.numpy()])
    y = torch.from_numpy(label_data[ix.numpy()])
#    x = torch.stack([torch.from_numpy(image_data[i:i+block_size].astype(np.float32)) for i in ix])
#    y = torch.tensor([torch.from_numpy(label_data[i:i+block_size].astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9


print("Initializing a new model from scratch")
config = ViTConfig()
model = ViT(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
batch_size = 64
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for _ in range(len(train_images) // batch_size):
        inputs, labels = get_batch('train', batch_size)
        
        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted y by passing x to the model
        logits, loss = model(inputs, targets=labels)

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        running_loss += loss.item()

    # Print statistics
    print(f'Epoch {epoch+1}, Loss: {running_loss / (len(train_images) // batch_size)}')

print('Finished Training')