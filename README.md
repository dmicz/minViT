
# minViT

Educational repository with a minimal implementation of a small-sized Vision Transformer (ViT) using PyTorch. See `minViT.ipynb` for a walkthrough of the implementation and `finetune.ipynb` on how to finetune larger ViTs using PyTorch to achieve decent performance on small dataset tasks. Notebooks are also available to view on [my blog](https://dmicz.github.io/machine-learning/minvit/).

Corresponding YouTube video [here](https://www.youtube.com/watch?v=krTL2uH-L40).

## install

```
pip install torch numpy torchvision
```

## usage

Currently, the only dataset code is provided for is CIFAR-10, located in `data/cifar-10/prepare.py`. Run this script from the minViT directory:

```
$ python data/cifar-10/prepare.py
```

This loads the cifar-10 dataset using `torchvision.datasets`. Then, a ViT can be configured by modifying the `ViTConfig` class at the beginning of `model.py` (TODO: add CLI config).

Config of the training hyperparameters is also done through modifying `train.py`, training can be started by running it:

```
$ python train.py
```

Model saving, other features to be added at a future date. :)