# LAMBC

This repository contains code for [LAMBC](https://arxiv.org/abs/2011.13584), an optimizer that trains with layer-wise adaptive learning rate. LAMBC improves upon [LAMB](https://arxiv.org/abs/1904.00962) by clipping the trust ratio which stabilizes its magnitude and prevent extreme values.

Experiments on image classification tasks (CIFAR-10 and down-sampled ImageNet-64) validate LAMBC's improvement over LAMB across different batch sizes and clipping bounds

## Installation

LAMBC requires the following:
* Linux
* Python 3.6 or later
* PyTorch 1.2 or later
* CUDA 10 or later

`d`






Compatible with Python 3.6 onwards. Pytorch implementation.

## Running the Program

Set the hyperparameters (lr, clipping bounds, epochs, batch size, etc.) in `main.py` or in the command line while running `main.py`.

`python3 main.py`

Selection of model for CIFAR10 is done in `model.py`. It is set to ResNet-18 currently. Do the same for ImageNet.
