import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from lamb_optim.lambc import Lambc





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LAMB with Adaptive Learning Rate Clipping')
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--n', type=int, default=3)
	parser.add_argument('--dataset', type=str, default='MNIST')
	parser.add_argument('--device', type=str, default='cpu')
	args = parser.parse_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('test')