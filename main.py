import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from lamb_optim.lambc import Lambc
from models import lenet
from model import Model


def train(args, model):
	print('train')

def test(args, model):
	print('test')

def main(args):
	model = Model(args)
	train(args, model)
	test(args, model)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LAMB with Adaptive Learning Rate Clipping')
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--epoch', type=int, default=150)
	parser.add_argument('--n', type=int, default=3)
	parser.add_argument('--dataset', type=str, default='MNIST')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--save_dir', type=str, default='./')
	args = parser.parse_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	main(args)