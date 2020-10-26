import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from lamb_optim.lambc import Lambc
from models import lenet
from model import Model


def train(model):
	print('train')

def test(model):
	print('test')

def main():
	model = Model(args)
	#model.save(args.save_dir + 'model.pt')

	# Dataset
	if args.dataset == 'MNIST':
		transform = transforms.Compose([transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))])
		train_set = torchvision.datasets.MNIST('../image_datasets', train=True, 
										download=True, transform=transform)
		test_set = torchvision.datasets.MNIST('../image_datasets', train=False, 
										download=True, transform=transform)
	elif args.dataset == 'CIFAR10':
		transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		transform_test = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		train_set = torchvision.datasets.CIFAR10('../image_datasets', train=True, 
									download=True, transform=transform_train)
		test_set = torchvision.datasets.CIFAR10('../image_datasets', train=False, 
									download=True, transform=transform_test)
	elif args.dataset == 'ImageNet':
		pass 		# KIV

	train_loader = torch.utils.data.DataLoader(train_set, 
					batch_size=args.batch_size, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(test_set, 
					batch_size=100*args.batch_size, shuffle=True, num_workers=2)

	train(model)
	test(model)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LAMB with Adaptive Learning Rate Clipping')
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--epoch', type=int, default=150)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--n', type=int, default=3)
	parser.add_argument('--dataset', type=str, default='MNIST')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--save_dir', type=str, default='./past_models/')
	args = parser.parse_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	main()