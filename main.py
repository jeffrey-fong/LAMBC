import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

from lamb_optim.lambc import Lambc
from models import lenet
from model import Model


def train(model, train_loader):
	print('train')
	model.train()
	opt = Lambc(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
					betas=(.9, .999), adam=False)
	criterion = nn.CrossEntropyLoss()
	losses = []

	for epoch in range(args.epochs):
		epoch_loss = 0
		total, correct = 0, 0
		for image, target in train_loader:
			image, target = image.to(args.device), target.to(args.device)
			opt.zero_grad()
			output = model.forward(image)
			loss = criterion(output, target)
			loss.backward()
			_, trust_ratio_list = opt.step()
			losses.append(loss.item())
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()

			# Record norm ratios
			norm_weights, norm_grads = [], []
			for param in model.parameters():
				weights = param.data.cpu().detach().numpy()
				grads = param.grad.cpu().detach().numpy()
				norm_weights.append(np.sum(np.power(weights, 2)))
				norm_grads.append(np.sum(np.power(grads, 2)))
			#print(norm_weights)
			#print(norm_grads)
			#print(np.divide(norm_weights, norm_grads))
			print(trust_ratio_list)
			break
		break

		# Print the current status
		print("-" * 25)
		print("Epoch:{:10}".format(epoch))
		print("Train Loss:{:10.6}\t".format(np.mean(losses)))
		print("Accuracy:{:10.6}\t".format(100.*correct/total))

	# Save and update the model after every full training round
	model.save(args.save_dir + "model" + ".pt")

def test(model, test_loader):
	if os.path.exists(args.save_dir + 'model.pt'):
			model.load(args.save_dir + 'model.pt')
	print('test')
	model.eval()
	criterion = nn.CrossEntropyLoss()
	losses = []
	total, correct = 0, 0
	for image, target in test_loader:
		image, target = image.to(args.device), target.to(args.device)
		output = model.forward(image)
		loss = criterion(output, target)
		losses.append(loss.item())
		_, predicted = output.max(1)
		total += target.size(0)
		correct += predicted.eq(target).sum().item()
	# Print the current status
	print("-" * 25)
	print("Test Loss:{:10.6}\t".format(np.mean(losses)))
	print("Accuracy:{:10.6}\t".format(100.*correct/total))


def main():
	# Network Model
	model = Model(args).to(args.device)
	#model.save(args.save_dir + 'model.pt')

	# Dataset
	if args.dataset == 'MNIST':
		transform = transforms.Compose([transforms.Resize((32, 32)),
						transforms.ToTensor(),
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

	#train(model, train_loader)
	#test(model, test_loader)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LAMB with Adaptive Learning Rate Clipping')
	parser.add_argument('--lr', type=float, default=0.0025)
	parser.add_argument('--weight_decay', type=float, default=0.0)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--n', type=int, default=3)
	parser.add_argument('--dataset', type=str, default='CIFAR10')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--save_dir', type=str, default='./past_models/')
	args = parser.parse_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	main()