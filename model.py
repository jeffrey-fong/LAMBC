import torch
import torch.nn as nn

import os

from models import lenet, resnet, alexnet

class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		# Model
		if args.dataset == 'MNIST':
			self.model = lenet.LeNet(input_dim=1)
		elif args.dataset == 'CIFAR10':
			#self.model = resnet.ResNet18()
			self.model = alexnet.AlexNet()

	def forward(self, image):
		output = self.model(image)
		return output

	def save(self, path):
		checkpoint = {'model': self.state_dict()}
		torch.save(checkpoint, path)

	def load(self, path):
		path = os.path.abspath(path)
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint['model'])