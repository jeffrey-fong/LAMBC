import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import json

import numpy as np

from dataset import ImageNetDataset
from lamb_optim.lambc import Lambc
from models import lenet, resnet
from model import Model
from torch.utils.tensorboard import SummaryWriter

avg_accuracy = 0.0
train_iter, test_iter = 0, 0
trust_ratio_list = []
writer = None


def train(model, train_loader):
    global train_iter, writer, trust_ratio_list
    model.train()
    opt = Lambc(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                betas=(.9, .999), adam=False, clip=args.clip,
                clip_bound=args.clip_bound)
    criterion = nn.CrossEntropyLoss()
    losses = []

    correct, total = 0, 0
    for batch_idx, (image, target) in enumerate(train_loader):
        image, target = image.to(args.device), target.to(args.device)
        opt.zero_grad()
        output = model.forward(image)
        loss = criterion(output, target)
        writer.add_scalar("Loss/train", loss, train_iter)
        loss.backward()
        _, trust_list = opt.step()
        if batch_idx == 0:
            trust_ratio_list.append([trust_list])
        else:
            trust_ratio_list[-1].append(trust_list)
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        writer.add_scalar("Accuracy/train", 100. * predicted.eq(target).int().sum().item() / target.size(0), train_iter)
        train_iter += 1

    # Print the current status
    print("Train Loss:{:10.6}\t".format(np.mean(losses)))
    print("Accuracy:{:10.6}\t".format(100. * correct / total))
    writer.flush()

    # Save and update the model after every full training round
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    model.save(args.save_dir + "model" + ".pt")


def test(model, test_loader):
    global avg_accuracy, test_iter, writer

    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct, total = 0, 0
    for image, target in test_loader:
        image, target = image.to(args.device), target.to(args.device)
        output = model.forward(image)
        loss = criterion(output, target)
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    # Print the current status
    print("Test Loss:{:10.6}\t".format(np.mean(losses)))
    print("Accuracy:{:10.6}\t".format(100. * correct / total))
    avg_accuracy += (100. * correct / total)
    writer.add_scalar("Accuracy/test", 100. * correct / total, test_iter)
    test_iter += 1
    writer.flush()


def main():
    global avg_accuracy, trust_ratio_list
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
        train_set = ImageNetDataset(train=True)
        test_set = ImageNetDataset(train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    model = Model(args).to(args.device)
    for epoch in range(args.epochs):
        print("Epoch:{:10}".format(epoch))
        train(model, train_loader)
        test(model, test_loader)
        print("-" * 25)

    # Save trust ratio log as (epoch x iter x layers)
    with open('./trust_ratio.json', 'w') as fout:
        json.dump(trust_ratio_list, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAMB with Trust Ratio Clipping')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip', type=bool, default=True)
    parser.add_argument('--clip_bound', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='./past_models/')
    args = parser.parse_args()

    seed = 1
    torch.manual_seed(seed)
    writer = SummaryWriter(log_dir=os.path.join("results", f"clip_{args.clip}_{args.dataset}_{seed}_{args.batch_size}"))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
