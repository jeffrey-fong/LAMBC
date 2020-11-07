import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
import numpy as np
from lamb_optim.lambc import Lambc
from models import lenet, resnet
from model import Model
from torch.utils.tensorboard import SummaryWriter

avg_accuracy = 0.0
num_update = 0
log_writer = None


def train(model, train_loader, test_loader):
    global num_update

    print('train')
    opt = Lambc(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                betas=(.9, .999), adam=False, clip=args.clip,
                clip_bound=args.clip_bound)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(args.epochs):
        # epoch_loss = 0
        # total, correct = 0, 0
        model.train()
        for image, target in train_loader:
            image, target = image.to(args.device), target.to(args.device)
            opt.zero_grad()
            output = model.forward(image)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            _, predicted = output.max(1)
            total = target.size(0)
            correct = predicted.eq(target).sum().item()
            num_update += 1

            # Print the current status
            # print("-" * 25)
            # print("Epoch:{:10}".format(epoch))
            # print("Train Loss:{:10.6}\t".format(np.mean(losses)))
            # print("Accuracy:{:10.6}\t".format(100. * correct / total))
            log_writer.add_scalar('train/acc', correct / total, num_update)

        test(model, test_loader)

    # Save and update the model after every full training round
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    model.save(args.save_dir + "model" + ".pt")


def test(model, test_loader):
    global avg_accuracy
    # if os.path.exists(args.save_dir + 'model.pt'):
    #     model.load(args.save_dir + 'model.pt')
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
    print("Accuracy:{:10.6}\t".format(100. * correct / total))
    avg_accuracy += (100. * correct / total)
    log_writer.add_scalar('test/acc', correct / total, num_update)


def main():
    global avg_accuracy
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
        pass  # KIV

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)

    # for i in range(10):
    #     print('Epoch:', i)
    # Network Model
    model = Model(args).to(args.device)
    train(model, train_loader, test_loader)
    time.sleep(1.0)
    test(model, test_loader)

    print('Average test accuracy:', avg_accuracy / 10.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAMB with Adaptive Learning Rate Clipping')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip', type=bool, default=True)
    parser.add_argument('--clip_bound', type=float, default=[0.01, 5.0])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='./past_models/')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 1
    os.makedirs("results", exist_ok=True)
    log_writer = SummaryWriter(log_dir=f"results/clip_{args.clip}_{args.dataset}_{seed}")

    torch.manual_seed(seed)
    main()
