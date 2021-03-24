import os
import torch
import csv
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import MLP, CNNMnist, Cifar_model, Cifar_model2, FMnist_model, FMnist_model2
from local_update import DatasetSplit

def get_data(args):
    if args.dataset == 'mnist':
        data_dir = './data'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=False,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=False,
                                      transform=apply_transform)

    elif args.dataset == 'cifar':
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                      transform=apply_transform)

    elif args.dataset == 'fmnist':
        data_dir = '/data/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    return train_dataset, test_dataset



def init_model(args, image_shape):
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist()
        elif args.dataset == 'cifar':
            # global_model = CNNCifar()
            global_model = Cifar_model()
        elif args.dataset == 'fmnist':
            global_model = FMnist_model()
    elif args.model == 'mlp':
        if args.dataset == 'mnist':
            img_size = image_shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)

    return global_model


def inference(model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().cuda()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

def save_info(data_list, name_list, file_dir, file_name):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    data = pd.DataFrame(index=name_list, data=data_list)

    data.to_csv(os.path.join(file_dir, file_name), mode='a')

def print_training_type(type_name):
    print('---------------------')
    print('the %s is training now....' % type_name)
    print('---------------------')
