import torch
from torch.utils.data import DataLoader, Dataset
from fedprox_optimizer import FedProxOptimizer
import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone(), torch.tensor(label)


class DatasetSplit_Fake(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if label + 1 > 9:
            label = 0
        else:
            label = int(label + 1)
        return torch.tensor(image), torch.tensor(label)



class Local_Update():
    def __init__(self, dataset, idxs, local_bs, local_ep, lr, momentum, mu):
        self.traindata = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'       
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.local_epochs = local_ep
        self.lr = lr
        self.momentum = momentum
        self.mu = mu

    def compute_loss(self, model):
        loss_list = []
        for batch_idx, (images, labels) in enumerate(self.traindata):
            images = images.cuda()
            labels = labels.cuda()

            log_probs = model(images)

            loss = self.criterion(log_probs, labels)
            loss_list.append(loss.item())
        return np.sum(loss_list)

    def update_weights(self, model, optim):

        model.train()
        epoch_loss = []

        if optim == 'fedprox':
            optimizer = FedProxOptimizer(model.parameters(), lr=self.lr, momentum=self.momentum, gmf=0, ratio=1, mu=self.mu)
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        
        loss_0 = self.compute_loss(model)
        epoch_loss.append(loss_0)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.traindata):
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss))
        print('local model training finished, the epoch loss is %s'%epoch_loss)
        return model.state_dict(), epoch_loss

    def update_weights_with_detection(self, model, epoch, optim):
        model.train()
        epoch_loss = []

        if optim == 'fedprox':
            optimizer = FedProxOptimizer(model.parameters(), lr=1e-2, momentum=0.5, gmf=0, ratio=1, mu=0.2)
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                        momentum=0.9)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=1e-4)

        loss_0 = self.compute_loss(model)
        epoch_loss.append(loss_0)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.traindata):
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss))
        print(epoch_loss)
        return model.state_dict(), epoch_loss

