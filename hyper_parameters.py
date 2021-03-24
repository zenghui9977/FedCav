import argparse

def arg_parser():

    parser = argparse.ArgumentParser()

    # client distribution hyperparameters
    parser.add_argument('--sigma', type=int, default=0, help='the measurement of imbalanced class distribution')
    parser.add_argument('--alpha', type=float, default=0, help='the fraction of new added class')

    # training hyperparameters, dataset, model ......
    parser.add_argument('--dataset', type=str, default='mnist', help='the training dataset:mnist, fmnist, cifar')
    parser.add_argument('--model', type=str, default='cnn', help='the training model of each client')
    
    parser.add_argument('--n', type=int, default=100, help='the total number of participants in the network')
    parser.add_argument('--q', type=float, default=0.1, help='the sample ratio of participants in each round')
    parser.add_argument('--epoch', type=int, default=20, help='the communication round in FL framework')
    parser.add_argument('--local_ep', type=int, default=5, help='the local epochs of the nodes/participants')
    parser.add_argument('--local_bs', type=int, default=10, help='the local batch size of the nodes/participants')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate of local model')
    parser.add_argument('--momentum', type=float, default=0.5, help='the momentum of SGD(default as 0.5)')

    parser.add_argument('--fedprox_mu', type=float, default=0, help='the hyperparameter of fedprox')
    parser.add_argument('--fedprox_momentum', type=float, default=0.9, help='the momentum of fedprox optimizer')
    

    # result saving hyperparameters
    parser.add_argument('--result_folder', type=str, default='/result/', help='the directory for saving the result')
    

    args = parser.parse_args()
    return args