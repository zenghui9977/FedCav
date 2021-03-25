import time
import copy
import sys
import torch
import numpy as np
from aggregation import FedAvgAggregation_mode_2, FedAvgAggregation_mode_1,FedCavAggregation_with_clip_loss
from local_update import Local_Update
from hyper_parameters import arg_parser
from utils import get_data, init_model, inference, save_info, print_training_type, Logger
from create_clients import noniid_equal_bias_mix, noniid_equal_label_sigma_alpha


start_time = time.time()
sys.stdout = Logger("sigma_150.log") 
args = arg_parser()

# use GPU

# read the dataset
trainset, testset = get_data(args)

user_group = noniid_equal_label_sigma_alpha(trainset, args.n, 5, args.sigma, args.alpha)

user_id_list = range(args.n)

model_dir = './model/'
FedAvg_model_dir = '/FedAvg/'
FedCav_model_dir = '/FedCav/'
FedProx_model_dir = '/FedProx/'

model_name = str(args.dataset) +'_sigma_' +str(args.sigma) + '.pth'

FedAvg_model = init_model(args, trainset[0][0].shape)
FedCav_model = init_model(args, trainset[0][0].shape)
FedProx_model = init_model(args, trainset[0][0].shape)

FedAvg_model = FedAvg_model.cuda()
FedCav_model = FedCav_model.cuda()
FedProx_model = FedProx_model.cuda()

FedAvg_model.train()
FedCav_model.train()
FedProx_model.train()

FedAvg_model_weights = FedAvg_model.state_dict()
FedCav_model_weights = FedCav_model.state_dict()
FedProx_model_weights = FedProx_model.state_dict()

FedAvg_accuracy_list, FedAvg_loss_list = [], []
FedCav_accuracy_list, FedCav_loss_list = [], []
FedProx_accuracy_list, FedProx_loss_list = [], []

for e in range(args.epoch):
    print(f'\n | Global Training Round : {e+1} |\n')

    m = max(int(args.q * args.n), 1)
    
    sample_user_idx = np.random.choice(user_id_list, m, replace=False)

    client_datanum = [len(user_group[sui]) for sui in sample_user_idx]

    fedavg_clients_weights, fedavg_clients_loss = [], []
    fedcav_clients_weights, fedcav_clients_loss = [], []
    fedprox_clients_weights, fedprox_clients_loss = [], []

    for u in sample_user_idx:
        print('----------------------------------------------------')
        print('Client %d training......'%u)
        client_temp = Local_Update(trainset, idxs=user_group[u], local_bs=args.local_bs, local_ep=args.local_ep, lr=args.lr, momentum=args.fedprox_momentum, mu=args.fedprox_mu)

        print_training_type('FedAvg')
        fedavg_local_w, fedavg_local_loss = client_temp.update_weights(model=copy.deepcopy(FedAvg_model), optim='sgd')

        print_training_type('FedCav')
        fedcav_local_w, fedcav_local_loss = client_temp.update_weights(model=copy.deepcopy(FedCav_model), optim='sgd')

        print_training_type('FedProx')
        fedprox_local_w, fedprox_local_loss = client_temp.update_weights(model=copy.deepcopy(FedProx_model), optim='fedprox')

        # record the w and local loss list
        # FedAvg
        fedavg_clients_weights.append(copy.deepcopy(fedavg_local_w))
        fedavg_clients_loss.append(copy.deepcopy(fedavg_local_loss))
        # FedCav
        fedcav_clients_weights.append(copy.deepcopy(fedcav_local_w))
        fedcav_clients_loss.append(copy.deepcopy(fedcav_local_loss))
        # FedProx
        fedprox_clients_weights.append(copy.deepcopy(fedprox_local_w))
        fedprox_clients_loss.append(copy.deepcopy(fedprox_local_loss))

    
    # Aggregation
    # FedAvg
    FedAvg_model_weights = FedAvgAggregation_mode_2(fedavg_clients_weights, client_datanum)
    FedAvg_model.load_state_dict(FedAvg_model_weights)
    # FedCav
    ev_loss = list(np.array(fedcav_clients_loss)[:,0])
    # FedCav_model_weights = FedCavAggregation(fedcav_clients_weights, ev_loss)
    FedCav_model_weights = FedCavAggregation_with_clip_loss(fedcav_clients_weights, ev_loss)
    # FedCav_model_weights = FedCavAggregation_with_clip_update(FedCav_model_weights, fedcav_clients_weights, ev_loss)
    FedCav_model.load_state_dict(FedCav_model_weights)
    # FedProx
    FedProx_model_weights = FedAvgAggregation_mode_1(fedprox_clients_weights)
    FedProx_model.load_state_dict(FedProx_model_weights)

    # whole testset Inference
    fedavg_whole_acc, fedavg_whole_loss = inference(FedAvg_model, testset)
    fedcav_whole_acc, fedcav_whole_loss = inference(FedCav_model, testset)
    fedprox_whole_acc, fedprox_whole_loss = inference(FedProx_model, testset)

    # spec test inference
    # fedavg_spec_acc, fedavg_spec_loss = inference(FedAvg_model, spec_test_dataset)
    # fedcav_spec_acc, fedcav_spec_loss = inference(FedCav_model, spec_test_dataset)
    # fedprox_spec_acc, fedprox_spec_loss = inference(FedProx_model, spec_test_dataset)

    # FedAvg result
    FedAvg_accuracy_list.append(fedavg_whole_acc)
    # FedAvg_spec_accuracy_list.append(fedavg_spec_acc)
    FedAvg_loss_list.append(fedavg_whole_loss)
    # FedAvg_spec_loss_list.append(fedavg_spec_loss)

    # FedCav result
    FedCav_accuracy_list.append(fedcav_whole_acc)
    # FedCav_spec_accuracy_list.append(fedcav_spec_acc)
    FedCav_loss_list.append(fedcav_whole_loss)
    # FedCav_spec_loss_list.append(fedcav_spec_loss)

    # FedProx result
    FedProx_accuracy_list.append(fedprox_whole_acc)
    # FedProx_spec_accuracy_list.append(fedprox_spec_acc)
    FedProx_loss_list.append(fedprox_whole_loss)
    # FedProx_spec_loss_list.append(fedprox_spec_loss)

    # print the result in this epoch
    print(f'|---- Training Result in epoch {e + 1} global rounds:')
    print("|------ FedAvg ------|")
    print("whole test set Accuracy: {:.2f}%".format(100 * fedavg_whole_acc))
    print("whole test set Loss: {:.2f}".format(fedavg_whole_loss))
    # print("part test set Accuracy: {:.2f}%".format(100 * fedavg_spec_acc))
    # print("part test set Loss: {:.2f}".format(fedavg_spec_loss))
    print("|------ FedCav ------|")
    print("whole test set Accuracy: {:.2f}%".format(100 * fedcav_whole_acc ))
    print("whole test set Loss: {:.2f}".format(fedcav_whole_loss))
    # print("part test set Accuracy: {:.2f}%".format(100 * fedcav_spec_acc))
    # print("part test set Loss: {:.2f}".format(fedcav_spec_loss))
    print("|------ FedProx ------|")
    print("whole test set Accuracy: {:.2f}%".format(100 * fedprox_whole_acc))
    print("whole test set Loss: {:.2f}".format(fedprox_whole_loss))
    # print("part test set Accuracy: {:.2f}%".format(100 * fedprox_spec_acc))
    # print("part test set Loss: {:.2f}".format(fedprox_spec_loss))

# model_name = str(args.dataset) + '_sigma_' + str(args.sigma) + '_alpha_' + str(args.alpha) + '.pth'
# fedprox_model_name = str(args.dataset) + '_sigma_' + str(args.sigma) + '_mu_' + str(args.fedprox_mu) + '.pth'

# torch.save(FedAvg_model, model_dir + FedAvg_model_dir + model_name)
# torch.save(FedCav_model, model_dir + FedCav_model_dir + model_name)
# torch.save(FedProx_model, model_dir + FedProx_model_dir + fedprox_model_name)

result_folder = '/result/'
name_list = ["FedAvg_accuracy", "FedCav_accuracy", "FedProx_accuracy", "FedAvg_loss", "FedCav_loss", "FedProx_loss"]

result_list = [FedAvg_accuracy_list, FedCav_accuracy_list, FedProx_accuracy_list, FedAvg_loss_list, FedCav_loss_list, FedProx_loss_list]

file_name = str(args.dataset) + '_sigma_' + str(args.sigma) + '_alpha_' + str(args.alpha) + '.csv'
save_info(result_list, name_list, result_list, file_name)
print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
