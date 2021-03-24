import copy
import torch
import numpy as np
# three part FedAvg FedCav FedProx

def compte_parameter_L2(w):
    w_temp = copy.deepcopy(w[0])
    for key in w_temp.keys():
        print("----------", key)
        # for i in range(len(w)):

            # print(torch.norm(w[i][key]))


def FedAvgAggregation_mode_1(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    # print(w_avg)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def FedAvgAggregation_mode_2(w, client_num):
    # print(client_num)
    client_avg_w = np.array(client_num) / sum(client_num)

    w_avg = copy.deepcopy(w[0])
    # print(w_avg)
    for key in w_avg.keys():
        w_avg[key] = client_avg_w[0] * w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += client_avg_w[i] * w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def FedCavAggregation(w, loss):
    """
    :param w:
    :param loss:
    :return:
    """
    loss_avg_w = torch.nn.functional.softmax(torch.Tensor(loss), dim=0)
    # print(loss_avg_w)
    w_avg = copy.deepcopy(w[0])
    # print(w_avg)
    for key in w_avg.keys():
        w_avg[key] = loss_avg_w[0] * w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += loss_avg_w[i] * w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def FedCavAggregation_with_clip_loss(w, loss):

    # clip loss first
    clip_num = np.mean(loss)
    loss = [min(i, clip_num) for i in loss]
    # print(loss)
    # compute the weights
    loss_avg_w = torch.nn.functional.softmax(torch.Tensor(loss), dim=0)
    print(loss_avg_w)
    w_avg = copy.deepcopy(w[0])
    # print(w_avg)
    for key in w_avg.keys():
        w_avg[key] = loss_avg_w[0] * w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += loss_avg_w[i] * w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def FedCavAggregation_with_clip_update(w_t, w_t1, loss):

    w_temp_update = copy.deepcopy(w_t1)
    # compute updates
    for key in w_t.keys():
        for i in range(len(w_t1)):
            w_temp_update[i][key] = w_t1[i][key] - w_t[key]

    print(w_temp_update)
    # clip by medium

    for key in w_t.keys():
        print(key)
        updates_norm_list = [torch.norm(w_temp_update[i][key]) for i in range(len(w_t1))]
        print(updates_norm_list)

        clip_bound = torch.median(torch.tensor(updates_norm_list))
        print(clip_bound)
        factor = [updates_norm_list[i]/clip_bound for i in range(len(w_t1))]
        for i in range(len(factor)):
            if factor[i] > 1.0:
                factor[i] = 1.0
            else:
                factor[i] = factor[i].numpy()
        print("factor", factor)
        break