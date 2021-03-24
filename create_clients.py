import torch
import collections
import numpy as np
from torchvision import datasets, transforms
from heapq import merge
import itertools
from matplotlib import pyplot as plt

from hyper_parameters import arg_parser
from utils import get_data

def split_data_by_label(dataset):
    # 将数据依据label分开，分成一个一个桶
    if torch.is_tensor(dataset.targets):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)
    # print(labels)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    mnist_counter = collections.Counter(idxs_labels[1, :])
    # print(mnist_counter)

    label_num = len(mnist_counter)

    label_idxs_buckets = []
    pos_temp = 0
    for i in range(label_num):
        label_idxs_buckets.append(list(idxs[pos_temp: pos_temp + mnist_counter[i]]))
        pos_temp += mnist_counter[i]
        # print(len(label_idxs_buckets[i]))
    return label_idxs_buckets


def distribute_label_table(label_num_list, user_num):
    label_table = []
    label_num = len(label_num_list)
    for i in label_num_list:
        temp = []
        for j in reversed(label_num_list):
            temp.append([i, j])

        if len(temp) < user_num//label_num:
            for k in range(user_num//label_num - len(temp)):
                temp.append(temp[k])
        label_table.append(temp)
    # print(label_table)
    label_table = [item for sublist in label_table for item in sublist]
    # print(label_table)
    return label_table


def iid_equal(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def noniid_equal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    if torch.is_tensor(dataset.targets):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0).astype(int)
    return dict_users


def noniid_unequal(dataset, num_users, min_size, batch_size):
    dict_users = {i: np.array([]) for i in range(num_users)}
    label_buckets = split_data_by_label(dataset)

    label_table = distribute_label_table(range(len(label_buckets)), num_users)

    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    # 每个user先分配一部分数据
    for i in range(num_users):
        user_label = label_table[i]
        shards_size = min_size // len(user_label)
        for u_l in user_label:
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
            label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

    # 把剩下的进行分配
    for i in range(len(label_buckets)):

        if len(label_buckets[i]) != 0:
            rest_size = len(label_buckets[i]) - len(label_buckets[i]) % batch_size
            rest_set = np.random.choice(list(label_buckets[i]), rest_size, replace=False)
            dict_users[i * i] = np.concatenate((dict_users[i], list(rest_set)),axis=0).astype(int)

    return dict_users


def noniid_eqaul_bias(dataset, num_users, alpha, beta):
    '''
    :param dataset:
    :param num_users: number of clients
    :param alpha: fraction of label
    :param beta: fraction of clients own the alpha label, we often set alpha = beta
    :return:
    '''
    dict_users = {i: np.array([]) for i in range(num_users)}
    label_buckets = split_data_by_label(dataset)

    label_num = len(label_buckets)
    items_num = len(dataset) // num_users

    label_buckets = [set(i) for i in label_buckets]

    label_index_list = list(range(label_num))

    pre_label_table_list = label_index_list[:int(alpha * label_num)]
    las_label_table_list = label_index_list[int(alpha * label_num):]

    pre_label_table = distribute_label_table(pre_label_table_list,int(beta * num_users))
    las_label_table = distribute_label_table(las_label_table_list, int((1 - beta) * num_users))

    label_table = list(merge(pre_label_table, las_label_table))

    for i in range(num_users):
        user_label = label_table[i]

        shards_size = items_num // len(user_label)
        for u_l in user_label:
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            # print(dict_users[i])
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

    return dict_users


def noniid_unequal_bias(dataset, num_users, alpha, beta, batch_size, min_size):
    '''
    :param dataset:
    :param num_users:
    :param alpha:
    :param beta:
    :param batch_size:
    :param min_size:
    :return:
    '''
    dict_users = {i: np.array([]) for i in range(num_users)}
    label_buckets = split_data_by_label(dataset)

    label_num = len(label_buckets)
    items_num = len(dataset) // num_users

    label_buckets = [set(i) for i in label_buckets]

    label_index_list = list(range(label_num))

    pre_label_table_list = label_index_list[:int(alpha * label_num)]
    las_label_table_list = label_index_list[int(alpha * label_num):]

    pre_label_table = distribute_label_table(pre_label_table_list,int(beta * num_users))
    las_label_table = distribute_label_table(las_label_table_list, int((1 - beta) * num_users))

    label_table = list(merge(pre_label_table, las_label_table))

    for i in range(num_users):
        user_label = label_table[i]
        shards_size = min_size // len(user_label)
        for u_l in user_label:
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

    for i in range(label_num):
        if len(label_buckets[i]) != 0:
            rest_size = len(label_buckets[i]) - len(label_buckets[i]) % batch_size
            print(rest_size)
            rest_set = np.random.choice(list(label_buckets[i]), rest_size, replace=False)
            print(len(rest_set))
            dict_users[i * i] = np.concatenate((dict_users[i], list(rest_set)),axis=0).astype(int)

    return dict_users


def noniid_equal_bias_mix(dataset, num_users, alpha, beta, theta):
    dict_users = {i: np.array([]) for i in range(num_users)}
    label_buckets = split_data_by_label(dataset)

    label_num = len(label_buckets)
    items_num = len(dataset) // num_users

    label_buckets = [set(i) for i in label_buckets]

    label_index_list = list(range(label_num))

    pre_label_table_list = label_index_list[:int(alpha * label_num)]
    las_label_table_list = label_index_list[int(alpha * label_num):]

    pre_label_table = distribute_label_table(pre_label_table_list, int(beta * num_users))
    las_label_table = distribute_label_table(las_label_table_list, int((1 - beta) * num_users))

    # 先分配有稀有label的节点
    # 方案先分配50%常见数据, 常见数据是IID的，在随机分配50%不常见数据
    for i in range(len(pre_label_table)):
        # 常见数据的数据量
        normal_datasize = int(theta * items_num)
        # 常见数据的标签个数
        normal_label_num = int((1 - alpha) * label_num)
        # 每个常见标签平均拿数据
        normal_data_account = normal_datasize
        for j in range(len(las_label_table_list)):
            n_l = las_label_table_list[j]
            if j == len(las_label_table_list) - 1:
                temp = normal_data_account
            else:
                temp = int(normal_datasize/normal_label_num)

            rand_set = np.random.choice(list(label_buckets[n_l]), temp, replace=True)
            normal_data_account -= temp
            label_buckets[n_l] = label_buckets[n_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)), axis=0).astype(int)
        # 不常见数据的分配
        abnormal_datasize = items_num - normal_datasize
        user_labels = pre_label_table[i]
        for u_l in user_labels:
            shards_size = abnormal_datasize // len(user_labels)
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

        # print(len(dict_users[i]))
    # 分配常用数据节点
    for i in range(len(las_label_table)):
        # 在dict_users上的index
        i_ = i + len(pre_label_table)
        user_labels = las_label_table[i]
        for u_l in user_labels:
            shards_size = items_num // len(user_labels)
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i_] = np.concatenate((dict_users[i_], list(rand_set)),axis=0).astype(int)

    return dict_users


def noniid_label_bias(dataset, num_users, delete_size, min_size):
    dict_users = {i: np.array([]) for i in range(num_users)}
    label_buckets = split_data_by_label(dataset)
    label_num = len(label_buckets)
    items_num = len(dataset) // num_users


    label_index_list = list(range(label_num))

    label_table = distribute_label_table(range(len(label_buckets)), num_users)
    label_buckets = [set(i) for i in label_buckets]

    for i in range(num_users):
        user_label = label_table[i]
        shards_size = min_size // len(user_label)
        for u_l in user_label:
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

    return dict_users


def seperate_major_mino(maj_list, mino_list, user_num):
    combination_for_node = list(itertools.product(maj_list, mino_list))
    comb_length = len(combination_for_node)
    label_table_list = []
    for u in range(user_num):
        temp = list(combination_for_node[u%comb_length])
        label_table_list.append(temp)
    return label_table_list

def distribute_data_with_certain_label(label_buckets, label, shard_size):
    if shard_size >= len(label_buckets[label]):
        rand_set = np.random.choice(list(label_buckets[label]), shard_size, replace=True)
    else:
        rand_set = np.random.choice(list(label_buckets[label]), shard_size, replace=False)
        label_buckets[label] = label_buckets[label] - set(rand_set)
    return label_buckets, rand_set


def noniid_equal_label_sigma_alpha(dataset, user_num, major_class_num, sigma, alpha):
    dict_users = {i: np.array([]) for i in range(user_num)}
    label_buckets = split_data_by_label(dataset)
    label_num = len(label_buckets)
    items_num = len(dataset) // user_num
    label_index_list = list(range(label_num))
     

    # generate class combination
    random_label_index_list = np.random.permutation(label_index_list)
    maj_list = random_label_index_list[:major_class_num]
    mino_list = random_label_index_list[major_class_num:]

    # in the label table list, the first item is the majority class, the second is the minority class
    label_table_list = seperate_major_mino(maj_list, mino_list, user_num)

    # generate class imbalanced set according the sigma
    label_buckets = [set(i) for i in label_buckets]

    for i in range(user_num):
        user_label = label_table_list[i]
        maj_shard_size, mino_shard_size = int(items_num//2) + sigma, int(items_num//2 - sigma)
        

        label_buckets, maj_rand_set = distribute_data_with_certain_label(label_buckets, user_label[0], maj_shard_size)
        label_buckets, mino_rand_set = distribute_data_with_certain_label(label_buckets, user_label[1], mino_shard_size)

        dict_users[i] = np.concatenate((maj_rand_set, mino_rand_set), axis=0).astype(int)

    return dict_users




      


def sample_iid_with_size(dataset, size):
    label_buckets = split_data_by_label(dataset)
    label_num = len(label_buckets)
    sub_size = int(size/ label_num)

    user_dict = []
    for i in range(label_num):
        rand_set = np.random.choice(list(label_buckets[i]), sub_size, replace=False)
        user_dict = np.concatenate((user_dict, list(rand_set)), axis=0)
    return user_dict



if __name__ == '__main__':
    args = arg_parser()

    train_dataset, test_dataset = get_data(args)

    # if torch.is_tensor(train_dataset.targets):
    #     labels = train_dataset.targets.numpy()
    # else:
    #     labels = np.array(train_dataset.targets)
    # user_dict = sample_iid_with_size(train_dataset, 600)

    # for i in range(len(user_dict)):
    #     print(labels[int(user_dict[i])])

    # label_list = range(10)
    # x = distribute_label_table(label_list[:7], 70)
    # x2 = distribute_label_table(label_list[7:], 30)
    # y = list(merge(x, x2))
    # print(y)
    # y = [subitem for item in y for subitem in item]
    # print(y)
    # coun = collections.Counter(y)
    # print(coun)
    # label_datasize = [coun[i] for i in range(len(coun))]
    # print(np.std(label_datasize))
    # print(type(train_dataset.targets))
    # if torch.is_tensor(train_dataset.targets):
    #     labels = train_dataset.targets.numpy()
    # else:
    #     labels = np.array(train_dataset.targets)
    #
    # print(labels)
    # print(train_dataset[0][0], train_dataset[0][0].shape)
    # plt.imshow(train_dataset[0][0].reshape(28,28), cmap='gray')
    # plt.grid(False)
    # plt.show()
    # label_buckets = split_data_by_label(train_dataset)
    # print(label_buckets)
    #
    # for i in range(len(label_buckets)):
    #     print(len(label_buckets[i]))
    #
    # distribute_label_table(8, 80)

    # noniid_unequal(train_dataset, 100, 550, 10)
    # noniid_eqaul_bias(train_dataset, 100, 0.3, 0.3)
    # noniid_unequal_bias(train_dataset, 100, 0.3, 0.3, 10, 400)
    # noniid_equal_bias_mix(train_dataset, 100, 0.2, 0.2, 0.5)
    user_dict = noniid_equal_label_sigma_alpha(train_dataset, 100, 5, 100, 0.1)
    for i in range(100):
        print(len(user_dict[i]))



