import random
import torch
import os
import numpy as np
import time
import argparse
import pickle
from DSTPP.Dataset import get_dataloader


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def current_time_str():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)

def normalization(x,MAX,MIN):
    return (x-MIN)/(MAX-MIN)

def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2',choices=['l1','l2','Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine',choices=['linear','cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices = [1,2,3])
    parser.add_argument('--dataset', type=str, default='Earthquake',choices=['Citibike','Earthquake','HawkesGMM','Pinwheel','COVID19','Mobility','HawkesGMM_2d','Independent'], help='')
    parser.add_argument('--batch_size', type=int, default=64,help='')
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=100, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args

def data_loader(writer, dataset, batch_size, dim):
    f = open('dataset/{}/data_train.pkl'.format(dataset),'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in train_data] # [t, \tau, ...] \tau_0=t_0

    f = open('dataset/{}/data_val.pkl'.format(dataset),'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in val_data]

    f = open('dataset/{}/data_test.pkl'.format(dataset),'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in test_data]

    data_all = train_data+test_data+val_data

    Max, Min = [], []
    for m in range(dim+2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] > 0
    
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    trainloader = get_dataloader(train_data, batch_size, D = dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data)<=1000 else 1000, D = dim, shuffle=False)
    valloader = get_dataloader(test_data, len(val_data) if len(val_data)<=1000 else 1000, D = dim, shuffle=False)

    return trainloader, testloader, valloader, (Max,Min)

def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num
