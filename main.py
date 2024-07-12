import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import wandb
import json

import dataset
from torch.utils.data import DataLoader
import train
import utils

from model import BIPN


'''
Implementation of multi-behavior recommendation system
'''

# seed setting
seed = 42
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # hyper parameter setting
    parser.add_argument('--embedding_size', type=int, default=64, help='Choose Embedding size')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--log_reg', type=float, default=0.5, help='')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)
    parser.add_argument('--omega', type=float, default=1)

    # data setting
    parser.add_argument('--data_name', type=str, default='tmall', help='choose data name')
    parser.add_argument('--loss', type=str, default='bpr', help='')
    parser.add_argument('--negative_cnt', type=int, default=4, help='Number of negative sample')

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='set batch size')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='tmall', help='')

    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_number')
    parser.add_argument('--train', default=True, type=eval, help='choose train or test')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')

    if args.data_name == 'tmall':
        args.data_pth = './data/Tmall'
        args.behaviors = ['buy', 'click', 'collect', 'cart']
    elif args.data_name == 'taobao':
        args.data_pth = './data/taobao'
        args.behaviors = ['buy', 'view', 'cart']
    elif args.data_name == 'beibei':
        args.data_pth = './data/beibei'
        args.behaviors = ['buy', 'view', 'cart']
    else:
        raise Exception('data_name cannot be None')
    
    # wandb.init(project = "diffusion project")
    # wandb_args = {
    #     "epochs": args.T,
    #     "dropout" : args.dropout,
    #     "lr" : args.lr,
    #     "dataset" : args.dataset,
    #     "batch_size" : args.batch_size,
    #     "num_workers" : args.num_workers,
    # }
    # wandb.run.name = (f'{args.model} with {args.dataset} & sampling steps : {args.T // args.steps}')
    # wandb.run.save()
    
    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()

    # Get number of user & items
    cnt_file_pth = os.path.join(args.data_pth, 'count.txt')
    with open(os.path.join(cnt_file_pth), encoding='utf-8') as r:
        cnt_data = json.load(r)
        N_user = cnt_data['user']
        N_item = cnt_data['item']

    if args.train:
        #train mode
        train_data = dataset.TrainDataset(args.behaviors, args.data_pth, args.negative_cnt)
        valid_data = dataset.TestDataset(os.path.join(args.data_pth, 'validation.txt'))

        # make dataloader
        train_dl = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True)
        valid_dl = DataLoader(dataset=valid_data,
                              batch_size=args.batch_size)
        
        # make matrices which we need
        inter_matrix, user_item_inter_set, all_inter_matrix, dicts = utils.make_inter_matrix(args.data_pth, args.behaviors, N_user, N_item)
        matrix_list = []
        matrix_list.append(inter_matrix)
        matrix_list.append(user_item_inter_set)
        matrix_list.append(all_inter_matrix)

        model = BIPN(args, device, N_user, N_item, matrix_list).to(device)
        
        train.train(args, device, train_dl, valid_dl, model, dicts)


    else:
        #test mode
        test_data = dataset.TestDataset(os.path.join(args.data_pth, 'test.txt'))
        test_dl = DataLoader(dataset=test_data,
                              batch_size=args.batch_size)