import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
# import wandb
import json

import dataset
from torch.utils.data import DataLoader
import train
import test
import utils

# from model.bipn import BIPN
# from model.mbcgcn import MBCGCN
from model.lightGCN import LightGCN
import setproctitle

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
    parser.add_argument('--gamma', type=float, default=1e-10)

    # data setting
    parser.add_argument('--data_name', type=str, default='tmall', help='choose data name')
    parser.add_argument('--loss', type=str, default='bpr', help='')
    parser.add_argument('--negative_cnt', type=int, default=4, help='Number of negative sample')
    parser.add_argument('--num_workers', default=4, type=int, help='workers of dataloader')

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='set batch size')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='BIPN', help='')

    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_number')
    parser.add_argument('--model', default='ligthGCN', type=str, help='model name')
    parser.add_argument('--behv', default='buy', type=str, help='behavior name')

    return parser.parse_args()


if __name__ == '__main__':
    setproctitle.setproctitle("jh")
    args = parse_args()
    # wandb.init(project = f"multi behavior recommendation project with {args.model} for {args.data_name}")
    # wandb.run.name = (f'{args.model} for {args.data_name} lr :{args.lr}, log_reg : {args.log_reg}, reg_weight : {args.reg_weight}')
    # wandb.run.save()
    device = torch.device(f'cuda:{args.gpu_id}')

    if args.data_name == 'tmall':
        args.data_pth = '/disks/ssd1/jahyeok/MBR_data/Tmall'
        args.behaviors = ['buy', 'click', 'collect', 'cart']
    elif args.data_name == 'taobao':
        args.data_pth = '/disks/ssd1/jahyeok/MBR_data/taobao'
        args.behaviors = ['buy', 'view', 'cart']
    elif args.data_name == 'beibei':
        args.data_pth = '/disks/ssd1/jahyeok/MBR_data/beibei'
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

    start = time.time()

    # Get number of user & items
    cnt_file_pth = os.path.join(args.data_pth, 'count.txt')
    with open(os.path.join(cnt_file_pth), encoding='utf-8') as r:
        cnt_data = json.load(r)
        N_user = cnt_data['user']
        N_item = cnt_data['item']

    # test_behavior = []
    # test_behavior.append(args.behv)

    #test mode
    test_data = dataset.TestDataset(os.path.join(args.data_pth, 'test.txt'))
    test_dl = DataLoader(dataset=test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size)
    
    # make matrices which we need
    inter_matrix, user_item_inter_set, all_inter_matrix, dicts = utils.make_inter_matrix(args.data_pth, args.behaviors, N_user, N_item)
    matrix_list = []
    matrix_list.append(inter_matrix)
    matrix_list.append(user_item_inter_set)
    matrix_list.append(all_inter_matrix)

    # if args.model == 'BIPN':
    #     model = BIPN(args, device, N_user, N_item, matrix_list).to(device)
    # elif args.model == 'MBCGCN':
    #     model = MBCGCN(args, device, N_user, N_item, matrix_list).to(device)
    # else:
    #     raise Exception('model name cannot be None')
    model = LightGCN(args, device, args.layers, N_user+1, N_item+1, matrix_list[2]).to(device)

    # uniform weight loading
    if args.data_name == 'tmall':
        raise Exception('data_name cannot be None')
    elif args.data_name == 'taobao':
        buy_pth = os.path.join(args.model_path, args.data_name, args.behaviors[0], str(args.lr), str(args.layers), 'model' + '.pth')
        view_pth = os.path.join(args.model_path, args.data_name, args.behaviors[1], str(args.lr), str(args.layers), 'model' + '.pth')
        cart_pth = os.path.join(args.model_path, args.data_name, args.behaviors[2], str(args.lr), str(args.layers), 'model' + '.pth')

        buy_dict = torch.load(buy_pth, weights_only=True)
        cart_dict = torch.load(cart_pth, weights_only=True)
        view_dict = torch.load(view_pth, weights_only=True)

        uniform_dict = buy_dict.copy()

        uniform_user_weight = buy_dict['user_embedding.weight'].to(device) + cart_dict['user_embedding.weight'].to(device) + view_dict['user_embedding.weight'].to(device)
        uniform_item_weight = buy_dict['item_embedding.weight'].to(device) + cart_dict['item_embedding.weight'].to(device) + view_dict['item_embedding.weight'].to(device)

        uniform_dict['user_embedding.weight'] = uniform_user_weight/3
        uniform_dict['item_embedding.weight'] = uniform_item_weight/3

    elif args.data_name == 'beibei':
        raise Exception('data_name cannot be None')
    else:
        raise Exception('data_name cannot be None')
    
    # model load
    model.load_state_dict(uniform_dict)

    test.test(args, device, test_dl, model, dicts[args.behaviors[0]])