import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
import utils
import setproctitle
import wandb


from metrics import metrics_dict


def test(args, device, test_dl, model, dicts):
    gt_length = utils.make_gt_length(args.data_pth, 0)
    model.eval()
    setproctitle.setproctitle('JH test....')

    with torch.no_grad():

        topk_list = []

        for idx, data in enumerate(test_dl):
            data = data.to(device)
            start = time.time()
            
            users = data[:, 0]

            scores = model.full_predict(users)

            for index, user in enumerate(users):
                user_score = scores[index]
                items = [int(item) for item in dicts.get(str(user.item()))]
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(args.topk), dim=-1)
                gt_items = data[index, 1]
                mask = np.isin(topk_idx.cpu().numpy(), gt_items.cpu().numpy())
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = calculate_result(args, topk_list, gt_length)
        
        print(f'test end, results : {metric_dict.__str__()}')



def calculate_result(args, topk_list, gt_len):
    result_list = []
    for metric in args.metrics:
        metric_fuc = metrics_dict[metric.lower()]
        result = metric_fuc(topk_list, gt_len)
        result_list.append(result)
    result_list = np.stack(result_list, axis=0).mean(axis=1)
    metric_dict = {}
    for topk in args.topk:
        for metric, value in zip(args.metrics, result_list):
            key = '{}@{}'.format(metric, topk)
            metric_dict[key] = np.round(value[topk - 1], 4)

    return metric_dict