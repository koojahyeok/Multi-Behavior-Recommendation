import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from tqdm import tqdm
import utils
import setproctitle
import wandb


from metrics import metrics_dict

def train(args, device, train_dl, valid_dl, model, dicts):
    gt_length = utils.make_gt_length(args.data_pth)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr,
                               weight_decay=args.decay)
    
    best_result = 0
    best_dict = {}
    best_epoch = 0
    best_model = None
    final_test = None

    for epoch in tqdm(range(args.epochs), dynamic_ncols=True):
        print(f"{epoch+1} training...")
        model.train()
        validate_metric_dict = train_one_epoch(args, device, train_dl, valid_dl, epoch, model, optimizer, dicts, gt_length)

        if validate_metric_dict is not None:
                result = validate_metric_dict['hit@20']
                # early stop
                if result - best_result > 0:
                    # final_test = test_metric_dict
                    best_result = result
                    best_dict = validate_metric_dict
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                # if epoch - best_epoch > 4:
                #     break
        # save the best model
        torch.save(best_model.state_dict(), os.path.join(args.model_path, args.model_name + '.pth'))
        logger.info(f"training end, curr iteration %d, results: %s" %
                    (epoch + 1, validate_metric_dict.__str__()))
        # logger.info(f"training end, best iteration %d, results: %s" %
        #             (best_epoch + 1, best_dict.__str__()))
        logger.info(f"best epoch : {best_epoch + 1}")
        # logger.info(f"final test result is:  %s" % final_test.__str__())

def train_one_epoch(args, device, train_dl, valid_dl, epoch, model, optimizer, dicts, gt_length):
    train_dl_iter = (
        tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
        )
    )
    start_time = time.time()
    total_loss = 0.0
    batch_no = 0
    for idx, data in train_dl_iter:
        setproctitle.setproctitle(f"JH | {epoch}/{args.epochs} | {idx}/{len(train_dl_iter)}")
        # start = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        # loss = loss.sum()
        loss.backward()
        optimizer.step()
        batch_no = idx + 1
        total_loss += loss.item()
    epoch_time = time.time() - start_time
    total_loss = total_loss / batch_no
    wandb.log({"Training loss": loss})

    logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

    # clear_parameter(model)
    # validate

    validate_metric_dict = evaluate(args, device, model, epoch, args.batch_size, valid_dl, dicts, gt_length)
    # logger.info(
    #     f"validate %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, validate_metric_dict.__str__()))
    wandb.log({"metric": validate_metric_dict})

    # # test
    # start_time = time.time()
    # test_metric_dict = evaluate(epoch, test_batch_size, dataset.test_dataset(),
    #                                     dataset.test_interacts, dataset.test_gt_length)
    # epoch_time = time.time() - start_time
    # logger.info(
    #     f"test %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

    return validate_metric_dict


def evaluate(args, device, model, epoch, batch_size, dl, dicts, gt_length):

    model.eval()
    with torch.no_grad():
        iter_data = (
            tqdm(
                enumerate(dl),
                total=len(dl),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []

        for idx, data in iter_data:
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
        return metric_dict



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