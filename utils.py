#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author:
# @Date  : 2023/9/23 11:26
# @Desc  :
import torch
import torch.nn as nn
import os
import scipy.sparse as sp
import json
import numpy as np

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=self.norm)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        return emb_loss


# return interaction matrix & user_item_set & all_interaction matrix
# it shows user-item interaction, if they interact 1, no 0
def make_inter_matrix(pth, behaviors, N_user, N_item):
    inter_matrices = []
    user_item_inter_set = []
    all_row = []
    all_col = []
    dicts = {}

    for behavior in behaviors:
        with open(os.path.join(pth, behavior + '.txt'), encoding='utf-8') as f:
            lines = f.readlines()
            row = []
            col = []
            bdict = {}

            for line in lines:
                line = line.strip('\n').strip().split()

                user = line[0]
                item = line[1]

                if user in bdict:
                    bdict[user].append(item)
                else:
                    bdict[user] = [item]

                row.append(int(line[0]))
                col.append(int(line[1]))
            
            dicts[behavior] = bdict
            
            values = torch.ones(len(row), dtype=torch.float32)
            inter_matrix = sp.coo_matrix((values, (row, col)), [N_user + 1, N_item + 1])
            # .tocsr() - change coo(Coordinate) style to csr(Compressed Sparse Row) style
            user_item_set = [list(row.nonzero()[1]) for row in inter_matrix.tocsr()]
            inter_matrices.append(inter_matrix)
            user_item_inter_set.append(user_item_set)
            all_row.extend(row)
            all_col.extend(col)


    all_edge_index = list(set(zip(all_row, all_col)))
    all_row = [sub[0] for sub in all_edge_index]
    all_col = [sub[1] for sub in all_edge_index]
    values = torch.ones(len(all_row), dtype=torch.float32)
    all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [N_user + 1, N_item + 1])

    return inter_matrices, user_item_inter_set, all_inter_matrix, dicts

def make_gt_length(pth):

    dict = {}

    with open(os.path.join(pth, 'test.txt'), encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip('\n').strip().split()

            user = line[0]
            item = line[1]

            if user in dict:
                dict[user].append(item)
            else:
                dict[user] = [item]

    gt_length = np.array([len(x) for _, x in dict.items()])

    return gt_length