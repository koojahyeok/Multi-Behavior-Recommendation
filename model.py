#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2023/9/23 16:16
# @Desc  :
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class BIPN(nn.Module):
    def __init__(self, args, device, N_user, N_item, matrix_list):
        super(BIPN, self).__init__()
        '''
        Input
        args : arguments
        N_user : Number of user
        N_item : Number of item
        matrix list = [inter_matrix, user_item_inter_set, all_inter_matrix]

        
        device : select gpu device
        layers : number of layers
        reg_weight :
        log_reg :
        node_dropout :
        message_dropour :
        embedding_size : size of embedding
        inter_matrix : interaction matrices of each behaviors
        user_item_inter_set : list of user_item set
        test_users : list of test_users
        behaviors : list of behavior

        user_embedding : initialization of user_embedding (N_user + 1 , embedding_size)
        item_embedding : initialization of item_embedding (N_item + 1 , embedding_size)

        bhv_embs : behavior embedding
        global_Graph : LightGCN for pretrain
        behavior_Graph : LightGCN for GCN Enhancement Module

        RZ : 
        U :

        storage_user_embeddings :
        storage_item_embeddings :
        '''

        self.device = device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.log_reg = args.log_reg
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.embedding_size = args.embedding_size
        self.n_users = N_user
        self.n_items = N_item
        self.inter_matrix = matrix_list[0]
        self.user_item_inter_set = matrix_list[1]
        self.test_users = [str(i) for i in range(1, self.n_users + 1)]
        self.behaviors = args.behaviors
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        self.bhv_embs = nn.Parameter(torch.eye(len(self.behaviors)))
        self.global_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, matrix_list[2])
        self.behavior_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, matrix_list[0][0])

        self.RZ = nn.Linear(2 * self.embedding_size + len(self.behaviors), 2 * self.embedding_size, bias=False)
        self.U = nn.Linear(2 * self.embedding_size + len(self.behaviors), self.embedding_size, bias=False)

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()

        # self.model_path = args.model_path
        # self.check_point = args.check_point
        # self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)


    def _init_weights(self, module):
        '''
        embedding & parameter initialization by xavier initialization
        '''

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def agg_info(self, u_emb, i_emb, bhv_emb):
        in_feature = torch.cat((u_emb, i_emb, bhv_emb), dim=-1)

        # Pre-filtering layer
        RZ = torch.sigmoid(self.RZ(in_feature))
        R, Z = torch.chunk(RZ, 2, dim=-1)
        RU = R * u_emb

        # Item-aware layer
        RU = torch.cat((RU, i_emb, bhv_emb), dim=-1)
        u_hat = torch.tanh(self.U(RU))

        # Post-filtering layer
        u_final = Z * u_hat

        return u_final


    def user_agg_item(self, user_samples, u_emb, ini_item_embs):

        keys = user_samples.tolist()
        user_item_set = self.user_item_inter_set[-1]
        agg_items = [user_item_set[x] for x in keys]
        degree = [len(x) for x in agg_items]
        degree = torch.tensor(degree).unsqueeze(-1).to(self.device)
        max_len = max(len(l) for l in agg_items)
        padded_list = np.zeros((len(agg_items), max_len), dtype=int)
        for i, l in enumerate(agg_items):
            padded_list[i, :len(l)] = l
        padded_list = torch.from_numpy(padded_list).to(self.device)
        mask = (padded_list == 0)
        agg_item_emb = ini_item_embs[padded_list.long()]
        u_in = u_emb.repeat(1, max_len, 1)
        bhv_emb = self.bhv_embs[-1].repeat(u_in.shape[0], u_in.shape[1], 1)

        u_final = self.agg_info(u_in, agg_item_emb, bhv_emb)

        u_final[mask] = 0
        u_final = torch.sum(u_final, dim=1)
        lamb = 1 / (degree + 1e-8)
        u_final = u_final.unsqueeze(1)
        u_final = u_final

        return u_final, lamb

    def forward(self, x):
        '''
        Input : x (B, 2 + neg, 4)
            B - batch size
            2 + neg - 1 positive sample, negative samples, 1 signal 

        embedding pretraining
        concatenatinb embeddings & Use lightGCN to pretraining
        '''
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(all_embeddings)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])


        '''
        GCN enhancement Module
        only using target behavior & Use lightGCN
        '''
        buy_embeddings = self.behavior_Graph(all_embeddings)
        user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])
        # pre_time = time.time() - pre_s_time
        # print(f'pre-train time : {pre_time}')

        '''
        BCIPN netwrok
        using positive & negative sample
        '''
        # bc_s_time = time.time()
        p_samples = x[:, 0, :]                                      # (B, 1, 4)
        n_samples = x[:, 1:-1, :].reshape(-1, 4)                    # (B, neg, 4)
        samples = torch.cat([p_samples, n_samples], dim=0)          # (B, neg+1, 4)

        u_sample, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)    # each : (B, neg+1, 1)
        u_emb = user_embedding[u_sample.long()].squeeze()
        i_emb = item_embedding[i_samples.squeeze().long()]
        bhv_emb = self.bhv_embs[b_samples.reshape(-1).long()]
        u_final = self.agg_info(u_emb, i_emb, bhv_emb) # BCIPN network
        # bc_f_time = time.time() - bc_s_time
        # print(f'BIPN net time : {bc_f_time}')

        # loss calculation
        # l_s_time = time.time()
        log_loss_scores = torch.sum((u_final * i_emb), dim=-1).unsqueeze(1)
        log_loss = self.cross_loss(torch.sigmoid(log_loss_scores), gt_samples.float())

        pair_samples = x[:, -1, :-1]
        mask = torch.any(pair_samples != 0, dim=-1)
        pair_samples = pair_samples[mask]
        bpr_loss = 0
        if pair_samples.shape[0] > 0:
            user_samples = pair_samples[:, 0].long()
            item_samples = pair_samples[:, 1:].long()
            u_emb = user_embedding[user_samples].unsqueeze(1)
            i_emb = item_embedding[item_samples]

            u_point, lamb = self.user_agg_item(user_samples, u_emb, item_embedding)
            u_gen_emb = u_emb + user_buy_embedding[user_samples].unsqueeze(1)
            i_final = i_emb + item_buy_embedding[item_samples]
            score_point = torch.sum((u_point * i_emb), dim=-1)
            score_gen = torch.sum((u_gen_emb * i_final), dim=-1)
            bpr_scores = (1 - lamb) * score_point + lamb * score_gen
            p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
            bpr_loss += self.bpr_loss(p_scores, n_scores)
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        loss = self.log_reg * log_loss + (1 - self.log_reg) * bpr_loss + self.reg_weight * emb_loss
        # l_f_time = time.time() - l_s_time
        # print(f'loss cal time : {l_f_time}')

        return loss
    
    def full_predict(self, users):
        
        # Pretraining module
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(all_embeddings)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])

        buy_embeddings = self.behavior_Graph(all_embeddings)
        user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])

        self.storage_user_embeddings = torch.zeros(self.n_users + 1, self.embedding_size).to(self.device)

        test_users = [int(x) for x in self.test_users]
        tmp_emb_list = []
        for i in range(0, len(test_users), 100):
            tmp_users = test_users[i: i + 100]
            tmp_users = torch.LongTensor(tmp_users)
            tmp_embeddings = user_embedding[tmp_users].unsqueeze(1)
            tmp_embeddings, _ = self.user_agg_item(tmp_users, tmp_embeddings, item_embedding)
            tmp_emb_list.append(tmp_embeddings.squeeze())
        tmp_emb_list = torch.cat(tmp_emb_list, dim=0)
        for index, key in enumerate(test_users):
            self.storage_user_embeddings[key] = tmp_emb_list[index]

        user_item_set = self.user_item_inter_set[-1]
        degree = [len(x) for x in user_item_set]
        degree = torch.tensor(degree).unsqueeze(-1).to(self.device)
        lamb = 1/(degree + 1e-8)

        user_embedding = user_embedding + user_buy_embedding
        user_embedding = lamb * user_embedding
        self.storage_user_embeddings = (1-lamb) * self.storage_user_embeddings


        self.storage_user_embeddings = torch.cat((self.storage_user_embeddings, user_embedding), dim=-1)
        self.storage_item_embeddings = torch.cat((item_embedding, item_embedding + item_buy_embedding), dim=-1)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores
    
class MBCGCN(nn.Module):
    def __init__(self, args, device, N_user, N_item, matrix_list):
        super(MBCGCN, self).__init__()
        '''
        Input
        args : arguments
        N_user : Number of user
        N_item : Number of item
        matrix list = [inter_matrix, user_item_inter_set, all_inter_matrix]

        device : select gpu device
        layers : number of layers
        reg_weight :
        log_reg :
        node_dropout :
        message_dropour :
        embedding_size : size of embedding
        inter_matrix : interaction matrices of each behaviors
        user_item_inter_set : list of user_item set
        test_users : list of test_users
        behaviors : list of behavior

        user_embedding : initialization of user_embedding (N_user + 1 , embedding_size)
        item_embedding : initialization of item_embedding (N_item + 1 , embedding_size)
        '''

        self.device = device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.log_reg = args.log_reg
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.embedding_size = args.embedding_size
        self.n_users = N_user
        self.n_items = N_item
        self.inter_matrix = matrix_list[0]
        self.user_item_inter_set = matrix_list[1]
        self.behaviors = args.behaviors
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0).to(device)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0).to(device)
        self.test_users = [str(i) for i in range(1, self.n_users + 1)]

        # Initialize lightGCN
        self.b_cnt = len(self.behaviors)
        self.GCN_blocks = []
        for i in range(self.b_cnt):
            if i == 0:
                GCN_target_block = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, matrix_list[0][i])
            else:
                GCN_block = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, matrix_list[0][i])
                self.GCN_blocks.append(GCN_block)
        self.GCN_blocks.append(GCN_target_block)

        # Initialize transformation matrix
        self.user_weight_matrices = []
        for i in range(self.b_cnt - 1):
            weight = nn.Linear(self.embedding_size, self.embedding_size, bias=False).to(device)
            self.user_weight_matrices.append(weight)

        self.item_weight_matrices = []
        for i in range(self.b_cnt - 1):
            weight = nn.Linear(self.embedding_size, self.embedding_size, bias=False).to(device)
            self.item_weight_matrices.append(weight)


        self.apply(self._init_weights)


    def _init_weights(self, module):
        '''
        embedding & parameter initialization by xavier initialization
        '''

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        '''
        Input : x (B, 2 + neg, 4)
            B - batch size
            2 + neg - 1 positive sample, negative samples, 1 signal 
        '''
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        GCN_user_results = []
        GCN_item_results = []

        for i in range(self.b_cnt):
            # Cascading GCN Blocks
            all_embeddings = self.GCN_blocks[i](all_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            GCN_user_results.append(user_embedding)
            GCN_item_results.append(item_embedding)

            # Feature Transformation - only auxilary behavior
            if i < self.b_cnt - 1:
                user_embedding = self.user_weight_matrices[i](user_embedding)
                item_embedding = self.item_weight_matrices[i](item_embedding)
                all_embeddings = torch.cat([user_embedding, item_embedding], dim=0)

        
        # embedding aggregation
        user_embedding = sum(GCN_user_results)
        item_embedding = sum(GCN_item_results)

        p_sample = x[:, 0, :]                       # (B, 1, 4)
        n_sample = x[:, 1:-1, :].reshape(-1, 4)     # (B, neg, 4)

        p_u_samples, p_i_samples, b, gt = torch.chunk(p_sample, 4, dim=-1)
        n_u_samples, n_i_samples, c, gc = torch.chunk(n_sample, 4, dim=-1)

        # samples = torch.cat([p_sample, n_sample], dim=0)

        # u_samples, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)    # each : (B, neg+1, 1)
        p_u_emb = user_embedding[p_u_samples.long()].squeeze()
        p_i_emb = item_embedding[p_i_samples.long()].squeeze()

        n_u_emb = user_embedding[n_u_samples.long()].squeeze()
        n_i_emb = item_embedding[n_i_samples.long()].squeeze()

        p_score = torch.sum((p_u_emb * p_i_emb), dim=-1)
        n_score = torch.sum((n_u_emb * n_i_emb), dim=-1)

        # Loss calculation
        p_score_expanded = p_score.unsqueeze(1).expand(-1, 4).reshape(-1)
        scores = p_score_expanded - n_score
        gamma = 1e-10

        bpr_score = -torch.log(gamma + torch.sigmoid(scores))
        loss = bpr_score.mean()
        return loss
    
    def full_predict(self, users):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        GCN_user_results = []
        GCN_item_results = []

        for i in range(self.b_cnt):
            # Cascading GCN Blocks
            all_embeddings = self.GCN_blocks[i](all_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            GCN_user_results.append(user_embedding)
            GCN_item_results.append(item_embedding)

            # Feature Transformation - only auxilary behavior
            if i < self.b_cnt - 1:
                all_embeddings = self.user_weight_matrices[i](user_embedding)
                all_embeddings = self.item_weight_matrices[i](item_embedding)
                all_embeddings = torch.cat([user_embedding, item_embedding], dim=0)
        
        # embedding aggregation
        user_embedding = sum(GCN_user_results)
        item_embedding = sum(GCN_item_results)

        scores = torch.matmul(user_embedding[users.long()], item_embedding.transpose(0, 1))

        return scores
