import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from .loss import BPRLoss, EmbLoss
from .lightGCN import LightGCN

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
        self.emb_loss = EmbLoss()

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


        # u_samples, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)    # each : (B, neg+1, 1)
        p_u_emb = user_embedding[p_u_samples.long()].squeeze()
        p_i_emb = item_embedding[p_i_samples.long()].squeeze()

        n_u_emb = user_embedding[n_u_samples.long()].squeeze()
        n_i_emb = item_embedding[n_i_samples.long()].squeeze()

        p_score = torch.sum((p_u_emb * p_i_emb), dim=-1)
        n_score = torch.sum((n_u_emb * n_i_emb), dim=-1)

        n_score = n_score.view(p_sample.shape[0], -1)                                  # (B, neg_cnt)
        n_score = n_score.sum(dim=-1)                                                   # (B, )

        # Loss calculation
        scores = p_score - n_score
        gamma = 1e-10

        bpr_score = -torch.log(gamma + torch.sigmoid(scores))
        bpr_loss = bpr_score.mean()
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        loss = bpr_loss + self.reg_weight * emb_loss

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