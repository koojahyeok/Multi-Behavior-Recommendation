import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from .loss import BPRLoss, EmbLoss

class LightGCN(nn.Module):

    def __init__(self, args, device, n_layers, n_user, n_item, interaction_matrix):
        super(LightGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.n_user = n_user
        self.n_item = n_item
        self.interaction_matrix = interaction_matrix
        self.adj_matrix = self.get_adj_matrix()
        self.gamma = args.gamma
        self.reg_weight = args.reg_weight
        self.embedding_size = args.embedding_size

        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size, padding_idx=0).to(device)
        self.item_embedding = nn.Embedding(self.n_item, self.embedding_size, padding_idx=0).to(device)

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

    def get_adj_matrix(self):
        """
        making adjecent matrix
        A : (N_user + N_item) X (N_user + N_item) sparse matrix, which elements are all zero
        inter_matrix   - row : user, col : item
        inter_matrix_t - row : item, col : user
        D : (N_user + N_item) X (N_user + N_item) diagonal matrix, each element denote number of non_zero row
        """
        A = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=float)
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.transpose()

        # make bidirectional graph
        data_dict = dict(zip(zip(inter_matrix.row, inter_matrix.col + self.n_user), [1] * inter_matrix.nnz))
        data_dict.update(dict(zip(zip(inter_matrix_t.row + self.n_user, inter_matrix_t.col), [1] * inter_matrix_t.nnz)))

        # update A matrix
        A._update(data_dict)

        # Make D matrix
        sum_list = (A > 0).sum(axis=1)
        diag = np.array(sum_list.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)

        # Normalization & change to pytorch sparse matrix
        A_adj = D * A * D
        A_adj = sp.coo_matrix(A_adj)
        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse_coo_tensor(index, data, torch.Size(A_adj.shape))
        return A_sparse

    def forward(self, x):

        in_embs = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        result = [in_embs]
        for i in range(self.n_layers):
            # calculation of sparse matrix(type : torch.sparse.FloatTensor) & dense matrix(type : torch.FloatTensor)
            in_embs = torch.sparse.mm(self.adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        user_emb, item_emb = torch.split(result, [self.n_user, self.n_item])

        p_samples = x[:, 0, :]                                      # (B, 1, 4)
        n_samples = x[:, 1:-1, :].reshape(-1, 4)                    # (B, neg, 4)
        
        p_u_samples, p_i_samples, _, _ = torch.chunk(p_samples, 4, dim=-1)
        n_u_samples, n_i_samples, _, _ = torch.chunk(n_samples, 4, dim=-1)

        p_u_emb = user_emb[p_u_samples.long()].squeeze()
        p_i_emb = item_emb[p_i_samples.long()].squeeze()

        n_u_emb = user_emb[n_u_samples.long()].squeeze()
        n_i_emb = item_emb[n_i_samples.long()].squeeze()

        p_score = torch.sum((p_u_emb * p_i_emb), dim=-1)
        n_score = torch.sum((n_u_emb * n_i_emb), dim=-1)

        n_score = n_score.view(p_samples.shape[0], -1)                                  # (B, neg_cnt)
        n_score = n_score.sum(dim=-1)          

        # u_emb = user_emb[u_sample.long()].squeeze()
        # i_emb = item_emb[i_samples.squeeze().long()]

        # score = torch.sum((u_emb * i_emb), dim=-1)
        # p_score, n_score = torch.split(score, [p_samples.shape[0], n_samples.shape[0]]) # B & neg_cnt * B
        # n_score = n_score.view(p_samples.shape[0], -1)                                  # (B, neg_cnt)
        # n_score = n_score.sum(dim=-1)                                                   # (B, )

        bpr_loss = self.bpr_loss(p_score, n_score)
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        loss = bpr_loss + self.reg_weight * emb_loss

        return loss

    def full_predict(self, users):
        in_embs = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        result = [in_embs]
        for i in range(self.n_layers):
            # calculation of sparse matrix(type : torch.sparse.FloatTensor) & dense matrix(type : torch.FloatTensor)
            in_embs = torch.sparse.mm(self.adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        user_emb, item_emb = torch.split(result, [self.n_user, self.n_item])

        scores = torch.matmul(user_emb[users.long()], item_emb.transpose(0, 1))

        return scores