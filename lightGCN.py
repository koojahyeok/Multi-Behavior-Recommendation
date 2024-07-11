import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):

    def __init__(self, device, n_layers, n_users, n_items, interaction_matrix):
        super(LightGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.n_user = n_users
        self.n_item = n_items
        self.interaction_matrix = interaction_matrix
        self.adj_matrix = self.get_adj_matrix()


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
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))
        return A_sparse

    def forward(self, in_embs):

        result = [in_embs]
        for i in range(self.n_layers):
            # calculation of sparse matrix(type : torch.sparse.FloatTensor) & dense matrix(type : torch.FloatTensor)
            in_embs = torch.sparse.mm(self.adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        return result