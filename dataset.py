import os
import random
import json
import torch
from torch.utils.data import Dataset
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class TrainDataset(Dataset):
    def __init__(self, behaviors, data_pth, negative_cnt):
        self.behaviors = behaviors
        self.pth = data_pth
        self.neg_cnt = negative_cnt
        
        # make all user_item interaction list
        all_u_i_inters = []

        for idx, behavior in enumerate(self.behaviors):
            with open(os.path.join(self.pth, behavior + '.txt'), encoding='utf-8') as r:
                lines = r.readlines()
                for line in lines:
                    user, item = line.split()
                    # user_id, item_id, behavior index, 1-positive sample, if 0, negative sample
                    all_u_i_inters.append([int(user), int(item), idx, 1])
        
        self.all_u_i_inters = np.array(all_u_i_inters)          # (#u_i_interactions x 4)

        # make cnt information
        with open(os.path.join(self.pth, 'count.txt'), encoding='utf-8') as r:
            cnt_data = json.load(r)
            self.u_cnt = cnt_data['user']
            self.i_cnt = cnt_data['item']
    
    def __getitem__(self, idx):
        total = []

        # get positive sample
        pos = self.all_u_i_inters[idx]
        u_id = pos[0]
        total.append(pos)

        u_i_inters = self.all_u_i_inters[self.all_u_i_inters[:, 0] == u_id]
        # get negative sampe
        for i in range(self.neg_cnt):
            item = random.randint(1, self.i_cnt)

            # check negativity
            while np.isin(item, u_i_inters[:, 1]):
                item = random.randint(1, self.i_cnt)
            
            neg = np.array([u_id, item, pos[2], 0])
            total.append(neg)

        # get interaction which only contain behavior 'buy'
        buy_inters = u_i_inters[u_i_inters[:, 2] == 0]
        if buy_inters.shape[0] == 0:
            signal = np.array([0, 0, 0, 0])
        else:
            p_item = random.choice(buy_inters[:, 1])
            n_item = random.randint(1, self.i_cnt)

             # check negativity
            while np.isin(n_item, u_i_inters[:, 1]):
                n_item = random.randint(1, self.i_cnt)
            signal = np.array([u_id, p_item, n_item, 0])
        
        total.append(signal)

        return np.array(total)
    
    def __len__(self):
        return self.all_u_i_inters.shape[0]
    
class TestDataset(Dataset):
    def __init__(self, data_pth):
        self.pth = data_pth

        test_u_i_inters = []

        with open(os.path.join(self.pth), encoding='utf-8') as r:
            lines = r.readlines()
            for line in lines:
                user, item = line.split()
                # test_u_i_inters.append(user)
                u_i = np.array([int(user), int(item)])
                test_u_i_inters.append(u_i)

        self.samples = np.array(test_u_i_inters)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return self.samples.shape[0]
    
