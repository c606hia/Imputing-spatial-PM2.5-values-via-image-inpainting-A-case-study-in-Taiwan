import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from maskedtensor import masked_tensor
import matplotlib.pyplot as plt
size = (1,512,512)
# size = (1,412,300)
m = ['','jan','feb']
class Dataset(torch.utils.data.Dataset):
    def __init__(self,training = True):
        super(Dataset, self).__init__()
        self.data_names_e = []
        self.data_names_egt = []
        self.data_names_e77m = []
        self.data_names_a = []
        self.data_names_m = []
        self.data_names_em = []
        self.train = training
        # airbox
        self.patha = './data/airbox_data/a_by_loc'
        # EPA-k and EPA-k mask
        self.pathe = './data/e+a_mask_gt_main'
        self.path_em = './data//mask'
        # pure EPA
        self.pathe_77 = './data/EPA_data/by_loc_main'
        for i in os.listdir(f'./{self.patha}'):
            if int(i.split('_')[1])<28:
                self.data_names_a.append(i)
        for i in os.listdir(f'./{self.pathe}'):
            if int(i.split('_')[1])<28:
                self.data_names_e.append(i)
        for i in os.listdir(f'./{self.pathe_77}'):
            if int(i.split('_')[1])<28:
                self.data_names_egt.append(i)
        for i in os.listdir(f'./{self.path_em}'):
            if int(i.split('_')[1])<28:
                self.data_names_em.append(i)
        # airbox
        # self.patha = '../airbox_data/feb/a_by_loc'
        # # EPA-k and EPA-k mask
        # self.pathe = '../EPA_data/feb/e_mask_gt_main'
        # self.path_em = '../EPA_data/feb/mask'
        # # pure EPA
        # self.pathe_77 = '../EPA_data/feb/by_loc_main'
        # for i in os.listdir(f'./{self.patha}'):
        #     if int(i.split('_')[1])<32:
        #         self.data_names_a.append(i)
        # for i in os.listdir(f'./{self.pathe}'):
        #     if int(i.split('_')[1])<32:
        #         self.data_names_e.append(i)
        # for i in os.listdir(f'./{self.pathe_77}'):
        #     if int(i.split('_')[1])<32:
        #         self.data_names_egt.append(i)
        # for i in os.listdir(f'./{self.path_em}'):
        #     if int(i.split('_')[1])<32:
        #         self.data_names_em.append(i)
        
    def __len__(self):
        return len(self.data_names_e)
        # return 1
    # def clean_up(self, index):
    #     # ans = np.load(f'./{self.ans}/{self.data_names_t[index]}',allow_pickle=True)
    #     a_data = np.load(f'./{self.patha}/{self.data_names_a[index]}',allow_pickle=True)
    #     e_data = np.load(f'./{self.pathe_77}/{self.data_names_egt[index]}',allow_pickle=True)
    #     ans = np.load(f'./{self.pathe}/{self.data_names_e[index]}',allow_pickle=True)
    #     e_gt = np.load(f'./{self.pathe_77}/{self.data_names_egt[index]}',allow_pickle=True)
    #     ans_m = np.load(f'./{self.path_em}/{self.data_names_em[index]}',allow_pickle=True)
    #     ans = masked_tensor(torch.tensor(ans.reshape(size)),torch.tensor(~ans_m.reshape(size)))
    #     # return data,torch.tensor(ans.reshape(1,512,512))
    #     # cat data
    #     data,e_idx, a_idx = self.gen_data(a_data, e_data,index)
    #     mask = self.gen_mask(data)
    #     self.item.append([torch.tensor(data.reshape(size)), torch.tensor(mask.reshape(size)), ans.to_tensor(0), torch.tensor(e_gt.reshape(size)), torch.tensor(a_data.reshape(size)), e_idx, a_idx])
    def __getitem__(self, index):
        # mon = int(self.data_names_a[index].split('_')[0])
        # self.patha = f'../airbox_data/{m[mon]}/a_by_loc'
        # self.pathe = f'../EPA_data/{m[mon]}/e_mask_gt_main'
        # self.path_em = f'../EPA_data/{m[mon]}/mask'
        # self.pathe_77 = f'../EPA_data/{m[mon]}/by_loc_main'
        a_data = np.load(f'./{self.patha}/{self.data_names_a[index]}',allow_pickle=True)
        e_data = np.load(f'./{self.pathe_77}/{self.data_names_egt[index]}',allow_pickle=True)
        ans = np.load(f'./{self.pathe}/{self.data_names_e[index]}',allow_pickle=True)
        ans_m = np.load(f'./{self.path_em}/{self.data_names_em[index]}',allow_pickle=True)
        ans = masked_tensor(torch.tensor(ans.reshape(size)),torch.tensor(~ans_m.reshape(size)))
        ans_m = 1-ans_m
        # cat data
        data,e_idx, a_idx = self.gen_data(a_data, e_data,index)
        mask = self.gen_mask(data)
        data = torch.cat((torch.tensor(a_data.reshape(size)),torch.tensor(e_data.reshape(size))),0)
        return data, torch.tensor(ans_m.reshape(size)), torch.nan_to_num(ans.to_tensor(0)), torch.tensor(e_data.reshape(size)), torch.tensor(a_data.reshape(size)), e_idx, a_idx
        # return self.item[index]
    def gen_mask(self, data):
        mask = np.zeros((512,512))
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j]>0:
                    mask[i][j] = 1
        return mask
    def gen_data(self, a, e,index):
        tmp = a
        idx = random.sample(range(0,74),74//2)
        count = 0
        e_idx = [[],[]]
        a_idx = [[],[]]
        for i in range(len(a)):
            for j in range(len(a[0])):
                if tmp[i][j] >0:
                    a_idx[0].append(i)
                    a_idx[1].append(j)
                if e[i][j] > 0:
                    if count in idx:
                        tmp[i][j] = e[i][j]
                    else:
                        e_idx[0].append(i)
                        e_idx[1].append(j)
                    count += 1
        for i in range(37-len(e_idx[0])):
            e_idx[0].append(0)
            e_idx[1].append(0)
        for i in range(3600-len(a_idx[0])):
            a_idx[0].append(0)
            a_idx[1].append(0)
        return tmp, torch.tensor(e_idx), torch.tensor(a_idx)
