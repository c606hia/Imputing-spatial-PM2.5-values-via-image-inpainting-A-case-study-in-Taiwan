import os
import glob
import scipy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
size = (1,512,512)
# size = (1,412,300)
class Dataset(torch.utils.data.Dataset):
    def __init__(self,training = True):
        super(Dataset, self).__init__()
        self.train = training
        if self.train:
            self.patha = './data/airbox_data/a_202101.csv'
            self.pathe = './data/EPA_data/E_202101.csv'
            self.a = pd.read_csv(self.patha).dropna(axis=0)
            self.e = pd.read_csv(self.pathe)
        else:
            self.patha = './data/airbox_data/a_202102.csv'
            self.pathe = './data/EPA_data/E_202102.csv'
            self.a = pd.read_csv(self.patha).dropna(axis=0)
            self.e = pd.read_csv(self.pathe)
    def __len__(self):
        if self.train:
            return 24*31
        else:
            return 24*28
    def __getitem__(self, index):
        day = index//24+1
        hour = index%24
        # ab
        data_a = self.a[self.a['DD']==day]
        data_a = data_a[data_a['HH']==hour]
        data_a = data_a[['Lon', 'Lat', 'PM25']]
        # epa
        data_e = self.e[self.e['DD']==day]
        data_e = data_e[data_e['HH']==hour]
        data_e = data_e[['Lon', 'Lat', 'PM25']]
        mean = np.mean(data_e['PM25'])
        data_e = data_e.fillna(mean)
        # seperate epa
        idx = random.sample(range(0,len(data_e)),len(data_e)//2)
        main_e = data_e.values[idx,:]
        idx2=list(set(range(len(data_e))) - (set(idx)))
        tar_e = data_e.values[idx2,:]
        main_data = pd.concat([pd.DataFrame(main_e,columns=['Lon', 'Lat', 'PM25']),data_a],ignore_index=True)
        return torch.tensor(main_data[['Lon', 'Lat', 'PM25']].values), torch.tensor(tar_e[:,:2]), torch.tensor(tar_e[:,2])
