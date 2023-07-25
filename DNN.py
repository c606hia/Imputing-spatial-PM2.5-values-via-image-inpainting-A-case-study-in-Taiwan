import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataset2_dnn import Dataset
from torch.utils.data import DataLoader
import time
from rich.table import Table
from rich.console import Console
import pandas as pd
from torchmetrics import R2Score
import kriging
import sys
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
device = 'cuda'
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.l1_all = nn.Linear(3, 64)
        self.l1_target = nn.Linear(2, 64)
        self.l2= nn.Linear(64, 256)
        self.l3= nn.Linear(256, 256)
        self.l4= nn.Linear(256, 128)
        self.output = nn.Linear(128, 37)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
    def forward(self, input_A, input_B,tar_size=37):
        hidden_A = self.relu(self.l1_all(input_A))
        hidden_B = self.relu(self.l1_target(input_B))
        merged = torch.cat((hidden_A, hidden_B), dim=1)
        hidden = self.drop(self.relu(self.l2(merged)))
        hidden = self.drop(self.relu(self.l3(hidden)))
        hidden = self.drop(self.relu(self.l4(hidden)))
        output = self.output(hidden)
        output = torch.div(torch.sum(output.view(-1,tar_size),dim=0),tar_size)
        return output

def draw(pred,gt,epoch,name='train'):
    fname = name
    count = 1
    if not os.path.exists(f'./results/{fname}'):
        os.makedirs(f'./results/{fname}')
    for i,j in zip(pred,gt):
        plt.subplot(len(pred),2,count)
        plt.title(f'iter {epoch} pred')
        plt.imshow(i[0].detach().cpu().numpy()[::-1])
        count += 1
        plt.subplot(len(pred),2,count)
        plt.title('gt')
        plt.imshow(j[0].detach().cpu().numpy()[::-1])
        count += 1
    plt.savefig(f'./results/{fname}/iter_{epoch}_train.png')
    plt.clf()

def validate(p,data,count,c):
    p.eval()
    with torch.no_grad():
        main_data, tar_loc, tar_pm  = __cuda__(*data)
        pred= p(main_data.float(), tar_loc.float(),len(tar_pm[0]))
        loss_G, r2 = get_r2_mse(pred, tar_pm.float())
        c.print(f'validate {count} Mse : {loss_G:.4f}, R2 : {r2:.4f}')
        # records + show result fig
        with open(f'./vali_record_.txt','a+') as f:
            c = Console(file = f)
            t = Table(title = f'record {count}')
            t.add_column('pred')
            t.add_column('gt')
            for p,g in zip(pred.view(37),tar_pm.view(37)):
                t.add_row(f'{p.item():.4f}', f'{g.item():.4f}')
            c.print(t)
            c.print(f'validate {count} Mse : {loss_G:.4f}, R2 : {r2:.4f}')

def __cuda__( *args):
        return (item.to('cuda') for item in args)

def get_r2_mse(pred, gt): 
    R2 = R2Score(num_outputs=1).to(device)
    mse = torch.nn.MSELoss()(gt.view(-1,37),pred.view(-1,37))
    r2s = R2(pred.view(37),gt.view(37)) 
    epa_mae = torch.mean(torch.abs(pred - gt)).item()
    return mse, r2s, epa_mae
def main(c):
    # preparation
    p = DNN().to('cuda')
    epoch = 300
    train_l = []
    opt = optim.Adam(p.parameters(),lr = 2e-4)
    dataset_train=Dataset(training=True)
    # loader_train=DataLoader(dataset=dataset_train,batch_size=1,shuffle=True,num_workers=2)
    # dataset_test=Dataset(training=False)
    # loader_test=DataLoader(dataset=dataset_test,batch_size=1,shuffle=True,num_workers=2)
    
    train_idx, val_idx = train_test_split(list(range(len(dataset_train))), test_size=0.2)
    datasets = {}
    datasets['train'] = Subset(dataset_train, train_idx)
    datasets['val'] = Subset(dataset_train, val_idx)
    
    loader_train=DataLoader(dataset=datasets['train'],batch_size=1,shuffle=True,num_workers=2)
    loader_test=DataLoader(dataset=datasets['val'],batch_size=1,shuffle=True,num_workers=2)
    p= nn.DataParallel(p)
    p.cuda()
    # load model
    # if os.path.isfile('./AutoEncoderVer1.pt'):
    #     stat_dicts = pt.load('./AutoEncoderVer1.pt')
    #     if isinstance(stat_dicts,dict):
    #         p.load_state_dict(stat_dicts)

    # train
    # p.start_training(loader_train, epoch)
    for i in range(1,epoch+1):
        stime = time.time()
        mse = 0
        r2s = 0
        mae = 0
        p.train()
        for data_num,items in enumerate(loader_train):
            main_data, tar_loc, tar_pm  = __cuda__(*items)
            opt.zero_grad()
            pred= p(main_data.float(), tar_loc.float(),len(tar_pm[0]))
            loss_G, r2, em = get_r2_mse(pred, tar_pm.float())
            (loss_G).backward()
            print(loss_G.item(), r2.item(), em)
            opt.step()
            mse += loss_G
            r2s += r2
            mae += em
        etime = time.time()
        int_time = etime-stime
        c.print(f"epoch:{i}, EPA mse:{(mse/len(loader_train)):.4f}, r2:{(r2s/len(loader_train)):.4f}, mae:{(mae/len(loader_train)):.4f}, time_taken:{(int_time):.2f}")
        r2s = 0
        mse = 0
        mae = 0
        if i % 20 == 0:
            with open(f'./record2_{i}.txt','a') as f:
                cc = Console(file = f)
                t = Table(title = f'record_iter_{i}')
                t.add_column('pred')
                t.add_column('gt')
                for pr,g in zip(pred.view(37),tar_pm.view(37)):
                    t.add_row(f'{pr.item():.4f}', f'{g.item():.4f}')
                cc.print(t)
        if i % 10 == 0:
            # save model file
            torch.save(p.state_dict(),'./DNN.pt')
    p.eval()
    mse =0
    r2s =0
    mae = 0
    with open('testing_final.txt','w+') as f:
        con = Console(file=f)
        with torch.no_grad():
            for data_num,items in enumerate(loader_test):
                main_data, tar_loc, tar_pm  = __cuda__(*items)
                pred= p(main_data.float(), tar_loc.float(),len(tar_pm[0]))
                loss_G, r2, em = get_r2_mse(pred, tar_pm.float())
                mse += loss_G
                r2s += r2
                mae += em
                con.print(f'testing {data_num} Mse : {loss_G:.4f}, R2 : {r2:.4f}, mae: {em:.4f}')
            con.print(f'avg Mse : {mse/len(loader_test):.4f}, R2 : {r2s/len(loader_test):.4f}, MAE : {mae/len(loader_test):.4f}')
        
if __name__ == '__main__':
    with open(sys.argv[1], "w") as report_file:
        c = Console(file=report_file)
        s = time.time()
        main(c)
        e = time.time()
        c.print(f'total time : {(e-s):.2f}')
