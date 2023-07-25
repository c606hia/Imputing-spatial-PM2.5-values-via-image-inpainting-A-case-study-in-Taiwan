import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from dataset2_ae import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import time
from rich.table import Table
from rich.console import Console
import pandas as pd
from torchmetrics import R2Score, MeanAbsolutePercentageError
import kriging
import sys
import os
import torch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
color_list = ['mediumblue', 'aqua', 'yellow', 'orange', 'chocolate', 'firebrick', 'fuchsia']
newcmp = LinearSegmentedColormap.from_list('testCmap', colors=color_list, N=256)
pt.cuda.set_device(0)
device = 'cuda'
n=1
size = (512,512)

class predictor_c(nn.Module):
    def __init__(self) -> None:
        super(predictor_c,self).__init__()
        self.c1 = nn.Conv2d(in_channels=2,out_channels=3,kernel_size=(3,3),stride=(1,1)) 
        
        for i in range(2,6):
            self.__setattr__(f'c{i}',nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=(1,1)))

        self.c6 = nn.AvgPool2d(3,stride=1,padding=0)

        for i in range(7,11):
            self.__setattr__(f'c{i}',nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=(1,1)))

        self.c11 = nn.AvgPool2d(3,stride=1,padding=0)

        for i in range(12,15):
            self.__setattr__(f'c{i}',nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=(1,1)))

        self.c15 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(3,3),stride=(1,1))



        self.d1 = nn.ConvTranspose2d(1,3,(3,3),(1,1))

        for i in range(2,11):
            self.__setattr__(f'd{i}',nn.ConvTranspose2d(3,3,(3,3),(1,1)))
        for i in range(11,15):
            self.__setattr__(f'd{i}',nn.ConvTranspose2d(3,3,(3,3),(1,1)))
        self.d15 = nn.ConvTranspose2d(3,1,(3,3),(1,1))

    def forward(self,data):
        data = data.to(device)

        for i in range(1, 16):
            data = self.__getattr__(f'c{i}')(data)
            data = nn.ELU()(data)
        # deconvolution
        # print(data.size())
        for i in range(1, 16):
            data = self.__getattr__(f'd{i}')(data)

        # print(data.size())
        return data
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

def __cuda__( *args):
        return (item.to('cuda') for item in args)

def l1_loss(f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)*torch.numel(mask)/torch.sum(mask)
def get_r2_mse(e_77, idx, fake_B, epa = True):
    R2 = R2Score(num_outputs=1)
    # 預設gt,pred空間，方便之後合併
    gt = torch.tensor([], dtype = torch.double).to(device)
    pred = torch.tensor([], dtype = torch.double).to(device)
    # 預設0，airbox站點不會計算mae跟R2，但還是需要回傳值
    r2s = 0
    epa_mae=0
    for i in range(len(idx)):
        # idx格式 => [batch size, row index, col index]
        # _idx0 => 在batch i 的所有row index
        _idx0 = idx[i,0][idx[i,0].nonzero()]
        # _idx1 => 在batch i 的所有col index
        _idx1 = idx[i,1][idx[i,1].nonzero()]
        # 根據_idx0,_idx1抓出資料然後跟預設空間合併
        pred = torch.concat([fake_B[i,0][_idx0,_idx1],pred])
        gt = torch.concat([e_77[i,0][_idx0,_idx1],gt])
    # 下面都是計算
    mse = torch.nn.MSELoss()(gt,pred)
    # 如果是epa就多算mae跟R2
    if epa:
        r2s = R2(pred.cpu(),gt.cpu()) 
        epa_mae = torch.mean(torch.abs(pred - gt)).item()
    return mse, r2s,epa_mae,pred,gt
def ground_loss(fake_B,real_b,mask):
    R2 = R2Score(num_outputs=1)
    idx = torch.argwhere(mask)
    gt = torch.tensor([], dtype = torch.double).to(device)
    pred = torch.tensor([], dtype = torch.double).to(device)
    for i in range(len(fake_B)):
        _idx0 = idx[idx[:,0]==i][:,2]
        _idx1 = idx[idx[:,0]==i][:,3]
        pred = torch.concat([fake_B[i,0][_idx0,_idx1],pred])
        gt = torch.concat([real_b[i,0][_idx0,_idx1],gt])
    return R2(pred.cpu(),gt.cpu()).item(), torch.mean(torch.abs(pred - gt)**2).item()
def main(c):
    global n

    # preparation
    p = predictor_c().to('cuda')
    epoch = 300
    train_l = []
    opt = optim.Adam(p.parameters(),lr = 2e-7)
    dataset_train=Dataset(training=True)
    # loader_train=DataLoader(dataset=dataset_train,batch_size=4,shuffle=True,num_workers=2)
    # dataset_test=Dataset(training=False)
    # loader_test=DataLoader(dataset=dataset_test,batch_size=1,shuffle=True,num_workers=2)
    
    train_idx, val_idx = train_test_split(list(range(len(dataset_train))), test_size=0.2)
    datasets = {}
    datasets['train'] = Subset(dataset_train, train_idx)
    datasets['val'] = Subset(dataset_train, val_idx)
    
    loader_train=DataLoader(dataset=datasets['train'],batch_size=8,shuffle=True,num_workers=2)
    loader_test=DataLoader(dataset=datasets['val'],batch_size=1,shuffle=True,num_workers=2)
    p= nn.DataParallel(p)
    p.cuda()
    # load model
    # if os.path.isfile('./AutoEncoderVer1.pt'):
    #     stat_dicts = pt.load('./AutoEncoderVer1.pt')
    #     if isinstance(stat_dicts,dict):
    #         p.load_state_dict(stat_dicts)

    # train
#     for i in range(1,epoch+1):
#         stime = time.time()
#         mse = 0
#         r2s = 0
#         l1_loss_val = 0
#         train_epa_mae = 0
#         g_mse = 0
#         g_r2 = 0
#         p.train()
#         for data_num,items in enumerate(loader_train):
#             # with torch.autograd.set_detect_anomaly(True):
#             masked_images, mask,  gt_images, e_77, a_gt, e_idx, a_idx = __cuda__(*items)
#             opt.zero_grad()
#             pred_img = p(masked_images.float())
            
#             hole_loss = l1_loss(gt_images, pred_img,mask)
#             e_mse_loss,r2,epa_mae,pred,gt = get_r2_mse(e_77,e_idx, pred_img)
#             a_mse_loss,w,x,y,z = get_r2_mse(a_gt,a_idx, pred_img, False)
#             gr,gm = ground_loss(pred_img, gt_images, mask)
#             loss_G = (a_mse_loss*0.5+ 0.1*hole_loss + e_mse_loss*2)
#             print(a_mse_loss.item(), hole_loss.item(), e_mse_loss.item())
#             loss_G.backward()
#             opt.step()
#             mse += e_mse_loss
#             r2s += r2
#             l1_loss_val += hole_loss
#             train_epa_mae += epa_mae
#             g_mse += gm
#             g_r2 += gr
#         etime = time.time()
#         int_time = etime-stime
#         c.print(f"epoch:{i}, l1_loss:{(l1_loss_val/len(loader_train)):.4f}, EPA mse:{(mse/len(loader_train)):.4f}, EPA mae:{(train_epa_mae/len(loader_train)):.4f}, \
# ground mse:{(g_mse/len(loader_train)):.4f}, ground r2:{(g_r2/len(loader_train)):.4f}, r2:{(r2s/len(loader_train)):.4f}, time_taken:{(int_time):.2f}")
#         r2s = 0
#         mse = 0
#         l1_loss_val = 0
#         train_epa_mae = 0
#         g_mse = 0
#         g_r2 = 0
#         if i % 20 == 0:
#             with open(f'./record_{i}.txt','a') as f:
#                 cc = Console(file = f)
#                 t = Table(title = f'record_iter_{i}')
#                 t.add_column('pred')
#                 t.add_column('gt')
#                 for pr,g in zip(pred,gt):
#                     t.add_row(f'{pr.item():.4f}', f'{g.item():.4f}')
#                 cc.print(t)
#         if i % 10 == 0:
#             # save model file
#             draw(pred_img,gt_images,i)
#             torch.save(p.state_dict(),'./AutoEncoderVer1.pt')
    p.eval()
    mse =0
    r2s =0
    l1_loss_val =0
    test_epa_mae = 0
    g_mse = 0
    g_r2 = 0
    with open('testing_final2.txt','w+') as f:
        con = Console(file=f)
        with torch.no_grad():
            for data_num,items in enumerate(loader_test):
                masked_images, mask,  gt_images, e_77, a_gt, e_idx, a_idx = __cuda__(*items)
                pred_img = p(masked_images.float())
                hole_loss = l1_loss(gt_images, pred_img,mask)
                e_mse_loss,r2,epa_mae,pred,gt = get_r2_mse(e_77,e_idx, pred_img)
                gr,gm = ground_loss(pred_img, gt_images, mask)
                mse += e_mse_loss
                r2s += r2
                l1_loss_val += hole_loss
                test_epa_mae += epa_mae
                g_mse += gm
                g_r2 += gr
                con.print(f'testing {data_num} l1_loss:{(hole_loss):.4f}, EPA mse:{(e_mse_loss):.4f}, EPA mae:{(epa_mae):.4f}, ground mse:{(gm):.4f}, ground r2:{(gr):.4f}, r2:{(r2):.4f}')
            con.print(f'avg l1_loss:{(l1_loss_val/len(loader_test)):.4f}, EPA mse:{(mse/len(loader_test)):.4f}, EPA mae:{(test_epa_mae/len(loader_test)):.4f}, ground mse:{(g_mse/len(loader_test)):.4f}, ground r2:{(g_r2/len(loader_test)):.4f}, r2:{(r2s/len(loader_test)):.4f}')
        
if __name__ == '__main__':
    with open(sys.argv[1], "w") as report_file:
        c = Console(file=report_file)
        s = time.time()
        main(c)
        e = time.time()
        c.print(f'total time : {(e-s):.2f}')
