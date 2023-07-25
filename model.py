import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
from torchmetrics import R2Score, MeanAbsolutePercentageError
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rich.table import Table
from rich.console import Console
import gc
import torch.nn as nn
import random
color_list = ['mediumblue', 'aqua', 'yellow', 'orange', 'chocolate', 'firebrick', 'fuchsia']
newcmp = LinearSegmentedColormap.from_list('testCmap', colors=color_list, N=256)

class RFRNetModel():
    def __init__(self):
        self.G = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.r2s = 0.0
        self.mse = 0.0
        self.ground_mse = 0.0
        self.ground_r2 = 0.0
        self.epa_mae = 0.0
    def initialize_model(self, c, path=None, train=True):
        self.c = c
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        try:
            
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                self.c.print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            self.c.print('No trained model, from start')
            self.iter = 1
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.c.print("Model moved to cuda")
            
            self.G = nn.DataParallel(self.G)

            self.G.cuda()
        else:
            self.device = torch.device("cpu")
            
    def validate(self,data,file_name,count):
        if not os.path.exists(f'./validate/{file_name}'):
            os.makedirs(f'./validate/{file_name}')
        self.G.eval()
        with torch.no_grad():
            masked_images, tw_masks,  gt_images, e_77, a_gt, e_idx, a_idx, masks = self.__cuda__(*data)
            fake_B, mask = self.G(masked_images.float(), masks.float())
            self.mse = self.get_r2_mse(e_77,e_idx,fake_B)
            # calc loss
            valid_loss = self.l1_loss(gt_images, fake_B, masks)
            hole_loss = self.l1_loss(gt_images, fake_B, (1 - masks))
            l1_loss = valid_loss.detach() + hole_loss.detach()
            self.c.print(f'validate {count} Mse : {self.mse:.4f}, R2 : {self.r2s:.4f}, l1_loss : {l1_loss:.4f}')
            # records + show result fig
            with open(f'./validate/{file_name}/record_{count}.txt','w+') as f:
                c = Console(file = f)
                t = Table(title = f'record {count}')
                t.add_column('pred')
                t.add_column('gt')
                for p,g in zip(self.pred,self.gt):
                    t.add_row(f'{p.item():.4f}', f'{g.item():.4f}')
                c.print(t)
                c.print(f'validate {count} Mse : {self.mse:.4f}, R2 : {self.r2s:.4f}, l1_loss : {l1_loss:.4f}')
    def train(self, train_loader, save_path, finetune = False, iters=450000, file_name = 'test'):
        self.fname = f'pic_{file_name}_{iters}'
        self.G.train()
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
        self.c.print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        while self.iter<iters+1:
            s_time = time.time()
            self.G.train()
            for data_num,items in enumerate(train_loader):
                # forward+backward
                masked_images, tw_masks,  gt_images, e_77, a_gt, e_idx, a_idx, masks = self.__cuda__(*items)
                self.forward(masked_images.float(), masks.float(), gt_images.float(),tw_masks)
                self.update_parameters(e_77, a_gt, e_idx, a_idx)
            e_time = time.time()
            int_time = e_time - s_time
            # records
            self.c.print(f"epoch:{self.iter}, l1_loss:{(self.l1_loss_val/len(train_loader)):.4f}, EPA mse:{(self.mse/len(train_loader)):.4f}, EPA mae:{(self.epa_mae/len(train_loader)):.4f}, \
r2:{(self.r2s/len(train_loader)):.4f}, ground mse: {(self.ground_mse/len(train_loader)):.4f}, \
ground r2: {(self.ground_r2/len(train_loader)):.4f}, time_taken:{(int_time):.2f}")
            self.l1_loss_val = 0.0
            self.r2s = 0.0
            self.mse = 0.0
            self.ground_mse = 0.0
            self.ground_r2 = 0.0
            self.epa_mae = 0.0
            if self.iter % 20 == 0:
                with open(f'./train_records/record_{file_name}_{iters}.txt','w+') as f:
                    c = Console(file = f)
                    t = Table(title = f'record_iter_{self.iter}')
                    t.add_column('pred')
                    t.add_column('gt')
                    for p,g in zip(self.pred,self.gt):
                        t.add_row(f'{p.item():.4f}', f'{g.item():.4f}')
                    c.print(t)
            if self.iter % 10 == 0:
                # save model file
                self.draw(self.fake_B*tw_masks,gt_images*tw_masks)
                if not os.path.exists('{:s}'.format(save_path)):
                    os.makedirs('{:s}'.format(save_path))
                save_ckpt('{:s}/g_10000.pth'.format(save_path, self.iter*10 ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
            self.iter += 1
            self.pred = None
            self.gr = None
            # clear memory
            gc.collect(2)
        # save model file
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
        save_ckpt('{:s}/g_10000.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
    def test(self, test_loader, result_save_path):
        # enble evaluate mode -> fix mean and std from batchnorm
        # self.G.eval()
        self.fname = result_save_path
        if not os.path.exists(f'./pic_results/{self.fname}'):
            os.makedirs(f'./pic_results/{self.fname}')
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        self.iter = 0
        M = []
        R2 = []
        EMAE = []
        GM = []
        GR = []
        GMAE = []
        for items in test_loader:
            # forward
            masked_images, tw_masks,  gt_images, e_77, a_gt, e_idx, a_idx, masks = self.__cuda__(*items)
            fake_B, mask = self.G(masked_images.float(), masks.float())
            self.mse = self.get_r2_mse(e_77,e_idx,fake_B)
            self.ground_loss(fake_B, gt_images, tw_masks)
            # calc loss
            valid_loss = self.l1_loss(gt_images, fake_B, tw_masks)
            l1_loss = valid_loss.detach()
            M.append(self.mse)
            R2.append(self.r2s)
            EMAE.append(self.epa_mae)
            GM.append(self.ground_mse)
            GR.append(self.ground_r2)
            GMAE.append(l1_loss)
            self.c.print(f"test :{self.iter}, l1_loss:{(l1_loss):.4f}, EPA mse:{(self.mse):.4f}, EPA mae:{(self.epa_mae):.4f}, \
r2:{(self.r2s):.4f}, ground mse: {(self.ground_mse):.4f}, ground r2: {(self.ground_r2):.4f}")
            # records + show result fig
            with open(f'./test_record/test_record_{count}.txt','w+') as f:
                c = Console(file = f)
                t = Table(title = f'record {count}')
                t.add_column('pred')
                t.add_column('gt')
                for p,g in zip(self.pred,self.gt):
                    t.add_row(f'{p.item():.4f}', f'{g.item():.4f}')
                c.print(t)
            count += 1
            self.iter += 1
            self.draw(fake_B*tw_masks, gt_images*tw_masks,e_idx,a_idx)
            self.r2s = 0.0
            self.mse = 0.0
            self.ground_mse = 0.0
            self.ground_r2 = 0.0
            self.epa_mae = 0.0
        self.c.print(f'Testing avg EPA mse: {sum(M)/len(M):.4f}, R2: {sum(R2)/len(R2):.4f}, MAE: {sum(EMAE)/len(EMAE):.4f}, ground mse: {sum(GM)/len(GM):.4f}, R2: {sum(GR)/len(GR):.4f}, MAE: {sum(GMAE)/len(GMAE):.4f}')
    def forward(self, masked_image, mask, gt_image, tw_mask):
        # set some 
        self.real_A = masked_image
        self.real_B = gt_image #true img not input
        self.mask = tw_mask
        # gen predict
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        # self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    def draw(self,pred,gt,e_idx=None, a_idx=None):
        # plt.figure(figsize=(20, 15))
        count = 1
        clist = [(0,"white"), (1./10.,"green"), (1, "red")]
        cmap = LinearSegmentedColormap.from_list("name", clist)
        if not os.path.exists(f'./results/{self.fname}'):
            os.makedirs(f'./results/{self.fname}')
        for i,j in zip(pred,gt):
            # pointwise
            # idx = int(count//2)
            # e_idx0 = e_idx[idx,0][e_idx[idx,0].nonzero()].detach().cpu().numpy()
            # e_idx1 = e_idx[idx,1][e_idx[idx,1].nonzero()].detach().cpu().numpy()
            # a_idx0 = a_idx[idx,0][a_idx[idx,0].nonzero()].detach().cpu().numpy()
            # a_idx1 = a_idx[idx,1][a_idx[idx,1].nonzero()].detach().cpu().numpy()
            # plt.title(f'iter {self.iter} pred epa')
            # plt.scatter(e_idx1,e_idx0,c=i[0][e_idx0,e_idx1].detach().cpu().numpy(),cmap=cmap,s=1)
            # plt.colorbar(fraction = 0.05)
            # plt.savefig(f'./results/{self.fname}/iter_{self.iter}_e.png')
            # plt.clf()
            # plt.title(f'iter {self.iter} pred ab')
            # plt.scatter(a_idx1,a_idx0,c=i[0][a_idx0,a_idx1].detach().cpu().numpy(),cmap=cmap,s=1)
            # plt.colorbar(fraction = 0.05)
            # plt.savefig(f'./results/{self.fname}/iter_{self.iter}_a.png')
            # plt.clf()
            # count += 1
            # plt.title(f'iter {self.iter} gt epa')
            # plt.scatter(e_idx1,e_idx0,c=j[0][e_idx0,e_idx1].detach().cpu().numpy(),cmap=cmap,s=1)
            # plt.colorbar(fraction = 0.05)
            # plt.savefig(f'./results/{self.fname}/iter_{self.iter}_e_gt.png')
            # plt.clf()
            # plt.title(f'iter {self.iter} gt ab')
            # plt.scatter(a_idx1,a_idx0,c=j[0][a_idx0,a_idx1].detach().cpu().numpy(),cmap=cmap,s=1)
            # plt.colorbar(fraction = 0.05)
            # plt.savefig(f'./results/{self.fname}/iter_{self.iter}_a_gt.png')
            # whole image
            plt.subplot(len(pred),2,count)
            plt.title(f'iter {self.iter} pred')
            plt.imshow(i[0].detach().cpu().numpy()[::-1],cmap=cmap)
            plt.colorbar(fraction = 0.05)
            count += 1
            plt.subplot(len(pred),2,count)
            plt.title('gt')
            plt.imshow(j[0].detach().cpu().numpy()[::-1],cmap=cmap,vmin = 0,vmax=np.max(i[0].detach().cpu().numpy()))
            plt.colorbar(fraction = 0.05)
            count += 1
        plt.savefig(f'./results/{self.fname}/iter_{self.iter}_train.png')
        plt.clf()
        plt.close()

    def update_parameters(self,e_77, a_gt, e_idx, a_idx):
        self.update_G(e_77, a_gt, e_idx, a_idx)
        self.update_D()
    
    def update_G(self,e_77, a_gt, e_idx, a_idx):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss(e_77, a_gt, e_idx, a_idx)
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_r2_mse(self,e_77, idx, fake_B, epa = True):
        R2 = R2Score(num_outputs=1)
        gt = torch.tensor([], dtype = torch.double).to(self.device)
        pred = torch.tensor([], dtype = torch.double).to(self.device)
        loss = 0
        for i in range(len(idx)):
            _idx0 = idx[i,0][idx[i,0].nonzero()]
            _idx1 = idx[i,1][idx[i,1].nonzero()]
            pred = torch.concat([fake_B[i,0][_idx0,_idx1],pred])
            gt = torch.concat([e_77[i,0][_idx0,_idx1],gt])
        mse = torch.nn.MSELoss()(gt,pred)
        if epa:
            self.pred = pred
            self.gt = gt
            self.epa_mae += torch.mean(torch.abs(self.pred - self.gt)).item()
            self.r2s += R2(self.pred.cpu(),self.gt.cpu()).item()
        return mse

    def get_g_loss(self,e_77, a_gt, e_idx, a_idx):
        real_B = self.real_B
        fake_B = self.fake_B
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        e_mse_loss = self.get_r2_mse(e_77,e_idx, fake_B)
        a_mse_loss = self.get_r2_mse(a_gt,a_idx, fake_B, False)
        self.ground_loss(fake_B, real_B, self.mask)
        self.mse += e_mse_loss
        loss_G = (  a_mse_loss * 15
                    + valid_loss * 1
                    + e_mse_loss * 20) 
        self.l1_loss_val += valid_loss.detach()
        return loss_G
    def ground_loss(self,fake_B,real_b,mask):
        R2 = R2Score(num_outputs=1)
        idx = torch.argwhere(mask)
        gt = torch.tensor([], dtype = torch.double).to(self.device)
        pred = torch.tensor([], dtype = torch.double).to(self.device)
        for i in range(len(fake_B)):
            _idx0 = idx[idx[:,0]==i][:,2]
            _idx1 = idx[idx[:,0]==i][:,3]
            pred = torch.concat([fake_B[i,0][_idx0,_idx1],pred])
            gt = torch.concat([real_b[i,0][_idx0,_idx1],gt])
        self.ground_r2 += R2(pred.cpu(),gt.cpu()).item()
        self.ground_mse += torch.mean(torch.abs(pred - gt)**2).item()
        del(pred,gt)
    def l1_loss(self, f1, f2, mask = 1):
        # print(torch.isnan(f1).any())
        # print(torch.isnan(f2).any())
        return torch.mean(torch.abs(f1 - f2)*mask)*torch.numel(mask)/torch.sum(mask)
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            