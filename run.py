import argparse
import os
from model import RFRNetModel
from dataset2_ar import Dataset
from torch.utils.data import DataLoader
from rich.console import Console
from torch.utils.data.distributed import DistributedSampler
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--model_save_path', type=str, default='./checkpoint')
    parser.add_argument('--result_save_path', type=str, default='./results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="./checkpoint/g_440000.pth")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--txt', type=str, default="test")
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model = RFRNetModel()
    if args.test:
        f = open(args.txt + '_' + str(args.num_iters) + '.txt','w')
        c = Console(file = f)
        model.initialize_model(c, args.model_path, False)
        model.cuda()
        train_data = Dataset()
        train_idx, val_idx = train_test_split(list(range(len(train_data))), test_size=0.2)
        datasets = {}
        datasets['train'] = Subset(train_data, train_idx)
        datasets['val'] = Subset(train_data, val_idx)
        
        loader_train=DataLoader(dataset=datasets['train'],batch_size=4,shuffle=True,num_workers=2)
        loader_test=DataLoader(dataset=datasets['val'],batch_size=1,shuffle=True,num_workers=2)
        dataloader = DataLoader(Dataset(training=False),batch_size = args.batch_size)
        model.test(dataloader, args.result_save_path)
    else:
        f = open('./training_note/'+args.txt + '_' + str(args.num_iters) + '.txt','w')
        c = Console(file = f)
        model.initialize_model(c, args.model_path, True)
        model.cuda()
        train_data = Dataset()
        train_idx, val_idx = train_test_split(list(range(len(train_data))), test_size=0.2)
        datasets = {}
        datasets['train'] = Subset(train_data, train_idx)
        datasets['val'] = Subset(train_data, val_idx)
        
        loader_train=DataLoader(dataset=datasets['train'],batch_size=4,shuffle=True,num_workers=2)
        loader_test=DataLoader(dataset=datasets['val'],batch_size=1,shuffle=True,num_workers=2)
        # dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = False, num_workers = args.n_threads)
        model.train(loader_train, args.model_save_path, args.finetune, args.num_iters,args.txt)
        f = open('./test_'+args.txt + '_' + str(args.num_iters) + '.txt','w')
        c = Console(file = f)
        model.c = c
        model.test(loader_test, args.result_save_path)

if __name__ == '__main__':
    run()
