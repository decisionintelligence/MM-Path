import torch
from torch.utils.data import  DataLoader
import argparse
from utils.utils import MMDataset
from models.MMPath import MMPath
import numpy as np
from utils.train import train
import random

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--opt',default='adam', type=str)
    parser.add_argument('--sigma',default=0.005,type=float)
    parser.add_argument('--city',default='Xian',type = str)
    parser.add_argument('--dropout',default=0.2,type = float)
    parser.add_argument('--nhead',default=4,type=int)
    parser.add_argument('--nlayer',default=5,type=int)
    parser.add_argument('--device',default='cuda:0',type = str)
    parser.add_argument('--epoch',default=50,type = int)
    parser.add_argument('--batch_size',default=0,type = int)
    opt, unknown = parser.parse_known_args()
    return opt

def set_seed(seed_value):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ =="__main__":

    set_seed(615)
    opt = parse_opt()
    device = opt.device
    
    city = opt.city
    dataset = MMDataset(f'data/{city}/pretrain.pkl')
    
    if city == 'Aalborg':
        model = MMPath(dataset.path_features,dataset.image_features,path_nhead=opt.nhead,img_nhead=opt.nhead,path_num_encoder_layers=opt.nlayer,path_num_decoder_layers=2,img_num_encoder_layers=opt.nlayer,path_max_len=134,
                     img_max_len=750,path_dropout=opt.dropout,img_dropout=opt.dropout,mask_prob = 0.4,img_freeze=True,path_freeze=True,
                     img_pre_train = True)
        batch_size=500
    elif city == 'Chengdu':
        model = MMPath(dataset.path_features,dataset.image_features,path_nhead=opt.nhead,img_nhead=opt.nhead,path_num_encoder_layers=opt.nlayer,path_num_decoder_layers=2,img_num_encoder_layers=opt.nlayer,path_max_len=65+2,
                     img_max_len=272+2,path_dropout=opt.dropout,img_dropout=opt.dropout,mask_prob = 0.4,img_freeze=True,path_freeze=True,
                     img_pre_train = True,city=city)
        batch_size=200
    elif city == 'Xian':
        model = MMPath(dataset.path_features,dataset.image_features,path_nhead=opt.nhead,img_nhead=opt.nhead,path_num_encoder_layers=opt.nlayer,path_num_decoder_layers=2,img_num_encoder_layers=opt.nlayer,path_max_len=124,
                     img_max_len=359,path_dropout=opt.dropout,img_dropout=opt.dropout,mask_prob = 0.4,img_freeze=True,path_freeze=True,
                     img_pre_train = True,city=city)
        batch_size=200
    if opt.batch_size==0:
        opt.batch_size = batch_size
    model.to(device)
    rand_loader = DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=True)
    train(model,rand_loader,device,opt.epoch,opt.lr,opt,use_wandb=False)