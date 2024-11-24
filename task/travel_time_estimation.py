import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch import nn 
import math
from torch.utils.data import  DataLoader
from utils import  mean_absolute_error, mean_absolute_relative_error, mean_absolute_percentage_error, mean_squared_error, MMDataset
from models.MMPath import MMPath
import argparse
import time
import json

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.0001,type=float)
    parser.add_argument('--opt',default='adam', type=str)
    parser.add_argument('--city',default='Xian',type = str)
    parser.add_argument('--nhead',default=4,type=int)
    parser.add_argument('--nlayer',default=5,type=int)
    parser.add_argument('--image_map',default='',type = str)
    parser.add_argument('--device',default='cuda:0',type = str)
    parser.add_argument('--dropout',default=0.2,type = float)
    parser.add_argument('--epoch',default=60,type = int)
    parser.add_argument('--batch_size',default=300,type = int)
    parser.add_argument('--pretain_weight',default='save/Xian/40epoches.pth',type = str)
    opt, unknown = parser.parse_known_args()
    return opt

class TimePredicter(nn.Module):
    def __init__(self, pre_param_path, dataset, opt,emb_size=64, mid_size=32, target_size=1):
        super(TimePredicter, self).__init__()
        
        if opt.city == 'Xian':
            img_max_len = 359
            path_max_len = 124
        elif opt.city == 'Aalborg':
            img_max_len = 750
            path_max_len = 134

        self.pre_model = MMPath(dataset.path_features, dataset.image_features, path_nhead=opt.nhead, img_nhead=opt.nhead, path_num_encoder_layers=opt.nlayer, path_num_decoder_layers=2, img_num_encoder_layers=opt.nlayer, path_max_len=path_max_len, img_max_len=img_max_len, path_dropout=opt.dropout, img_dropout=opt.dropout, mask_prob=0.4, predict=True, path_freeze=False, img_freeze=False,city=opt.city,mae_pretrain=True)
        pre_param = torch.load(pre_param_path)
        new_state_dict = {}
        for key in pre_param.keys():
            new_key = key.replace('module.', '', 1)
            if new_key in self.pre_model.state_dict().keys():
                new_state_dict[new_key] = pre_param[key]

        self.pre_model.load_state_dict(new_state_dict)
        self.fc1 = nn.Linear(emb_size, mid_size)
        self.fc2 = nn.Linear(mid_size, target_size)

    def forward(self, path, image, path2Image, graph_edges):
        x, path, path_emb, image, image_emb = self.pre_model(path, image, path2Image, graph_edges)
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        return output


seed = 615
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
opt = parse_opt()
city = opt.city


batch_size = opt.batch_size
fine_tune_dataset = MMDataset(f'data/{city}/travel_time_estimation/finetune.pkl')
val_dataset = MMDataset(f'data/{city}/travel_time_estimation/valid.pkl',times_max=fine_tune_dataset.times_max,times_min=fine_tune_dataset.times_min)
test_dataset = MMDataset(f'data/{city}/travel_time_estimation/test.pkl',times_max=fine_tune_dataset.times_max,times_min=fine_tune_dataset.times_min)
train_loader= DataLoader(dataset=fine_tune_dataset, batch_size = batch_size)
val_loader= DataLoader(dataset=val_dataset, batch_size = batch_size)
test_loader= DataLoader(dataset=test_dataset, batch_size = batch_size)
device = opt.device
pre = opt.pretain_weight
model = TimePredicter(pre, opt=opt, dataset=fine_tune_dataset)

model.to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

best_loss = float('inf')
best_mae = float('inf')
best_mare = float('inf')
best_mape = float('inf')
best_rmse = float('inf')
best_epoch=0
tbar = range(opt.batch_size)

all_train_time,all_val_time,all_test_time = 0.0, 0.0, 0.0

for epoch in tbar:
        model.train()
        train_loss,train_mae,train_mare,train_mape,train_batches,train_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        print(f'Epoch:{epoch}============>')
        for i, batch in enumerate(train_loader):
            path, image, times, path2Image, graph_edges= batch  
            path = path.to(device)
            image = image.to(device)
            path2Image = path2Image.to(device)
            graph_edges = graph_edges.to(device)
            times = times.to(device).float()
            start_time = time.time()
            output = model(path, image, path2Image, graph_edges).squeeze(1)
            optimizer.zero_grad() 
            loss = criterion(output, times)       
            loss.backward()
            optimizer.step()
            train_time +=time.time() - start_time
                    
            train_loss += loss.item()
            train_batches +=1
        all_train_time += train_time
        model.eval()
        val_mae1,val_loss,val_mae,val_mare_error,val_mare_truth,val_mape,val_rmse,val_batches,val_time =0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                path, image, times, path2Image, graph_edges= batch  
                path = path.to(device)
                image = image.to(device)
                path2Image = path2Image.to(device)
                graph_edges = graph_edges.to(device)
                times = times.to(device).float()
                start_time = time.time()
                output = model(path, image, path2Image, graph_edges).squeeze(1)
                val_time +=time.time() - start_time
                loss = criterion(output, times)

                times = times * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
                output = output * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
                val_loss += loss.item() * path.size(0)
                val_mae += mean_absolute_error(times,output)
                tmp = mean_absolute_relative_error(times,output)
                val_mare_error +=tmp[0]
                val_mare_truth +=tmp[1]
                val_mape += mean_absolute_percentage_error(times,output)
                val_rmse += mean_squared_error(times,output)
                val_batches +=1
            total = len(val_loader.dataset)
            train_loss = train_loss / total
            val_loss = val_loss / total
            val_mae = val_mae / total
            val_mare = val_mare_error / val_mare_truth
            val_mape = val_mape / total
            val_rmse = math.sqrt(val_rmse / total)    

            if val_mae < best_mae:
                best_epoch = epoch+1
                best_loss = val_loss
                best_state = model.state_dict()
            best_mae = min(val_mae,best_mae)
            best_mare = min(val_mare,best_mare)
            best_mape = min(val_mape,best_mape)
            best_rmse = min(val_rmse,best_rmse)      
            print(json.dumps({
                "val_loss":val_loss,
                "val_mae":val_mae,
                "val_mare":val_mare,
                "val_mape":val_mape,
                "val_rmse":val_rmse,
                }))
        
            scheduler.step(val_loss)
            model.eval()
            test_mae1,test_loss,test_mae,test_mare_error,test_mare_truth,test_mape,test_rmse,test_time,test_batches =0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0
            for batch in test_loader:
                
                path, image, times, path2Image, graph_edges= batch  
                path = path.to(device)
                image = image.to(device)
                path2Image = path2Image.to(device)
                graph_edges = graph_edges.to(device)
                times = times.to(device).float()
                start_time = time.time()
                output = model(path, image, path2Image, graph_edges).squeeze(1)
                test_time +=time.time() - start_time
                loss = criterion(output, times)

                times = times * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
                output = output * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
                test_loss += loss.item() * path.size(0)
                test_mae += mean_absolute_error(times,output)
                tmp = mean_absolute_relative_error(times,output)
                test_mare_error +=tmp[0]
                test_mare_truth +=tmp[1]
                test_mape += mean_absolute_percentage_error(times,output)
                test_rmse += mean_squared_error(times,output)
                test_batches +=1
            all_test_time+=test_time
            total = len(test_loader.dataset)
            test_loss = test_loss / total
            test_mae = test_mae / total
            test_mare = test_mare_error / test_mare_truth
            test_mape = test_mape / total
            test_rmse = math.sqrt(test_rmse / total)
       
            print(json.dumps({
                "test_loss":test_loss,
                "test_mae":test_mae,
                "test_mare":test_mare,
                "test_mape":test_mape,
                "test_rmse":test_rmse,
                }))

            