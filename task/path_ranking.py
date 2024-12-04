import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from utils.utils import  mean_absolute_error,MMDataset,kendall_tau_torch,spearman_r_torch
from models.MMPath import MMPath
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.003,type=float)
    parser.add_argument('--opt',default='adam', type=str)
    parser.add_argument('--city',default='Xian',type = str)
    parser.add_argument('--nhead',default=4,type=int)
    parser.add_argument('--nlayer',default=5,type=int)
    parser.add_argument('--dropout',default=0.2,type = float)
    parser.add_argument('--image_map',default='',type = str)
    parser.add_argument('--device',default='cuda:0',type = str)
    parser.add_argument('--epoch',default=60,type = int)
    parser.add_argument('--batch_size',default=100,type = int)
    parser.add_argument('--pretain_weight',default='save/Xian/40epoches.pth',type = str)
    opt, unknown = parser.parse_known_args()

    return opt

class PathPredicter(nn.Module):
    def __init__(self, pre_param_path, dataset, opt, emb_size=64, mid_size=32, target_size=1):
        super(PathPredicter, self).__init__()
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

        output = F.sigmoid(self.fc1(x))
        output = F.sigmoid(self.fc2(output))
        return output
seed = 615
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    

opt = parse_opt()

city = opt.city
batch_size = opt.batch_size
fine_tune_dataset = MMDataset(f'data/{city}/path_ranking/finetune.pkl')
val_dataset = MMDataset(f'data/{city}/path_ranking/valid.pkl',times_max=fine_tune_dataset.times_max,times_min=fine_tune_dataset.times_min)
test_dataset = MMDataset(f'data/{city}/path_ranking/test.pkl',times_max=fine_tune_dataset.times_max,times_min=fine_tune_dataset.times_min)

train_loader= DataLoader(dataset=fine_tune_dataset, batch_size = batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
TopK=8
device = opt.device
pre = opt.pretain_weight

model = PathPredicter(pre, opt=opt, dataset=fine_tune_dataset)
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
model.to(device)

tbar = range(60)
for epoch in tbar:   
    model.train()
    train_loss,train_mae,train_mare,train_mape,train_batches,train_time = 0.0, 0.0, 0.0, 0.0, 0,0.0
    print(f'Epoch:{epoch}===============>')
    for i, batch in enumerate(train_loader):
        
        path, image, jaccards, path2Image, graph_edges= batch    
        path = path.reshape(-1,path.size(2))
        image = image.reshape(-1,image.size(2))
        path2Image = path2Image.reshape(-1,path2Image.size(2))
        jaccards = jaccards.reshape(-1)
        graph_edges = graph_edges.reshape(-1,graph_edges.size(2),graph_edges.size(3))
        path = path.to(device)
        image = image.to(device)
        path2Image = path2Image.to(device)
        graph_edges = graph_edges.to(device)
        jaccards = jaccards.to(device)
        output = model(path, image, path2Image, graph_edges).squeeze(1)
        optimizer.zero_grad() 
        loss = criterion(output, jaccards)       
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches +=1

    model.eval()
    val_loss,val_mae,val_tau,val_rou,val_mrr,train_time =0.0, 0.0, 0.0, 0.0, 0.0,0.0
    with torch.no_grad(): 
        for batch in val_loader:
            path, image, jaccards, path2Image, graph_edges= batch    
        
            path = path.reshape(-1,path.size(2))
            image = image.reshape(-1,image.size(2))
            path2Image = path2Image.reshape(-1,path2Image.size(2))
            jaccards = jaccards.reshape(-1)
            graph_edges = graph_edges.reshape(-1,graph_edges.size(2),graph_edges.size(3))

            path = path.to(device)
            image = image.to(device)
            path2Image = path2Image.to(device)
            graph_edges = graph_edges.to(device)
            jaccards = jaccards.to(device)

            output = model(path, image, path2Image, graph_edges).squeeze(1)
            loss = criterion(output, jaccards)

            jaccards = jaccards * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
            output = output * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
            
            sorted_output, predict_indices = torch.sort(output.reshape(-1,TopK), dim=1, descending=True)
            sorted_output, true_indices = torch.sort(jaccards.reshape(-1,TopK), dim=1, descending=True)

            tau = kendall_tau_torch(true_indices,predict_indices).item()
            rou = spearman_r_torch(true_indices,predict_indices).item()
            mrr = (1/(predict_indices[:,0] + 1)).sum().item()

            val_loss += loss.item() * path.size(0)
            val_mae += mean_absolute_error(jaccards,output)
            val_tau += tau
            val_rou += rou
            val_mrr += mrr
    
        total = len(val_loader.dataset)
        val_loss = val_loss / total
        val_mae = val_mae / total / TopK
        val_tau = val_tau / total
        val_rou = val_rou / total
        val_mrr = val_mrr / total
    
        print({
            "val_loss":val_loss,
            "val_mae":val_mae,
            "val_mrr":val_mrr,
            "val_tau":val_tau,
            'val_rou':val_rou
            })
        
        scheduler.step(val_loss)
            
test_loss,test_mae,test_tau,test_rou,test_mrr,test_time =0.0, 0.0, 0.0, 0.0, 0.0,0.0

for batch in test_loader:
    path, image, jaccards, path2Image, graph_edges= batch    

    path = path.reshape(-1,path.size(2))
    image = image.reshape(-1,image.size(2))
    path2Image = path2Image.reshape(-1,path2Image.size(2))
    jaccards = jaccards.reshape(-1)
    graph_edges = graph_edges.reshape(-1,graph_edges.size(2),graph_edges.size(3))

    path = path.to(device)
    image = image.to(device)
    path2Image = path2Image.to(device)
    graph_edges = graph_edges.to(device)
    jaccards = jaccards.to(device)
    output = model(path, image, path2Image, graph_edges).squeeze(1)
    loss = criterion(output, jaccards)

    jaccards = jaccards.reshape(-1,TopK) * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
    output = output.reshape(-1,TopK) * (fine_tune_dataset.times_max-fine_tune_dataset.times_min) + fine_tune_dataset.times_min
    
    sorted_output, predict_indices = torch.sort(output.reshape(-1,TopK), dim=1, descending=True)
    sorted_output, true_indices = torch.sort(jaccards.reshape(-1,TopK), dim=1, descending=True)

    tau = kendall_tau_torch(true_indices,predict_indices).item()
    rou = spearman_r_torch(true_indices,predict_indices).item()
    mrr = (1/(predict_indices[:,0] + 1)).sum().item()


    test_loss += loss.item() * path.size(0)
    test_mae += mean_absolute_error(jaccards,output)
    test_tau += tau
    test_rou += rou
    test_mrr += mrr
total = len(test_loader.dataset)
test_loss = test_loss / total
test_mae = test_mae / total / TopK
test_tau = test_tau / total  
test_rou = test_rou / total
test_mrr = test_mrr / total   

print({
    "test_loss":test_loss,
    "test_mae":test_mae,
    'test_mrr':test_mrr,
    "test_tau":test_tau,
    'test_rou':test_rou
    })
            