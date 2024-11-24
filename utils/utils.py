import torch
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
from torch import nn
from scipy.stats import kendalltau, spearmanr

class MMDataset(Dataset):
    def __init__(self,file_path,device='cpu',times_max=0,times_min=0):

        file = open(f'{file_path}', 'rb')
        data = pickle.load(file)
        self.path = torch.tensor(data['path'])
        self.image = torch.tensor(data['image'])
        self.path2Image = torch.tensor(data['path2Image'])
        self.times = data["times"]
        self.graph_edges = torch.tensor(data["graph_edges"])
        self.image_features = data['image_features']
        self.path_features = data['path_features']
        self.pad_token_id = 0
        self.mask_token_id = 1
        #0[ped] 1[mask] 2[cls] 3[sep]
        path_append = torch.randn(4, self.path_features.size(1))
        image_append = torch.randn(4, self.image_features.size(1))
        self.path_features = torch.cat((path_append,self.path_features),dim=0)
        self.image_features = torch.cat((image_append,self.image_features),dim=0)
        
        if times_max ==0:
            self.times_max = self.times.max()
            self.times_min = self.times.min()
        else:
            self.times_max = times_max
            self.times_min = times_min
        self.times = (self.times - self.times_min) / (self.times_max-self.times_min)

    def __len__(self):
        return self.path.size(0)

    def __getitem__(self, idx):
    #0[ped] 1[mask] 2[cls] 3[sep]
        path = self.path[idx]
        image = self.image[idx]
        time = self.times[idx]
        path2Image = self.path2Image[idx]
        graph_edges = self.graph_edges[idx]
        return path,image,time,path2Image,graph_edges


def mask_tokens(inputs, mask_prob=0.15):
    pad_token_id = 0
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    target_masked = inputs.clone()
    maskable = (inputs != pad_token_id) & (inputs != cls_token_id) & (inputs != sep_token_id)
    input_mask = torch.full(inputs.shape, fill_value=pad_token_id, device=inputs.device)
    mask = torch.bernoulli(torch.full(inputs.shape, mask_prob).to(inputs.device) * maskable).bool()
    mask_indices = torch.multinomial(maskable.float(), num_samples=1, replacement=False).to(inputs.device)
    mask = torch.scatter(mask, dim=1, index=mask_indices, src=torch.ones(mask_indices.size(),dtype=torch.bool,device=inputs.device))
    input_mask[mask] = mask_token_id
    input_mask[~mask] = inputs[~mask]
    target_masked = inputs.clone()
    target_masked[~mask] = pad_token_id
    return input_mask, inputs.clone(), target_masked

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def shuffle_tensor(tensor):
    original_indices = torch.arange(tensor.size(0), device=tensor.device)
    while True:
        indices = np.random.permutation(tensor.size(0)) 
        shuffled_indices = torch.tensor(indices).to(tensor.device)
        if not torch.any(shuffled_indices == original_indices):
            break
    shuffled_tensor = tensor[shuffled_indices]
    
    return shuffled_tensor

class TimeDataset(Dataset):
    def __init__(self,path,device='cpu'):
        f = open(path,'rb')
        emb ,time = pickle.load(f)
        self.emb = emb
        self.time = time
        self.min = self.time.min()
        self.max = self.time.max()
        self.time = (self.time - self.min) / (self.max - self.min)

    def __len__(self):
        return self.emb.size(0)

    def __getitem__(self, idx):
    #0[ped] 1[mask] 2[cls] 3[sep]
        emb = self.emb[idx]
        time = self.time[idx]
        return emb,time



class TimeDataset_tensor(Dataset):
    def __init__(self,path,time,device='cpu'):
        self.emb = path
        self.time = time
        self.min = self.time.min()
        self.max = self.time.max()
        self.time = (self.time - self.min) / (self.max - self.min)

    def __len__(self):
        return self.emb.size(0)

    def __getitem__(self, idx):
    #0[ped] 1[mask] 2[cls] 3[sep]
        emb = self.emb[idx]
        time = self.time[idx]
        return emb,time
    
class PathTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
        super(PathTimePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def mean_squared_error(y_true, y_pred):
    """
    计算均方根误差（RMSE）
    Args:
    - y_true: 真实值
    - y_pred: 预测值
    Returns:
    - rmse: 均方根误差
    """
    mse = torch.sum((y_true - y_pred) ** 2)
    return mse.item()

def mean_absolute_error(y_true, y_pred):
    """
    计算平均绝对误差（MAE）
    Args:
    - y_true: 真实值
    - y_pred: 预测值
    Returns:
    - mae: 平均绝对误差
    """
    mae = torch.sum(torch.abs(y_true - y_pred))
    return mae.item()

def mean_absolute_relative_error(y_true, y_pred):
    """
    计算平均绝对相对误差（MARE）
    Args:
    - y_true: 真实值
    - y_pred: 预测值
    Returns:
    - mare: 平均绝对相对误差
    """
    mask = y_true != 0
    abs_relative_error = torch.abs((y_true[mask] - y_pred[mask]) )
    ground_truth =  y_true[mask]
    abs_relative_error = torch.sum(abs_relative_error)
    ground_truth = torch.sum(ground_truth)
    return abs_relative_error.item(),ground_truth.item()


def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差（MAPE）
    Args:
    - y_true: 真实值
    - y_pred: 预测值
    Returns:
    - mape: 平均绝对百分比误差
    """
    mask = y_true != 0
    abs_percentage_error = torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0
    mape = torch.sum(abs_percentage_error)
    return mape.item()

def kendall_tau_torch(y_true, y_pred):
    """计算每组的肯达尔τ指数"""
    tau = []
    for i in range(y_true.shape[0]):
        tau.append(kendalltau(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy())[0])
    return torch.tensor(tau).sum()

def spearman_r_torch(y_true, y_pred):
    """计算每组的斯皮尔曼指数"""
    rho = []
    for i in range(y_true.shape[0]):
        rho.append(spearmanr(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy())[0])
    return torch.tensor(rho).sum()

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        positive_similarity = F.cosine_similarity(anchor, positive).unsqueeze(1)
        

        negative_similarity = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)
        

        similarities = torch.cat([positive_similarity, negative_similarity], dim=1)
        

        probabilities = F.softmax(similarities / self.temperature, dim=1)
        

        positive_probabilities = probabilities[:, 0]

        loss = -torch.log(positive_probabilities).mean()
        return loss