import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from utils.utils import shuffle_tensor,InfoNCELoss
import os

def train(model,dataloader,gpu,epoches_num,lr,opt,use_wandb=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    beta=1
    device = opt.device
    tripletLoss = nn.TripletMarginLoss(margin = beta,reduction='mean').to(opt.device)
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0,reduction='mean')
    cos_loss =nn.CosineEmbeddingLoss()
    infonce_loss = InfoNCELoss(temperature=0.05)
    tbar =tqdm(range(epoches_num))
    os.makedirs(f'save/{opt.city}/', exist_ok=True)
    model.train()
    for epoch in tbar:

        epoch_loss,epoch_loss_mask,epoch_loss_fuse,epoch_loss_fine,epoch_loss_medium,epoch_loss_coarse = [0.0] * 6
        
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            path,image,time,path2Image,graph_edges = batch    
            path = path.to(gpu)
            image = image.to(gpu)
            time = time.to(gpu)
            path2Image = path2Image.to(gpu)
            graph_edges = graph_edges.to(gpu)
            # 0[ped] 1[mask] 2[cls] 3[sep]
            sep_mask = (path[:,:] == 3)
            node_mask = (path[:,:]>3)

            predict, targets_masked, path, path_emb, image, image_emb, y, z = model(path,image,path2Image,graph_edges)
            # Task 1: Mask
            loss_mask = cross_entropy_loss(predict.transpose(1, 2), targets_masked) 
    
            # Task 2: Fusion of Similarities
            yn = shuffle_tensor(y)
            zn = shuffle_tensor(z)
            loss_fuse = tripletLoss(y,z,zn) + tripletLoss(z,y,yn)
                
            # Task 3: Multi-granularity Alignment
            path_node = path_emb[node_mask]
            image_node = image_emb[node_mask]
            path_sep = path_emb[sep_mask]
            image_sep = image_emb[sep_mask]

            node_target = torch.ones([path_node.size(0)]).to(device)
            sep_target = torch.ones([path_sep.size(0)]).to(device)
            loss_fine = cos_loss(path_node,image_node,node_target)
            loss_medium = cos_loss(path_sep,image_sep,sep_target)

            path_cls = path_emb[:,0,:]
            image_cls = image_emb[:,0,:]
            num = path_cls.size(0)
            negative_samples_nums = 10
            path_negative_samples_index = torch.randint(0, num - 1, (num, negative_samples_nums),device=device)
            image_negative_samples_index = torch.randint(0, num - 1, (num, negative_samples_nums),device=device)
            pos_template = torch.arange(num).repeat(negative_samples_nums, 1).T.to(device)

            path_negative_samples_index = (path_negative_samples_index + (path_negative_samples_index == pos_template).long())%num
            image_negative_samples_index = (image_negative_samples_index + (image_negative_samples_index == pos_template).long())%num

            path_negative_samples = path_cls[path_negative_samples_index]
            image_negative_samples = image_cls[image_negative_samples_index]

            loss_coarse = infonce_loss(path_cls,image_cls,image_negative_samples) + infonce_loss(image_cls,path_cls,path_negative_samples)

          
            loss_fuse = 2 * loss_fuse * 3
            loss_mask = 5e-1 * loss_mask * 3
            loss_fine = 5 * loss_fine
            loss_medium = 5 * loss_medium
            loss_coarse = loss_coarse
            
            loss = loss_mask  +  loss_fuse +  (loss_coarse + loss_fine + loss_medium)

            loss.backward()
            optimizer.step()

            if path.size(0) == opt.batch_size:
                tbar.set_postfix({"batch": i,"loss":loss.item()}, refresh=True)
            epoch_loss += loss.item() 
            epoch_loss_mask += loss_mask.item() 
            epoch_loss_fuse += loss_fuse.item() 
            epoch_loss_fine += loss_fine.item() 
            epoch_loss_medium += loss_medium.item() 
            epoch_loss_coarse += loss_coarse.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i+1)
        epoch_loss_mask = epoch_loss_mask / (i+1)
        epoch_loss_fuse = epoch_loss_fuse / (i+1)
        epoch_loss_fine = epoch_loss_fine / (i+1)
        epoch_loss_medium = epoch_loss_medium / (i+1) 
        epoch_loss_coarse = epoch_loss_coarse / (i+1)        
        tbar.set_postfix({
            "epoch": epoch,
            'epoch_loss':epoch_loss}, refresh=True)    
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'save/{opt.city}/{epoch+1}epoches.pth')

