import torch
from torch import nn
from models.PathTransformer import PathTransformer
from models.ImageTransformer import ImageTransformer
from torch import nn
import torch.nn.functional as F
from models.GCN import BatchGCN

class MMPath(nn.Module):
    def __init__(self, path_features=None,img_features=None, path_nhead=4, path_num_encoder_layers=5, path_num_decoder_layers=2, path_dim_feedforward=256, path_dropout=0.1, path_max_len=66,path_pre_train= True,path_freeze=False,
                     img_nhead=4, img_num_encoder_layers=5, img_dim_feedforward=256, img_dropout=0.1, img_max_len=256,img_pre_train=True,img_freeze=False
                    ,nhid = 64,nhead=4,predict=False,mask_prob=0.5,city="Aalborg",mae_pretrain=True):
        super(MMPath, self).__init__()

        path_num_tokens = path_features.size(0)
        path_embedding_size = path_features.size(1)
        img_num_tokens = img_features.size(0)
        img_embedding_size = img_features.size(1)
        img_reduce_size = path_features.size(1)
        
        self.path_model = PathTransformer(path_features,path_num_tokens, path_embedding_size, path_nhead, path_num_encoder_layers, path_num_decoder_layers, path_dim_feedforward, path_dropout, path_max_len,path_pre_train,path_freeze,predict=predict,mask_prob=mask_prob,city=city,mae_pretrain=mae_pretrain)
        self.image_model = ImageTransformer(img_features,img_num_tokens, img_embedding_size,img_reduce_size, img_nhead, img_num_encoder_layers, img_dim_feedforward, img_dropout, img_max_len,img_pre_train,img_freeze)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=nhid, num_heads=nhead,batch_first=True)
        self.fc1 = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, nhid)

        self.gcn = BatchGCN(64,128,64)

        self.predict = predict
        self.path_max_len = path_max_len

    def forward(self, path, image, path2image,graph_edges):
        path_index = path.clone()
        image_index = image.clone()
        path2image = torch.clamp(path2image, min=0)
        image_index = torch.gather(image_index,1,path2image)

        if self.predict:
            path, path_emb = self.path_model(path)
        else:
            predict, targets_masked, path, path_emb = self.path_model(path)

        image, image_emb = self.image_model(image)

        path2Image = path2image.unsqueeze(-1).expand(-1,-1 , image_emb.size(2))

        
        if self.predict:

            x_output = self.gcn(torch.cat((path_emb,image_emb),dim=1),graph_edges)
            x_path = x_output[:,:self.path_max_len,:]
            x_image = x_output[:,self.path_max_len:,:]
            x_image = torch.gather(x_image, 1, path2Image)
            x = torch.cat((x_path,x_image),dim=1).mean(1)
            return x, path, path_emb, image, image_emb

        else:  
            y_output = self.gcn(torch.cat((path_emb,image),dim=1),graph_edges)
            y_path = y_output[:,:self.path_max_len,:]
            y_image = y_output[:,self.path_max_len:,:]

            
            y_image = torch.gather(y_image, 1, path2Image)
            y = self.fc1(torch.cat((y_path,y_image),dim=1).mean(1))

            z_output = self.gcn(torch.cat((path,image_emb),dim=1),graph_edges)
            z_path = z_output[:,:self.path_max_len,:]
            z_image = z_output[:,self.path_max_len:,:]
            z_image = torch.gather(z_image, 1, path2Image)
            z = self.fc2(torch.cat((z_path,z_image),dim=1).mean(1))
        

            image_emb = torch.gather(image_emb, 1, path2Image)
            image = torch.gather(image, 1, path2Image)
            return predict, targets_masked, path, path_emb, image, image_emb, y, z
