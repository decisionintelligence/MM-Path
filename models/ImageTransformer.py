import torch
import torch.nn
from models.PositionalEncoding import PositionalEncoding
import math
from torch import nn;
import torch.nn.functional as F

class encoderLayer(nn.Module):
    def __init__(self,pretrained_weights,num_tokens, embedding_size,pre_train,freeze):
        super(encoderLayer, self).__init__()

        self.embedding = nn.Embedding(num_tokens, embedding_size)
        if pretrained_weights is not None:
            if pre_train:
                self.embedding.weight.data.copy_(pretrained_weights)
        self.embedding.weight.data[4:].requires_grad = (not freeze)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImageTransformer(nn.Module):
    def __init__(self,pretrained_weights, num_tokens, embedding_size,reduced_size, nhead, num_encoder_layers, dim_feedforward, dropout, max_len,pre_train=True,freeze=False):
        super(ImageTransformer, self).__init__()
        self.reduced_size = reduced_size

        self.embedding = encoderLayer(pretrained_weights,num_tokens,embedding_size,pre_train,freeze)

        self.pos_encoder = PositionalEncoding(reduced_size, dropout, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
    def forward(self, src ):
        src_key_padding_mask = (src == 0)
        src = self.embedding(src) * math.sqrt(self.reduced_size)
        src = self.pos_encoder(src)
        emb = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return src.clone(), emb 