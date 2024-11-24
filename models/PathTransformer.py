import torch
from models.PositionalEncoding import PositionalEncoding
import math
from utils.utils import mask_tokens,generate_square_subsequent_mask
from torch import nn
import torch.nn.functional as F
from functools import partial
import math
import torch
import torch.nn as nn
import warnings
from timm.models.vision_transformer import Block

class PatchEmbed(nn.Module):
    def __init__(self, pre_weight):
        super().__init__()
        self.token = nn.Embedding(pre_weight.size(0), pre_weight.size(1))
    def forward(self, seq):
        x = self.token(seq)
        return x 
    
class MaskedAutoencoderViT(nn.Module):

    def __init__(self,vec=None, num_patches=515,
                 embed_dim=64, depth=24, num_heads=16,
                 mlp_ratio=4., embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6),city="Aalborg",mae_pretrain=True):
        super().__init__()

        num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        embbed_weight = torch.zeros((4,64),dtype=vec.dtype,device=vec.device)
        embbed_weight = torch.cat((embbed_weight,vec),dim=0)
        self.patch_embed = embed_layer(embbed_weight)

    def emb(self,x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        return x
    
    def forward_encoder(self, x):

        x = self.emb(x)
        xx = x.clone()
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x,xx

    def forward(self, intputs):
        
        return self.forward_encoder(intputs)


class OutPutLayer(nn.Module):
    def __init__(self,output):
        super(OutPutLayer, self).__init__()
        self.fc1 = nn.Linear(64, 2048)
        self.fc3 = nn.Linear(2048, output)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    
class encoderLayer(nn.Module):
    def __init__(self,pretrained_weights,num_tokens, embedding_size,pre_train,freeze,dropout,max_len):
        super(encoderLayer, self).__init__()

        self.embedding = nn.Embedding(num_tokens, embedding_size)
        if pretrained_weights is not None:
            if pre_train:
                self.embedding.weight.data.copy_(pretrained_weights)
        self.embedding.weight.data[4:].requires_grad = (not freeze)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_len)


    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding_size)
        x = self.pos_encoder( x)
        return x
    
class PathTransformer(nn.Module):
    def __init__(self,pretrained_weights, num_tokens, embedding_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len,pre_train=True,freeze=True,predict=False,mask_prob=0.4,city='Aalborg',mae_pretrain=True):
        super(PathTransformer, self).__init__()
        self.embedding_size = embedding_size
        
        self.encoder = MaskedAutoencoderViT( 
            vec = pretrained_weights,
            num_patches=134,
            embed_dim = 64,
            depth = 12,
            num_heads = nhead,
            city=city,
            mae_pretrain = mae_pretrain
        )

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = OutPutLayer(num_tokens + 4)
        self.predict = predict
        self.mask_prob =mask_prob

    def forward(self,path):

        if self.predict:
            emb,path = self.encoder(path)
            return  path,emb 
        
        else:
            src = path.clone()
            pad_token_id = 0
            mask_token_id = 1
            input_masked, targets, targets_masked = mask_tokens(src,mask_prob = self.mask_prob)
            src_key_padding_mask = (input_masked == pad_token_id)
            src_key_padding_mask = src_key_padding_mask.float().masked_fill(src_key_padding_mask, float('-inf'))
            tgt_key_padding_mask = (targets == pad_token_id)
            tgt_key_padding_mask = tgt_key_padding_mask.float().masked_fill(tgt_key_padding_mask, float('-inf'))

            memory,_ = self.encoder(input_masked)
            targets = self.encoder.emb(targets)
            tgt_mask = generate_square_subsequent_mask(targets.size(1)).to(targets.device)
            output = self.decoder(targets, memory,tgt_mask=tgt_mask,  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
            predict = self.output_layer(output)

            src_key_padding_mask = (path == 0)
            src_key_padding_mask = src_key_padding_mask.float().masked_fill(src_key_padding_mask, float('-inf'))
            emb,path = self.encoder(path)
            return predict, targets_masked, path, emb 
