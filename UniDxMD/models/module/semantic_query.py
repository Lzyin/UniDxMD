import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        # assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)


        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale #
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)                                                                                               
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))

        return x




class Semantic_query(nn.Module):
    def __init__(self,
                 transformer_width=128,  
                 transformer_heads=4, 
                 transformer_layers=6,
                 input_dim=128, 
                 dropout=0,
                 ):
        super().__init__()


        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02) # 
            if isinstance(m, nn.Linear) and m.bias is not None: # 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # 
            nn.init.constant_(m.weight, 1.0)

 

    def forward(self, embedding, prototype_2d):

        semantic_embedding = embedding.weight.detach()
        prototype_2d = prototype_2d.detach()

        # [128, 128] x [6, 128].T -> [128, 6]
        logits = torch.matmul(semantic_embedding, prototype_2d.transpose(0, 1))
        weights = F.softmax(logits, dim=-1)

        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1).mean()
        entropy_reg_coef = 0.001
        entropy_loss = - entropy_reg_coef * entropy    

        semantic_weight = torch.matmul(weights, prototype_2d)
        return semantic_weight, entropy_loss

