import torch
import torch.nn as nn

class SparseKGAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., topk_ratio=0.5):
        super(SparseKGAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.topk_ratio = topk_ratio
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, context_feats, context_mask=None):
        B, N, C = x.shape
        # [修改] B_kg -> B_ctx
        B_ctx, N_ctx, C_ctx = context_feats.shape
        
        # [修改] kg_feats -> context_feats
        y = self.layer_norm(context_feats)
        
        # Q: [B, Heads, N_img, C_head]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # K, V: [B, Heads, N_kg, C_head]
        kv = self.kv(y).reshape(B, N_ctx, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Top-K Sparse Mask
        k_num = int(N_ctx * self.topk_ratio)
        if k_num < 1: k_num = 1
        
        topk_val, topk_idx = torch.topk(attn, k=k_num, dim=-1, largest=True)
        
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        
        if context_mask is not None:
            mask = mask & (context_mask[:, None, None, :] > 0)

        attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return x + out