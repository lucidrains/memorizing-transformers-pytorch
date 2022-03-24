import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

# t5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias * self.scale

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        sim = self.rel_pos_bias(sim) + sim

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# approximate nearest neighbor attention

class KNNAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        index_search_topk = 32
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.index_search_topk = index_search_topk
        self.combine_attn_output_gate = nn.Parameter(torch.randn(heads, 1, 1))

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5)
        self.dropout = nn.Dropout(dropout)

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, *, index):
        h = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate local attention

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        sim = self.rel_pos_bias(sim) + sim

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max

        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        local_values = einsum('b h i j, b h j d -> b h i d', attn, v)

        # calculate knn attention over memory

        k_mem, v_mem, mem_mask = index.search(q, k = self.index_search_topk)

        # use null key / value to protect against empty memory

        null_k, null_v = map(lambda t: rearrange(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (self.null_k, self.null_v))

        k_mem = torch.cat((null_k, k_mem), dim = -2)
        v_mem = torch.cat((null_v, v_mem), dim = -2)
        mem_mask = F.pad(mem_mask, (1, 0), value = True)

        sim_mem = einsum('b h i d, b h j d -> b h i j', q, mem_k) * self.scale
        sim_mem = sim_mem - sim_mem.amax(dim = -1, keepdim = True).detach()

        sim = sim.masked_fill(~mem_mask, mask_value)
        attn_mem = sim_mem.softmax(dim = -1)
        attn_mem = self.dropout(attn_mem)

        mem_values = einsum('b h i j, b h j d -> b h i d', attn_mem, v_mem)

        # do head-wise gating, as described in paper

        gate = self.combine_attn_output_gate.sigmoid()
        out = local_values * gate + mem_values * (1 - gate)

        # combine heads and project out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class MemorizingTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        block_wrapper = partial(PreNormResidual, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                block_wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                block_wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return self.to_logits(x)
