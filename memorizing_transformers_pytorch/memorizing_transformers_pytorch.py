import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from memorizing_transformers_pytorch.knn_memory import KNNMemory

# constants

DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# helper functions

def exists(val):
    return val is not None

def unique(arr):
    return list({el: True for el in arr}.keys())

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def l2norm(t):
    return F.normalize(t, dim = -1)

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return F.softmax(t, dim = dim)

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

    def forward(self, i, j, *, device):
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

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, *, rel_pos_bias = None):
        h, device = self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(rel_pos_bias):
            sim = rel_pos_bias + sim

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)
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
        num_retrieved_memories = 32,
        intra_attn_values_gating = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)

        self.null_k = nn.Parameter(torch.randn(dim_head))
        self.null_v = nn.Parameter(torch.randn(dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.intra_attn_values_gating = intra_attn_values_gating

        if not intra_attn_values_gating:
            # this was the type of gating proposed in the paper
            self.combine_attn_output_gate = nn.Parameter(torch.randn(heads, 1, 1))
        else:
            # proposed in alphafold2, where each token gates the aggregated values
            # for memorizing transformers, this can be both the KNN memories and local values
            self.values_gating = nn.Sequential(
                nn.Linear(dim, inner_dim * 2, bias = False),
                Rearrange('b n (h d) -> b h n d', h = heads),
                nn.Sigmoid()
            )

    def forward(
        self,
        x,
        *,
        knn_memory = None,
        add_memory = True,
        rel_pos_bias = None
    ):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # calculate local attention

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = rel_pos_bias + sim

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        local_values = einsum('b h i j, b j d -> b h i d', attn, v)

        # calculate knn attention over memory, if index is passed in

        if exists(knn_memory):
            knn_queries = rearrange(q, 'b h n d -> b (h n) d')

            mem_kv, mem_mask = knn_memory.search(knn_queries, self.num_retrieved_memories)

            mem_mask = rearrange(mem_mask, 'b (h i) j -> b h i j', h = h)
            mem_k, mem_v = rearrange(mem_kv, 'b (h i) j kv d -> b h i j kv d', h = h).unbind(dim = -2)

            # use null key / value to protect against empty memory

            null_k, null_v = map(lambda t: repeat(t, 'd -> b h i 1 d', b = b, h = h, i = n), (self.null_k, self.null_v))

            mem_k = torch.cat((null_k, mem_k), dim = -2)
            mem_v = torch.cat((null_v, mem_v), dim = -2)
            mem_mask = F.pad(mem_mask, (1, 0), value = True)
            sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * self.scale

            sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)
            attn_mem = stable_softmax(sim_mem)
            attn_mem = self.dropout(attn_mem)

            mem_values = einsum('b h i j, b h i j d -> b h i d', attn_mem, mem_v)

            # do head-wise gating, as described in paper

            if self.intra_attn_values_gating:
                local_gate, mem_gate = self.values_gating(x).chunk(2, dim = -1)
                out = local_values * local_gate + mem_values * mem_gate
            else:
                gate = self.combine_attn_output_gate.sigmoid()
                out = local_values * gate + mem_values * (1 - gate)

            # add new memories to KNNMemory

            if add_memory:
                kv = torch.stack((k, v), dim = -2)
                knn_memory.add(l2norm(kv))  # in paper, they showed that normalizing the keys / values led to more stable training

        else:
            out = local_values

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
        ff_dropout = 0.,
        memorizing_layers = None,
        max_knn_memories = 2048,
        num_retrieved_memories = 32,
        clear_memories_on_sos_token_id = None,
        knn_memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        knn_use_gpu = False,
        pad_id = 0,
        intra_attn_values_gating = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pad_id = pad_id

        block_wrapper = partial(PreNormResidual, dim)

        memorizing_layers = default(memorizing_layers, (depth // 2,)) # default KNN attention layer to midpoint of transformer
        memorizing_layers = cast_tuple(memorizing_layers)

        self.dim_head = dim_head

        self.max_knn_memories = max_knn_memories
        self.knn_memories_directory = knn_memories_directory
        self.memorizing_layers = unique(memorizing_layers)
        self.num_memory_layers = len(memorizing_layers)
        self.knn_use_gpu = knn_use_gpu

        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)

        # layers

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            use_knn_attention = (idx + 1) in memorizing_layers

            if use_knn_attention:
                attn = KNNAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, num_retrieved_memories = num_retrieved_memories, intra_attn_values_gating = intra_attn_values_gating)
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))


        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def create_knn_memories(
        self,
        *,
        batch_size,
        knn_memories_directory = None
    ):
        knn_memories_directory = default(knn_memories_directory, self.knn_memories_directory)
        memories_dir = Path(knn_memories_directory)

        return [KNNMemory(dim = self.dim_head, max_memories = self.max_knn_memories, knn_use_gpu = self.knn_use_gpu, num_indices = batch_size, memmap_filename = str(memories_dir / f'knn.memory.layer.{ind + 1}.memmap')) for ind in range(self.num_memory_layers)]

    def forward(
        self,
        x,
        knn_memories = None,
        labels = None
    ):
        batch_size, seq_len, *_, device = *x.shape, x.device
        x = self.token_emb(x)

        # validate KNN memories to have enough indices for batch size

        if exists(knn_memories):
            assert all([memory.num_indices == batch_size for memory in knn_memories]), f'you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices'

        # if KNN memories are passed in, and researcher wants memories auto-cleared on <sos> token detection
        # do the appropriate logic

        if exists(knn_memories) and exists(self.clear_memories_on_sos_token_id):
            clear_memory = (x == self.clear_memories_on_sos_token_id).any(dim = -1)
            batch_indices, _ = clear_memory.nonzero(as_tuple = True)
            batch_indices_to_clear = batch_indices.tolist()

            if len(batch_indices_to_clear) > 0:
                for knn_memory in knn_memories:
                    knn_memory.clear(batch_indices_to_clear)

        # if KNN memories are not instantiated (on first pass), create fresh memories

        if not exists(knn_memories):
            knn_memories = self.create_knn_memories(batch_size)

        # iterate through the memories in order of the ascending layers that contain KNNAttention

        knn_memories_iter = iter(knn_memories)

        # positional bias

        rel_pos_bias = self.rel_pos_bias(seq_len, seq_len, device = device)

        # go through all layers

        for ind, (attn, ff) in enumerate(self.layers):
            layer_num = ind + 1
            attn_kwargs = dict(rel_pos_bias = rel_pos_bias)
            is_memorizing_layer = layer_num in self.memorizing_layers

            if is_memorizing_layer:
                attn_kwargs = {**attn_kwargs, 'knn_memory': next(knn_memories_iter)}

            x = attn(x, **attn_kwargs)
            x = ff(x)

        logits = self.to_logits(x)

        if not exists(labels):
            return logits, knn_memories

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)
        return loss, knn_memories
