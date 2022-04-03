import math
from functools import partial
from contextlib import contextmanager
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
        out = self.fn(self.norm(x), **kwargs)

        if not isinstance(out, tuple):
            return out + x

        head, *tail = out
        return (head + x, *tail)

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
        dropout = 0.,
        xl_max_memories = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, *, xl_memory = None, rel_pos_bias = None):
        h, device = self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
            k = torch.cat((k_xl_mem, k), dim = -2)
            v = torch.cat((v_xl_mem, v), dim = -2)

        sim = einsum('b h i d, b j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # new xl memories

        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        if self.xl_max_memories > 0:
            new_xl_kv_memories = new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_xl_kv_memories = None

        return self.to_out(out), new_xl_kv_memories

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
        intra_attn_values_gating = False,
        xl_max_memories = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

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
        knn_memory,
        xl_memory = None,
        add_knn_memory = True,
        rel_pos_bias = None
    ):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # handle xl memory

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
            k = torch.cat((k_xl_mem, k), dim = -2)
            v = torch.cat((v_xl_mem, v), dim = -2)

        # calculate local attention

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        mask_value = -torch.finfo(sim.dtype).max

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        local_values = einsum('b h i j, b j d -> b h i d', attn, v)

        # calculate knn attention over memory, if index is passed in

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)

        # use null key / value to protect against empty memory

        null_k, null_v = map(lambda t: repeat(t, 'd -> b h i 1 d', b = b, h = h, i = n), (self.null_k, self.null_v))

        mem_k = torch.cat((null_k, mem_k), dim = -2)
        mem_v = torch.cat((null_v, mem_v), dim = -2)
        mem_mask = F.pad(mem_mask, (1, 0), value = True)
        sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * self.scale

        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)
        attn_mem = stable_softmax(sim_mem)
        attn_mem = self.knn_mem_dropout(attn_mem)

        mem_values = einsum('b h i j, b h i j d -> b h i d', attn_mem, mem_v)

        # do head-wise gating, as described in paper

        if self.intra_attn_values_gating:
            local_gate, mem_gate = self.values_gating(x).chunk(2, dim = -1)
            out = local_values * local_gate + mem_values * mem_gate
        else:
            gate = self.combine_attn_output_gate.sigmoid()
            out = local_values * gate + mem_values * (1 - gate)

        # calculate new XL memories, as well as memories to be discarded

        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        if self.xl_max_memories > 0:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories[:, :-self.xl_max_memories], new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(l2norm(new_kv_memories_discarded))  # in paper, they showed that normalizing the keys / values led to more stable training

        # combine heads and project out

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), new_xl_kv_memories

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
        shift_knn_memories_down = 0.,
        pad_id = 0,
        intra_attn_values_gating = False,
        xl_max_memories = 0,
        xl_memory_layers = None,
        shift_xl_memories_down = 0.,
        memory_expiration_fn = None
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pad_id = pad_id

        block_wrapper = partial(PreNormResidual, dim)
        valid_layers = set(range(1, depth + 1))

        memorizing_layers = default(memorizing_layers, (depth // 2,)) # default KNN attention layer to midpoint of transformer
        memorizing_layers = cast_tuple(memorizing_layers)
        memorizing_layers = tuple(filter(lambda i: i in valid_layers, memorizing_layers))

        self.dim_head = dim_head

        # xl memory hyperparameter

        if xl_max_memories > 0:
            xl_memory_layers = default(xl_memory_layers, tuple(range(1, depth + 1)))
            xl_memory_layers = unique(xl_memory_layers)
            self.xl_memory_layers = tuple(filter(lambda i: i in valid_layers, xl_memory_layers))            
            self.num_xl_memory_layers = len(self.xl_memory_layers)
        else:
            self.xl_memory_layers = tuple()
            self.num_xl_memory_layers = 0

        # knn memory hyperparameters

        self.max_knn_memories = max_knn_memories
        self.knn_memories_directory = knn_memories_directory
        self.memorizing_layers = unique(memorizing_layers)
        self.num_memory_layers = len(memorizing_layers)

        self.knn_use_gpu = knn_use_gpu
        self.memory_expiration_fn = memory_expiration_fn
        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)

        # layers

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer_num = idx + 1

            use_xl_memories = layer_num in self.xl_memory_layers
            use_knn_attention = layer_num in memorizing_layers
            xl_max_memories_layer = 0 if not use_xl_memories else xl_max_memories

            if use_knn_attention:
                attn = KNNAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, num_retrieved_memories = num_retrieved_memories, intra_attn_values_gating = intra_attn_values_gating, xl_max_memories = xl_max_memories_layer)
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, xl_max_memories = xl_max_memories_layer)

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        # memory layer shifting
        # from a little known paper https://arxiv.org/abs/2012.15688

        self.shift_knn_memories_down = shift_knn_memories_down
        self.shift_xl_memories_down = shift_xl_memories_down

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

        return [KNNMemory(dim = self.dim_head, max_memories = self.max_knn_memories, knn_use_gpu = self.knn_use_gpu, num_indices = batch_size, expire_memory_fn = self.memory_expiration_fn, memmap_filename = str(memories_dir / f'knn.memory.layer.{ind + 1}.memmap')) for ind in range(self.num_memory_layers)]

    @contextmanager
    def knn_memories_context(
        self,
        **kwargs
    ):
        knn_memories = self.create_knn_memories(**kwargs)
        yield knn_memories
        for memory in knn_memories:
            del memory

    def forward(
        self,
        x,
        knn_memories,
        xl_memories = None,
        labels = None,
        add_knn_memory = True
    ):
        batch_size, seq_len, *_, device = *x.shape, x.device
        x = self.token_emb(x)

        # validate KNN memories to have enough indices for batch size

        assert all([memory.num_indices == batch_size for memory in knn_memories]), f'you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices'

        # if KNN memories are passed in, and researcher wants memories auto-cleared on <sos> token detection
        # do the appropriate logic

        if exists(self.clear_memories_on_sos_token_id):
            clear_memory = (x == self.clear_memories_on_sos_token_id).any(dim = -1)
            batch_indices, _ = clear_memory.nonzero(as_tuple = True)
            batch_indices_to_clear = batch_indices.tolist()

            if len(batch_indices_to_clear) > 0:
                for knn_memory in knn_memories:
                    knn_memory.clear(batch_indices_to_clear)

        # handle XL memories

        xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
        assert len(xl_memories) == self.num_xl_memory_layers
        has_xl_memories = len(xl_memories) > 0

        # shifting memories a number of layers down, little known technique shown to enhance memories from Ernie-Doc paper

        if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
            knn_memories = [*knn_memories[self.shift_knn_memories_down:], *knn_memories[:self.shift_knn_memories_down]]

        if len(xl_memories) > 0 and self.shift_xl_memories_down > 0:
            xl_memories = [*xl_memories[self.shift_xl_memories_down:], *xl_memories[:self.shift_xl_memories_down]]

        # iterate through the memories in order of the ascending layers that contain KNNAttention

        xl_memories_iter = iter(xl_memories)
        knn_memories_iter = iter(knn_memories)

        # positional bias

        max_context_len = max([seq_len, *map(lambda t: (t.shape[-3] if exists(t) else 0) + seq_len, xl_memories)])
        rel_pos_bias = self.rel_pos_bias(seq_len, max_context_len, device = device)

        # keep track of new xl memories

        new_xl_memories = [] if has_xl_memories else None

        # go through all layers

        for ind, (attn, ff) in enumerate(self.layers):
            layer_num = ind + 1

            is_memorizing_layer = layer_num in self.memorizing_layers
            is_xl_memory_layer = layer_num in self.xl_memory_layers

            attn_kwargs = dict(rel_pos_bias = rel_pos_bias)

            if is_memorizing_layer:
                attn_kwargs = {**attn_kwargs, 'knn_memory': next(knn_memories_iter), 'add_knn_memory': add_knn_memory}

            if is_xl_memory_layer:
                attn_kwargs = {**attn_kwargs, 'xl_memory': next(xl_memories_iter)}

            # attention

            x, xl_mem = attn(x, **attn_kwargs)

            # add new XL memories if needed

            if exists(xl_mem):
                new_xl_memories.append(xl_mem)

            # feedforward

            x = ff(x)

        # to logits

        logits = self.to_logits(x)

        if not exists(labels):
            if exists(new_xl_memories):
                return logits, new_xl_memories

            return logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)

        if exists(new_xl_memories):
            return loss, new_xl_memories

        return loss
