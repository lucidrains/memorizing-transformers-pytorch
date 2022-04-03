import os
import math
import torch
import faiss
import numpy as np

from einops import rearrange
from memorizing_transformers_pytorch.utils import rearrange_with_dim_list

# constants

FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def check_shape(t, pattern, **kwargs):
    return rearrange(t, f"{pattern} -> {pattern}", **kwargs)

def count_intersect(x, y):
    # returns an array that shows how many times an element in x is contained in tensor y
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis = -1)

# logic for expiring memories
# function takes in list of ids, which ranges from newest to oldest
# and returns the list of ids that should be removed

def expire_strategy_remove_oldest(
    ids,
    *,
    max_num_entries,
    num_hits,
    ages,
    ages_since_last_hit
):
    ids_to_remove = ids[max_num_entries:]
    return ids_to_remove

# a wrapper around faiss IndexIVFFlat
# taking care of expiring old keys automagically

class KNN():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        use_gpu = False,
        expire_memory_fn = expire_strategy_remove_oldest
    ):
        expire_memory_fn = default(expire_memory_fn, expire_strategy_remove_oldest)
        assert callable(expire_memory_fn)

        nlist = math.floor(math.sqrt(max_num_entries))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        self.use_gpu = use_gpu

        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), FAISS_INDEX_GPU_ID, index) if use_gpu else index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.expire_memory_fn = expire_memory_fn

        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.ids = np.empty((0,), dtype = np.int32)
        self.hits = np.empty((0,), dtype = np.int32)
        self.age_num_iterations = np.empty((0,), dtype = np.int32)
        self.ages_since_last_hit = np.empty((0,), dtype = np.int32)
        return self.index.reset()

    def train(self, x):
        return self.index.train(x)

    def add(self, x, ids):
        if not self.index.is_trained:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))
        self.hits = np.concatenate((np.zeros_like(ids), self.hits))
        self.age_num_iterations = np.concatenate((np.zeros_like(ids), self.age_num_iterations))
        self.ages_since_last_hit = np.concatenate((np.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:

            remove_ids = self.expire_memory_fn(
                self.ids,
                max_num_entries = self.max_num_entries,
                num_hits = self.hits,
                ages = self.age_num_iterations,
                ages_since_last_hit = self.ages_since_last_hit
            )

            keep_mask = count_intersect(self.ids, remove_ids) == 0

            self.ids = self.ids[keep_mask]
            self.hits = self.hits[keep_mask]
            self.age_num_iterations = self.age_num_iterations[keep_mask]
            self.ages_since_last_hit = self.ages_since_last_hit[keep_mask]

            assert len(self.ids) <= self.max_num_entries
            self.remove(remove_ids)

        return self.index.add_with_ids(x, ids = ids)

    def remove(self, ids):
        self.index.remove_ids(ids)

    def search(
        self,
        x,
        topk,
        nprobe = 8,
        return_distances = False,
        increment_hits = False,
        increment_age = True
    ):
        if not self.index.is_trained:
            return np.full((x.shape[0], topk), -1)

        self.index.nprobe = nprobe
        distances, indices = self.index.search(x, k = topk)

        if increment_hits:
            hits = count_intersect(self.ids, rearrange(indices, '... -> (...)'))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices

# KNN memory layer, where one can store key / value memories
# can automatically take care of a collection of faiss indices (across batch dimension)

class KNNMemory():
    def __init__(
        self,
        dim,
        max_memories = 16000,
        num_indices = 1,
        memmap_filename = './knn.memory.memmap',
        knn_use_gpu = False,
        expire_memory_fn = expire_strategy_remove_oldest
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.knns = [KNN(dim = dim, max_num_entries = max_memories, use_gpu = knn_use_gpu, cap_num_entries = True, expire_memory_fn = expire_memory_fn) for _ in range(num_indices)]

    def clear(self, indices = None):
        if not exists(indices):
            indices = list(range(self.num_indices))

        indices = cast_list(indices)

        for index in indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[indices] = 0

    def add(self, memories):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = self.num_indices)

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        knn_insert_ids = np.arange(num_memories)
        keys = np.ascontiguousarray(memories[..., 0, :])

        for key, db_offset, knn in zip(keys, self.db_offsets, self.knns):
            knn.add(key, ids = knn_insert_ids + db_offset)

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets, 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.arange(self.num_indices), 'i -> i 1'), add_indices] = memories
        self.db.flush()

        self.db_offsets += num_memories

    def search(
        self,
        queries,
        topk,
        nprobe = 8,
        increment_hits = True,
        increment_age = True
    ):
        _, *prec_dims, _ = queries.shape
        check_shape(queries, 'b ... d', d = self.dim, b = self.num_indices)
        queries = rearrange(queries, 'b ... d -> b (...) d')

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        for ind, (query, knn) in enumerate(zip(queries, self.knns)):
            indices = knn.search(query, topk, nprobe, increment_hits = increment_hits, increment_age = increment_age)
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[ind, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values = rearrange_with_dim_list(all_key_values, 'b (...p) ... -> b ...p ...', p = prec_dims)
        all_masks = rearrange_with_dim_list(all_masks, 'b (...p) ... -> b ...p ...', p = prec_dims)

        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db
