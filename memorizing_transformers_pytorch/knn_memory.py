import math
import torch
import faiss
import numpy as np
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def cast_list(val):
    return val if isinstance(val, list) else [val]

def check_shape(t, pattern, **kwargs):
    return rearrange(t, f"{pattern} -> {pattern}", **kwargs)

# a wrapper around faiss IndexIVFFlat
# taking care of expiring old keys automagically

class KNN():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        use_gpu = False
    ):
        nlist = math.floor(math.sqrt(max_num_entries))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        self.use_gpu = use_gpu
        self.index = index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries

        self.reset()

    def cleanup(self):
        del self.index

    def reset(self):
        if self.cap_num_entries:
            self.ids = np.empty((0,), dtype = np.int32)

        return self.index.reset()

    def train(self, x):
        return self.index.train(x)

    def add(self, x, ids):
        if not self.index.is_trained:
            self.train(x)

        if self.cap_num_entries:
            self.ids = np.concatenate((ids, self.ids))
            remove_ids = self.ids[self.max_num_entries:]
            self.remove(remove_ids)

        return self.index.add_with_ids(x, ids = ids)

    def remove(self, ids):
        self.index.remove_ids(ids)

    def search(self, x, topk, nprobe = 8, return_distances = False):
        if not self.index.is_trained:
            return np.full((x.shape[0], topk), -1)

        search_index = faiss.index_cpu_to_all_gpus(self.index) if self.use_gpu else self.index

        search_index.nprobe = nprobe
        distances, indices = search_index.search(x, k = topk)

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
        knn_use_gpu = False
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.knns = [KNN(dim = dim, max_num_entries = max_memories, use_gpu = knn_use_gpu, cap_num_entries = True) for _ in range(num_indices)]

    def clear(self, indices = None):
        if not exists(indices):
            indices = list(range(self.num_indices))

        indices = cast_list(indices)

        for index in indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[indices] = 0

    def add(self, memories):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2)

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

    def search(self, queries, topk, nprobe = 8):
        check_shape(queries, 'b n d', d = self.dim)

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        for ind, (query, knn) in enumerate(zip(queries, self.knns)):
            indices = knn.search(query, topk, nprobe)
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[ind, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        return all_key_values.to(device), all_masks.to(device)

    def cleanup(self):
        for knn in self.knns:
            knn.cleanup()
        del self.db
