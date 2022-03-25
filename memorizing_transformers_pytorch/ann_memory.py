import math
import torch
import faiss
import numpy as np

# helper functions

def exists(val):
    return val is not None

def cast_list(val):
    return val if isinstance(val, list) else [val]

# a wrapper around faiss IndexFlatL2
# taking care of expiring old keys automagically

class ANN():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        use_gpu = False
    ):
        nlist = math.floor(math.sqrt(max_num_entries))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        if use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)

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

    def add(self, x, ids = None):
        if not self.index.is_trained:
            self.train(x)

        if self.cap_num_entries:
            self.ids = np.concatenate((ids, self.ids))
            remove_ids = self.ids[self.max_num_entries:]
            self.remove(remove_ids)

        return self.index.add_with_ids(x, ids = ids)

    def remove(self, ids):
        self.index.remove_ids(ids)

    def search(self, x, topk, nprobe = 8):
        self.index.nprobe = nprobe
        return self.index.search(x, k = topk)

# ANN memory layer, where one can store key / value memories
# can automatically take care of a collection of faiss indices (across batch dimension)

class ANNMemory():
    def __init__(
        self,
        dim,
        max_num_entries = 16000,
        num_indices = 1,
        memmap_filename = './ann.memory.memmap',
        ann_use_gpu = False
    ):
        self.num_indices = num_indices
        self.max_num_entries = max_num_entries
        self.shape = (num_indices, max_num_entries, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.anns = [ANN(dim = dim, max_num_entries = max_num_entries, use_gpu = ann_use_gpu, cap_num_entries = True) for _ in range(num_indices)]

    def clear(self, indices = None):
        if not exists(indices):
            indices = list(range(self.num_indices))

        indices = cast_list(indices)

        for index in indices:
            ann = self.anns[index]
            ann.reset()

        self.db_offsets[indices] = 0

    def add(self, memories):
        memories = memories.numpy()
        memories = memories[:, -self.max_num_entries:]
        num_memories = memories.shape[1]

        ann_insert_ids = np.arange(num_memories)
        keys = np.ascontiguousarray(memories[..., 0, :])

        for key, db_offset, ann in zip(keys, self.db_offsets, self.anns):
            ann.add(key, ids = ann_insert_ids + db_offset)

        add_indices = (np.arange(num_memories)[None, :] + self.db_offsets[:, None]) % self.max_num_entries
        self.db[np.arange(self.num_indices)[:, None], add_indices] = memories
        self.db.flush()

        self.db_offsets += num_memories

    def search(self, queries, topk, nprobe = 8):
        device = queries.device
        queries = queries.cpu().numpy()

        all_masks = []
        all_key_values = []

        for ind, (query, ann) in enumerate(zip(queries, self.anns)):
            dist, indices = ann.search(query, topk, nprobe)
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[ind, db_indices % self.max_num_entries]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~all_masks[..., None, None], 0.)

        return all_key_values.to(device), all_masks.to(device)

    def cleanup(self):
        for ann in self.anns:
            ann.cleanup()
        del self.db
