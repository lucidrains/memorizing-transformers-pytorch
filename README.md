<img src="./diagram.png" width="500px"></img>

## Memorizing Transformers - Pytorch

Implementation of <a href="https://arxiv.org/abs/2203.08913">Memorizing Transformers</a> (ICLR 2022), attention net augmented with indexing and retrieval of memories using approximate nearest neighbors, in Pytorch

## Install

```bash
$ pip install memorizing-transformers-pytorch
```

## Usage

```python
import torch
from memorizing_transformers_pytorch import MemorizingTransformer

model = MemorizingTransformer(
    num_tokens = 20000,                 # number of tokens
    dim = 512,                          # dimension
    dim_head = 64,                      # dimension per attention head
    depth = 8,                          # number of layers
    memorizing_layers = (4, 5),         # which layers to have ANN memories
    max_knn_memories = 2048,            # maximum ANN memories to keep (oldest ones will be discarded)
    num_retrieved_memories = 32,        # number of ANN memories to retrieve
    clear_memories_on_sos_token_id = 1, # clear passed in ANN memories automatically for batch indices which contain this specified SOS token id - otherwise, you can also manually iterate through the ANN memories and clear the indices before the next iteration
)

data = torch.randint(0, 20000, (2, 1024)) # mock data

knn_memories = model.create_knn_memories(batch_size = 2) # create collection of KNN memories with the correct batch size (2 in example)

logits, _ = model(data, knn_memories = knn_memories) # (1, 1024, 20000)
# ... and so on
```

## KNN Memory

This repository contains a wrapper around Faiss that can automatically store and retrieve key / values

```python
import torch
from memorizing_transformers_pytorch import KNNMemory

memory = KNNMemory(
    dim = 64,                   # dimension of key / values
    max_memories = 1024,        # maximum number of memories to keep (will throw out the oldest memories for now if it overfills)
    num_indices = 2             # this should be equivalent to batch dimension, as each batch keeps track of its own memories, expiring when it sees a new document
)

memory.add(torch.randn(2, 512, 2, 64))  # (batch, seq, key | value, feature dim)
memory.add(torch.randn(2, 512, 2, 64))

memory.clear([0]) # clear batch 0, if it saw an <sos>

memory.add(torch.randn(2, 512, 2, 64))
memory.add(torch.randn(2, 512, 2, 64))

key_values, mask = memory.search(torch.randn(2, 512, 64), topk = 32)
```

## Todo

- [x] take care of cross entropy loss if labels passed in
- [ ] write alternative gating that takes into account number of retrieved memories as well as positions using continuous MLP representation
- [ ] complete transformer-xl with appropriate memory storing and retrieval strategies + https://github.com/lucidrains/x-transformers#enhanced-recurrence
- [ ] enwik8 demo

## Citations

```bibtex
@article{wu2022memorizing,
  title   = {Memorizing transformers},
  author  = {Wu, Yuhuai and Rabe, Markus N and Hutchins, DeLesley and Szegedy, Christian},
  journal = {arXiv preprint arXiv:2203.08913},
  year    = {2022}
}
```

```bibtex
@article{Shazeer2019FastTD,
  title   = {Fast Transformer Decoding: One Write-Head is All You Need},
  author  = {Noam M. Shazeer},
  journal = {ArXiv},
  year    = {2019},
  volume  = {abs/1911.02150}
}
```
