(cudf-polars-pytorch)=
# PyTorch

{class}`~cudf_polars.engine.spmd.SPMDEngine` can run inside a `torch.distributed` job and
hand its results to PyTorch without leaving the GPU. A single `torchrun` job can do
distributed ETL with Polars on the GPUs that will train the model, and the training loop
reads the result as `torch.Tensor` objects that share the query's GPU memory.

This is **experimental** and may change without notice. It builds on the {doc}`spmd_engine`, which
covers how ranks are launched, how scans are partitioned across ranks, and the query symmetry
requirement.

## Quickstart

```python
# Launch: torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) script.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import polars as pl

from cudf_polars.engine.spmd import SPMDEngine, use_gpu
from cudf_polars.engine.torch_interop import persisted_to_torch

# Give this rank its GPU before any CUDA call, including NCCL init.
use_gpu(int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(0)
dist.init_process_group(backend="nccl")

with SPMDEngine.from_torch_distributed() as engine:
    # Distributed GPU ETL. The result stays on the GPU.
    result = engine.execute(
        pl.scan_parquet("s3://bucket/data/*.parquet")
        .filter(pl.col("label").is_not_null())
        .select(pl.col("feat").cast(pl.Float32), pl.col("label").cast(pl.Float32))
    )
    # Zero-copy views of that GPU memory. Clone so the engine's memory can be released.
    tensors = persisted_to_torch(result, engine=engine, ensure_sharded=True)
    feat, label = tensors["feat"].clone(), tensors["label"].clone()
    del tensors

# Train as usual. Every rank holds its own rows, and DDP averages the gradients.
model = DDP(torch.nn.Linear(1, 1).cuda(), device_ids=[0])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = feat.unsqueeze(1), label.unsqueeze(1)
for _ in range(100):
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

dist.destroy_process_group()
```

{meth}`~cudf_polars.engine.spmd.SPMDEngine.from_torch_distributed` uses the torch process
group to bootstrap the engine's communicator. It takes an optional
{class}`~cudf_polars.engine.options.StreamingOptions`, read the same way as in
{meth}`~cudf_polars.engine.spmd.SPMDEngine.from_options`.

{func}`~cudf_polars.engine.torch_interop.persisted_to_torch` views each column of this rank's
result as a tensor. Columns must be numeric or boolean and contain no nulls. It consumes the
result, so the result cannot also be collected.

A result is either sharded, each rank holding different rows, or replicated, every rank
holding the same full copy. Which one a query produces depends on how the engine partitioned
it. Replicated is right for values every rank needs, such as normalization constants, so
leave `ensure_sharded` off for those. For training data, `ensure_sharded=True` cuts a
replicated result down to this rank's share, split the way `torch.chunk` would split it, and
returns a sharded result unchanged.

## Selecting a GPU

The usual `torchrun` idiom leaves every GPU visible and selects one with
`torch.cuda.set_device(LOCAL_RANK)`. That does not work here, because the engine runs on
CUDA device ordinal 0 (see [Selecting a GPU](spmd_engine.md#selecting-a-gpu)).
{func}`~cudf_polars.engine.spmd.use_gpu` restricts the process to its GPU instead, as in the
quickstart. Call it before any CUDA call, including `dist.init_process_group(backend="nccl")`.
Afterwards the process sees exactly one GPU, numbered 0, so use `device="cuda:0"` for tensors
and `device_ids=[0]` for `DistributedDataParallel`.

## Things to know

- Launch with `torchrun`, not `rrun`. A job launched with `rrun` should use plain
  {class}`~cudf_polars.engine.spmd.SPMDEngine` instead.
- All ranks must issue the same queries in the same order, as for any
  {class}`~cudf_polars.engine.spmd.SPMDEngine` use.
- Each rank's share of the result must fit in its GPU memory, see
  {ref}`cudf-polars-pytorch-larger-results`.
- The tensors returned by `persisted_to_torch` keep the engine's GPU memory alive.
  `.clone()` the ones you keep and drop the views.
- `engine.rank` need not equal `dist.get_rank()`. Each rank reads its own data either way,
  but do not use one numbering to index something keyed by the other.
- cuDF and torch manage their GPU memory separately. Torch can allocate from RMM too, which
  changes its global allocator:

  ```python
  import rmm.allocators.torch

  torch.cuda.memory.change_current_allocator(rmm.allocators.torch.rmm_torch_allocator)
  ```

## Use cases

The snippets below run inside the `with` block from the quickstart. `events`, `users` and
`catalog` stand for `pl.scan_parquet(...)` inputs, and names such as `train` and
`send_for_labeling` stand for application code.

### Collapse two-cluster ETL-to-training pipelines into one

A common production layout has a CPU cluster build training tables, write them to object
storage as Parquet, and a GPU cluster load and train on them. With cudf-polars and PyTorch in
one process group, a single `torchrun` job does the ETL on the GPUs that will train the model.

```python
features = (
    events.group_by("user_id")
    .agg(
        pl.col("amount").sum().alias("spend"),
        pl.len().alias("n_events"),
        pl.col("label").max(),
    )
    .select(pl.col("spend", "n_events", "label").cast(pl.Float32))
)
tensors = persisted_to_torch(engine.execute(features), engine=engine, ensure_sharded=True)
```

### Per-epoch reshuffle and feature recompute

Tabular deep learning often wants a fresh global shuffle each epoch, and sometimes wants to
recompute features such as target encodings between epochs. On the training GPUs this is
cheap enough to do every epoch.

```python
for epoch in range(num_epochs):
    # Target encoding, recomputed from the current events.
    merchant_rate = events.group_by("merchant_id").agg(
        pl.col("label").mean().alias("merchant_rate")
    )
    epoch_query = (
        events.join(merchant_rate, on="merchant_id")
        # A fresh global shuffle: sort on a hash seeded by the epoch.
        .sort(pl.col("event_id").hash(seed=epoch))
        .select(pl.col("amount", "merchant_rate", "label").cast(pl.Float32))
    )
    batch = persisted_to_torch(engine.execute(epoch_query), engine=engine, ensure_sharded=True)
    train_one_epoch(model, batch)
```

### Big distributed joins for training data prep

Joining users, events and a catalog for a recommender often does not fit on one GPU. The
engine runs the join across the ranks, and the result lands on the same ranks as the model.

```python
train_table = (
    events.join(users, on="user_id")
    .join(catalog, on="item_id")
    .select(pl.col("age", "price", "label").cast(pl.Float32))
)
tensors = persisted_to_torch(engine.execute(train_table), engine=engine, ensure_sharded=True)
```

(cudf-polars-pytorch-larger-results)=
### Results larger than GPU memory

The query itself can be larger than GPU memory, but the result handed to torch has to fit.
For a larger result, split the input into batches whose results fit and run one query per
batch. Hashing a key works for any data. A filter that skips whole files or row groups, such
as one on a date column, avoids reading all of the input for every batch.

```python
nbatches = 16
for i in range(nbatches):
    batch = engine.execute(
        events.filter(pl.col("event_id").hash(seed=0) % nbatches == i)
        .select(pl.col("amount", "label").cast(pl.Float32))
    )
    tensors = persisted_to_torch(batch, engine=engine, ensure_sharded=True)
    amount, label = tensors["amount"].clone(), tensors["label"].clone()
    del tensors
    train(model, amount, label)
```

### Active learning

An active learning loop trains, scores the whole corpus, picks the uncertain rows, relabels
them and retrains. The whole loop becomes a plain Python `for` loop in one script.

```python
for _ in range(num_rounds):
    train(model, labeled_tensors())
    # Score this rank's share of the unlabeled pool.
    pool = persisted_to_torch(
        engine.execute(unlabeled.select(pl.col("id"), pl.col("feat").cast(pl.Float32))),
        engine=engine,
        ensure_sharded=True,
    )
    with torch.no_grad():
        p = model(pool["feat"].unsqueeze(1)).sigmoid().squeeze(1)
    # `scores` is this rank's share. The engine sorts across all ranks, so every rank
    # gets the same, globally most uncertain rows.
    scores = pl.LazyFrame({
        "id": pool["id"].cpu().numpy(),
        "margin": (p - 0.5).abs().cpu().numpy(),
    })
    picked = scores.sort("margin").head(1000).collect(engine=engine)
    if dist.get_rank() == 0:
        send_for_labeling(picked["id"])
```

### Last-mile transforms

Transforms right before the forward pass, such as normalization or frequency capping, can run
in the query.

```python
query = events.select(
    # Normalize with the global mean and standard deviation.
    ((pl.col("amount") - pl.col("amount").mean()) / pl.col("amount").std()).cast(pl.Float32),
    # Frequency cap.
    pl.col("clicks").clip(upper_bound=100).cast(pl.Float32),
    pl.col("label").cast(pl.Float32),
)
tensors = persisted_to_torch(engine.execute(query), engine=engine, ensure_sharded=True)
```

### Tabular features next to torch models

Tabular features from cudf-polars and embedding lookups in torch can serve the same batch on
the same ranks.

```python
rows = persisted_to_torch(
    engine.execute(
        events.select(pl.col("item_id").cast(pl.Int64), pl.col("amount", "label").cast(pl.Float32))
    ),
    engine=engine,
    ensure_sharded=True,
)
item_embedding = torch.nn.Embedding(num_items, 64).cuda()
x = torch.cat([item_embedding(rows["item_id"]), rows["amount"].unsqueeze(1)], dim=1)
```
