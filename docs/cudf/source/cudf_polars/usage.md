(cudf-polars-usage)=
# Usage

`cudf-polars` runs your Polars `LazyFrame` queries on GPU. You select GPU execution by passing
an `engine=` argument to `.collect()` or `.sink_*()`. See {doc}`engines` for the conceptual
picture, this page walks through running your first query.

We always recommend constructing an engine object and using it in a context manager to ensure proper
resource cleanup. The engine constructor is where you specify {class}`~cudf_polars.engine.options.StreamingOptions`
such as `spill_to_pinned_memory` or `fallback_mode`. Ray is the showcased example below, see also
{doc}`other_engines`.

## Your first GPU query

```python
import polars as pl
from cudf_polars.engine.ray import RayEngine

query = (
    pl.scan_parquet("/data/dataset/*.parquet")
    .filter(pl.col("amount") > 100)
    .group_by("customer_id")
    .agg(pl.col("amount").sum())
)

with RayEngine() as engine:
    result = query.collect(engine=engine)
print(result)
```

{class}`~cudf_polars.engine.ray.RayEngine` with no arguments uses every
GPU visible to the process, so the example above runs on one GPU if that's all that's available
and scales automatically to every GPU on the node otherwise. It also attaches to an existing
Ray cluster if one is already running (see [Attaching to an existing Ray cluster](#attaching-to-an-existing-ray-cluster)).

```{note}
The examples on this page use {class}`~cudf_polars.engine.ray.RayEngine`. `cudf-polars` supports
multiple engines for GPU execution. See {doc}`other_engines` for alternatives, or {doc}`engines` for a conceptual overview of when to pick which.
```

```{note}
`.collect()` pulls the full result back to the client process. For large distributed outputs,
prefer `.sink_*()` or aggregate/sample inside the query before `.collect()`. See
[Result collection](engines.md#result-collection).
```

## Configuring `RayEngine`

{class}`~cudf_polars.engine.ray.RayEngine` with no arguments starts a
local [Ray][ray-docs] cluster and creates one GPU worker per visible GPU.

For custom configuration, build a
{class}`~cudf_polars.engine.options.StreamingOptions` and use
`RayEngine.from_options()`:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with RayEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/dataset/*.parquet").collect(engine=engine)
```

See {doc}`options` for the available fields.

```{note}
`RayEngine` is an object you create and pass to `.collect(engine=engine)`. Prefer the
context-manager form so the Ray cluster and GPU workers are torn down automatically.
```

The same `from_options()` / `StreamingOptions` pattern shown here works for every streaming
engine. See {doc}`other_engines` for the DaskEngine and SPMDEngine variants.

## Attaching to an existing Ray cluster

For multi-node runs, start a Ray cluster separately (for example with `ray start` on each
node) and attach to it from your client script. When Ray is already initialized,
{class}`~cudf_polars.engine.ray.RayEngine` connects to the running
cluster and leaves it untouched on exit:

```python
import ray
import polars as pl
from cudf_polars.engine.ray import RayEngine

ray.init(address="auto")  # attach to a running cluster
with RayEngine() as engine:
    result = (
        pl.scan_parquet("s3://bucket/*.parquet")
            .group_by("customer_id")
            .agg(pl.col("amount").sum())
            .collect(engine=engine)
    )
```

{class}`~cudf_polars.engine.ray.RayEngine` creates one rank per GPU in the Ray cluster.
It raises `RuntimeError` if no GPUs are available.

## Manual Engine Lifetime Control

When you need to control the engine lifetime explicitly, for example in a Jupyter notebook
where a `with` block cannot span multiple cells, construct a `RayEngine` once and reuse it,
then call `engine.shutdown()` when you are done:

```python
# Cell 1: start the engine
from cudf_polars.engine.ray import RayEngine

engine = RayEngine()
```

```python
# Cell 2: run a query
import polars as pl

result = (
    pl.scan_parquet("/data/*.parquet")
      .group_by("customer_id")
      .agg(pl.col("amount").sum())
      .collect(engine=engine)
)
result
```

```python
# Cell 3: run another query reusing the same engine
other = pl.scan_parquet("/data/other/*.parquet").collect(engine=engine)
```

```python
# Final cell: tear everything down
engine.shutdown()
```

`engine.shutdown()` stops the GPU worker processes (rank actors) and, if the engine started Ray itself,
also calls `ray.shutdown()`. It is idempotent, so calling it twice is safe.

## Sink behavior

When a streaming engine is used, sink operations such as `df.sink_parquet("my_path")` always produce
a directory containing one or more files. It is not currently possible to disable this behavior, and
setting `sink_to_directory=False` raises a `ValueError`.

The in-memory engine, by contrast, follows standard Polars semantics and writes to a single file at
the specified path.


## Cluster diagnostics

{meth}`~cudf_polars.engine.ray.RayEngine.gather_cluster_info` returns
a list of {class}`~cudf_polars.engine.core.ClusterInfo`, one per rank
actor, with fields `hostname`, `pid`, `cuda_visible_devices`, and `gpu_uuid`:

```python
with RayEngine() as engine:
    print(f"cluster has {engine.nranks} ranks")
    for i, info in enumerate(engine.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info.hostname}, pid={info.pid}, "
            f"cuda_visible_devices={info.cuda_visible_devices}, "
            f"gpu_uuid={info.gpu_uuid}"
        )
# rank 0: hostname=node-0, pid=12345, cuda_visible_devices=0, gpu_uuid=GPU-abc123...
# rank 1: hostname=node-0, pid=12346, cuda_visible_devices=1, gpu_uuid=GPU-def456...
```

[ray-docs]: https://docs.ray.io/en/latest/
