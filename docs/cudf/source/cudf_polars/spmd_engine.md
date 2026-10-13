(cudf-polars-spmd-engine)=
# SPMD

{class}`~cudf_polars.engine.spmd.SPMDEngine` runs the streaming executor
in [SPMD][spmd-wiki] mode: the same Python script runs once per GPU, and each process owns its
local data fragment. Collective operations (shuffles, allgathers, joins) coordinate across
processes to produce a globally consistent result.

On startup, `SPMDEngine` pins the process to the CPU cores and NUMA node closest to its GPU.
Under `rrun` this binding is delegated to the launcher; outside `rrun` (single-process mode)
`SPMDEngine` performs it itself. See
{class}`~cudf_polars.engine.hardware_binding.HardwareBindingPolicy`
to override this behaviour.

## Selecting a GPU

`SPMDEngine` runs on CUDA device ordinal 0, so the GPU it should use has to come first in
`CUDA_VISIBLE_DEVICES`. Selecting a GPU by ordinal instead, by leaving the list alone and
calling `cudaSetDevice`, is not supported, and constructing an engine that way raises. Other
GPUs may stay visible: `CUDA_VISIBLE_DEVICES=1,0` runs the engine on GPU 1 while both remain
available to the process.

`rrun` assigns a GPU to each rank, as do the Dask and Ray frontends for their workers, so
this is usually taken care of. A launcher that does not, `torchrun` being the common case,
leaves it to the script, which can call {func}`~cudf_polars.engine.spmd.use_gpu` before its
first CUDA call. {doc}`pytorch` covers that setup.

The requirement exists because the engine runs its actors on threads created by rapidsmpf,
and the current CUDA device is per-thread: a new thread always starts on ordinal 0 whatever
the thread that created it had selected. Putting the engine's GPU first makes that safe,
because ordinal 0 is then the device every thread already uses.

## Single-GPU setup

To use {class}`~cudf_polars.engine.spmd.SPMDEngine` on a single GPU, create the engine and
run your Python script as normal. You still get the full streaming executor (partitioned inputs,
spilling, scaling past device memory), you just don't need any multi-process coordination:

```python
# python my_script.py
import polars as pl
from cudf_polars.engine.spmd import SPMDEngine

with SPMDEngine() as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
          .filter(pl.col("amount") > 100)
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
```

With a single rank, the [Query symmetry requirement](#query-symmetry-requirement) and
[Collecting distributed results](#collecting-distributed-results) steps below do not apply,
`collect()` returns the full result directly.

## Multi-GPU with `rrun`

To run on more than one GPU, the same Python script must be launched collectively, and all
processes must be informed that they are participating in the cluster. This is the role of the
`rrun` launcher: it starts one process per GPU,
{class}`~cudf_polars.engine.spmd.SPMDEngine` detects this and bootstraps
a UCXX communicator across all ranks.

When the same script is launched without `rrun`, `SPMDEngine` falls back to a single-process,
single-GPU communicator that requires no external communication library. This mode is useful
for local development, unit tests, and single-GPU pipelines (see [Single-GPU setup](#single-gpu-setup) above).

```python
# multi-GPU launch: rrun -n 4 python my_script.py
# single-GPU:       python my_script.py
import polars as pl
from cudf_polars.engine.spmd import SPMDEngine

with SPMDEngine() as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
          .filter(pl.col("amount") > 100)
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
```

File-based sources (`scan_parquet`, `scan_csv`, …) are automatically partitioned so that each
rank reads a different file or row-group range. In-memory `DataFrame` objects are already
rank-local, so each rank processes its own copy.

## Configuring `SPMDEngine`

For custom configuration, build a
{class}`~cudf_polars.engine.options.StreamingOptions` and use
`SPMDEngine.from_options()`:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with SPMDEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/dataset/*.parquet").collect(engine=engine)
```

See {doc}`options` for the available fields.

{class}`~cudf_polars.engine.spmd.SPMDEngine` exposes a few properties
that are useful in SPMD code:

* `engine.nranks` / `engine.rank`: cluster size and local rank index.
* `engine.comm`: the active `rapidsmpf.communicator.Communicator`.
* `engine.context`: the active `rapidsmpf.streaming.core.context.Context`.

## Query symmetry requirement

All ranks must execute the **same sequence of queries in the same order**. Collective operations
are matched using internal operation IDs. If one rank executes a collective that another rank
does not, the program will deadlock.

In practice:

* Avoid rank-conditional `collect()` or `sink*()` calls.
* Avoid branches that change the query graph.
* Keep the client script deterministic.

```python
# OK: every rank runs the same query in the same order.
with SPMDEngine() as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
```

```python
# DEADLOCKS: rank 0 issues a group_by collective the other ranks never see.
with SPMDEngine() as engine:
    df = pl.scan_parquet("/data/*.parquet")
    if engine.rank == 0:        # don't do this
        df = df.group_by("customer_id").agg(pl.col("amount").sum())
    result = df.collect(engine=engine)
```

## Collecting distributed results

Unlike `RayEngine` / `DaskEngine`, where `.collect()` gathers every partition to the client,
here each rank's `.collect()` returns *its own* fragment. If you want to keep processing the
data rank-by-rank, just use that fragment directly. If you need a single concatenated view,
use the helper below.

`collect()` returns a rank-local result. Use
{func}`~cudf_polars.engine.spmd.allgather_polars_dataframe` to assemble
the full dataset on every rank:

```python
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.engine.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)

with SPMDEngine() as engine:
    result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)

    with reserve_op_id() as op_id:
        full = allgather_polars_dataframe(
            engine=engine,
            local_df=result,
            op_id=op_id,
        )
```

`op_id` identifies the collective across ranks. All ranks must pass the same value.
{func}`~cudf_polars.streaming.actor_graph.collectives.common.reserve_op_id` draws from the same
pool that cudf-polars uses internally for shuffle and join collectives, so there is no risk of
collision. Do not pass hardcoded integers: they may silently collide with an ID reserved by an
active collective inside `collect()`.

The result is a `pl.DataFrame` containing rows from all ranks in rank order (rank 0 first, then
rank 1, …, rank N-1).

## Reusing a communicator

By default {class}`~cudf_polars.engine.spmd.SPMDEngine` bootstraps a new
UCXX communicator on every construction. When running multiple engines in sequence (for example
in a test suite or interactive session), repeated bootstrapping is unnecessary and can race on
the file-based coordination layer shared by all ranks.

Pass a pre-created communicator via `comm=` to skip the bootstrap entirely. The engine does
**not** close the communicator on shutdown. The caller retains ownership and can reuse it
across multiple {class}`~cudf_polars.engine.spmd.SPMDEngine` lifetimes:

```python
from rapidsmpf import bootstrap
from rapidsmpf.progress_thread import ProgressThread
from cudf_polars.engine.spmd import SPMDEngine

# Bootstrap once.
comm = bootstrap.create_ucxx_comm(progress_thread=ProgressThread())

# Reuse across multiple engine lifetimes, no re-bootstrap between them.
with SPMDEngine(comm=comm) as engine:
    result1 = df1.lazy().collect(engine=engine)

with SPMDEngine(comm=comm) as engine:
    result2 = df2.lazy().collect(engine=engine)
```

## Cluster diagnostics

{meth}`~cudf_polars.engine.spmd.SPMDEngine.gather_cluster_info` returns
placement information for every rank:

```python
with SPMDEngine() as engine:
    if engine.rank == 0:
        for i, info in enumerate(engine.gather_cluster_info()):
            print(
                f"rank {i}: hostname={info['hostname']}, pid={info['pid']}, "
                f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
            )
```

[spmd-wiki]: https://en.wikipedia.org/wiki/Single_program,_multiple_data
