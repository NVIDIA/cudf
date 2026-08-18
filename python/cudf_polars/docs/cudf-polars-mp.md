# Multi-GPU Polars

Multi-GPU Polars extends Polars query execution to multiple GPUs.

Multi-GPU execution distributes a query across several GPU workers. Each worker
owns a disjoint fragment of the data and participates in collective operations
such as shuffles, all-gathers, and joins to produce a globally correct result.

The entry point in all cases is the Polars `GPUEngine` with `executor="streaming"`.
The `cluster` option selects the execution model:

| `cluster`       | Description                                          | Status            |
| --------------- | ---------------------------------------------------- | ----------------- |
| `"single"`      | Single-GPU, in-process execution                     | Stable (legacy)   |
| `"distributed"` | Multi-GPU via [Dask Distributed][dask-distributed]   | Stable (legacy)   |
| `"ray"`         | Multi-GPU via [Ray][ray-docs] actors                 | Preview (new API) |
| `"dask"`        | Multi-GPU via [Dask Distributed][dask-distributed]   | Preview (new API) |
| `"spmd"`        | Multi-GPU via [SPMD][spmd-wiki] launched with `rrun` | Preview (new API) |

Three preview execution modes are available:

* **Ray mode** — a single-client model where a driver program coordinates GPU
  workers implemented as Ray actors.
* **Dask mode** — a single-client model where a driver program coordinates GPU workers
  running on a Dask distributed cluster.
* **SPMD mode** — each GPU runs the same script as an independent process.
  When launched with `rrun` a full UCXX communicator connects the ranks.
  Without `rrun` it falls back to a single-rank communicator with no external
  dependencies, which is useful for local development and testing.

This document describes these three execution modes.

* [Unified configuration (StreamingOptions)](#unified-configuration-streamingoptions)
* [Ray execution mode](#ray-execution-mode)
* [Dask execution mode](#dask-execution-mode)
* [SPMD execution mode](#spmd-execution-mode)

---

## Unified configuration (`StreamingOptions`)

`StreamingOptions` is the recommended way to configure Ray, Dask, and SPMD engines.
It provides a single typed object covering all configuration knobs across three
categories:

| Category    | Controls                                                              |
| ----------- | --------------------------------------------------------------------- |
| `rapidsmpf` | Threads, CUDA streams, spilling, pinned memory, log level             |
| `executor`  | Partitioning, fallback behavior, dynamic planning                     |
| `engine`    | Polars integration, IO, RMM, hardware binding, thread-pool sizing     |

All fields default to `UNSPECIFIED`, which means: use the corresponding
environment variable if set, otherwise let the underlying library apply its
own built-in default.

```python
from cudf_polars.engine.options import StreamingOptions

opts = StreamingOptions(
    num_streaming_threads=8,
    log="DEBUG",
    fallback_mode="silent",
    spill_device_limit="70%",
)
```

Pass the options object to `from_options()` on any engine — this is the
recommended constructor for typical use:

```python
from cudf_polars.engine.dask import DaskEngine
from cudf_polars.engine.ray import RayEngine
from cudf_polars.engine.spmd import SPMDEngine

with RayEngine.from_options(opts) as engine:
    result = df.lazy().collect(engine=engine)

# or, in Dask mode:
with DaskEngine.from_options(opts) as engine:
    result = df.lazy().collect(engine=engine)

# or, in SPMD mode:
with SPMDEngine.from_options(opts) as engine:
    result = df.lazy().collect(engine=engine)
```

### Building from a dictionary

`StreamingOptions.from_dict()` accepts a flat dict of field names. Unknown keys
raise `TypeError`; `None` values are treated as `UNSPECIFIED`:

```python
opts = StreamingOptions.from_dict({
    "num_streaming_threads": 8,
    "fallback_mode": "silent",
})
```

### Memory resource configuration

Use `memory_resource_config` to control the RMM memory resource used by the
engine. It accepts a `MemoryResourceConfig` object that specifies the fully
qualified class name and optional constructor arguments:

```python
from cudf_polars.utils.config import MemoryResourceConfig

opts = StreamingOptions(
    memory_resource_config=MemoryResourceConfig(
        qualname="rmm.mr.CudaAsyncMemoryResource",
    ),
)
```

Nested resources (e.g. a pool wrapping a managed resource) are supported:

```python
opts = StreamingOptions(
    memory_resource_config=MemoryResourceConfig(
        qualname="rmm.mr.PoolMemoryResource",
        options={
            "upstream_mr": {
                "qualname": "rmm.mr.ManagedMemoryResource",
            },
        },
    ),
)
```

When no `memory_resource_config` is provided:

- **SPMDEngine** uses `rmm.mr.get_current_device_resource()` (the in-process
  default — useful when user code has already configured a resource).
- **DaskEngine** and **RayEngine** default to `rmm.mr.CudaAsyncMemoryResource()`
  (workers start in a fresh process with no pre-configured resource). The initial
  pool size is null and the release threshold is set to 90% of the device's
  memory.

### Hardware binding

All three engines automatically bind each worker process to the CPU cores,
NUMA memory nodes, and network devices that are topologically close to the
worker's GPU. This is done via `rapidsmpf.rrun.rrun.bind()` and improves
performance by ensuring memory allocations and network traffic stay local to
the GPU's NUMA node.

Binding is controlled by the `hardware_binding` executor option, which accepts
a `HardwareBindingPolicy` instance:

```python
from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
)
```

The default policy (`HardwareBindingPolicy()`) skips binding when running under `rrun`,
which already handles binding at launch. Otherwise, it binds once per process based on
`CUDA_VISIBLE_DEVICES`. If `CUDA_VISIBLE_DEVICES` is unset, binding falls back to GPU 0.


| Field             | Default | Description                                                                                                        |
| ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `skip_under_rrun` | `True`  | Skip binding when launched via `rrun` (which already performs binding). If skipped, all other options are ignored. |
| `enabled`         | `True`  | Enable or disable hardware binding.                                                                                |
| `enable_once`     | `True`  | Perform binding at most once per process. Subsequent calls are no-ops.                                             |
| `raise_on_fail`   | `False` | Surface binding failures by enabling `verbose=True` in `rrun.bind()`.                                              |


Examples:

```python
# Disable binding entirely:
opts = StreamingOptions(hardware_binding=HardwareBindingPolicy(enabled=False))

# Enable failure reporting:
opts = StreamingOptions(
    hardware_binding=HardwareBindingPolicy(raise_on_fail=True),
)
```

Via the environment variable (JSON):

```bash
# Disable binding:
export CUDF_POLARS__HARDWARE_BINDING='{"enabled": false}'
```

Via the CLI:

```bash
python my_script.py --hardware-binding '{"raise_on_fail": true}'
```

---

## Ray execution mode

Ray mode uses a single client process that drives execution across multiple ranks.
Each rank corresponds to one GPU worker and participates in collective operations
through a shared UCXX communicator.

In the Ray implementation each rank is implemented as a [**Ray actor**][ray-actors],
with one actor created per available GPU.

Conceptually the system looks like this:

```
                 ┌──────────────────────────────┐
                 │        User script           │
                 │   (single client process)    │
                 │  LazyFrame.collect(engine=…) │
                 └──────────────┬───────────────┘
                                │ IR dispatched to all actors
               ┌────────────────|─────────────────┐
               ↓                ↓                 ↓
        ┌─────────────┐  ┌─────────────┐   ┌─────────────┐
        │  RankActor  │  │  RankActor  │   │  RankActor  │
        │   rank 0    │  │   rank 1    │   │  rank N-1   │
        │   run IR    │  │   run IR    │   │   run IR    │
        └──────┬──────┘  └──────┬──────┘   └──────┬──────┘
               ↓                ↓                 ↓
┌────────────────────────────────────────────────────────────────┐
│                     RapidsMPF streaming engine                 │
│   shuffle / all-gather · UCXX communicator · RMM GPU memory    │
└────────────────────────────────────────────────────────────────┘
               ↑                ↑                 ↑
             GPU 0            GPU 1            GPU N-1
```

The client broadcasts the query plan to all ranks. The ranks execute the pipeline
collectively through UCXX, and their outputs are streamed back and concatenated on
the client process.

The driver script runs as a normal Python program with no `rrun` launcher. Query
symmetry is handled automatically: the client serializes the complete query plan and
broadcasts it to all actors, so every rank always executes the same query.

### Prerequisites

* Ray (`ray`) installed
* RapidsMPF and UCXX available on all GPU nodes

### Running in Ray mode

`RayEngine` is imported from `cudf_polars.engine.ray`. On construction it:

1. Calls `ray.init()` if Ray is not already running
2. Creates one `RankActor` per GPU
3. Bootstraps a UCXX communicator across the actors

Actors are shut down when `shutdown()` is called or the context manager exits. If the
engine started Ray, it also calls `ray.shutdown()`.

The recommended way to construct a `RayEngine` is via `from_options()`:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with RayEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

print(result)
```

With no options, `RayEngine()` uses all built-in defaults:

```python
with RayEngine() as engine:
    result = pl.scan_parquet(...).collect(engine=engine)
```

### Ray lifecycle

If Ray is already initialized, `RayEngine` attaches to the existing cluster and
does not call `ray.shutdown()` on exit.

```python
import ray
import polars as pl
from cudf_polars.engine.ray import RayEngine

ray.init(address="auto")

try:
    with RayEngine() as engine:
        result = pl.scan_parquet(...).collect(engine=engine)
finally:
    ray.shutdown()
```

`RayEngine` raises `RuntimeError` if created inside an `rrun` cluster or if no
GPUs are available.

### Cluster diagnostics

`RayEngine.gather_cluster_info()` returns placement information for all rank actors:

```python
with RayEngine() as engine:
    print(f"cluster has {engine.nranks} ranks")
    for i, info in enumerate(engine.gather_cluster_info()):
        print(
            f"rank {i}: hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

Each entry includes `pid`, `hostname`, `cuda_visible_devices`, and `node_id`.

### Passing options

Prefer `RayEngine.from_options()` with a `StreamingOptions` object (see
[Unified configuration](#unified-configuration-streamingoptions)). For
fine-grained control, the `__init__` parameters accept raw dicts:

```python
from rapidsmpf.config import Options

with RayEngine(
    rapidsmpf_options=Options(num_streaming_threads=8),
    executor_options={"num_py_executors": 2},
    executor_options={"max_rows_per_partition": 500_000},
    engine_options={"raise_on_fail": True},
    ray_init_options={"num_cpus": 4},
) as engine:
    ...
```

`ray_init_options` is forwarded to `ray.init()` when Ray is not already
initialized. It is kept separate from streaming behavior options and has no
`StreamingOptions` equivalent.

`executor_options` is forwarded directly to `pl.GPUEngine` as its `executor_options`
argument; user-supplied keys are merged with reserved entries set by `RayEngine`.

---

## Dask execution mode

Dask mode uses a single client process that drives execution across multiple ranks.
Each rank corresponds to one GPU worker and participates in collective operations
through a shared UCXX communicator. In the Dask implementation each rank is implemented
as a **Dask worker**, with one worker per available GPU.

Conceptually the system looks like this:

```
                 ┌──────────────────────────────┐
                 │        User script           │
                 │   (single client process)    │
                 │  LazyFrame.collect(engine=…) │
                 └──────────────┬───────────────┘
                                │ IR dispatched to all workers
               ┌────────────────|─────────────────┐
               ↓                ↓                 ↓
        ┌─────────────┐  ┌─────────────┐   ┌─────────────┐
        │ Dask worker │  │ Dask worker │   │ Dask worker │
        │   rank 0    │  │   rank 1    │   │  rank N-1   │
        │   run IR    │  │   run IR    │   │   run IR    │
        └──────┬──────┘  └──────┬──────┘   └──────┬──────┘
               ↓                ↓                 ↓
┌────────────────────────────────────────────────────────────────┐
│                     RapidsMPF streaming engine                 │
│   shuffle / all-gather · UCXX communicator · RMM GPU memory    │
└────────────────────────────────────────────────────────────────┘
               ↑                ↑                 ↑
             GPU 0            GPU 1            GPU N-1
```

### Prerequisites

* Dask distributed (`distributed`) installed
* RapidsMPF and UCXX available on all GPU nodes

### Running in Dask mode

`DaskEngine` is imported from `cudf_polars.engine.dask`. On construction it:

1. If `dask_client` is `None`, creates a `distributed.LocalCluster` (one worker per visible GPU) and a `distributed.Client`
2. Bootstraps a UCXX communicator across all workers

`DaskEngine` is a `StreamingEngine` subclass (and therefore a `pl.GPUEngine`) that can be used directly or as a context manager.

```python
import polars as pl
from cudf_polars.engine.dask import DaskEngine

with DaskEngine() as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

print(result)
```

The context manager yields a `DaskEngine` with:

* `engine.nranks` — number of Dask workers at bootstrap time
* `engine.gather_cluster_info()` — cluster diagnostics

### Dask lifecycle

Bring-your-own-client variant:

```python
from distributed import Client
import polars as pl
from cudf_polars.engine.dask import DaskEngine

with Client("scheduler-address:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = pl.scan_parquet(...).collect(engine=engine)
```

Jupyter/manual style:

```python
engine = DaskEngine()
result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
engine.shutdown()
```

`DaskEngine` raises `RuntimeError` if created inside an `rrun` cluster.

### Hardware binding with pre-configured clusters

When using a pre-configured cluster that already performs its own hardware
binding — such as `dask_cuda.LocalCUDACluster`, which pins CPU affinity and
sets `CUDA_VISIBLE_DEVICES` per worker — disable the built-in binding to
avoid conflicts:

```python
from cudf_polars.engine.dask import DaskEngine
from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
)

with DaskEngine(
    dask_client=dc,
    engine_options={
        "hardware_binding": HardwareBindingPolicy(enabled=False),
    },
) as engine:
    ...
```

### Manually launched Dask clusters

When launching workers manually (e.g. on a multi-node HPC cluster), use the
built-in nanny preload to assign one GPU per worker. The preload sets
`CUDA_VISIBLE_DEVICES` on each worker before the process spawns:

```bash
# On each node — launch one worker per GPU with a single thread each:
dask worker SCHEDULER:8786 --nworkers N --nthreads 1 \
    --preload-nanny cudf_polars.engine.dask
```

Then connect from the client:

```python
from distributed import Client
from cudf_polars.engine.dask import DaskEngine

with Client("SCHEDULER:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = lf.collect(engine=engine)
```

Hardware binding (CPU affinity, NUMA, network) is handled automatically by
`DaskEngine` via `HardwareBindingPolicy` — the nanny preload only handles
GPU assignment.

See the [Dask CLI deployment guide][dask-cli] for more on `dask worker` options.

### Cluster diagnostics

```python
with DaskEngine() as engine:
    print(f"cluster has {engine.nranks} workers")
    for info in engine.gather_cluster_info():
        print(
            f"hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

Each entry includes `pid`, `hostname`, and `cuda_visible_devices`.

### Passing options

Prefer `DaskEngine.from_options()` with a `StreamingOptions` object (see
[Unified configuration](#unified-configuration-streamingoptions)). For
fine-grained control, the `__init__` parameters accept raw dicts:

```python
from rapidsmpf.config import Options

with DaskEngine(
    rapidsmpf_options=Options(num_streaming_threads=8),
    executor_options={"num_py_executors": 2},
    executor_options={"max_rows_per_partition": 500_000},
    engine_options={"raise_on_fail": True},
) as engine:
    ...
```

`executor_options` is forwarded directly to `pl.GPUEngine` as its `executor_options`
argument; user-supplied keys are merged with reserved entries set by `DaskEngine`.

---

## SPMD execution mode

In SPMD (Single Program, Multiple Data) execution, the same Python script runs once
per GPU and each process owns its local data fragment. Collective operations
(shuffles, all-gathers, joins) coordinate across processes to produce a globally
consistent result.

`SPMDEngine` selects the communicator automatically:

* **With `rrun`** — the `rrun` launcher starts one process per GPU and
  `SPMDEngine` bootstraps a UCXX communicator across all ranks.
* **Without `rrun`** — `SPMDEngine` falls back to a single-rank
  communicator that requires no external communication library (no UCXX,
  Ray, or Dask). This mode is useful for local development, unit tests,
  and single-GPU pipelines.

File-based sources (`scan_parquet`, `scan_csv`, etc.) are automatically partitioned
so that different ranks read different file or row-group ranges. In-memory
`DataFrame` objects are already rank-local, so each rank processes its own copy.

Conceptually the setup looks like this:

```
       rank 0               rank 1       ...      rank N-1
┌─────────────────┐  ┌─────────────────┐     ┌─────────────────┐
│   User script   │  │   User script   │     │   User script   │
│ (same code on   │  │ (same code on   │     │ (same code on   │
│  every rank)    │  │  every rank)    │     │  every rank)    │
└────────┬────────┘  └────────┬────────┘     └────────┬────────┘
         │                    │                       │
┌────────┴────────────────────┴───────────────────────┴────────┐
│              LazyFrame.collect(engine=engine)                │
└────────┬────────────────────┬───────────────────────┬────────┘
         ↓                    ↓                       ↓
┌─────────────────┐  ┌─────────────────┐     ┌─────────────────┐
│     run IR      │  │     run IR      │     │     run IR      │
└────────┬────────┘  └────────┬────────┘     └────────┬────────┘
         │                    │                       │
         ↓                    ↓                       ↓
┌────────────────────────────────────────────────────────────────┐
│                     RapidsMPF streaming engine                 │
│   shuffle / all-gather · UCXX communicator · RMM GPU memory    │
└────────────────────────────────────────────────────────────────┘
         ↑                    ↑                       ↑
      GPU 0                GPU 1                   GPU N-1
```

After `collect`, results are **rank-local**. To assemble the full dataset on
every rank, call `allgather_polars_dataframe()`.

### Prerequisites

* RapidsMPF (`rapidsmpf`) installed
* UCXX available when using `rrun` for multi-GPU execution
  (usually installed with RapidsMPF; not required for single-rank use)
* `rrun` launcher available for multi-GPU use (`rrun --help` should succeed)

### Running in SPMD mode

`SPMDEngine` is the primary entry point for SPMD execution. It is a context
manager imported from `cudf_polars.engine.spmd`. On construction it:

1. Bootstraps a communicator: UCXX when running under `rrun`, otherwise a
   single-rank communicator that requires no external library.
   Pass an already-bootstrapped communicator via `comm=` to skip this step and
   reuse an existing one (see [Reusing a communicator](#reusing-a-communicator) below).
2. Creates a RapidsMPF streaming `Context` that owns GPU memory and a CUDA stream pool.

All resources except the (optionally) caller-supplied communicator are released when
the context exits (or `shutdown()` is called).

The recommended way to construct an `SPMDEngine` is via `from_options()`:

```python
# multi-GPU launch: rrun -n 4 python my_script.py
# single-GPU (no rrun needed): python my_script.py
import polars as pl
from cudf_polars.streaming.collectives.common import reserve_op_id
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with SPMDEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )

    with reserve_op_id() as op_id:
        full = allgather_polars_dataframe(
            engine=engine,
            local_df=result,
            op_id=op_id,
        )
```

With no options, `SPMDEngine()` uses all built-in defaults:

```python
with SPMDEngine() as engine:
    result = pl.scan_parquet(...).collect(engine=engine)
```

`SPMDEngine` provides:

* `engine.comm` — [`rapidsmpf.communicator.Communicator`][rapidsmpf-communicator]
* `engine.context` — [`rapidsmpf.streaming.core.context.Context`][rapidsmpf-context]
* `engine.nranks` / `engine.rank` — cluster size and local rank index

Pass `engine` to every `LazyFrame.collect()` or `sink*()` call inside the context block.

### Query symmetry requirement

All ranks must execute the **same sequence of queries in the same order**. Collective
operations are matched using internal operation IDs. If one rank executes a collective
that another rank does not, the program will deadlock.

In practice:

* Avoid rank-conditional `collect()` or `sink*()` calls
* Avoid branches that change the query graph
* Keep the driver script deterministic

**Example that works correctly:**

```python
# Every rank executes the same query in the same order.
with SPMDEngine() as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
        .collect(engine=engine)
    )
```

**Example that deadlocks:**

```python
# Rank 0 executes a group_by collective; other ranks do not.
# The collective IDs go out of sync → deadlock.
with SPMDEngine() as engine:
    df = pl.scan_parquet("/data/*.parquet")
    if engine.rank == 0:        # DON'T DO THIS
        df = df.group_by("customer_id").agg(pl.col("amount").sum())
    result = df.collect(engine=engine)
```

### Collecting distributed results

`collect()` returns a rank-local result. Use
`allgather_polars_dataframe()` to gather all fragments:

```python
from cudf_polars.streaming.collectives.common import reserve_op_id
from cudf_polars.engine.spmd import (
    SPMDEngine,
    allgather_polars_dataframe,
)

with SPMDEngine() as engine:
    with reserve_op_id() as op_id:
        full = allgather_polars_dataframe(
            engine=engine,
            local_df=result,
            op_id=op_id,
        )
```

`op_id` identifies this collective across ranks — all ranks must pass the same value.
Use `reserve_op_id()` to obtain a safe ID. It draws from the same pool that cudf-polars uses internally for shuffle
and join collectives, so there is no risk of collision. Do not pass hardcoded integers: they
may silently collide with an ID already reserved by an active collective inside `collect()`.

The result is guaranteed to be a `pl.DataFrame` containing rows from all ranks in rank order
(rank 0 first, then rank 1, …, rank N-1).

### Reusing a communicator

By default `SPMDEngine` bootstraps a new UCXX communicator on every construction.
When running multiple engines in sequence (for example in a test suite or an
interactive session), bootstrapping repeatedly is unnecessary and can cause race
conditions in the file-based coordination layer shared by all ranks.

Pass a pre-created communicator via the `comm=` argument to skip the bootstrap
entirely. The engine **does not** close the communicator on shutdown — the caller
retains ownership and can reuse it across multiple `SPMDEngine` lifetimes.

```python
from rapidsmpf import bootstrap
from rapidsmpf.progress_thread import ProgressThread
from cudf_polars.engine.spmd import SPMDEngine

# Bootstrap once.
comm = bootstrap.create_ucxx_comm(progress_thread=ProgressThread())

# Reuse across multiple engine lifetimes — no re-bootstrap between them.
with SPMDEngine(comm=comm) as engine:
    result1 = df1.lazy().collect(engine=engine)

with SPMDEngine(comm=comm) as engine:
    result2 = df2.lazy().collect(engine=engine)
```

### Passing options

Prefer `SPMDEngine.from_options()` with a `StreamingOptions` object (see
[Unified configuration](#unified-configuration-streamingoptions)). For
fine-grained control, the `__init__` parameters accept raw dicts:

```python
from rapidsmpf.config import Options

with SPMDEngine(
    rapidsmpf_options=Options(num_streaming_threads=8),
    executor_options={"num_py_executors": 2},
    executor_options={"max_rows_per_partition": 500_000},
    engine_options={"parquet_options": {}},
) as engine:
    ...
```

**Memory resource:** All engines accept a `memory_resource_config` option (via
`StreamingOptions` or `engine_options`) that controls the RMM memory resource.
See [Memory resource configuration](#memory-resource-configuration) for details.
When no config is provided, `SPMDEngine` falls back to
`rmm.mr.get_current_device_resource()`, while `DaskEngine` and `RayEngine`
default to `rmm.mr.CudaAsyncMemoryResource()`.

`comm` is an already-bootstrapped communicator. When provided, the bootstrap step
is skipped and the caller retains ownership (see
[Reusing a communicator](#reusing-a-communicator)). Defaults to `None`.

`rapidsmpf_options` is an `Options` object passed to the RapidsMPF `Context`. Defaults
to `None` (uses RapidsMPF defaults).

`executor_options` is forwarded directly to `pl.GPUEngine` as its `executor_options`
argument; user-supplied keys are merged with reserved entries set by `SPMDEngine`.

`engine_options` is forwarded as keyword arguments to `pl.GPUEngine`.

<!-- Reference links -->
[dask-cli]: https://docs.dask.org/en/latest/deploying-cli.html
[dask-distributed]: https://distributed.dask.org/en/stable/
[spmd-wiki]: https://en.wikipedia.org/wiki/Single_program,_multiple_data
[ray-docs]: https://docs.ray.io/en/latest/
[ray-actors]: https://docs.ray.io/en/latest/ray-core/actors.html
[rapidsmpf-communicator]: https://docs.rapids.ai/api/rapidsmpf/stable/glossary/#term-Communicator
[rapidsmpf-context]: https://docs.rapids.ai/api/rapidsmpf/stable/glossary/#term-Context
[polars-gpuengine]: https://docs.pola.rs/api/python/stable/reference/api/polars.GPUEngine.html
