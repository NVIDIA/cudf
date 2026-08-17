(cudf-polars-dask-engine)=
# Dask

{class}`~cudf_polars.engine.dask.DaskEngine` runs the streaming executor
on a [Dask distributed][dask-distributed] cluster: one Dask worker per GPU, coordinated by a
single client process. Partitions are streamed through the query plan and collective operations
(shuffles, allgathers, joins) run across workers over a shared UCXX communicator. On startup,
each worker is pinned to the CPU cores and NUMA node closest to its GPU (see
[Pre-configured GPU clusters](#pre-configured-gpu-clusters) below).

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

With no arguments, {class}`~cudf_polars.engine.dask.DaskEngine` creates a
`distributed.LocalCluster` with one worker per visible GPU, a `distributed.Client`, and
bootstraps a UCXX communicator across all workers. On exit, everything it created is torn down.

```{note}
`.collect()` pulls the full result back to the client process. For large distributed outputs,
prefer `.sink_*()` or aggregate/sample inside the query before `.collect()`. See
[Result collection](engines.md#result-collection).
```

## Configuring `DaskEngine`

For custom configuration, build
{class}`~cudf_polars.engine.options.StreamingOptions` and use
`DaskEngine.from_options()`:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.dask import DaskEngine

opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")

with DaskEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/dataset/*.parquet").collect(engine=engine)
```

See {doc}`options` for the available fields.

## Bring your own Dask client

Pass an existing `distributed.Client` via `dask_client=` to attach to an already-running
scheduler:

```python
from distributed import Client
import polars as pl
from cudf_polars.engine.dask import DaskEngine

with Client("scheduler-address:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
```

When you supply the client, {class}`~cudf_polars.engine.dask.DaskEngine`
leaves it (and the cluster) alone on exit.

(pre-configured-gpu-clusters)=
### Pre-configured GPU clusters

Some Dask launchers, notably `dask_cuda.LocalCUDACluster`, already pin CPU affinity and set
`CUDA_VISIBLE_DEVICES` per worker. Disable the built-in hardware binding via
{class}`~cudf_polars.engine.hardware_binding.HardwareBindingPolicy`
to avoid having both layers fight over each worker's affinity (the second to run wins, which
makes the resulting placement non-deterministic):

```python
from dask_cuda import LocalCUDACluster
from distributed import Client
from cudf_polars.engine.dask import DaskEngine
from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
)

with Client(LocalCUDACluster()) as dc, DaskEngine(
    dask_client=dc,
    engine_options={
        "hardware_binding": HardwareBindingPolicy(enabled=False),
    },
) as engine:
    ...
```

### Manually launched workers

When launching workers yourself (for example on a multi-node HPC cluster), use the built-in nanny
preload to assign one GPU per worker. The preload sets `CUDA_VISIBLE_DEVICES` on each worker
before the process spawns:

```bash
# On each node, launch one worker per GPU with a single thread each:
dask worker SCHEDULER_ADDRESS:8786 --nworkers N --nthreads 1 \
    --preload-nanny cudf_polars.engine.dask
```

Then connect from the client:

```python
import polars as pl
from distributed import Client
from cudf_polars.engine.dask import DaskEngine

with Client("SCHEDULER_ADDRESS:8786") as dc:
    with DaskEngine(dask_client=dc) as engine:
        result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
```

Hardware binding (CPU affinity, NUMA, network) is handled automatically by
{class}`~cudf_polars.engine.dask.DaskEngine`; the nanny preload only
deals with GPU assignment.

See the [Dask CLI deployment guide][dask-cli] for more on `dask worker` options.

#### Using `dask-cuda-worker`

As an alternative to the built-in nanny preload, you can launch workers with
[`dask-cuda-worker`][dask-cuda-worker] from the [dask-cuda][dask-cuda] project. It launches one
worker per visible GPU and installs a set of plugins on every worker: a `CPUAffinity` plugin
that pins the worker to the NUMA node of its GPU, an `RMMSetup` plugin, and a nanny preload that
configures UCX.

`DaskEngine` sets up the same things for its own streaming runtime, so the two need to be
coordinated or they will fight:

* **CPU affinity is unconditional in `dask-cuda-worker`**, the `CPUAffinity` plugin is always
  installed and there is no CLI flag to turn it off. Pass `hardware_binding=HardwareBindingPolicy(enabled=False)`
  to `DaskEngine` so it does not try to re-pin affinity on top of dask-cuda's binding.
* **Do not pass `--rmm-pool-size`, `--rmm-managed-memory`, or similar RMM flags** to
  `dask-cuda-worker`. Let `DaskEngine` own the memory resource via its `memory_resource_config`
  (see {doc}`options`) otherwise two different memory resources will be installed on the same
  worker.
* **Do not pass `--enable-tcp-over-ucx`, `--enable-infiniband`, `--enable-nvlink`, or
  `--enable-rdmacm`** to `dask-cuda-worker`. `DaskEngine` bootstraps its own UCXX communicator
  and will select transports itself. Enabling them on both sides can produce inconsistent UCX
  configuration across the cluster.

```bash
# On each node, GPU assignment + CPU affinity only (no RMM, no UCX flags):
dask-cuda-worker SCHEDULER_ADDRESS:8786
```

```python
import polars as pl
from distributed import Client
from cudf_polars.engine.dask import DaskEngine
from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
)

with Client("SCHEDULER_ADDRESS:8786") as dc:
    with DaskEngine(
        dask_client=dc,
        engine_options={
            # dask-cuda-worker always pins CPU affinity; disable DaskEngine's
            # binding so the two don't conflict.
            "hardware_binding": HardwareBindingPolicy(enabled=False),
        },
    ) as engine:
        result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
```

## Cluster diagnostics

{meth}`~cudf_polars.engine.dask.DaskEngine.gather_cluster_info` returns
placement information for every worker:

```python
with DaskEngine() as engine:
    print(f"cluster has {engine.nranks} workers")
    for info in engine.gather_cluster_info():
        print(
            f"hostname={info['hostname']}, pid={info['pid']}, "
            f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}"
        )
```

{class}`~cudf_polars.engine.dask.DaskEngine` raises `RuntimeError` if
created inside an `rrun` cluster.

[dask-distributed]: https://distributed.dask.org/en/stable/
[dask-cli]: https://docs.dask.org/en/latest/deploying-cli.html
[dask-cuda]: https://docs.rapids.ai/api/dask-cuda/nightly/
[dask-cuda-worker]: https://docs.rapids.ai/api/dask-cuda/nightly/quickstart/#dask-cuda-worker
