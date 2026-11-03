(cudf-polars-default-singleton-engine)=
# Default `engine="gpu"`

`.collect(engine="gpu")` (and `engine=pl.GPUEngine()`) is the API you invoke when you don't
construct a streaming engine explicitly. It runs the same streaming executor as the explicit
engines (Ray, Dask, SPMD), conceptually similar to
[Polars' own streaming engine](https://docs.pola.rs/user-guide/concepts/streaming/) but on the
GPU. Under the hood it's backed by {class}`~cudf_polars.engine.default_singleton_engine.DefaultSingletonEngine`,
a process-wide singleton specialization of {class}`~cudf_polars.engine.spmd.SPMDEngine`. At most one live
instance exists per process, which is created lazily on first use and torn down at interpreter
exit. Ray is the showcased explicit engine (see {doc}`usage`); this page documents what
`engine="gpu"` does *without* you having to construct anything.

```{important}
`engine="gpu"` is meant for trivial setup: single-GPU execution with no
configuration or engine object to manage.
For any non-trivial workflow, construct an engine explicitly. To tune
options, use
{meth}`RayEngine.from_options(...) <cudf_polars.engine.ray.RayEngine.from_options>`.
`engine="gpu"` accepts no options, so settings such as
`spill_to_pinned_memory=True` for spill-heavy workloads require an
explicit engine. See {doc}`usage` and {doc}`options`.
```

## What you get without an explicit engine

When you just write:

```python
import polars as pl

result = (
    pl.scan_parquet("/data/*.parquet")
      .group_by("customer_id")
      .agg(pl.col("amount").sum())
      .collect(engine="gpu")
)
```

cudf-polars uses
{class}`~cudf_polars.engine.default_singleton_engine.DefaultSingletonEngine`
under the hood. No cluster is set up, the rapidsmpf `Context` is bootstrapped on first use,
and subsequent `.collect()` calls in the same process reuse it.

## Explicit handle

If you genuinely want the singleton (for example in tests or scripts that need to call
`.shutdown()` deterministically) you can obtain it via the factory:

```python
from cudf_polars.engine.default_singleton_engine import (
    DefaultSingletonEngine,
)

engine = DefaultSingletonEngine.get_or_create()
result = query.collect(engine=engine)
```

`get_or_create()` is idempotent: calling it again returns the same instance.

For anything beyond defaults, prefer an explicit engine. See {doc}`usage`.

## Lifecycle

The singleton is bootstrapped once per process. The rapidsmpf `Context`, RMM adaptor, and
Python thread-pool executor are reused across every `.collect()` call.

Shutdown is automatic: the engine registers an `atexit` hook that tears it down at interpreter
exit. To shut it down explicitly (for example to release resources before constructing a
multi-GPU engine), call the static method:

```python
from cudf_polars.engine.default_singleton_engine import (
    DefaultSingletonEngine,
)

DefaultSingletonEngine.shutdown()
```

`shutdown()` is idempotent (calling it twice is safe) and a no-op if no live engine exists.

## Mutual exclusion with explicit engines

`DefaultSingletonEngine`, {class}`~cudf_polars.engine.ray.RayEngine`,
{class}`~cudf_polars.engine.dask.DaskEngine`, and
{class}`~cudf_polars.engine.spmd.SPMDEngine` cannot coexist in the same
process. Concretely:

- Constructing `RayEngine` / `DaskEngine` / `SPMDEngine` while the singleton is alive raises
  `RuntimeError`.
- `DefaultSingletonEngine.get_or_create()` raises `RuntimeError` if any explicit streaming
  engine is alive.

Recommended pattern: pick one engine for the lifetime of the program. If you need to switch,
shut down the active engine first:

```python
DefaultSingletonEngine.shutdown()
explicit_engine = SPMDEngine.from_options(opts)
```

## No options

`DefaultSingletonEngine.get_or_create()` takes no arguments. To tune `StreamingOptions` such
as `spill_to_pinned_memory`, `fallback_mode`, `max_rows_per_partition`, or any rapidsmpf
runtime knob, construct an explicit
{class}`~cudf_polars.engine.ray.RayEngine` via
{meth}`RayEngine.from_options(...) <cudf_polars.engine.ray.RayEngine.from_options>`.
See {doc}`options` for the available fields.
