(cudf-polars-execute)=
# Keeping results on the GPU with `engine.execute()`

```{warning}
`engine.execute()` is **experimental**. Its API may change or be removed in a
future release. Using it emits a `cudf_polars.UnstableWarning`; set the
`CUDF_POLARS_WARN_UNSTABLE` environment variable to `1` to have that warning
raised as an error instead. See {doc}`options`.
```

[`LazyFrame.collect`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html)
runs a query on the GPU and then copies the
full result back to host memory as a `pl.DataFrame`. When you want to use this
result as the input to a subsequent query that will also execute on the GPU, this copy to host is unnecessary and necessitates a subsequent copy back to the GPU.

`engine.execute()` avoids that overhead. It runs the query and returns a
{class}`~cudf_polars.engine.persisted_result.PersistedQueryResult` whose
data remain on the GPU in the process that produced them. You can then
chain further work through {meth}`~cudf_polars.engine.persisted_result.PersistedQueryResult.lazy`
without a host copy in between.

It is available on the streaming engines:
{class}`~cudf_polars.engine.spmd.SPMDEngine`,
{class}`~cudf_polars.engine.ray.RayEngine`, and
{class}`~cudf_polars.engine.dask.DaskEngine`.

```{note}
This is distinct from Polars' own `LazyFrame.execute()`, which you call directly
on the frame (`df.execute()`) rather than on a cudf-polars engine. cudf-polars
does not accelerate `LazyFrame.execute()` yet: calling it always moves the data
back to the client and host memory, just like
[`LazyFrame.collect`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html).
To keep results GPU-resident, use `engine.execute()` as described here.
```

## Basic usage

```python
import polars as pl
from cudf_polars.engine.ray import RayEngine

with RayEngine() as engine:
    # Runs on the GPU, the result stays there.
    result = engine.execute(
        pl.scan_parquet("/data/dataset/*.parquet")
        .filter(pl.col("amount") > 100)
        .group_by("customer_id")
        .agg(pl.col("amount").sum())
    )

    # Run more work on the GPU using the persisted result as input.
    df = (
        result.lazy()
        .sort("amount")
        .head(10)
        .collect(engine=engine)
    )
```

The `LazyFrame` returned by `result.lazy()` can be collected **only** with the
engine that produced it, never with a different engine. The data never
leave their owning processes, so the default (host) Polars engine cannot read
them, and collecting or executing the result with any other engine is unsupported.

## Chaining into another `execute()`

`result.lazy()` supports the usual `LazyFrame` API, so you can add further
operations and pass the resulting query directly to `engine.execute()` without
collecting it first.

Only materialization is special. Like any persisted result, it must run on the
engine that produced it, as described above, and cannot be collected directly
on the host. The query executes on the GPU, and its output remains there as a
new persisted result. This makes it useful for building multi-step pipelines
without transferring intermediate results to host memory:

```python
with RayEngine() as engine:
    a = engine.execute(pl.scan_parquet("/data/*.parquet"))

    # Run more GPU work and keep the result on the GPU (no host copy).
    b = engine.execute(a.lazy().filter(pl.col("amount") > 100))

    # Collect only when you actually need the data on the host.
    df = b.lazy().collect(engine=engine)
```

Each `execute()`/`collect()` consumes the result it reads, so
`a` is spent by the second `execute()` above and cannot be read again.

## Results are one-shot

A persisted result can only be collected once. Attempting to collect the same
`LazyFrame` twice, or to use it in an operation that reads it multiple times
(such as a self-join), raises a `RuntimeError`. Call `engine.execute()` again
to produce a fresh result.

```python
with RayEngine() as engine:
    result = engine.execute(pl.scan_parquet("/data/*.parquet"))
    lazy = result.lazy()

    df = lazy.collect(engine=engine)        # OK: consumes the partitions.
    df = lazy.collect(engine=engine)        # RuntimeError: already consumed.

    # Need it again? Re-run the query for a fresh result.
    df = engine.execute(pl.scan_parquet("/data/*.parquet")).lazy().collect(engine=engine)
```

## Releasing partitions

Persisted partitions occupy GPU memory until they are released. If you never
collect a result, its partitions are freed when the result (and any `LazyFrame`
derived from it) is garbage-collected. To free them deterministically, use the
context-manager protocol or call
{meth}`~cudf_polars.engine.persisted_result.PersistedQueryResult.release`:

```python
with RayEngine() as engine:
    with engine.execute(pl.scan_parquet("/data/*.parquet")) as result:
        df = result.lazy().select("a").collect(engine=engine)
    # Partitions are released here on scope exit.
```

## SPMD note

On {class}`~cudf_polars.engine.spmd.SPMDEngine`, `execute()` is collective:
every rank must call it with an equivalent query, and each rank's result holds
that rank's own partition (see
[Query symmetry requirement](spmd_engine.md#query-symmetry-requirement)).
