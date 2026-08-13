(cudf-polars-io-plugins)=
# Python IO Sources

Polars lets you feed a query from a custom Python function instead of a file, by
registering an *IO plugin* with [`polars.io.plugins.register_io_source`][register-io-source].
cudf-polars executes such sources on the GPU, so you can generate or load data
directly into a query, for example from a custom file format, a remote store, or
data produced on the fly.

See the [Polars IO plugins guide][polars-io-plugins] for the full description of
the IO-source contract. This page covers the cudf-polars-specific behavior.

## Plain IO Sources

An IO source is a callable with the signature `(with_columns, predicate, n_rows, batch_size)`
that yields one or more chunks (i.e. polars DataFrames):

```python
import polars as pl
from polars.io.plugins import register_io_source


def source(with_columns, predicate, n_rows, batch_size):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    if predicate is not None:
        df = df.filter(predicate)
    if n_rows is not None:
        df = df.head(n_rows)
    if with_columns is not None:
        df = df.select(with_columns)
    yield df


lf = register_io_source(source, schema={"a": pl.Int64, "b": pl.Int64})
result = lf.select("a").collect(engine=pl.GPUEngine())
```

Chunks are normally returned as regular host-resident `polars.DataFrame`
objects, but cudf-polars internally represents GPU-resident data using
`cudf_polars.containers.DataFrame`.

An IO source may return `cudf_polars.containers.DataFrame` objects directly.
Doing so avoids the host-to-device copy, but restricts the source to
cudf-polars engines, since the default Polars CPU engine cannot consume
`cudf_polars.containers.DataFrame` objects.

The source may yield multiple chunks, which cudf-polars combines into the scan
output (under a streaming engine the chunks are forwarded individually; see
{ref}`io-plugins-sized-chunks`).

## Schema Validation

The columns a source emits (after applying `with_columns`) must match the
registered schema in name, order, and dtype. cudf-polars validates the produced
output against that schema and raises [`polars.exceptions.SchemaError`](https://docs.pola.rs/api/python/stable/reference/exceptions.html)
on a mismatch.

Polars itself only validates when `register_io_source(..., validate_schema=True)`
is used, but that flag is not carried into the GPU plan, so cudf-polars validates
unconditionally. A source that deliberately yields a dtype different from its
declared schema (only valid with `validate_schema=False`) therefore cannot run on
the GPU and must be collected with the default Polars CPU engine. This is tracked
in [cudf#22917](https://github.com/rapidsai/cudf/issues/22917).

## Rank-Aware Sources

With a multi-GPU streaming engine (see {doc}`engines`), every rank runs the scan
function. Plain Python sources are not rank-aware, so cudf-polars executes them
on rank 0 only.

For distributed loading, use a rank-aware source that subclasses
{class}`~cudf_polars.streaming.rank_aware_source.RankAwareSource`. Its
`__call__` method follows the `register_io_source` contract and adds two
optional trailing arguments:

* `rank`: the zero-based rank running the source.
* `nranks`: the total number of ranks in the query.

Both default to `rank=0`, `nranks=1` for single-rank execution, the
in-memory cudf-polars engine, and the default Polars CPU engine.

```python
import polars as pl
from polars.io.plugins import register_io_source
from cudf_polars.streaming.rank_aware_source import RankAwareSource


class PartitionedFrame(RankAwareSource):
    def __init__(self, frame: pl.DataFrame):
        self.frame = frame

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int = 0,
        nranks: int = 1,
    ):
        df = self.frame.gather_every(nranks, offset=rank)
        if predicate is not None:
            df = df.filter(predicate)
        if n_rows is not None:
            # Only reached on the Polars CPU engine (see the Row-Limit Pushdown section).
            df = df.head(n_rows)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df


source = PartitionedFrame(pl.DataFrame({"a": range(10)}))
lf = register_io_source(source, schema={"a": pl.Int64})
```

Pass the `RankAwareSource` instance directly to `register_io_source`; do not
wrap it in another callable. To inject rank information, cudf-polars must
recover the `RankAwareSource` instance from the registered callable. Only an
unwrapped instance is recognized; wrapping it in anything else (for example, a
`functools.partial`, closure, lambda, or decorator) hides the instance, in which
case the source is treated as rank-unaware and executes on rank 0 only.

This limitation is tracked in [cudf#22917](https://github.com/rapidsai/cudf/issues/22917).


(io-plugins-sized-chunks)=
## Streaming Chunks With a Known Count

Under a streaming executor, cudf-polars forwards each chunk produced by an IO
source through the pipeline individually. However, a Polars source is a lazy
iterator whose length is unknown until fully consumed, so cudf-polars normally
has to drain the source before it can start forwarding chunks.

A source that already knows its chunk count can avoid this by returning a
`SizedChunks`: a thin iterator wrapper that also reports its length.
cudf-polars can then stream one chunk at a time, keeping only a single chunk
resident on the GPU at once. Since `SizedChunks` is still a normal iterator, the
default Polars CPU engine handles it unchanged.

```python
import polars as pl
from polars.io.plugins import register_io_source
from cudf_polars.streaming.rank_aware_source import SizedChunks


def source(with_columns, predicate, n_rows, batch_size):
    def chunks():
        # Produced lazily, but the total count is known up front.
        for path in ("a.parquet", "b.parquet"):
            yield pl.read_parquet(path)

    return SizedChunks(2, chunks())


lf = register_io_source(source, schema={"a": pl.Int64})
```

## Row-Limit Pushdown

Polars can push `head` / `limit` operations into a Python scan via the `n_rows`
parameter. cudf-polars does not currently support this pushdown and rejects such
plans during translation. Standard GPU fallback behavior applies: by default the
query falls back to the Polars CPU engine, and in raise-on-fail mode
cudf-polars raises `NotImplementedError`.

As a result, `n_rows` is always `None` on GPU engines. However, IO sources intended
to also work with the default Polars CPU engine must still handle `n_rows` correctly.

Distributed support for a global row limit is tracked in [cudf#22918](https://github.com/rapidsai/cudf/issues/22918).

## Threading

Under a streaming engine, cudf-polars runs IO sources on a worker thread pool.
A source is created on a worker thread, and successive chunks are pulled on
worker threads that may differ from the one that created the source and from
each other. A source must therefore not depend on thread-affine state that is
created up front and reused across chunks, for example a `sqlite3.Connection`
(which by default may only be used on the thread that opened it). Open such
resources inside the function that produces each chunk, or use a thread-safe
equivalent.

## API

```{eval-rst}
.. autoclass:: cudf_polars.streaming.rank_aware_source.RankAwareSource
   :special-members: __call__

.. autoclass:: cudf_polars.streaming.rank_aware_source.SizedChunks
```

[register-io-source]: https://docs.pola.rs/api/python/stable/reference/api/polars.io.plugins.register_io_source.html
[polars-io-plugins]: https://docs.pola.rs/user-guide/plugins/io_plugins/
