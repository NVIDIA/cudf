(cudf-polars-in-memory-engine)=
# In-memory engine

The in-memory engine (`engine=pl.GPUEngine(executor="in-memory")`) is
the only non-streaming path in cudf-polars. It materializes the whole query in device memory
on a single GPU.

For most workflows, prefer a streaming engine. Use the in-memory engine when:

- The data comfortably fits in device memory and you want minimum setup.
- You need `LazyFrame.profile` (see {doc}`profiling`).
- You are debugging and want the simpler, non-streaming execution path.

```python
result = query.collect(engine=pl.GPUEngine(executor="in-memory"))
```

This is the path documented in Polars' own [GPU support guide][polars-gpu]. By contrast,
`engine="gpu"` (or `engine=pl.GPUEngine()`) selects the default streaming path on a single GPU
(see {doc}`default_singleton_engine`). That default accepts no options, so for anything beyond
a quick script, construct an explicit engine.

## Configuration

The in-memory engine does not accept
{class}`~cudf_polars.engine.options.StreamingOptions`. Pass keyword
arguments to `pl.GPUEngine(...)` directly:

```python
import polars as pl

engine = pl.GPUEngine(executor="in-memory", parquet_options={"chunked": True})
```

See the [Polars GPU support guide][polars-gpu] for the full in-memory usage story.

[polars-gpu]: https://docs.pola.rs/user-guide/gpu-support/
