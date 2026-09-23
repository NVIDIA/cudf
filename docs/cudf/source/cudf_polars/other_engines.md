(cudf-polars-other-engines)=
# Other Engines

The examples in {doc}`usage` use
{class}`~cudf_polars.engine.ray.RayEngine`. The pages below cover
other ways to run cudf-polars:

* **{doc}`dask_engine`** runs on a [Dask distributed][dask] cluster with one Dask worker per
  GPU. Use this when you already have a Dask deployment or a preferred Dask launcher.
* **{doc}`spmd_engine`** is single program, multiple data: the same script runs once per GPU,
  typically launched with `rrun`. Single-rank mode needs no external cluster at all.
* **{doc}`default_singleton_engine`** documents what `engine="gpu"` does under the hood when no
  engine is constructed explicitly. Useful to *understand*, but for any non-trivial workflow we
  recommend constructing an explicit engine so you can pass {class}`~cudf_polars.engine.options.StreamingOptions`.
* **{doc}`in_memory_engine`** (`engine=pl.GPUEngine(executor="in-memory")`) is the only non-streaming
  path. Suitable for small queries (data that fits in device memory), debugging, or when you specifically
  need `LazyFrame.profile`.

See {doc}`engines` for the conceptual comparison with `RayEngine` (cluster model, runtime
dependencies, typical use), and {doc}`options` for the shared
{class}`~cudf_polars.engine.options.StreamingOptions` configuration
(the in-memory engine does not accept `StreamingOptions`).

```{toctree}
:maxdepth: 1
:hidden:

dask_engine
spmd_engine
default_singleton_engine
in_memory_engine
```

[dask]: https://distributed.dask.org/en/stable/
