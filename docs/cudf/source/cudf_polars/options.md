(cudf-polars-options)=
# Configuration Options

{class}`~cudf_polars.engine.options.StreamingOptions` is the recommended
way to configure the streaming engines (Ray, Dask, SPMD. The default `engine="gpu"` accepts no
options, see the note below). Build one and pass it to `RayEngine.from_options()`
to construct a {class}`~cudf_polars.engine.ray.RayEngine`:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(
    num_streaming_threads=8,
    fallback_mode="silent",
    spill_device_limit="70%",
)

with RayEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
          .filter(pl.col("amount") > 100)
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )
```

```{note}
`engine="gpu"` (the default when no engine is constructed) accepts no
{class}`~cudf_polars.engine.options.StreamingOptions`. Many of the
fields below have a noticeable runtime impact (for example `spill_to_pinned_memory=True`
significantly speeds up spill-heavy workflows), so to use any non-default value construct one
of the engines listed below.
```

{class}`~cudf_polars.engine.options.StreamingOptions` covers three
categories of fields:

| Category    | Scope                                                                                  | Env var prefix            |
| ----------- | -------------------------------------------------------------------------------------- | ------------------------- |
| `executor`  | Query execution and partitioning, e.g. `max_rows_per_partition`, `fallback_mode`, ...  | `CUDF_POLARS__EXECUTOR__` |
| `engine`    | `pl.GPUEngine` kwargs, e.g. Parquet, memory resource, CUDA streams, hardware binding   | `CUDF_POLARS__`           |
| `rapidsmpf` | Streaming runtime, e.g. threads, CUDA streams, spilling, pinned memory, log level      | `RAPIDSMPF_`              |

The `engine` category surfaces the same tuning knobs as plain `pl.GPUEngine(...)`. For example,
`parquet_options` and `memory_resource_config`. Configure these settings through
{class}`~cudf_polars.engine.options.StreamingOptions` rather than
passing them to `pl.GPUEngine(...)` directly.

The `rapidsmpf` category adds lower-level configuration for the streaming runtime that has no equivalent on
the plain `pl.GPUEngine`. Most users will not need to touch these directly. See the
[streaming runtime configuration reference][rapidsmpf-config] for the underlying meaning of each
`RAPIDSMPF_*` field.

Every option has a corresponding environment variable. When an option is not set explicitly, its
value is read from the environment variable if present; otherwise the underlying library applies
its built-in default. Boolean environment variables accept `{"1", "true", "yes", "y"}` as true
and `{"0", "false", "no", "n"}` as false.


## Building from a dictionary

{meth}`~cudf_polars.engine.options.StreamingOptions.from_dict` accepts a flat dict of field names.
Unknown keys raise `TypeError` and `None` values leave the field unspecified:

```python
opts = StreamingOptions.from_dict({
    "num_streaming_threads": 8,
    "fallback_mode": "silent",
})
```

This is convenient when options come from a config file or CLI.

## Engine keyword arguments

Each engine ({class}`~cudf_polars.engine.ray.RayEngine`,
{class}`~cudf_polars.engine.dask.DaskEngine`, or
{class}`~cudf_polars.engine.spmd.SPMDEngine`) accepts
`rapidsmpf_options`, `executor_options`, and `engine_options` as raw keyword arguments.
We recommend using this only when you need fine-grained control that doesn't fit the
{class}`~cudf_polars.engine.options.StreamingOptions` schema.
Otherwise, prefer the engine's `from_options` constructor with
{class}`~cudf_polars.engine.options.StreamingOptions`.

For the in-memory engine,
{class}`~cudf_polars.engine.options.StreamingOptions` does not apply.
See {doc}`in_memory_engine` for how to configure it.


## Options Reference

Environment variables follow these patterns:

* `executor`: `CUDF_POLARS__EXECUTOR__<OPTION_NAME>` (e.g. `CUDF_POLARS__EXECUTOR__FALLBACK_MODE`)
* `engine`: `CUDF_POLARS__<OPTION_NAME>` (e.g. `CUDF_POLARS__RAISE_ON_FAIL`; nested prefixes for structured options)
* `rapidsmpf`: `RAPIDSMPF_<OPTION_NAME>` (e.g. `RAPIDSMPF_NUM_STREAMING_THREADS`)

### Category: `executor`

| Field                    | Description                                                                                                                                         | Default     |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `num_py_executors`       | Workers for the internal Python `ThreadPoolExecutor`.                                                                                               | `8`         |
| `fallback_mode`          | When an unsupported operation forces a fallback to CPU execution: `"warn"`, `"raise"`, `"silent"`.                                                  | `"warn"`    |
| `max_rows_per_partition` | Maximum number of rows per partition. Only used for in-memory `DataFrame` sources, never for disk IO or dynamic planning.                           | `1_000_000` |
| `broadcast_limit`        | Maximum number of bytes for broadcast joins.                                                                                                        | auto        |
| `target_partition_size`  | Target partition size in bytes. Used for IO and dynamic planning. `0` means auto.                                                                   | auto        |
| `dynamic_planning`       | Dynamic planning configuration, dict or {class}`~cudf_polars.utils.config.DynamicPlanningOptions`. `None` disables.                                 | enabled     |
| `join_filter_pushdown`   | Configuration for join filter pushdown plan rewrites, dict or {class}`~cudf_polars.utils.config.JoinFilterPushdownOptions`. `None` disables.        | enabled     |
| `sink_to_directory`      | Whether `.sink_*()` writes its output as a directory. The `spmd`, `ray`, and `dask` engines always use `True`; passing `False` raises `ValueError`. | `True`      |

### Category: `engine`

| Field                    | Description                                                                                                                   | Default                   |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| `raise_on_fail`          | Raise an error instead of falling back to CPU execution.                                                                      | `False`                   |
| `parquet_options`        | Parquet configuration, dict or {class}`~cudf_polars.utils.config.ParquetOptions`.                                             | —                         |
| `memory_resource_config` | RMM configuration, dict or {class}`~cudf_polars.utils.config.MemoryResourceConfig`.                                           | —                         |
| `hardware_binding`       | Hardware binding policy. Pass a {class}`~cudf_polars.engine.hardware_binding.HardwareBindingPolicy` for fine-grained control. | `HardwareBindingPolicy()` |
| `allow_gpu_sharing`      | When `False` (default), the engine raises if multiple ranks share the same physical GPU.                                      | `False`                   |

### Category: `rapidsmpf`

Lower-level streaming runtime knobs. Most users will not need to touch these directly. See the
[streaming runtime configuration reference][rapidsmpf-config] for the full list of fields and defaults.

## Developer Options

These environment variables are intended for library developers and advanced users.

| Environment variable        | Description                                                                                                    | Default |
|-----------------------------|----------------------------------------------------------------------------------------------------------------|---------|
| `CUDF_POLARS_WARN_UNSTABLE` | Raises a `cudf_polars.UnstableWarning` whenever an unstable cudf-polars feature is used. Set to `1` to enable. | `0`     |

<!-- Reference links -->
[rapidsmpf-config]: https://docs.rapids.ai/api/rapidsmpf/nightly/configuration/
