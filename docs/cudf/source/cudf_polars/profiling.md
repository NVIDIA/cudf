(cudf-polars-profiling)=
# Profiling and Tracing

## Streaming Statistics

When a query runs on a streaming engine
({class}`~cudf_polars.engine.ray.RayEngine`,
{class}`~cudf_polars.engine.dask.DaskEngine`,
{class}`~cudf_polars.engine.spmd.SPMDEngine`, or the default
`engine="gpu"`), the underlying streaming runtime can record detailed per-rank statistics:
shuffle byte counts, allgather participation, memory-pool high-water marks, and more. See the
[underlying statistics reference][rapidsmpf-stats] for the full list of metrics.

Statistics collection is off by default. Enable it by setting `statistics=True` on
{class}`~cudf_polars.engine.options.StreamingOptions` (or exporting
`RAPIDSMPF_STATISTICS=1`), then call `gather_statistics()` on the engine to pull the per-rank
records:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(statistics=True)

with RayEngine.from_options(opts) as engine:
    result = (
        pl.scan_parquet("/data/*.parquet")
          .group_by("customer_id")
          .agg(pl.col("amount").sum())
          .collect(engine=engine)
    )

    per_rank = engine.gather_statistics(clear=True)
    for rank, stats in enumerate(per_rank):
        print(f"rank {rank}:\n{stats}")
```

`gather_statistics(*, clear=False)` returns a list of `rapidsmpf.statistics.Statistics` objects,
one per rank, in rank order. Passing `clear=True` resets each rank's counters after the gather —
useful when you want to scope statistics to a single query.

Use `global_statistics(*, clear=False)` when you only need the cluster-wide picture. It gathers
and merges the per-rank statistics into a single `Statistics` (counts and values summed, maxima
reduced with `max`). Capture it inside the engine context, then print after exit:

```python
import polars as pl
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.ray import RayEngine

opts = StreamingOptions(statistics=True)

with RayEngine.from_options(opts) as engine:
    result = pl.scan_parquet("/data/*.parquet").collect(engine=engine)
    total = engine.global_statistics(clear=True)
print(total)
```


## GPU Profiling

For streaming queries, we recommend profiling with [NVIDIA NSight Systems][nsight]. `cudf-polars`
includes [nvtx][nvtx] annotations to help you understand where time is being spent. Streaming
engines do not support `LazyFrame.profile`, since `profile` requires a single in-memory pass.

If you specifically need [`LazyFrame.profile`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.profile.html),
the in-memory engine supports it. This is useful for small queries during development:

```python
import polars as pl
q = pl.scan_parquet("ny-taxi/2024/*.parquet").filter(pl.col("total_amount") > 15.0)
profile = q.profile(engine=pl.GPUEngine(executor="in-memory"))
```

The result is `(result_df, timings_df)`, see the Polars docs link above for the schema.

## Tracing

cudf-polars can optionally trace execution of each node in the query plan. To enable tracing, set
the environment variable ``CUDF_POLARS_LOG_TRACES`` to a true value ("1", "true", "y", "yes")
before starting your process.

cudf-polars logs traces at three scopes (levels):

1. `plan`: These generally happen once per query. This will include things like the (serialized)
   query plan.
2. `actor`: (streaming engines only). There will be roughly one `actor` trace per node in the
   logical plan.
3. `evaluate_ir_node`: Logs the evaluation of a physical node in the query plan. Note that one
   logical node might expand to more than one physical nodes.

Each trace includes a `scope` key indicating which level that trace belongs to. `actor`-scoped
nodes will be nested under a `plan`-scoped node. When using a streaming engine,
`evaluate_ir_node`-scoped nodes will be nested under an `actor`-scoped node.

### Schemas

The different scopes have different schemas. Fields in **bold** are required / always present.

#### scope=plan

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope**  | Literal["plan"] | The string literal `"plan"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **plan**   | `PlanObject` | A serialized representation of the query plan. |
| **event**  | String | A message like "Query Plan" |

#### scope=actor

`actor`-scoped traces only appear when running on a streaming engine.

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope**      | Literal["actor"] | The string literal `"actor"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **start**      | int   | A nanosecond-resolution counter indicating when the actor started. Note: actors generally start early in the query and suspend waiting for data. |
| **stop**      | int   | A nanosecond-resolution counter indicating when the actor completed. |
| **event**      | String | A message like "Streaming Actor". |
| **actor_ir_type** | String | The type of the actor, like `"Scan"`. |
| **actor_ir_id**   | int    | A unique identifier for the actor. All traces logged under this actor will include this value. |
| chunk_count | int | A counter for how many table chunks have been processed by this actor at the time of logging. |
| duplicated | bool | Whether the output rows are duplicated across ranks (e.g. after an allgather). |
| row_count       | int  | Total row count produced by this node during execution. |

#### scope=evaluate_ir_node

| Field Name | Type  | Description |
| ---------- | ----- | ----------- |
| **scope** | `Literal["evaluate_ir_node"]` | The string literal `"evaluate_ir_node"`. Useful for distinguishing from other types of traces. |
| **cudf_polars_query_id** | UUID4 | A unique identifier for the polars query being executed. All traces logged as part of this query use this ID. |
| **type**       | string | The name of the IR node |
| **start**      | int    | A nanosecond-precision counter indicating when this node started executing |
| **stop**       | int    | A nanosecond-precision counter indicating when this node finished executing |
| **overhead_duration**   | int    | The overhead, in nanoseconds, added by tracing |
| `count_frames_{phase}` | int | The number of dataframes for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_DATAFRAMES=0`. |
| `frames_{phase}` | `list[dict]` | A list with dictionaries with "shape" and "size" fields, one per input dataframe, for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_DATAFRAMES=0`. |
| `total_bytes_{phase}` | int | The sum of the size (in bytes) of the dataframes for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_current_bytes_{phase}` | int | The current number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_current_count_{phase}` | int | The current number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_peak_bytes_{phase}` | int | The peak number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_peak_count_{phase}` | int | The peak number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_total_bytes_{phase}` | int | The total number of bytes allocated by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `rmm_total_count_{phase}` | int | The total number of allocations made by RMM Memory Resource used by cudf-polars for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| `nvml_current_bytes_{phase}` | int | The device memory usage of this process, as reported by NVML, for the input / output `phase`. This metric can be disabled by setting `CUDF_POLARS_LOG_TRACES_MEMORY=0`. |
| actor_ir_id   | int    | A unique identifier for the parent actor (streaming engines only). |

Setting `CUDF_POLARS_LOG_TRACES=1` enables all the metrics. Depending on the query, the overhead
from collecting the memory or dataframe metrics can be measurable. You can disable some metrics
through additional environment variables. For example, to disable the memory-related metrics, set:

```bash
CUDF_POLARS_LOG_TRACES=1 CUDF_POLARS_LOG_TRACES_MEMORY=0
```

And to disable the memory and dataframe metrics, which essentially leaves just the duration
metrics, set
```bash
CUDF_POLARS_LOG_TRACES=1 CUDF_POLARS_LOG_TRACES_MEMORY=0 CUDF_POLARS_LOG_TRACES_DATAFRAMES=0
```

Note that tracing still needs to be enabled with `CUDF_POLARS_LOG_TRACES=1`.

The implementation uses [structlog] to build log records. You can configure the output using
structlog's [configuration][structlog-configure] and enrich the records with
[context variables][structlog-context].

```python
>>> df = pl.DataFrame({"a": ["a", "a", "b"], "b": [1, 2, 3]}).lazy()
>>> df.group_by("a").agg(pl.col("b").min().alias("min"), pl.col("b").max().alias("max")).collect(engine=pl.GPUEngine(executor="in-memory"))
2025-09-10 07:44:01 [info     ] Execute IR      count_frames_input=0 count_frames_output=1 ... type=DataFrameScan
2025-09-10 07:44:01 [info     ] Execute IR      count_frames_input=1 count_frames_output=1 ... type=GroupBy
shape: (2, 3)
┌─────┬─────┬─────┐
│ a   ┆ min ┆ max │
│ --- ┆ --- ┆ --- │
│ str ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ b   ┆ 3   ┆ 3   │
│ a   ┆ 1   ┆ 2   │
└─────┴─────┴─────┘
```

[nsight]: https://developer.nvidia.com/nsight-systems
[nvtx]: https://nvidia.github.io/NVTX/
[rapidsmpf-stats]: https://docs.rapids.ai/api/rapidsmpf/nightly/statistics/
[structlog]: https://www.structlog.org/en/stable/
[structlog-configure]: https://www.structlog.org/en/stable/configuration.html
[structlog-context]: https://www.structlog.org/en/stable/contextvars.html
