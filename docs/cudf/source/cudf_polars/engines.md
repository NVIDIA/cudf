(cudf-polars-engines)=
# Engines

## What is an engine?

`cudf-polars` executes Polars `LazyFrame` queries on GPU. You select GPU execution by passing an
`engine=` argument to `.collect()` or `.sink_*()`. The `engine` you pass decides *how* the
query runs: whether it streams through partitioned inputs or fits everything in device memory,
whether it runs in-process or distributes work across a cluster of GPU workers, and which
cluster backend coordinates those workers.

## Execution modes

### Streaming

Streaming engines partition their inputs (Parquet files or in-memory `DataFrame`s) and process
those partitions through the query graph in chunks. This lets queries scale past device memory
and (on Ray, Dask, and SPMD) across multiple GPUs and multiple nodes. cudf-polars' streaming
executor is its own GPU implementation, but conceptually parallels
[Polars' CPU streaming engine](https://docs.pola.rs/user-guide/concepts/streaming/): the same
partition-and-stream model, just on the GPU.

All four ways of running cudf-polars use this same streaming executor:
{class}`~cudf_polars.engine.ray.RayEngine`,
{class}`~cudf_polars.engine.dask.DaskEngine`,
{class}`~cudf_polars.engine.spmd.SPMDEngine`, and the default
`engine="gpu"` (backed internally by
{class}`~cudf_polars.engine.default_singleton_engine.DefaultSingletonEngine`).
They differ only in how their GPU worker(s) are provisioned.
{class}`~cudf_polars.engine.ray.RayEngine` with no arguments uses every
GPU visible to the process, so on a single node with N GPUs it runs the query on all N of them
without any extra configuration. Launching a multi-node cluster simply means pointing the
engine at that cluster; the user-facing code is the same.

### In-memory

The in-memory engine (`engine=pl.GPUEngine(executor="in-memory")`) is
the only non-streaming path. It runs the query on a single GPU, materializing intermediates in
device memory. Use it for small queries (data that fits in device memory), debugging, or when
you specifically need `LazyFrame.profile` support (see {doc}`profiling`). For production
workloads on any nontrivial dataset, use a streaming engine. See {doc}`in_memory_engine` for
details.

## Cluster backends

| Engine                                        | Cluster model                                           | Extra runtime dependency | Typical use                                                                     |
| --------------------------------------------- | --------------------------------------------------------| ------------------------ | ------------------------------------------------------------------------------- |
| {class}`~cudf_polars.engine.ray.RayEngine`    | Single client, one Ray actor per GPU                    | [Ray][ray-docs]          | Works from a laptop to a cloud cluster. No separate cluster setup needed.       |
| {class}`~cudf_polars.engine.dask.DaskEngine`  | Single client, one Dask worker per GPU                  | [Dask distributed][dask] | Teams with an existing Dask deployment or a preferred Dask launcher.            |
| {class}`~cudf_polars.engine.spmd.SPMDEngine`  | Same script runs once per GPU, joined by a communicator | UCXX (under `rrun`)      | HPC / SPMD launchers such as `rrun`. Single-rank mode needs no cluster at all.  |
| [`engine="gpu"`](default_singleton_engine.md) | Implicit process-wide singleton on one GPU; no cluster  | None                     | Default when no engine is constructed. Short scripts and notebooks. No options. |

All four approaches use the same execution model under the hood, so which to select depends
on your preferred deployment method, not performance tradeoffs. For any non-trivial workflow,
construct one of the first three engines explicitly (see {doc}`usage`). `engine="gpu"` is a
convenience and accepts no options, so it cannot be tuned. See {doc}`default_singleton_engine`
for details on the implementation that backs it.

## Result collection

`.collect()` returns a single `pl.DataFrame` on the **caller's process**. On the streaming
engines that has two flavors:

- **`RayEngine` / `DaskEngine`** (single client): every partition is pulled from the
  cluster workers back to the client and concatenated there. This is convenient for small
  results but does not scale to large queries. E.g., calling `.collect()` on a 1 TB query
  result sends 1 TB through your client. Sink the result
  (`.sink_parquet("path/")`, `.sink_csv(...)`, …) so each rank writes its own partition
  directly, or reduce/sample the data inside the query before `.collect()`.
- **`SPMDEngine`** (one process per GPU): each rank's `.collect()` returns *that rank's*
  local fragment. There is no client to gather to. If you need a single concatenated
  `pl.DataFrame` across ranks, call
  {func}`~cudf_polars.engine.spmd.allgather_polars_dataframe` explicitly (see
  [Collecting distributed results](spmd_engine.md#collecting-distributed-results)). If you
  want to keep processing the data rank-by-rank, just stay in `SPMDEngine` and use its
  MPI-style model: each rank already owns its fragment.
- **`engine="gpu"`**: single GPU, no cluster, so `.collect()` is the only sensible option.

Rules of thumb for multi-machine `RayEngine` / `DaskEngine` runs:

- For exports: prefer `.sink_*()` over `.collect()`.
- For analysis: aggregate, sample, or `limit()` the result inside the lazy query before
  `.collect()` so the client only sees a small DataFrame.
- For further distributed processing in Python: switch to `SPMDEngine` so each rank keeps
  its fragment.

## Where to go next

- {doc}`usage`: tutorial that walks through running your first GPU query end-to-end.
- {doc}`other_engines`: per-engine reference pages for DaskEngine and SPMDEngine.
- {doc}`options`: the `StreamingOptions` configuration object and every field it surfaces.

[ray-docs]: https://docs.ray.io/en/latest/
[dask]: https://distributed.dask.org/en/stable/
