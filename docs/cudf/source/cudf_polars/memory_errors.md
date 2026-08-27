(cudf-polars-memory-errors)=
# Understanding Memory Use in the GPU Streaming Engine

The GPU streaming engine is designed to execute queries on datasets that are larger
than GPU memory. This page explains how the engine manages memory, why out-of-memory
errors occur, and how spilling to host memory works, so that you can reason about
and tune memory-related behaviour in your workloads.

## How the engine uses GPU memory

Rather than loading an entire dataset into GPU memory at once, the streaming engine
decomposes input data into a sequence of *chunks* and processes them independently.
`target_partition_size` controls the target size of each chunk in bytes. Chunks flow
through the query graph, being filtered, transformed, aggregated, and joined.
When operations change the size of a chunk, chunks may be split or combined to
be approximately `target_partition_size` bytes.

The default chunk size is the lesser of 1.5 GB and 2.5% of the smallest GPU in the
cluster. For most workloads this leaves enough headroom for intermediate results
without sacrificing compute efficiency, but it is not optimal for every query.
The optimal chunk size is much smaller than total memory of a single GPU. This is
because RapidsMPF overlaps the execution of multiple operations across the actor graph,
and each operation requires temporary allocations for its input, output, and
intermediate data buffers.

## Why out-of-memory errors occur

The `target_partition_size` setting establishes a *target*, not a hard ceiling.
Some situations lead to oversized chunks, or require the engine to hold many
chunks in memory simultaneously.

### Shuffle-heavy operations

A *shuffle* is a cluster-wide redistribution of data so that rows with matching keys
land on the same GPU. During a shuffle each GPU must hold both the data it is sending
and the data it is receiving, which can temporarily increase memory pressure.

Shuffles arise in several common operations. A `sort` over the full dataset must
redistribute rows into a globally ordered sequence. A `group_by` over a column
with many unique values must arrange for every distinct key to be on the same GPU
before aggregating. A `join` must align rows from both tables by their join keys before
comparing them. Queries with skewed value distributions are especially prone to memory
spikes because a single GPU (or chunk) may receive a disproportionately large share of
the shuffled data.

### Buffering for multiple consumers

When the output of an operation is consumed by more than one downstream operation,
the engine must keep that output in memory until all consumers have consumed it.
A query plan with a shared sub-plan or a self-join can therefore hold multiple
chunks in memory simultaneously even if each individual operation is within budget.

### File format constraints

The minimum amount of data the engine can read at one time depends on the file format.
For Parquet, the engine can read as little as one *column chunk* at a time, so files written
with reasonable row-group sizes give the engine fine-grained control over how much data
enters the pipeline at once. For formats that do not support partial reads, such as CSV,
the engine must load an entire file before it can begin processing, which may produce
chunks much larger than `target_partition_size`.

### Concurrent file reads

Each scan node may read more than one input chunk at a time. At least two IO
producer tasks help overlap IO with GPU compute. Larger
`max_concurrent_io_tasks` values may improve high-latency IO throughput but
increase memory use per Scan actor. See
{class}`~cudf_polars.engine.options.StreamingOptions` for current defaults.

## Spilling to host memory

When GPU memory pressure rises above a configurable threshold
(`RAPIDSMPF_SPILL_DEVICE_LIMIT`, default `80%`), the engine begins *spilling*: moving
chunks that are not currently being processed from GPU to host (CPU) memory. Once
the operation that needs them resumes, the chunks are then unspilled back to the GPU.

Spilling lets the engine handle datasets larger than GPU memory, but every spill and
unspill requires a GPU-to-host data transfer, and the operation depending on that
chunk must wait until the transfer completes.

If spilling is occurring but the GPU still runs out of memory, it usually means the
peak allocation within a single operation is too large. Reducing `target_partition_size`
makes each operation smaller, and lowering `RAPIDSMPF_SPILL_DEVICE_LIMIT` causes
spilling to start earlier, leaving more headroom for those peaks.

## Pinned memory and transfer speed

By default, host memory used for spilling is *pageable*. The operating system may
swap it to disk, and GPU-to-host transfers must go through an extra copy. *Pinned*
(page-locked) memory bypasses this: the GPU DMA engine can transfer directly to and
from it, significantly increasing bandwidth.

The trade-off is that registering pinned memory with the driver has a substantial
up-front cost. Pinned memory also reduces the amount of host memory available for
other work on the system: pinned pages cannot be swapped out or migrated by the
operating system. These costs are therefore only worthwhile if spilling is a clear
bottleneck in your workflow, or if you are executing many queries in the same session.
Consequently, if we want to spill to pinned memory we must explicitly opt in when
constructing the GPU engine for queries.

## Configuration reference

| Option | Default | Effect |
|---|---|---|
| `target_partition_size` (executor option or `CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE`) | 1.5 GB or 2.5% of smallest GPU | Target chunk size in bytes. Smaller values reduce peak memory at some cost to compute efficiency. |
| `max_concurrent_io_tasks` (executor option or `CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS`) | auto | Number of concurrent IO producer tasks for each scan node. Larger values may improve high-latency IO throughput but increase memory pressure. |
| `RAPIDSMPF_SPILL_DEVICE_LIMIT` | `80%` | GPU memory fraction at which spilling begins. Lower values give more headroom for peaks. |
| `RAPIDSMPF_PINNED_MEMORY` | disabled | Set to `true` to enable pinned host memory for spill buffers. |
| `RAPIDSMPF_PINNED_INITIAL_POOL_SIZE` | (none) | Size of the pinned memory pool to pre-allocate (e.g. `32GB`). |

For the full list of engine configuration options, including `target_partition_size`
and `max_concurrent_io_tasks`, see {doc}`options`. For the full list of memory
and spill configuration options see the [RapidsMPF configuration reference](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/#general).
