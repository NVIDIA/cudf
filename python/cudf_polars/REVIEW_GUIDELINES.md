# cudf_polars Review Guidelines

These guidelines apply only to `python/cudf_polars`. They complement
`python/REVIEW_GUIDELINES.md` as library specific guidelines.

Use the severity labels from `python/REVIEW_GUIDELINES.md` when you comment.
Code examples are simplified mock code. Always report violations in this guide.

## CUDA Streams

### Pass `stream` to every pylibcudf call

Every pylibcudf operation that accepts a `stream` argument and can enqueue or
synchronize device work must receive that argument explicitly. Do not rely on
the default stream for those operations. This also covers calls
that do not look like kernels:
`Scalar.to_py`, `Scalar.from_py`, `Column.obj_scalar`, `Column.astype`,
and `plc.Table.from_arrow`.

```python
# Bad: these silently use the default stream
value = result.to_py()
lit = plc.Scalar.from_py(0, dtype.plc_type)

# Good
value = result.to_py(stream=stream)
lit = plc.Scalar.from_py(0, dtype.plc_type, stream=stream)
```

### Run work on several inputs inside `stream_ordered_after`

Inputs can live on different streams. `stream_ordered_after` orders the new
stream after every input and joins it back before the inputs are freed. A
bare pool stream, or `get_joined_cuda_stream`, misses one of the two joins.
The input is then freed while kernels that read it are still queued.

```python
# Bad: inputs can be freed while concatenate is still queued
stream = ir_context.get_cuda_stream()
table = plc.concatenate.concatenate([df.table for df in dfs], stream=stream)

# Good
with ir_context.stream_ordered_after(*dfs) as stream:
    table = plc.concatenate.concatenate([df.table for df in dfs], stream=stream)
```

Related rules:

- Evaluate kernel-launching key expressions on the input's own stream
  *before* entering the joined block.
- Single-input work stays on `chunk.stream`. Do not use
  `stream_ordered_after` with one upstream.
- Join every stream that used a buffer or scalar before freeing it,
  including on the error path.
- Get streams from `ir_context.get_cuda_stream()`. Do not construct `Stream()`.

### Every `stream.synchronize()` needs a reason

Synchronize only before freeing host memory used by an async copy, or
before handing device memory to a consumer that does not know about
streams. A sync on the actor event loop blocks every actor. `to_polars()`
already synchronizes.

```python
# Bad: redundant, to_polars() already synchronized
out = df.to_polars()
stream.synchronize()

# Good: an external library reads the buffer without a stream
stream.synchronize()  # torch reads this memory on its own stream
tensor = torch.from_dlpack(column)
```

### Document stream requirements of helpers

A helper that takes device columns must say which stream its inputs must
be valid on, and whether it syncs to the host internally. A collective
helper must say that every rank has to call it.

## Streaming Actors

### Pass every channel to `shutdown_on_error`, grouped by role

This includes channels the actor creates itself: `Lineariser` inputs,
replay channels, and side or filter channels. Side channels go in
`chs_aux`. Always pass `ir_context`.

```python
# Bad: the replay channel is not shut down on error, so peers hang
ch_replay = context.create_channel()
async with shutdown_on_error(context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir):
    ...

# Good
ch_replay = context.create_channel()
async with shutdown_on_error(
    context,
    chs_in=(ch_in,),
    chs_out=(ch_out,),
    chs_aux=(ch_replay,),
    trace_ir=ir,
    ir_context=ir_context,
) as actor_scope:
    ...
```

### Offload blocking work with `ir_context.to_thread`

Run blocking or native work from an actor with `ir_context.to_thread`. It
reuses the shared executor and drains the future on cancellation. Do not
create new thread pools or call `loop.run_in_executor` directly.

```python
# Bad: a new pool per call, and the future is not drained on cancellation
with ThreadPoolExecutor() as pool:
    result = await loop.run_in_executor(pool, ir.do_evaluate, *args)

# Good
result = await ir_context.to_thread(ir.do_evaluate, *args, context=ir_context)
```

### Keep message sequence numbers

Actors that buffer or replay messages must keep the input
`sequence_number`; use `replay_buffered_channel`. Two-input chunkwise
actors must assert that both channels end together and that paired
sequence numbers match. More than one producer into a channel needs a
`Lineariser`.

```python
# Bad: renumbers, so downstream order-sensitive actors reorder rows
for i, msg in enumerate(buffered):
    await ch_out.send(Message(i, msg.payload))

# Good
assert left.sequence_number == right.sequence_number
```

### Reuse shared streaming helpers

Use `replay_buffered_channel`, `ChunkStore`, `names_to_indices`,
`empty_table_chunk`, and `make_empty_column` instead of hand-written
copies. Do not special-case `nranks == 1`: actors and collectives must be
correct for one rank without a separate branch.

## Device Memory and Spilling

### Reserve memory asynchronously, sized to the real allocation

Every device allocation in an actor (unspill, pack, sort, concat, key
evaluation) must be covered by an awaited reservation. A synchronous
`make_available_and_spill(..., allow_overbooking=True)` blocks every actor
on the event loop. `net_memory_delta` and `reserve_extra` must match the
real memory effect. Use no unexplained multipliers, and no extra
reservation for a single-chunk concat.

```python
# Bad: blocks the event loop and can overbook
table = chunk.make_available_and_spill(br, allow_overbooking=True)

# Good
with opaque_memory_usage(
    await reserve_memory(context, size=input_bytes, net_memory_delta=0)
):
    df = await ir_context.to_thread(ir.do_evaluate, *args, *dfs, context=ir_context)
```

### Do not defeat spilling

- Do not unspill a `TableChunk` only to read its size or row count. Both
  are available while the chunk is spilled.
- Delete the local `DataFrame` after wrapping it in an
  `exclusive_view=True` `TableChunk`.
- Release device chunks before long host-only work.
- Finish reading `chunk.table_view()` before moving the chunk into a
  `Message` or `ChunkStore`. Spilling can free the buffer after that.

```python
# Bad: unspills just to count rows
n = chunk.make_available(...).table_view().num_rows()

# Good
chunk = TableChunk.from_pylibcudf_table(
    df.table, stream, exclusive_view=True, br=context.br()
)
del df
```

### Install and restore the RMM resource with the shared helper

Swap memory resources only with the `set_memory_resource` context manager,
which restores the previous one in `finally`. Engine `reset()` and reuse
paths must not rebuild the memory resource, or reuse the adaptor of a
`Context` that was shut down. Pass `br=context.br()` to rapidsmpf chunk
and message factories.

```python
# Bad
rmm.mr.set_current_device_resource(mr)

# Good
with set_memory_resource(mr):
    ...
```

## rapidsmpf Collectives

### Always finish collective inserts

If a shuffle, allgather, or `SparseAlltoall` is never told that insertion
is finished, the worker aborts. Finish insertion on every path, including
errors and cancellation. Clear temporary `ChunkStore`s in `finally`.

```python
# Bad: an exception in the loop skips insert_finished()
for msg in messages:
    shuffler.insert(msg)
shuffler.insert_finished()

# Good
async with manager.inserting() as inserter:
    for msg in messages:
        inserter.insert(msg)
```

### Collective IDs are deterministic and per query

Register each new collective-using IR type with `ReserveOpIDs`. The number
of reserved IDs must not depend on `nranks` or runtime state, because every
rank must get the same IDs. Do not share an ID between concurrent
collectives, and do not release an ID before its collective completes.

### Every rank builds and runs the same graph

A branch that decides whether to enter a collective (barrier, allgather,
context reset) must use a value every rank agrees on. Lowering and ID
assignment must be deterministic. Do not key IR state by `id(node)`: nodes
are rebuilt on workers. Use `get_stable_id()` or `get_stable_plan_id()`.

```python
# Bad: rank 0 has data and enters the allgather; rank 1 has none and waits forever
if local_rows > 0:
    stats = await allgather(local_stats)

# Good: every rank takes part, with empty input if needed
stats = await allgather(local_stats)
```

Multi-input actors must receive metadata from all channels at once with
`gather_in_task_group`, not one after another; shared fanouts otherwise
deadlock.

## Partitioning and Ordering Metadata

### Remap or drop partitioning when columns change

A node that reorders, adds, drops, or recomputes columns must remap
`HashScheme`/`OrderScheme` indices with `maybe_remap_partitioning`, or drop
the scheme. Do not hard-code `partitioning=None` for outputs that keep the
key order. Never remap by output name: aggregations reuse input names.

```python
# Bad: column indices changed, but the metadata still points at old indices
await send_metadata(ch_out, ChannelMetadata(partitioning=input_meta.partitioning))

# Good
partitioning = maybe_remap_partitioning(ir, input_meta.partitioning, context=context)
```

### Skip a shuffle only on a strict key match

Skipping a shuffle, sort, or reduction for input that is "already
partitioned" requires the scheme's keys to match the node's keys, in
order. For `OrderScheme`, `order` and `null_order` must match too. A prefix
match needs `strict_boundaries`. A chunkwise join needs both sides
hash-partitioned.

```python
# Bad: partitioned on (a, b), but the groupby keys are (b, a) or (a,)
if meta.partitioning is not None:
    return local_groupby(...)

# Good
if partitioning_matches(meta.partitioning, keys=ir.keys, strict=True):
    return local_groupby(...)
```

(`partitioning_matches` is an illustrative name.)

### Ordering and sortedness claims must be provable

- Mark an `Ordering` strict only if the boundaries are unique.
- Assemble ordered partitions by concatenating key-range-ordered chunks,
  not with a merge that can reorder equal keys.
- Utilities that depend on ordering must raise when the requested ordering
  is not met.

### Propagate sortedness flags only when the operation keeps the order

- Call `set_sorted`/`sorted_like` only on results of order-preserving
  operations.
- Flag a result of an order-preserving operation, such as a filter, a
  slice, or a copy, that drops the sortedness of its input when it could
  keep it with `sorted_like`.

```python
# Bad: replace can break the order
return Column(result, name=col.name).sorted_like(col)

# Bad: a filter keeps the order, but the result loses the sorted flag
return Column(filtered, name=col.name)

# Good: propagate only through order-preserving operations
return Column(col.obj.copy(), name=col.name).sorted_like(col)
```

## Multi-rank Semantics

### Dataset-wide operations are not per rank

Global `Slice`, `head`, and `n_rows` limits must not be copied into
per-rank or per-chunk nodes: slices do not commute with distribution.
Order-sensitive operations (ordered concat, `first`/`last`, `over()`
without `order_by`) must keep global row order, so tie-break on the
original row position. Gather per-rank results in rank order.

```python
# Bad: each rank returns its own first 10 rows, so 10 * nranks rows in total
lowered = Slice(schema, offset=0, length=10, child=local_scan)

# Good: slice once, after the global order is known
lowered = Slice(schema, offset=0, length=10, child=GlobalOrderedConcat(local_scan))
```

### Join strategies must not drop rows

Broadcast a single-partition side whenever the other side has more than
one partition. Checking only one side drops rows silently on peer ranks.
Reject hash-partitioned joins on computed (non-`Col`) keys. Do not pre-filter
the side whose rows must be preserved (left of a left join, either side of
a full join).

```python
# Bad
if left_count == 1:
    return broadcast_join(...)

# Good
if left_count == 1 and right_count == 1:
    return local_join(...)
if left_count == 1 or right_count == 1:
    return broadcast_join(...)
```

### Empty ranks and side-effecting actors

- A rank or channel with no chunks must still send a correctly typed empty
  chunk (`empty_table_chunk(ir, context, stream)`) and still take part in
  collectives.
- Sink, write, and callback actors must act on one rank only when the input
  metadata is `duplicated`. The other ranks must still drain their input.

## Translation

### GPU results must match CPU Polars

Every Polars IR node and expression that the translator accepts must
produce the same result on the GPU as on CPU Polars.

## Lowering and IR Structure

### Keep row alignment in streaming decomposition

Build `HConcat` with `hconcat_streaming_safe(...)`, not a hard-coded
`streaming_safe=True`. Non-pointwise expressions on multi-partition input
must be lowered or raise `NotImplementedError` to fall back; they must not
be evaluated per partition. `UnaryFunction._pointwise_fns` is an allowlist:
never add a function that changes the row count (`unique`, `drop_nulls`,
reductions).

```python
# Bad: unique() returns fewer rows, then gets broadcast against the full input
_pointwise_fns = frozenset({"abs", "round", "unique"})
```

### IR identity, `partition_info`, and serialization

- When adding an IR field, keep `get_hashable` and `is_equal` consistent.
  Keep runtime cache state, such as prefetched metadata, out of IR identity.
- Passes that replace nodes after lowering must rebuild `partition_info`
  for the new nodes and their ancestors.
- IR nodes and cross-rank payloads must be picklable and deterministic.
- Call `_lower_ir_fallback` with `msg=`, so the fallback warning is emitted.

```python
# Bad: new field affects equality but not the hash
class Scan(IR):
    _non_child = (..., "parquet_metadata")  # cache object becomes part of identity
```

### Stay under the libcudf row limit

Bound concatenation and chunk grouping by `CUDF_ROW_LIMIT`, using the
actual accumulated row counts. Do not add new `2**31 - 1` literals. Do not
limit chunk size and then concatenate all the per-chunk results anyway.

## Expression Semantics and Polars Parity

### Validate at translation time

Expression `__init__` must reject input dtypes and literal arguments that the
GPU path does not support, but CPU Polars accepts, with `NotImplementedError`.
Inputs that CPU Polars rejects must raise the corresponding CPU-compatible
exception. A check in `do_evaluate` is too late: it causes
silent casts, libcudf `logic_error`s, or device-to-host copies. Do not
assume an argument is a `Literal` when Polars allows any expression.

```python
# Bad
def do_evaluate(self, df, *, context):
    if self.children[1].value < 0:
        raise ValueError(...)

# Good
def __init__(self, dtype, name, options, *children):
    if not isinstance(children[1], Literal):
        raise NotImplementedError("non-literal n is not supported")
```

### Unsupported is not the same as invalid

Unknown plan variants, options, and dtypes must raise
`NotImplementedError` at translation. Invalid operations must raise the same exception as CPU Polars.

### Broadcast scalars correctly

Broadcast length-1 or literal operands before passing them to libcudf with
full-length columns. If every input is a scalar, the result has one row,
not `df.num_rows`.

```python
# Bad: pl.struct(pl.lit(1), pl.col("a")) passes a length-1 column to libcudf
plc.Column.struct_from_children([lit.obj, a.obj])

# Good
lit, a = broadcast(lit, a, target_length=df.num_rows, stream=stream)
```

### Window (`over`) expressions

Pattern matching must look past the direct child. A nested shape that is
not supported must fall back instead of being applied globally.
Order-sensitive window functions (`cum_*`, `shift`, `diff`) without
`order_by` must restore the per-group input order in multi-partition
execution.

### Result dtypes: cast with `Column.astype`

libcudf result types often differ from the Polars schema: `argmin` returns
INT32, and integer `sum` returns INT64. Cast with
`Column.astype(dtype, stream)`; it is a no-op when the types already
match. Never label a buffer with a type of a different width.

```python
# Bad: labels an INT32 result as the Polars index dtype (UInt32/UInt64)
return Column(result, dtype=self.dtype)

# Also bad: calls pylibcudf directly instead of Column.astype
plc.unary.cast(result, self.dtype.plc_type, stream=stream)

# Good
return Column(result, dtype=DataType(...)).astype(self.dtype, stream)
```

Streaming aggregation decompositions must keep one intermediate dtype
between the partial and combine stages. Global counts use the Polars index
dtype or 64-bit integers.

### Zero rows and zero columns

`do_evaluate` must work on 0-row input. Results with zero columns must keep `num_rows`.

## Efficient pylibcudf Use

### Fuse elementwise chains and avoid materializing for a count

- Chains of full-column `plc.binaryop`/`plc.unary` calls can often be one
  `plc.expressions` AST evaluated with `plc.transform.compute_column`.
- Do not materialize a table only to read its size or a boolean.
- Do not call `drop_nulls()` before an aggregation that takes a null policy.
- Do cheap host-side checks (null count, dtype) before launching kernels.

```python
# Bad: three kernels and two full-size intermediate columns for abs(a * b) + c
ab = plc.binaryop.binary_operation(a, b, plc.binaryop.BinaryOperator.MUL, dtype, stream=stream)
abs_ab = plc.unary.unary_operation(ab, plc.unary.UnaryOperator.ABS, stream=stream)
out = plc.binaryop.binary_operation(abs_ab, c, plc.binaryop.BinaryOperator.ADD, dtype, stream=stream)

# Good: one AST, evaluated in one kernel
Op, Col = plc.expressions.Operation, plc.expressions.ColumnReference
ast_op = plc.expressions.ASTOperator
expr = Op(ast_op.ADD, Op(ast_op.ABS, Op(ast_op.MUL, Col(0), Col(1))), Col(2))
out = plc.transform.compute_column(plc.Table([a, b, c]), expr, stream=stream)
```

### Do not convert to pyarrow

Do not convert pylibcudf or Polars objects to pyarrow objects, for example
with a `to_arrow` method.

```python
# Bad
col = plc.Column.from_arrow(series.to_arrow(), stream=stream)

# Good
col = plc.Column.from_arrow(series, stream=stream)
```

### No per-chunk host round-trips or identity GPU work

Avoid reading device scalars per chunk, running reductions only to choose a
branch, and identity work such as identity gathers or rebuilding the same
`DataFrame`. Hoist values that do not change within a query out of
per-chunk loops.

## Scans and Parquet

### Parquet metadata: fetch once, cache per path

Key footer caches by individual path, not `tuple(paths)`: lowering splits
scans, so a tuple key misses. Do not read footers inside `IR.do_evaluate`.
Shared caches hold immutable metadata, not `HybridScanReader` objects,
which keep per-read state. New scan features must update
`hybrid_scan_eligible`.

```python
# Bad
cache[tuple(ir.paths)] = metadata

# Good
for path, md in zip(ir.paths, metadata, strict=True):
    cache[path] = md
```

## Configuration and Engines

### Resolve options once, in the config dataclasses

Resolve defaults, environment variables, and validation in the config
dataclasses (`StreamingExecutor`, `ParquetOptions`, `StreamingOptions`), so
bad values fail at engine creation. Reject incompatible combinations in
`__post_init__`. Name new environment variables
`CUDF_POLARS__<CATEGORY>__<NAME>` and document them in `options.md`.

```python
# Bad: read during lowering, so bad values fail deep in a query
threshold = int(os.environ.get("CUDF_POLARS_BROADCAST_THRESHOLD", 8))

# Good
@dataclasses.dataclass(frozen=True)
class StreamingExecutor:
    broadcast_join_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            "CUDF_POLARS__EXECUTOR__BROADCAST_JOIN_LIMIT", int, default=8
        )
    )
```

### Do not change process-global state

Library and engine code must not set `os.environ`, change third-party
logger levels, or add global warning filters. Pass explicit parameters to
the underlying API instead.

```python
# Bad
os.environ["UCX_TLS"] = "tcp"
logging.getLogger("distributed").setLevel(logging.ERROR)
```

### Keep SPMD, Dask, and Ray engines consistent

New behavior and options must reach every frontend; put shared logic in
`StreamingEngine`. Engines must not mutate caller-supplied
`engine_options`/`executor_options`, ignore `rapidsmpf_options` when a
caller-owned communicator is passed, or silently drop reserved keys.

```python
# Bad
executor_options.pop("cluster")

# Good
executor_options = {k: v for k, v in executor_options.items() if k != "cluster"}
```

### Engine lifecycle

- `shutdown()` and `_reset()` are idempotent; methods on an engine that was
  shut down raise a clear `RuntimeError`.
- Tear down a rapidsmpf `Context` on the thread that created it.
- Close clients and clusters the engine created, in `finally`; do not close
  ones the user passed in.
- Frontend errors are raised, not printed or swallowed, and say how to
  recover.
- Dask: `Client.scheduler_info()` needs `n_workers=-1`; by default it
  reports at most 5 workers.

## Dependencies and Polars Versions

### Optional dependencies stay optional

`import cudf_polars` must not import `ray`, `distributed`, `structlog`, or
other optional packages. Keep `except ImportError` narrow. Tests that fake
a missing package by changing `sys.modules` must run in a subprocess.

## Testing

### Do not hide row-order differences

Do not `.sort()` results, or pass `check_row_order=False`, when Polars
defines the row order. `assert_frame_equal` ignores sorted flags, so
sortedness tests must compare `.flags`.

```python
# Bad
assert_gpu_result_equal(q.sort("a"), engine=engine)

# Good
assert_gpu_result_equal(q, engine=engine)
assert got["a"].flags["SORTED_ASC"]
```

### Use the shared engine fixtures

- Use `engine` for supported behavior, `streaming_engine` for
  streaming-only behavior, and `in_memory_engine` for translation errors.
- Do not construct engines or communicators in tests, and do not call
  `shutdown()` on fixture engines. Use
  `streaming_engine_factory(StreamingOptions(...))` instead.
- Use `warns_on_spmd` instead of `pytest.warns` for fallback warnings.
- Use `timeout_seconds` instead of hard-coded timeouts.
- Do not `pytest.skip` on setup errors; use `pytest.importorskip` for
  optional packages.

```python
# Bad
def test_join():
    engine = SPMDEngine(...)
    try:
        ...
    except Exception:
        pytest.skip("engine unavailable")
    finally:
        engine.shutdown()

# Good
def test_join(engine):
    assert_gpu_result_equal(q, engine=engine)
```
