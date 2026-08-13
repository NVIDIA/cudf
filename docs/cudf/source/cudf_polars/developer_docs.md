# Getting started

You will need:

1. Rust development environment. If you use the rapids [combined
   devcontainer](https://github.com/rapidsai/devcontainers/), add
   `"./features/src/rust": {"version": "latest", "profile": "default"},` to your
   preferred configuration. Or else, use
   [rustup](https://rust-lang.org/tools/install/)
2. A [cudf development
   environment](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#setting-up-your-build-environment).
   The combined devcontainer works, or whatever your favourite approach is.

```{note}
These instructions will get simpler as we merge code in.
```

## Installing polars

The `cudf-polars` `pyproject.toml` advertises which polars versions it
works with. So for pure `cudf-polars` development, installing as
normal and satisfying the dependencies in the repository is
sufficient. For development, if we're adding things to the polars side
of things, we will need to build polars from source:

```sh
git clone https://github.com/pola-rs/polars
cd polars
```

We will install build dependencies in the same environment that we created for
building cudf. Note that polars offers a `make build` command that sets up a
separate virtual environment, but we don't want to do that right now. So in the
polars clone:

```sh
# cudf environment (conda or pip) is active
pip install --upgrade uv
uv pip install --upgrade -r py-polars/requirements-dev.txt
```

```{note}
plain `pip install` works fine, but `uv` is _much_ faster!
```

Now we have the necessary machinery to build polars
```sh
cd py-polars
# build in debug mode, best option for development/debugging
maturin develop -m Cargo.toml
```

For benchmarking purposes we should build in release mode
```sh
RUSTFLAGS='-C target-cpu=native' maturin develop -m Cargo.toml --release
```

After any update of the polars code, we need to rerun the `maturin` build
command.

## Installing the cudf polars executor

The executor for the polars logical plan lives in the cudf repo, in
`python/cudf_polars`. Build cudf as normal and then install the
`cudf_polars` package in editable mode:

```sh
cd cudf/python/cudf_polars
pip install --no-build-isolation --no-deps -e .
```

You should now be able to run the tests in the `cudf_polars` package:
```sh
pytest -v tests
```

# Executor design

The polars `LazyFrame.collect` functionality offers configuration of
the engine to use for collection through the `engine` argument. At a
low level, this provides for configuration of a "post-optimization"
callback that may be used by a third party library to replace a node
(or more, though we only replace a single node) in the optimized
logical plan with a Python callback that is to deliver the result of
evaluating the plan. This splits the execution of the plan into two
phases. First, a symbolic phase which translates to our internal
representation (IR). Second, an execution phase which executes using
our IR.

The translation phase receives the a low-level Rust `NodeTraverser`
object that delivers Python representations of the plan nodes (and
expressions) one at a time. During translation, we endeavour to raise
`NotImplementedError` for any unsupported functionality. This way, if
we can't execute something, we just don't modify the logical plan at
all: if we can translate the IR, it is assumed that evaluation will
later succeed.

The usage of the cudf-based executor is therefore selected with the
gpu engine:

```python
import polars as pl

result = q.collect(engine="gpu")
```

This should either transparently run on the GPU and deliver a polars
dataframe, or else fail (but be handled) and just run the normal CPU
execution. If `POLARS_VERBOSE` is true, then fallback is logged with a
`PerformanceWarning`.

As well as a string argument, the engine can also be specified with a
polars `GPUEngine` object. This allows passing more configuration in.
Currently, the public properties are `device`, to select the device,
and `memory_resource`, to select the RMM memory resource used for
allocations during the collection phase.

For example:
```python
import polars as pl

result = q.collect(engine=pl.GPUEngine(device=1, memory_resource=mr))
```

Uses device-1, and the given memory resource. Note that the memory
resource provided _must_ be valid for allocations on the specified
device, no checking is performed.

For debugging purposes, we can also pass undocumented keyword
arguments, at the moment, `raise_on_fail` is also supported, which
raises, rather than falling back, during translation:

```python
result = q.collect(engine=pl.GPUEngine(raise_on_fail=True))
```

This is mostly useful when writing tests, since in that case we want
any failures to propagate, rather than falling back to the CPU mode.

## IR versioning

On the polars side, the `NodeTraverser` object advertises an internal
version (via `NodeTraverser.version()` as a `(major, minor)` tuple).
`minor` version bumps are for backwards compatible changes (e.g.
exposing new nodes), whereas `major` bumps are for incompatible
changes. We can therefore attempt to detect the IR version
(independently of the polars version) and dispatch, or error
appropriately. This should be done during IR translation in
`translate.py`.

# IR design

As noted, we translate the polars DSL into our own IR. This is both so
that we can smooth out minor version differences (advertised by
`NodeTraverser` version changes) within `cudf-polars`, and so that we
have the freedom to introduce new IR nodes and rewrite rules as might
be appropriate for GPU execution.

To that end, we provide facilities for definition of nodes as well as
writing traversals and rewrite rules. The abstract base class `Node`
in `dsl/nodebase.py` defines the interface for implementing new nodes,
and provides many useful default methods. See also the docstrings of
the `Node` class.

```{note}
This generic implementation relies on nodes being treated as
*immutable*. Do not implement in-place modification of nodes, bad
things will happen.
```

## Defining nodes

A concrete node type (`cudf-polars` has expression nodes, `Expr`;
and plan nodes, `IR`), should inherit from `Node`. Nodes have
two types of data:

1. `children`: a tuple (possibly empty) of concrete nodes;
2. non-child: arbitrary data attached to the node that is _not_ a
   concrete node.

The base `Node` class requires that one advertise the names of the
non-child attributes in the `_non_child` class variable. The
constructor of the concrete node should take its arguments in the
order `*_non_child` (ordered as the class variable does) and then
`*children`. For example, the `Sort` node, which sorts a column
generated by an expression, has this definition:

```python
class Expr(Node):
    children: tuple[Expr, ...]

class Sort(Expr):
    _non_child = ("dtype", "options")
    children: tuple[Expr]
    def __init__(self, dtype, options, column: Expr):
        self.dtype = dtype
        self.options = options
        self.children = (column,)
```

By following this pattern, we get an automatic (caching)
implementation of `__hash__` and `__eq__`, as well as a useful
`reconstruct` method that will rebuild the node with new children.

If you want to control the behaviour of `__hash__` and `__eq__` for a
single node, override (respectively) the `get_hashable` and `is_equal`
methods.

## Adding new translation rules from the polars IR

### Plan nodes

Plan node definitions live in `cudf_polars/dsl/ir.py`, these all
inherit from the base `IR` node. The evaluation of a plan node is done
by implementing the `do_evaluate` method. This method takes in
the non-child arguments specified in `_non_child_args`, followed by
pre-evaluated child nodes (`DataFrame` objects), and finally a
keyword-only `context` argument (an `IRExecutionContext` object
containing runtime execution context). To perform the
evaluation, one should use the base class (generic) `evaluate` method
which handles the recursive evaluation of child nodes.

Plan nodes must also declare an `_n_non_child_args` attribute giving
the length of the `_non_child_args` tuple. This is used by tracing to know
how many non-child (dataframe) inputs to expect without introspection.

To translate the plan node, add a case handler in `translate_ir` that
lives in `cudf_polars/dsl/translate.py`.

As well as child nodes that are plans, most plan nodes contain child
expressions, which should be transformed using the input to the plan as a
context. The translation of expressions is handled via
`translate_expr` in `cudf_polars/dsl/translate.py`. So that data-type
resolution is performed correctly any expression should be translated
with the correct plan node "active" in the visitor. For example, when
translating a `Join` node, the left keys (expressions) should be
translated with the left input active (and right keys with right
input). To facilitate this, use the `set_node` context manager.

### Expression nodes

Adding a handle for an expression node is very similar to a plan node.
Expressions are defined in `cudf_polars/dsl/expressions/` and exported
into the `dsl` namespace via `expr.py`. They inherit
from `Expr`.

Expressions are evaluated by implementing a `do_evaluate` method that
takes a `DataFrame` as context (this provides columns) along with an
`ExecutionContext` parameter (indicating what context we're evaluating
this expression in, currently unused) and a `mapping` from
expressions to evaluated `Column`s. This approach enables a simple form of
expression rewriting during evaluation of expressions that is used in
evaluation of, for example, groupby-aggregations. To perform the
evaluation, one should use the base class (generic) `evaluate` method
which handles the boilerplate for looking up in the substitution
`mapping`.

To simplify state tracking, all columns should be considered immutable
on construction. This matches the "functional" description coming from
the logical plan in any case, so is reasonably natural.

## Traversing and transforming nodes

In addition to representing and evaluating nodes. We also provide
facilities for traversing a tree of nodes and defining transformation
rules in `dsl/traversal.py`. The simplest is `traversal`, a
[pre-order](https://en.wikipedia.org/wiki/Tree_traversal) visit of all
unique nodes in an expression. Use this if you want to know some
specific thing about an expression. For example, to determine if an
expression contains a `Literal` node:

```python
def has_literal(node: Expr) -> bool:
    return any(isinstance(e, Literal) for e in traversal(node))
```

It is often convenient to provide (immutable) state to a visitor, as
well as some facility to perform DAG-aware rewrites (reusing a
transformation for an expression if we have already seen it). We
therefore adopt the following pattern of writing DAG-aware visitors.
Suppose we want a rewrite rule (`rewrite`) between expressions
(`Expr`) and some new type `T`. We define our general transformation
function `rewrite` with type `Expr -> (Expr -> T) -> T`:

```python
from cudf_polars.typing import GenericTransformer
from typing import TypedDict

class State(TypedDict):
    ...


@singledispatch
def rewrite(e: Expr, rec: GenericTransformer[Expr, T, State]) -> T:
    ...
```

Note in particular that the function to perform the recursion is
passed as the second argument. Rather than defining methods on each
node in turn for a particular rewrite rule, we prefer free functions
and use `functools.singledispatch` to provide dispatching. We now, in
the usual fashion, register handlers for different expression types.
To use this function, we need to be able to provide both the
expression to convert and the recursive function itself. To do this we
must convert our `rewrite` function into something that only takes a
single argument (the expression to rewrite), but carries around
information about how to perform the recursion. To this end, we have
two utilities in `traversal.py`:

- `make_recursive` and
- `CachingVisitor`.

These both implement the `GenericTransformer` protocol, and can be
wrapped around a transformation function like `rewrite` to provide a
function `Expr -> T`. They also allow us to attach arbitrary
*immutable* state to our visitor by passing a `state` dictionary. The
`state` dictionary should be given as some `TypedDict` so that the
transformation function knows which fields are available.
`make_recursive` is very simple, and provides no caching of
intermediate results (so any DAGs that are visited will be viewed as
trees). `CachingVisitor` provides the same interface, but maintains a
cache of intermediate results, and reuses them if the same expression
is seen again.

Finally, for writing transformations that take nodes and deliver new
nodes (e.g. rewrite rules), we have a final utility
`reuse_if_unchanged` that can be used as a base case transformation
for node to node rewrites. It is a depth-first visit that transforms
children but only returns a new node with new children if the rewrite
of children returned new nodes.

To see how these pieces fit together, let us consider writing a
`rename` function that takes an expression (potentially with
references to columns) along with a mapping defining a renaming
between (some subset of) column names. The goal is to deliver a new
expression with appropriate columns renamed.

To start, we define the dispatch function
```python
from collections.abc import Mapping
from functools import singledispatch
from cudf_polars.dsl.traversal import (
    CachingVisitor, make_recursive, reuse_if_unchanged
)
from cudf_polars.dsl.expr import Col, Expr
from cudf_polars.dsl.to_ast import ExprTransformer


@singledispatch
def _rename(e: Expr, rec: ExprTransformer) -> Expr:
    raise NotImplementedError(f"No handler for {type(e)}")
```
then we register specific handlers, first for columns:
```python
@_rename.register
def _(e: Col, rec: ExprTransformer) -> Expr:
    mapping = rec.state["mapping"] # state set on rec
    if e.name in mapping:
        # If we have a rename, return a new Col reference
        # with a new name
        return type(e)(e.dtype, mapping[e.name])
    return e
```
and then for the remaining expressions
```python
_rename.register(Expr)(reuse_if_unchanged)
```

```{note}
In this case, we could have put the generic handler in the `_rename`
function, however, then we would not get a nice error message if we
accidentally sent in an object of the incorrect type.
```

Finally we tie everything together with a public function:

```python
from typing import TypedDict

class State(TypedDict):
    mapping: Mapping[str, str]


def rename(e: Expr, mapping: Mapping[str, str]) -> Expr:
    """Rename column references in an expression."""
    mapper = CachingVisitor(_rename, state=State(mapping=mapping))
    # or
    # mapper = make_recursive(_rename, state=State(mapping=mapping))
    return mapper(e)
```

# IO partition statistics

```{note}
IO partition statistics are experimental and the details are
likely to change in the future.
```

The streaming executor samples Parquet footer metadata to estimate
per-column storage sizes, then uses those estimates together with
`target_partition_size` (bytes) on the streaming executor to decide how
to split or fuse file-level partitions so that each input partition is
close to that target size. Tune `target_partition_size` via
`executor_options` or `CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE`.

```python
engine = pl.GPUEngine(
    executor="streaming",
    executor_options={"target_partition_size": 2_000_000_000},
)
```

Internally, `collect_statistics` walks the IR graph, groups Parquet
`Scan` nodes that share the same file paths (unioning projected columns
for sampling), and builds one `DataSourceInfo` per path group. It then
attaches that source to every `Scan` in the group. Leaf `DataFrameScan`
nodes are handled separately. Results live in `StatsCollector.scan_stats`.
Parquet metadata sampling is shared across scans that read the same files.

# Containers

Containers should be constructed as relatively lightweight objects
around their pylibcudf counterparts. We have three (in
`cudf_polars/containers/`):

1. `Scalar` (a wrapper around a pylibcudf `Scalar`)
2. `Column` (a wrapper around a pylibcudf `Column`)
3. `DataFrame` (a wrapper around a pylibcudf `Table`)

The interfaces offered by these are somewhat in flux, but broadly
speaking, a `DataFrame` is just a mapping from string `name`s to
`Column`s, and thus also holds a pylibcudf `Table`. Names are only
attached to `Column`s and hence inserted into `DataFrames` via
`NamedExpr`s, which are the top-level expression nodes that live
inside an `IR` node. This means that the expression evaluator never
has to concern itself with column names: columns are only ever
decorated with names when constructing a `DataFrame`.

The columns keep track of metadata (for example, whether or not they
are sorted). We could imagine tracking more metadata, like minimum and
maximum, though perhaps that is better left to libcudf itself.

We offer some utility methods for transferring metadata when
constructing new dataframes and columns, both `DataFrame` and `Column`
offer a `sorted_like(like: Self)` call which copies metadata from the
template.

All methods on containers that modify in place should return `self`,
to facilitate use in a ["fluent"
style](https://en.wikipedia.org/wiki/Fluent_interface). It makes it
much easier to write iteration over objects and collect the results if
everyone always returns a value.

# CUDA Streams

CUDA [Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams)
are used to manage concurrent operations. These build on libcudf's and
pylibcudf's usage of streams when performing operations on pylibcudf `Column`s
and `Table`s.

In `cudf-polars`, we attach a `Stream` to `cudf_polars.containers.DataFrame`.
This stream (or a new stream that it's joined into) is used for all pylibcudf
operations on the data backing that `DataFrame`.

When creating a `cudf_polars.containers.DataFrame` you *must* ensure that all
the provided pylibcudf Tables / Columns are valid on the provided `stream`.

Take special care when creating a `DataFrame` that combines pylibcudf `Table`s
or `Column`s from multiple `DataFrame`s, or "bare" pylibcudf objects that don't
come from a `DataFrame` at all. This also applies to `DataFrame` methods like
`DataFrame.with_columns` and `DataFrame.filter` which accept
`cudf_polars.containers.Column` objects that might not be valid on the
`DataFrame`'s original stream.

Here's an example of the simpler case where a `pylibcudf.Table` is created
on some CUDA stream and that same stream is used for the `DataFrame`:

```python
import polars as pl
import pyarrow as pa
import pylibcudf as plc
from rmm.pylibrmm.stream import Stream

from cudf_polars.containers import DataFrame, DataType

stream = Stream()
t = plc.Table.from_arrow(
    pa.Table.from_pylist([{"a": 1, "b": 0}, {"a": 1, "b": 1}, {"a": 2, "b": 0}]),
    stream=stream
)
# t is valid on `stream`. So we must provide `stream` or some CUDA Stream that's
# downstream of it
df = DataFrame.from_table(
    t,
    names=['a', 'b'],
    dtypes=[DataType(pl.Int64()), DataType(pl.Int64())],
    stream=stream
)
```

Managing multiple containers, which are potentially valid on different streams,
is more challenging. We have some utilities that can help correctly handle data
from multiple independent sources. For example, to add a new `Column` to `df`
that's valid on some independent CUDA stream, we'd use
`cudf_polars.utils.cuda_stream.get_joined_cuda_stream` to get a new CUDA stream
that's downstream of both the original `stream` and `stream_b`.


```python
from cudf_polars.containers import Column
from cudf_polars.utils.cuda_stream import get_joined_cuda_stream

stream_b = Stream()
col = Column(plc.Column.from_arrow(pa.array([1, 2, 3]), stream=stream_b), dtype=pl.Int64(), name="c")

new_stream = get_joined_cuda_stream(upstreams=(stream, stream_b))
df2 = df.with_columns([col], stream=new_stream)
```

The same principle applies to using the `cudf_polars.containers.DataFrame`
constructor with multiple `cudf_polars.containers.Column` objects that are valid
on multiple streams. It's the caller's responsibility to provide a stream that
all the `Column`s are valid on, likely by joining together the streams that each
individual stream is valid on.

# Writing tests

We use `pytest`, tests live in the `tests/` subdirectory,
organisationally the top-level test files each handle one of the `IR`
nodes. The goal is that they are parametrized over all the options
each node will handle, to have reasonable coverage. Tests of
expression functionality should live in `tests/expressions/`.

To write a test an assert correctness, build a lazyframe as a query,
and then use the utility assertion function from
`cudf_polars.testing.asserts`. This runs the query using both the cudf
executor and polars CPU, and checks that they match. So:

```python
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_whatever():
    query = pl.LazyFrame(...).(...)

    assert_gpu_result_equal(query)
```

## Test coverage and asserting failure modes

Where translation of a query should fail due to the feature being
unsupported we should test this. To assert that _translation_ raises
an exception (usually `NotImplementedError`), use the utility function
`assert_ir_translation_raises`:

```python
from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_whatever():
    unsupported_query = ...
    assert_ir_translation_raises(unsupported_query, NotImplementedError)
```

This test will fail if translation does not raise.

# Debugging

If the callback execution fails during the polars `collect` call, we
obtain an error, but are not able to drop into the debugger and
inspect the stack properly: we can't cross the language barrier.

However, we can drive the translation and execution of the DSL by
hand. Given some `LazyFrame` representing a query, we can first
translate it to our intermediate representation (IR), and then execute
and convert back to polars:

```python
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.ir import IRExecutionContext
from rmm.pylibrmm.stream import DEFAULT_STREAM
import polars as pl

q = ...

# Convert to our IR
translator = Translator(q._ldf.visit(), pl.GPUEngine())
ir = translator.translate_ir()

# DataFrame living on the device
result = ir.evaluate(cache={}, timer=None, context=IRExecutionContext())

# Polars dataframe
host_result = result.to_polars()
```

If we get any exceptions, we can then debug as normal in Python.

# Configuration

Polars users can configure various options about how the plan is executed
through the `pl.GPUEngine()`. This includes some configuration options
defined in polars itself: https://docs.pola.rs/api/python/dev/reference/lazyframe/api/polars.lazyframe.engine_config.GPUEngine.html

All additional keyword arguments are made available to cudf-polars through
`engine.config`.

To centralize validation and keep things well-typed internally, we model our
additional configuration as a set of dataclasses defined in
`cudf_polars/utils/config.py`. To transition from user-provided options to our
(validated) internal options, use `ConfigOptions.from_polars_engine`.

# Profiling

You can profile `cudf_polars` using NVIDIA NSight Systems. Each `.collect()` or
`.sink()` call has two top-level ranges under the `cudf_polars` domain:

1. `ConvertIR`: measures the time spent converting the polars query plan to
   cudf-polars' IR.
2. `ExecuteIR`: measures the time spent executing the cudf-polars IR.

The majority of time should be spent in the `ExecuteIR` range. Within
`ExecuteIR`, each individual IR node's `do_evaluate` method is wrapped in
another `nvtx` range (e.g. `Scan.do_evaluate`, `GroupBy.do_evaluate`, etc.).
These provide a higher-level grouping over the lower-level libcudf calls (e.g.
`read_chunk`, `aggregate`).

Finally, if using [rapidsmpf](https://docs.rapids.ai/api/rapidsmpf/nightly/)
for shuffling, the methods inserting and extracting partitions to shuffle are
annotated with nvtx ranges.

# Query Plans

The module `cudf_polars.experimental.explain` contains functions for dumping
the query for a given `LazyFrame`.


## Structured Output

`cudf_polars.experimental.explain.serialize_query` can be used to output
the query plan in a structured format.

```python
>>> import dataclasses
>>> import polars as pl
>>> from cudf_polars.experimental.explain import serialize_query
>>> q = pl.LazyFrame({"a": ['a', 'b', 'a'], "b": [1, 2, 3]}).group_by("a").agg(pl.len())
>>> dataclasses.asdict(serialize_query(q, engine=pl.GPUEngine()))
{'roots': ['526964741'],
 'nodes': {'526964741': {'id': '526964741',
   'children': ['1694929589'],
   'schema': {'a': 'STRING', 'len': 'UINT32'},
   'properties': {'columns': ['a', 'len']},
   'type': 'Select'},
  '1694929589': {'id': '1694929589',
   'children': ['2632275007'],
   'schema': {'a': 'STRING', '___0': 'UINT32'},
   'properties': {'keys': ['a']},
   'type': 'GroupBy'},
  '2632275007': {'id': '2632275007',
   'children': [],
   'schema': {'a': 'STRING'},
   'properties': {},
   'type': 'DataFrameScan'}},
 'partition_info': {'526964741': {'count': 1, 'partitioned_on': ()},
  '1694929589': {'count': 1, 'partitioned_on': ()},
  '2632275007': {'count': 1, 'partitioned_on': ()}}}
```

The structured schema has three top-level fields:

1. `roots`: the integer ID for the "root" (final) nodes in the query plan
2. `partition_info`: partitioning information at each stage of the query
3. `nodes`: A mapping from integer node id to node details. Each node ID
   that appears in the output will be present in this mapping.
   Inspect `children` to understand which nodes this node depends on.

Note that all integers are stored as strings to make round-tripping
to JSON easier.
