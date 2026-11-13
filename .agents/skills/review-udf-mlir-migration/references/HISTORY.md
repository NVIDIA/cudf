# History and intent

This file records design intent, not current merge status. Verify PRs, branches,
and source before reporting status.

This reference is migration-era context; once the full cutover to
numba-cuda-mlir lands and the legacy backend is removed, it can be retired
(folded into a short note or dropped).

## Why the migration happened

numba-cuda entered maintenance mode and would not carry cuDF indefinitely across
new CUDA versions and GPU architectures. numba-cuda-mlir was built as its
replacement. cuDF could not merely switch decorators because its UDF stack
depends deeply on the compiler's low-level extension surface.

Before the cuDF port could work, numba-cuda-mlir itself needed substantial
features for cuDF:

- custom extension models and registries;
- device NRT and automatic lifetime operations;
- string representations and operations;
- cooperative arrays and block-level reductions;
- linking/runtime support;
- later experiments with arena allocation.

The cuDF migration is therefore the consumer-side half of a cross-repository
compiler project.

## Milestones

### NRT and string groundwork

The early numba-cuda-mlir compiler had a device-side NRT runtime but lacked
complete compiler-driven refcount insertion. The project ported ownership
discovery, incref/decref placement, stats/testing, destructor handling, and the
cuDF string path.

The first working string port preserved legacy C++ shim concepts:
`StringView`, `UDFString`, and `ManagedUDFString`. Later work unified these into
`MLIRString = {meminfo, data, nbytes}`, where null `meminfo` means borrowed.

Historical names should remain historical:

- `numba-cuda-mlir`: current project/package;
- `ManagedUDFString`: legacy row/groupby production and first-port owning
  representation, superseded specifically in the incremental MLIR design;
- `mlir_string`/`MLIRStringType`: unified incremental design.

### Full answer-key rewrite

In May 2026, `cudf-numba-cuda-mlir-extension-backend` became a working full
rewrite against contemporary numba-cuda-mlir. It remains valuable as an answer
key and as integration-CI input, but it was too large to review directly and
contains transitional choices superseded by later incremental design.

### Decomposition for review

The first proposed split followed large subsystem boundaries. During review
planning, the strategy changed for two important reasons:

1. reviewers should learn foundational nullable semantics before strings and
   groupby;
2. reviewers should not be taught a complicated legacy string model only to
   replace it a few PRs later.

The settled row/groupby principles were:

- Masked/NA first;
- focused tests at every layer;
- no partial row/groupby production cutovers;
- exactly one final row/groupby cutover;
- new behavior should not surface accidentally during replumbing.

### Masked series

Historical PR/commit subjects identify this sequence:

1. dependency/availability plumbing;
2. `MaskedType`, constructor, attributes, data model;
3. `NAType`, `cudf.NA`, unification and casts;
4. binary operators;
5. unary operations, truth, casts, and `abs`;
6. datetime/timedelta operations;
7. membership;
8. `pack_return`.

The series was deliberately granular because each item teaches a reusable
extension pattern and has distinct semantics. `pack_return`, for example,
exists specifically to normalize a UDF that may return either a bare scalar or
a Masked value.

### String series

After the Masked work, string changes moved to independent feature branches
rather than retaining a long stacked chain. The early slices established:

1. the unified `MLIRStringType`, ABI, wrapper, and argument handler;
2. data model plus UTF-8-aware `len`, including nullable strings;
3. read-only operations such as comparisons, search, affix, membership, and
   character predicates.

String-producing operations and full apply marshalling require the allocating
NRT path and are a materially different review layer from read-only methods.

### Cutover experiments

The later `mlir-udf-full-cutover` work reconstructs the real row/groupby paths,
removes legacy modules and C++ string infrastructure, and exposes remaining
integration gaps. Treat it as evidence for the final vertical path, not as a
license to copy all changes into an earlier feature PR.

## Why one final row/groupby cutover

A partial row/groupby cutover creates two production paths with mixed semantics
and forces every intermediate feature PR to solve unrelated integration.
Instead:

- feature PRs prove types, models, and lowerings in focused kernels;
- limited monkeypatching may exercise a pipeline seam;
- row/groupby production remains on the legacy backend during those feature PRs;
- one final PR switches dispatch, runs full API tests, and removes obsolete
  implementation.

This keeps each review about one semantic layer while making the final cutover
explicitly responsible for integration. Rolling migrated separately and already
has a production numba-cuda-mlir path; it is not waiting for this cutover.

## Strategic significance

The migration demonstrates a broader compiler-platform constraint: libraries
that only compile primitive numeric kernels barely notice an extension API
change; cuDF pays for a full type-system/runtime rewrite.

The cost pattern is:

```text
type
  -> data model
  -> typing and unification
  -> lowering
  -> kernel ABI and marshalling
  -> device ownership/runtime
  -> integration and cleanup
```

The deepest cost is lowering and lifetime integration. The user-visible
features—nullable UDFs, string UDFs, grouped reductions, and rolling
operations—are the reason that infrastructure exists.

## Historical review lessons

- A physical ABI is a contract shared by models, builders, wrappers, shims, and
  cleanup code.
- A package rename can expose class-identity bugs that look like lowering bugs.
- A passing inlined function test may not exercise argument-binding refcounts.
- Refcount statistics can balance despite use-after-free.
- Borrowed string pointers require their source column to remain alive.
- Literal-to-string conversion can accidentally allocate in a read-only path.
- A generic typing template can steal a string-specific operation.
- Tests on one development GPU do not prove cuDF's supported CUDA/architecture
  matrix.
- Cross-repository source CI is necessary because compiler unit tests do not
  exercise cuDF's extension surface deeply enough.
