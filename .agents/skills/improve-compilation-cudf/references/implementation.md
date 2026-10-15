# Implementation patterns and discovery

Read only the pattern matching the measured bottleneck. PR numbers link through
[the historical reference](pr-history.md); verify the current call sites before copying
old file names, traits, aggregation sets, or CUDA qualifiers.

## Choose a mechanism that fits the evidence

Prefer this order when several techniques address the same bottleneck:

1. Remove avoidable work: redundant instantiations, dispatch dimensions that do not affect
   the operation, unnecessary heavy includes, or repeated calls that can use an existing
   compiled overload without new materialization.
2. Change where necessary work is compiled: split independent instantiations to expose
   parallelism, or co-locate equivalent kernel families to avoid duplicate emission.
   Choose according to the target metric and measure the effect on both time and size.
3. Change the device computation boundary: materialize hashes/flags/bounds, introduce
   indirection, change inlining, or swap algorithms only when the diagnosed cost warrants
   the additional runtime and memory investigation.

This orders investigation by expected disruption, not guaranteed safety or benefit.
Dispatch removal can change codegen, and TU moves can affect optimization. Skip directly
to a later technique when evidence already rules out the earlier ones; do not make
unrelated cleanups or run every experiment just to complete this list.

| Observed cause | Candidate change | Start with |
| --- | --- | --- |
| Dispatch does not use the selected C++ type | Remove that dispatch; retain exceptional preprocessing such as dictionary key normalization | #23282 |
| Cartesian product of independent dispatch dimensions | Stage dispatches, derive intermediate types, or normalize an index representation | #10756, #6457, #6727 |
| Unsupported type/aggregation combinations instantiate heavy code | Guard instantiation with `if constexpr`/constraints or a local dispatcher matching the caller's supported set | #23330, #17753, #11489 |
| Policy flag duplicates an expensive algorithm | Consider a runtime flag, `nullate::DYNAMIC`, or optional iterator | #6835, #9623, #21312, #9324 |
| The same implementation is compiled in callers and explicit-instantiation TUs | Select one ownership scheme; remove redundant TUs or use declaration-only callers | #23323 versus #23331 |
| Equivalent kernels occur in multiple TUs | Share an out-of-line host helper or co-locate wrappers that instantiate the same kernel family | #23706, #23733 |
| One large TU dominates parallel builds | Split independent operations/specializations; keep heavy definitions out of common declaration headers | #21804, #23322, #23343 |
| Recursive row operators expand inside CUB/cuCO templates | Isolate expensive computation, selectively materialize hashes/bounds, or test a narrow noinline boundary | #12900, #23320, #23448 |
| Device code recomputes metadata known on the host | Precompute and transfer compact metadata, preserving alignment and lifetime | #17234 |
| Excessive parsing or repeated host-only compilation | Narrow headers, use precompiled overloads, move host-only code to `.cpp`, or share test utility objects | #20166, #9299, #23420, #20491, #18131 |

Choose a small, reviewable change with an explicit hypothesis. Splitting and co-location
solve different problems: splitting exposes parallelism, while co-location can prevent
duplicate device code. Measure both effects rather than treating either as a universal rule.

## Refresh the relevant history

Search the affected symbols/files as well as title keywords. Include open work to avoid
duplicating an ongoing change and inspect reverts/successors before adopting a design.

```sh
gh search prs --repo NVIDIA/cudf --merged '"build time"' --limit 1000
gh search prs --repo NVIDIA/cudf --merged '"instantiation"' --limit 1000
gh search prs --repo NVIDIA/cudf --merged '"binary size"' --limit 1000
gh pr view 23706 -R NVIDIA/cudf --json state,mergedAt,body,files,url
gh pr diff 23706 -R NVIDIA/cudf
gh pr view 23706 -R NVIDIA/cudf --comments
gh api --paginate repos/NVIDIA/cudf/pulls/23706/comments
```

The last call retrieves inline review comments; `gh pr view --comments` alone does not
cover those threads. Use GraphQL review threads when resolution status or thread grouping
matters. Paginate or partition date ranges when a search hits its cap. For local discovery:

```sh
git log --all --oneline -- cpp/src/path/to/affected_file.cu
rg -n 'extern template|template (class|struct)|launch_affected_operation' cpp
```

Replace the path/symbol with the target. Local `--all` history includes unmerged branches;
it does not prove merge status. Read #21973 and #23419 for hypotheses and failed experiments,
then check current support and measurements. Old hotspot rankings are not today's backlog.

## Own expensive instantiations once

First list the concrete specializations, definition owners, launch sites, and consumers.
The two common choices are:

- **Consumer ownership (#23323):** let a caller instantiate the required implementation,
  and remove separate instantiation TUs only when they emit the same unnecessary work.
- **Dedicated ownership (#23331, #16603):** expose a declaration-only host entry point;
  the owning CUDA TU sees the heavy implementation, launches its kernels, and defines the
  required host-wrapper specializations. Consumers call the host entry point.

A typical layout for the second choice is:

```text
operation.hpp           host entry-point declarations; lightweight parameter types
operation_impl.cuh      template implementation, private to definition-owning CUDA TUs
operation_variant.cu    definitions/explicit instantiations for a bounded combination set
caller.cpp or caller.cu includes operation.hpp and calls the host entry point
```

These file names are illustrative. Do not split a short instantiation TU again unless it
contains independently separable instantiation work. Keep common alias declarations
consistent and update the current CMake source list for additions/removals.
Move shared non-template initialization, copy, or fill work into ordinary functions when
that avoids recompiling it for every specialization. Preserve internal visibility and
stream/resource parameters, and keep launchers with the device definitions they need.

`extern template` can suppress applicable implicit instantiations even when a definition
is visible; this is a valid C++ design. It does not remove header parsing, guarantee that
all inline work disappears, or let a CUDA whole-compilation kernel be launched from a TU
without its device definition. Verify emitted objects and the actual CUDA build mode.
Avoid inventing an explicit-instantiation framework for a handful of specializations.

For equivalent device code emitted by different APIs, try a shared non-template helper or
co-locate the wrappers (#23706 variance/std, #23733 stable/non-stable segmented sort).
Add a short source comment explaining intentional ownership/co-location so a later split
does not silently undo it. Check the changed longest TU as well as linked size.

## Remove one dispatch dimension at a time

Trace what each selected type actually controls. #23282's non-dictionary path ignored the
dispatched C++ type and called a type-erased table operation; a direct dictionary check
preserved required preprocessing without instantiating the common path for every type.
Do not delete dispatch just because its immediate call is type-erased: preceding logic,
overload selection, validation, and error behavior may still depend on the type.

Before introducing a custom type map, check whether `dispatch_storage_type` or
`row::primitive::dispatch_primitive_type` already provides the required representation.
Verify its current supported types and semantics against the caller; a storage or primitive
mapping is not appropriate for every operation.

For a supported subset, a custom `IdTypeMap` can map excluded IDs to a sentinel type.
The functor must guard that sentinel with `if constexpr` before mentioning unsupported
heavy code. Keep the return type consistent across the dispatched set: an unsupported
specialization that only calls `CUDF_UNREACHABLE` can deduce `void` with `auto`, conflicting
with value-returning supported specializations. Declare the common return type when needed.
A runtime rejection inside an otherwise unrestricted specialization does
not prevent that code from being instantiated. For #23330, derive support from the current
shared-memory compatibility checks and aggregation decomposition; never reuse that set
for global memory without tracing its different callers.

If the axes factor mathematically, avoid their full Cartesian product. #10756 staged
binary-op support checks and calculated a three-type common type through pairwise steps.
For indices, existing index/offset-normalizing iterators can remove a width dimension
(#6457). For nullable inputs, `nullate::DYNAMIC` or optional iterators can collapse
otherwise equivalent kernel families (#11482, #9324). Preserve width/signedness, output
type, overflow, and null behavior and measure device codegen afterward.

Changing an if/switch cascade to `dispatch_bool`/`dispatch_enum` may improve readability
without reducing the instantiated set. Measure the set and generated work, not syntax.
The current helpers in `cudf/detail/utilities/dispatchers.hpp` are host-only; do not call
them from a device dispatcher. Check execution-space qualifiers and use an appropriate
device-capable implementation when the dispatch must occur on the GPU.

## Break expensive composition at a useful boundary

- **Precompute expensive metadata:** host-known AST arity (#17234), or selected nested
  row hashes (#23320/#23322), can remove work from a much larger device template graph.
  Retain the original metadata semantics and include its preparation cost.
- **Materialize selectively:** producing flags/bounds/hashes for a simple downstream
  CUB/cuCO operation can reduce optimizer complexity (#12900/#23343/#23532). It also adds
  allocations, launches, and memory traffic. Keep cheap/bandwidth-bound paths fused when
  that is faster; ordered nested distinct is a concrete case where hashing stayed inline.
- **Control inlining narrowly:** #23448 stores the comparator object in device memory and
  passes an object pointer to a noinline device wrapper. This is not a device function-pointer
  API. Preserve async object/upload lifetime and measure small-input overhead. Debug-only
  workarounds, measured Release noinline, and force-inline wins all exist in the history.
- **Try equivalent algorithms as an experiment:** #21793 replaced selected tabulate calls
  with transform over a counting iterator. Do not assume equivalent APIs use the same
  CCCL kernel, vectorization, or launch policy on today's version.

Favor an existing precompiled overload when input is already materialized: gather maps
(#9299/#23420), strings sizes/offsets (#23706). Materializing a lazy iterator just to reach
such an overload needs its own runtime and memory justification.

## Narrow dependencies without changing the build contract

Separate lightweight host declarations from device implementations so host-only code can
compile as `.cpp` (#20491). Include focused row-operator and CUB headers rather than large
umbrella headers, while retaining direct includes for every used symbol. Use forward
declarations only where complete types are unnecessary. A header containing declarations
alone does not guarantee its transitive dependencies are host-compatible.

Consult the current CMake IWYU integration and CI artifacts rather than assuming there
is a standalone `iwyu` target. Preserve existing include guards; do not add duplicate
guarding or change unrelated headers just to follow a historical pragma-once cleanup.
Use the current `rapids_cuda_enable_fatbin_compression` configuration; compression,
visibility, architecture selection, and Release assertion policy are separate mechanisms
from removing redundant specializations.

## cuDF-specific correctness boundaries

- **Reachability comes first.** Prove that an excluded combination is rejected or handled
  elsewhere before the affected kernel. A runtime early return alone does not prevent
  template instantiation. Do not remove a public specialization merely because no internal
  call site uses it. Preserve empty-input validation and error behavior.
- **Restrict locally.** Shared-memory aggregation support is narrower than global-memory
  support. Follow decomposition and compatibility checks before narrowing a dispatcher;
  do not copy an old aggregation list into a current branch. Preserve dictionary, decimal,
  nested, count, and overflow semantics wherever supported.
- **Make ownership explicit.** Document the host entry point, definition-owning TU, explicit
  specializations, CMake sources, and consumers. In CUDA whole-program compilation, keep
  a kernel launch with its device definition and expose a host launch wrapper across TUs
  (#16603). Definitions may legitimately remain visible with `extern template`; verify
  which instantiations are suppressed and what is still parsed/inlined. This is not a
  general device-linking solution.
  Check exact template parameters, including cooperative-group size (#19518).
- **Prefer existing precompiled overloads.** A resident gather map or strings-size buffer
  may already fit a span overload. Do not materialize a lazy iterator just to reach it
  without measuring the additional allocation, pass, and traffic.
- **Keep comparisons exact.** Preserve null/NaN equality, signed-zero/hash consistency,
  dictionary key normalization, nested child nulls, stability/keep policy, decimal scale,
  chrono units, and window boundary rules. Erase a representation only where these remain
  equivalent. Avoid blanket per-element device dispatch in bandwidth-bound reductions.
- **Treat inlining as an experiment.** The Debug-only fixes (#21197, #22675), measured
  Release follow-up (#22699), and indirect comparator (#23448) have different
  scopes. #9530 needed force-inlining for runtime performance. Do not generalize
  either attribute across all row operators or alter CUB policies to mask a regression.
- **Account for scratch and lifetime.** Cached hashes/bounds and device-resident comparators
  must survive asynchronous use on the correct stream and honor resource ownership.
  Include preprocessing and temporary memory in runtime comparisons. #23322 kept ordered
  nested hashing inline because materialization regressed that path.
- **Keep headers honest.** Follow the current developer/review guides and include what is
  used directly. Minimize heavy implementation headers without relying on transitive includes.
  Do not use anonymous namespaces in shared headers as a deduplication strategy (#22418).
