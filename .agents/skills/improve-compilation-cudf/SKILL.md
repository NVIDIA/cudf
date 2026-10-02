---
name: improve-compilation-cudf
description: Reduce libcudf C++/CUDA build time, redundant template instantiations, compiler memory use, and binary size using cuDF PR precedents and controlled validation. Use for source-level compilation and code-size work, rather than ordinary DataFrame runtime tuning.
---

# cuDF build and binary-size optimization

Reduce compiler work or shipped binary size while preserving supported behavior and runtime
performance. Follow this checklist for the selected change; use the linked references for
commands, implementation details, and historical evidence.

## Workflow

- **Check prior work and current scope.** Inspect the branch, working-tree changes, and
  repository instructions. Read relevant entries in the [PR reference](references/pr-history.md)
  and refresh live history for the affected files, including open work, reverts, and review
  discussions. Confirm the optimization is still needed. Do not resume unrelated experiments.

- **Choose the metric and capture the baseline.** State whether the target is full build
  wall time, summed compiler work, longest TU, compiler peak memory, incremental rebuilds,
  or linked/packaged size. These are different measurements. Record exact revisions,
  toolchain/dependencies, architecture flags, build mode, compression, and concurrency.
  **Never use sccache, ccache, or distributed compiler caches for compilation timing.**
  Clear launchers and verify actual commands using the [measurement guide](references/measurement.md).

- **Find the expensive work.** Rank current TUs, inspect their compile commands and object
  sizes, and examine fatbins when size is the target. Use fresh measurement logs and generate
  each Ninja report from its own build directory before objects change. A small source file
  can instantiate a large template graph; distinguish parsing, instantiation, and optimizer
  cost. Capture baseline runtime results before changing GPU behavior.

- **Explain the cause before editing.** List the type, aggregation, nullability, nestedness,
  comparator, policy, and iterator dimensions that multiply the expensive implementation.
  Trace their callers and guards to identify unreachable, equivalent, or duplicated work.
  Identify definition-owning TUs, explicit instantiations, consumers, and launch sites.
  A runtime guard alone does not prevent compilation of an unsupported specialization.

- **Make one focused change.** Use the [implementation patterns](references/implementation.md)
  to start with redundant instantiations, unused dispatch dimensions, existing compiled
  overloads, and unnecessary includes. Consider TU splitting or co-location next; reserve
  materialization and inlining changes for evidence that the simpler changes cannot address.
  This is a preferred order, not a requirement to try every technique. Preserve CUDA
  kernel-launch ownership, public APIs, supported architectures, exact type/null semantics,
  streams, memory resources, and async lifetimes. Explain intentional inlining or ownership
  choices in source comments so later refactors can reassess them with measurements.

- **Remeasure the complete affected set.** Compare all removed and replacement TUs with
  identical settings. Report longest TU, summed compile durations, object bytes, and linked
  library/fatbin size as relevant. Incremental builds are useful for isolated iteration;
  full clean-build wall-time claims require comparable clean builds. Snapshot reports and
  distinguish full-architecture measurements from single-architecture prototypes or estimates.

- **Validate according to the change.** Follow the
  [validation matrix](references/measurement.md#correctness-and-runtime-coverage).
  Use affected builds, link checks, and tests for mechanical source/ownership changes;
  GPU benchmarks are not mandatory for every cleanup. Compare runtime when device code,
  launch behavior, preprocessing, or allocation can change, or when equivalence is uncertain.
  Include preparation and scratch costs, not only the optimized kernel.

- **Decide whether to keep the change, then report it.** Apply the
  [acceptance criteria](references/measurement.md#decide-whether-to-keep-the-change): retain
  demonstrated gains with acceptable runtime, memory, and maintenance costs; rework or
  discard changes that fail those checks. Label inconclusive results without claiming a win.
  State the cause, final mechanism, before/after numbers, exact validation scope,
  runtime/memory tradeoffs, and evidence locations. Use the
  [reporting template](references/measurement.md#result-format) where helpful. Separate newly
  measured results from historical reports and estimates. A closed, unmerged proposal is
  not a merged success. This workflow does not authorize commits, pushes, or PR publication.

## Companion workflows

Consult [build/test commands](../build-test-cudf/SKILL.md),
[NVBench comparison guidance](../perf-compare-cudf/SKILL.md), and
[review conventions](../review-cudf/SKILL.md) only when applicable to the task and environment.
Preserve the exact chosen baseline: do not merge another branch, switch toolchains, or enable
compiler caches as a side effect of following a build recipe.
