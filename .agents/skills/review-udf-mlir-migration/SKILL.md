---
name: review-udf-mlir-migration
description: >-
  Reviews and explains cuDF UDF migration changes from numba-cuda to
  numba-cuda-mlir from first principles. Use for PRs or code touching
  cudf.core.udf, mlir_backend, MaskedType, NAType, mlir_string, string UDFs,
  NRT, row/groupby/rolling apply kernels, CooperativeArray, compiler extension
  typing/lowering/models, or the final MLIR UDF cutover.
---

# Review the cuDF UDF MLIR migration

This migration preserves cuDF's Python UDF features while replacing the
compiler substrate beneath them. It is not an import rename. cuDF is an unusually
deep extension-API consumer: its nullable values, strings, group operations,
kernel wrappers, and memory ownership are compiler extensions.

## Required review posture

Explain before judging. A cuDF reviewer should be able to understand:

1. the user-facing operation this code enables;
2. the old execution path;
3. the new execution path;
4. why this PR is a separate layer;
5. the correctness contracts at that layer;
6. what the tests prove and do not prove.

Do not begin with MLIR syntax. Begin with a small UDF or dataframe operation and
follow it down to the changed code.

Current code, branch ancestry, and the PR diff are authoritative. Historical
branches and this skill explain intent; they do not override current source.

## Read the references

Always read:

- [FOUNDATIONS.md](references/FOUNDATIONS.md): first-principles mental model.
- [REVIEW_PLAYBOOK.md](references/REVIEW_PLAYBOOK.md): investigation and report
  method.

Read as needed:

- [ARCHITECTURE.md](references/ARCHITECTURE.md): row, Masked/NA, strings/NRT, groupby,
  registration, ABI, and tests.
- [HISTORY.md](references/HISTORY.md): why the work exists and why the PR sequence has its
  present shape.

## Start every review by locating the layer

Inspect the PR base and diff; do not infer scope from its title. Classify each
change into one or more layers:

1. dependency and availability plumbing;
2. extension type and physical data model;
3. typing and type unification;
4. lowering and casts;
5. kernel ABI and host argument marshalling;
6. ownership, NRT, and result cleanup;
7. row/groupby/rolling kernel construction;
8. production dispatch/cutover;
9. removal of the legacy implementation.

Then identify the feature slice: Masked/NA, read-only string, allocating string,
groupby/cooperative array, rolling, or generic compiler integration.

## Reconstruct the vertical path

Trace the changed feature through all applicable stages:

```text
cuDF API call
  -> generated kernel / device UDF
  -> numba typing and extension registry
  -> extension data model
  -> MLIR lowering and linked device functions
  -> kernel argument/result ABI
  -> launch
  -> cuDF column + validity mask
  -> cleanup of temporary owned values
```

When one stage changes, search every producer and consumer of that representation.
A type declaration without a matching model, lowering, marshaller, or cleanup
path is incomplete even if its unit test passes.

## Core review invariants

### Backend identity

- Types, signatures, registries, and models must come from one
  `numba_cuda_mlir`/vendored-Numba universe.
- Equivalent-looking classes imported from stock Numba can fail dispatch by
  identity.
- Registration must happen before compilation and against the registry the
  active compiler actually uses.

### Type and model agreement

- Logical type, MLIR representation, byte size, field order, alignment,
  argument handler, and output marshaller must agree.
- Ask the active model manager for a type where possible; do not independently
  recreate an identified struct and accidentally create a distinct `.1` type.
- An extension struct crossing a kernel boundary is normally pointer-marshalled,
  not represented as a memref/NumPy array element.

### Null semantics

- `Masked(value, valid)` carries a payload and validity bit.
- Invalid payload contents are unspecified. Tests must not assert them.
- Operator typing resolves the payload operation; lowering propagates validity
  according to the operation.
- `cudf.NA`, branch unification, casts, and `pack_return` are part of the same
  semantic chain.

### Ownership

- Determine whether every value is borrowed or owned.
- A borrowed string/group view must not be freed and must not outlive its source.
- An owned value must expose its `MemInfo`, survive any escape into an output
  slot, and be decref'd exactly once per owned reference.
- Balanced allocation/free statistics do not prove absence of use-after-free.
  Use behavioral checks and Compute Sanitizer where ownership changes.

### Cooperative execution

- Row apply is conceptually one thread per row.
- Groupby apply is conceptually one block per group.
- A block-shared allocation must have one allocator/owner, a pointer broadcast,
  and synchronization that prevents free-before-last-consumer.
- Generic per-thread NRT reasoning does not by itself prove a cooperative
  operation safe.

### Test scope

- A typing test proves resolution, not representation or execution.
- A tiny kernel proves a registration/lowering slice, not cuDF marshalling.
- An integration test proves the path it reaches; verify it actually reaches the
  new backend.
- New tests should fail on the PR base for the intended reason.
- Test borrowed and owned cases, literal and runtime validity, nullable inputs,
  and unsupported types where applicable.

## Preserve the row/groupby migration strategy

The row/groupby feature PRs intentionally stay outside their production apply
paths until one final cutover. Intermediate tests may compile focused kernels or
monkeypatch the minimum pipeline seam. Do not request a partial production
cutover merely to make a feature PR look end-to-end.

Rolling UDFs have a separate numba-cuda-mlir production path and must be
reviewed as their own vertical slice.

Do not accept a final row/groupby cutover based only on focused extension tests.
It must exercise the real cuDF API, marshalling, launch, output construction,
cleanup, and legacy removal.

## Required review response

Lead with a plain-language verdict, then use this order:

1. **What users can do** — a minimal UDF/API example.
2. **Where this PR sits** — feature and layer in the migration.
3. **How it works** — old path versus new path, from type to launched kernel.
4. **Why it matters** — prerequisite unlocked or legacy dependency removed.
5. **Correctness contracts** — the specific invariants above that apply.
6. **Findings** — concrete bugs or risks, ordered by severity, with paths/lines.
7. **Verification** — what was run, base-versus-head behavior, and limitations.

Do not confuse unfamiliarity with a defect. Do not approve merely because the
new code resembles the answer-key branch. Verify semantics against tests and
the current compiler API.
