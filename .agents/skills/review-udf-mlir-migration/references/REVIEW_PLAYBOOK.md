# Review playbook

## 1. Establish repository truth

Before explaining the PR:

```bash
git status --short --branch
git log --oneline --decorate -10
git diff --stat <base>...HEAD
git diff --name-status <base>...HEAD
```

For a GitHub PR, inspect its actual base/head and dependency PRs. Do not assume
the local branch names or historical PR numbering remain current.

Identify whether the diff is:

- a focused extension slice;
- a restack/rebase artifact;
- a compiler compatibility update;
- an integration/cutover;
- legacy removal.

Exclude merge-only noise before reasoning about semantics.

## 2. Build the user story

Find or write the smallest Python UDF that needs this change. Examples:

```python
lambda row: row["a"] + row["b"]          # Masked binary
lambda x: cudf.NA if x < 0 else x        # NA unification
lambda s: len(s)                          # borrowed string
lambda s: s.upper()                       # owned string
lambda g: (g["x"] - g["y"]).mean()       # cooperative group
```

State whether the PR adds user-visible behavior or only rebuilds existing
behavior on the new backend.

## 3. Trace producers and consumers

For every changed type or representation, search:

- constructor and `typeof`;
- typing registration;
- data-model registration;
- lowering and cast registration;
- field extraction/construction;
- argument preparation;
- output preparation;
- cleanup/free path;
- direct and integration tests.

When a PR changes how a thing is created, audit every site that consumes it.
When it changes a consumer, audit every producer.

## 4. Review by layer

### Type/model PR

- Is equality/hash/unification stable?
- Does unsupported input fail during typing rather than deep in lowering?
- Is the model's field order/size shared with every marshaller?
- Is ownership discoverable?
- Are borrowed null owners supported?
- Is the model registered in the active compiler context?

### Typing PR

- Does each accepted signature have a lowering?
- Does each lowering have an accepted signature?
- Are literals unliteralized only where semantics permit?
- Are result dtypes delegated to the compiler's scalar typing?
- Can a broad template shadow a more specific one?
- Are kwargs and unsupported arities rejected?

### Lowering PR

- Does it use the target model rather than reconstructing a conflicting type?
- Are operands converted according to typed semantics?
- Are temporal units and signedness preserved?
- Is validity propagated correctly?
- Is undefined invalid payload kept unobserved?
- Does delegated scalar lowering use the exact resolved signature?
- Are temporary variable names unique within one expression?

### ABI/marshalling PR

- Does host byte size equal device layout?
- Are device pointers passed without accidental host dereference?
- Is an extension struct pointer-marshalled?
- Does nullable input carry the correct mask?
- Is the source owner kept alive for borrowed pointers?
- Does result conversion read every owner before cleanup?

### NRT/ownership PR

For each owned value, narrate the reference count:

1. allocation;
2. assignment/copy;
3. argument passing;
4. tuple/attribute extraction;
5. overwrite;
6. return/output escape;
7. scope exit;
8. final host cleanup.

Distinguish aliases from fresh results. Copying, extracting, or passing an
existing owner creates another reference and needs the corresponding incref.
A freshly allocated call result already supplies the target reference and must
not be incremented a second time merely because it is stored.

Then test a branch where different allocations escape. Check actual object
refcounts or sanitizer behavior, not only global allocation/free totals.

### Cooperative/groupby PR

- Which thread allocates and decrefs?
- How is pointer/owner broadcast?
- Is every consumer inside the synchronization lifetime?
- Can one path skip a barrier?
- Is the result block-uniform?
- Is shared-memory size correct for every reduction stage?
- What happens for group size zero, one, and larger than one block stride?

### Cutover PR

- Does real `Series.apply`/`DataFrame.apply`/`groupby.apply` select MLIR?
- Is return-type inference also on the intended compiler?
- Are dependency and packaging paths available in all CI variants?
- Are legacy wrappers, shims, build rules, and imports removed together?
- Are unsupported legacy features explicitly handled?
- Does the final diff preserve public behavior and supported dtype coverage?

### Rolling PR

- Do both return-type inference and execution use numba-cuda-mlir?
- Are window bounds correct at the beginning and end of the column?
- Do `min_periods`, null counts, and output-mask validity agree?
- Do nullable inputs still raise the documented `NotImplementedError`, unless
  the PR intentionally adds and tests nullable rolling UDF support?
- Are empty inputs handled without an invalid launch or write?
- Does the kernel cache key include every compilation-relevant property?
- Does the public rolling test exercise the compiled path rather than fallback?

## 5. Prove tests are meaningful

For new tests:

1. run on head;
2. run against the PR base and require failure;
3. inspect that the base failure is the intended missing feature;
4. confirm the test imports the working checkout;
5. confirm the target registry/lowering was selected.

Use a test ladder:

1. typing-only;
2. direct tiny kernel;
3. marshalling boundary;
4. public cuDF API;
5. NRT stats/refcount probe;
6. Compute Sanitizer where ownership changes.

Useful edge matrices:

- valid/invalid;
- scalar/Masked/both Masked;
- literal/runtime operand;
- signed/unsigned/float/bool;
- UTF-8 ASCII/multibyte/empty/null;
- borrowed/owned;
- branch returns first allocation/second allocation;
- group size 0/1/multiple warps;
- base/head.

## 6. Diagnose environment before blaming the diff

Use repository instructions for the current environment. Historical local
practice used an activated `cudf_dev` environment because compiler imports could
load CUDA and C++ libraries at module scope.

Check:

- `cudf.__file__` and `numba_cuda_mlir.__file__`;
- active Python and package versions;
- stale editable compiler builds or `_mlir` artifacts;
- generated/fatbin artifacts after C++ changes;
- CUDA architecture/toolkit compatibility;
- whether CI has the numba-cuda-mlir package at all.

Do not repair environment failures by changing source semantics.

## 7. Report template

```markdown
Verdict: [correct / correct with gap / blocking issue]

What users can do
[minimal public API example and behavior]

Where this sits
[feature, migration layer, prerequisites and dependents]

How it works
[old path -> new path, starting from the UDF]

Why it matters
[what this unlocks or removes]

Findings
- [severity] path:line — defect, consequence, evidence

Verification
- [base/head tests and result]
- [what was not verified and why]
```

If there are no defects, still explain the architecture and evidence. The
reviewer's lack of compiler context is an expected input to this skill, not a
reason to return only "looks good."
