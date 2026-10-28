# Foundations

## Why a dataframe library has compiler internals

A numeric CUDA kernel over primitive arrays can often be handed directly to a
compiler. cuDF UDFs need semantics the compiler does not natively know:

- a column scalar is nullable;
- `cudf.NA` participates in type inference and branch unification;
- string inputs may borrow column storage while string results own allocations;
- grouped columns are operated on collectively by a CUDA block;
- Python values must be marshalled from and back into cuDF columns.

cuDF teaches these semantics to the compiler through custom types, data models,
typing templates, lowerings, argument handlers, and device-runtime hooks.

The migration rewrites that teaching layer from numba-cuda's llvmlite-oriented
extension API to numba-cuda-mlir's MLIR-oriented extension API.

## Supported surface

The target is cuDF's existing JIT UDF feature set, by feature class rather than
a per-operation list (individual operations land incrementally, so confirm a
given op against the current typing/lowering registrations):

- nullable scalars (`Masked`/`NA`) and the operators, casts, and `pack_return`
  over them;
- read-only string operations (`len`, comparisons, membership, search,
  character predicates);
- allocating string operations that produce new strings (concatenation,
  `strip`/`lstrip`/`rstrip`, character indexing, and further ops in progress);
- grouped/cooperative reductions through `groupby(...).apply`;
- rolling-window UDFs through the rolling path.

Out of scope for a given PR is anything not yet registered; a typing template
alone does not mean an operation is executable.

## Four separate questions

For any extension operation, answer these separately:

1. **Typing:** Given operand types, is the operation legal and what type results?
2. **Representation:** What fields and ABI represent that type in generated code?
3. **Lowering:** Which MLIR operations implement the runtime behavior?
4. **Lifetime:** Does the value borrow storage or own it, and where is ownership
   retained or released?

A correct answer to one does not imply the others.

## Example: nullable row UDF

```python
def f(row):
    if row["a"] is cudf.NA:
        return 0
    return (row["a"] + row["b"]) * 2

df.apply(f, axis=1)
```

Conceptually, the generated kernel:

1. reads each column's data and validity bit;
2. constructs `Masked(value, valid)` scalars;
3. runs the user's operations through Masked typing/lowering;
4. normalizes either a scalar or Masked result with `pack_return`;
5. writes payload and validity to the output column.

Thus a PR adding `Masked + Masked` is not an isolated operator convenience. It
implements one node in the type algebra used by ordinary `DataFrame.apply`.

## Example: string ownership

```python
def f(s):
    result = (s + "!") if len(s) > 5 else s
    return result
```

Input `s` can borrow bytes from an existing cuDF chars buffer. The `s + "!"`
branch allocates new bytes and therefore needs an owner; the other branch
returns `s` unchanged and stays borrowed. Which result escapes is
data-dependent. Compiler-inserted decrefs can be statically placed on control
flow paths, but whether a decref is the final reference is known only at runtime.
That is why refcounting is required.

The unified string representation is:

```text
{ ptr meminfo, ptr data, i64 nbytes }
```

- null `meminfo`: borrowed storage;
- non-null `meminfo`: NRT-owned storage.

Read-only operations can remain allocation-free. String-producing operations
must allocate and participate in NRT.

## Example: grouped UDF

```python
def f(group):
    diff = group["x"] - group["y"]
    return (diff * diff).mean()

df.groupby("key").apply(f, engine="jit")
```

cuDF arranges each group as a contiguous range and launches one block per group.
Input grouped columns are borrowed views. Elementwise operations create
block-cooperative temporary arrays; reductions combine their values across the
block.

This differs fundamentally from row apply:

- the useful value is shared by a block, not private to one thread;
- one thread allocates and broadcasts the storage;
- every participating thread must finish before the owner is released;
- barriers are part of the memory-safety contract.

## The NRT contract

NRT uses a `MemInfo` owner containing a reference count, data pointer, size, and
destructor information. An extension model advertises ownership with concepts
equivalent to:

```python
def has_nrt_meminfo(self):
    return True

def get_nrt_meminfo(self, value):
    return extract_field_zero(value)
```

The compiler then inserts lifetime operations around IR events. The migration
work identified nine important classes of insertion point:

- variable deletion/scope exit;
- overwrite/store of an existing value;
- tuple element deletion;
- variable copy;
- argument binding;
- cast;
- call result;
- tuple construction;
- attribute extraction.

Copied or extracted aliases need an incref before the new reference is stored.
A freshly allocated call result already begins with the target's reference and
must not receive an extra incref merely because it is assigned. Review the
compiler's NRT context and tests when changing this distinction.

Missing one can leak or free too early. Output storage is a special escape:
storing an owned value must preserve the owner beyond cleanup of the local
temporary.

## Why the lowerings look unfamiliar

The old backend used llvmlite and Numba helpers such as struct proxies and
`compile_internal`. The new backend builds SSA values with MLIR dialect
operations such as `llvm.extractvalue`, `llvm.insertvalue`, and `arith.*`.

The semantic steps are usually unchanged:

1. unpack fields;
2. resolve or delegate the scalar operation;
3. compute validity;
4. repack the extension value;
5. store it in the target variable.

Review semantic equivalence, not textual similarity.
