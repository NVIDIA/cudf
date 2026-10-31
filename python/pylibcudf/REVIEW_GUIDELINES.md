# pylibcudf Review Guidelines

These guidelines apply only to `python/pylibcudf`. They complement
`python/REVIEW_GUIDELINES.md` as library specific guidelines.

Use the severity labels from `python/REVIEW_GUIDELINES.md` when you comment.
Code examples are simplified mock code. Always report violations in this guide.

## Object Lifetime and Ownership

### Keep the owner alive when C++ stores a reference

Some libcudf objects keep a reference, span, or raw pointer to their inputs
instead of a copy. If the wrapper does not hold the Python owner, the owner
can be garbage collected while C++ still points to it.

```cython
# Bad: options only stores a reference to the filter expression
def set_filter(self, Expression filter):
    self.c_obj.set_filter(dereference(filter.c_obj))

# Good: keep the Python owner alive for as long as the options object lives
def set_filter(self, Expression filter):
    self.c_obj.set_filter(dereference(filter.c_obj))
    self._filter = filter
```

Sub-views of device memory, such as `gpumemoryview` slices, must also keep a
reference to the owner and clamp bounds to the buffer size.

### Name host data passed to asynchronous copies

`rmm.DeviceBuffer.to_device` and other asynchronous memory copy APIs copy
on a non-default stream. Keep a Python reference to the host data alive
until the copy finishes. Watch for temporaries: slicing `bytes` makes a
copy, and `to_pybytes()` returns a new object. Either can be freed before
the copy finishes. This applies to tests too.

```python
# Bad: buf[4:] is a temporary copy that can be freed during the copy
dbuf = rmm.DeviceBuffer.to_device(buf[4:], stream=stream)

# Good: bind the source to a name that outlives the stream sync
payload = buf[4:]
dbuf = rmm.DeviceBuffer.to_device(payload, stream=stream)
stream.synchronize()
```

### Lifetime tests use `weakref`, not `gc.collect()`

```python
# Good
owner = make_owner()
ref = weakref.ref(owner)
view = owner.view()
del owner
assert ref() is not None  # view must keep the owner alive
```

## Streams and Memory Resources

### New APIs take `stream` and `mr`

Every new `cpdef` API that launches device work or allocates device memory takes
`stream`. If it allocates memory for returned objects, it also takes `mr` and
binds the libcudf overload that accepts a memory resource.

```cython
# Bad: no mr, so the result always comes from the current device resource
cpdef Table pack(Table input, object stream=None):
    ...

# Good
cpdef Table pack(
    Table input,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    cdef Stream _stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    ...
```

### Declarations follow the C++ move to `cuda::stream_ref`

`cdef extern` declarations must use the stream type in the current header.
A "default stream" must be built from `cudaStreamDefault` (or `nullptr`),
not `cudaStreamLegacy`. The legacy stream changes per-thread default stream
(PTDS) behavior.

```cython
# Bad: the header now takes cuda::stream_ref
cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
    unique_ptr[table] gather(..., cuda_stream_view stream, ...)

# Good
cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
    unique_ptr[table] gather(..., stream_ref stream, ...) \
        except +libcudf_exception_handler
```

### Tests that allocate with CuPy use `patch_cupy_stream`

CuPy allocations in pylibcudf tests must run on the same stream as cuDF.
Use the `patch_cupy_stream` fixture instead of bare `cp.asarray(...)`.

## GIL and `nogil` Blocks

### Run long-running libcudf calls inside `with nogil:`

Run long-running libcudf calls inside `with nogil:`, after all Python object
conversion and validation is complete.

### Keep Python locks out of `with nogil:`

- Do not hold Python-level locks while entering `with nogil:` libcudf calls
  that may allocate device memory or synchronize CUDA work.
- Watch for lock-order inversions: one path holds the GIL while it waits
  for CUDA or RMM state, and another path holds CUDA or RMM state while it
  tries to reacquire the GIL.
- If a binding must reacquire the GIL after it launches CUDA work, check
  that the work is ordered on the provided stream and that no
  Python-visible object can see partially completed device state.

### Release the GIL around IO setup and large metadata copies

Creating a datasource (`make_datasources`, file or S3 access) and copying
large C++ structs such as `FileMetaData` also need `with nogil:`. Callers
run these from thread pools, so holding the GIL there serializes the
threads. When callers need per-row-group or per-column-chunk data, add a
bulk accessor instead of a Python loop.

```cython
# Bad: one Python call per row group, GIL held for each copy
for i in range(metadata.num_row_groups()):
    sizes.append(metadata.row_group(i).total_byte_size())

# Good: one bulk call (illustrative name), copy done without the GIL
with nogil:
    c_sizes = c_metadata.row_group_byte_sizes()
```

## Exceptions and the Python C API

### Use `libcudf_exception_handler`, not bare `except +`

A bare `except +` maps libcudf errors to generic `RuntimeError`s. Mark a
function `noexcept` only if nothing it calls can throw.

```cython
# Bad
int64_t count(...) except +
size_type num_rows(...) noexcept   # calls a function that can throw

# Good
int64_t count(...) except +libcudf_exception_handler
```

### Use checked casts on untyped arguments

```cython
# Bad: unchecked cast, a wrong type crashes
if isinstance(obj, Table):
    tbl = <Table>obj

# Good: raises TypeError on a wrong type
tbl = <Table?>obj
```

## Matching libcudf

### Names, defaults, and struct shapes mirror libcudf

Cython bindings must match the C++ API signatures and semantics. Function,
enum, parameter names, and defaults must match libcudf. Wrapper classes
should follow the shape of the libcudf struct they wrap.

```python
# Bad: name and default differ from cudf::make_empty_lists_column
def empty_list_column(child_type=None): ...

# Good: same name and parameters as libcudf
def make_empty_lists_column(DataType child_type, stream=None, mr=None): ...
```

### Update every mirror when a libcudf enum or signature changes

When libcudf adds, removes, or reorders an enum member, update
`libcudf/*.pxd`, `.pyx`, `.pyi`, and any hard-coded value together. Do not
copy enum or bit-flag values into Python literals.

```python
# Bad: breaks silently when libcudf reorders the enum
if kind == 3:  # TIMESTAMP_DAYS
    ...

# Good
if kind == TypeId.TIMESTAMP_DAYS:
    ...
```

### Do not bind libcudf `detail` APIs

Bind only public libcudf headers. If a needed API is in `detail`, make it
public in libcudf or write a small Cython loop. Inline C++ in a `.pxd` goes
in an anonymous namespace with its own `#include`s.

```cython
# Bad
cdef extern from "cudf/detail/utilities/vector_factories.hpp" ...
```

## API Stability

### New `cdef class` fields in a `.pxd` change the ABI

Adding or reordering attributes on a public `cdef class` in a `.pxd`
changes its C layout. That breaks downstream packages that `cimport` it.
Flag it as an ABI change.

## Type Stubs and Docstrings

### Keep `.pyi` stubs in step with `.pyx` signatures

Every signature change needs the matching `.pyi` and numpydoc update.
Annotations must not be wider than the Cython type: Cython checks builtin
types at runtime. Type `stream` as `CudaStreamLike | None` from
`pylibcudf.typing`.

```python
# Bad: the .pyx parameter is typed `list`, so passing a tuple raises TypeError
def concatenate(objects: Sequence[Table], stream: Stream | None = None) -> Table: ...

# Good
def concatenate(
    objects: list[Table],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
```
