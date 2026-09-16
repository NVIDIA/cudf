# String UDF memory management

Inside UDFs, some string methods like ``concat()`` and ``replace()`` produce new
strings.  For a CUDA thread to create a new string, it must dynamically allocate
memory on the device to hold the string's data. The cleanup of this memory by the
thread later on must preserve Python's semantics, for example when the variable
corresponding to the new string goes out of scope. To accomplish this in cuDF, UDF
memory management (allocation and freeing of the underlying data) is handled
transparently for the user, via a reference counting mechanism. This reference
counting implementation is distinct from the one in python and has its own interface
and requirements,

Along with the code generated from the functions and operations within the passed UDF,
numba-cuda will automatically weave the necessary reference counting operations into
the final device function that each thread will ultimately run. This allows the
programmer to pass a UDF that may utilize memory allocating types such strings
generally as one would in python:

```python
def udf(string):
  if len(string) > 2:
    result = string.upper() # new allocation
  else:
    result = string + string # new allocation
  return result + 'abc'
```


## Numba memory management and the Numba Runtime (NRT)

The API functions used to update the reference count associated with a variable
derive from [Numba's memory management for nopython mode
code](https://numba.readthedocs.io/en/stable/developer/numba-runtime.html#memory-management).
This runtime library (NRT or Numba Runtime) provides implementations for operators
that increase and decrease a variable's reference count (INCREF/DECREF), and numba
analyzes the passed UDF to determine where the calls targeting these implementations
should go and what objects they should operate on. Below are some examples of situations
where numba-cuda would detect a reference counting operation needs to be applied to
an object:

- **The creation of a new object**: During object creation, memory is allocated
and a structure to track the memory is created and initialized.
- **When new references are created**: For example during assignments, the
reference count of the assigned-from object is incremented.
- **When references are destroyed**: For example when an object goes out of
scope, or when an object holding a reference is destroyed. During these
events, the reference count of the tracked object is decremented. If the
reference count of an object falls to zero, the Numba Runtime will invoke its destructor.
- **When an intermediate variable is no longer needed**: For example when creating
a new variable for inspection then disposing of it, as in `string.upper() == 'A'`


Numba does not reference count every variable, as only variables with an associated
heap memory allocation need to be tracked. Numba determines if this is true for a
variable during compilation by querying the properties of the datamodel underlying
the variable's type. We provide a string type ``ManagedUDFString`` that implements
the required properties and backs any new string that is created on the device. Its
datamodel is defined under the data structures section below and is registered to
the extension type as shown.


## Data structures
The core concept is a ``ManagedUDFString`` numba extension type that fulfills the
requirements to be reference counted by NRT. It is composed of a `cudf::udf_string`
that owns the string data and a pointer to a ``MemInfo`` object, which the NRT API
uses for reference counting.

```python

from cudf.core.udf.strings_typing import ManagedUDFString
from numba.cuda.descriptor import cuda_target

@register_model(ManagedUDFString)
class managed_udf_string_model(models.StructModel):
  _members = (("meminfo", types.voidptr), ("udf_string", udf_string))

  def __init__(self, dmm, fe_type):
      super().__init__(dmm, fe_type, self._members)

  def has_nrt_meminfo(self):
      return True

  def get_nrt_meminfo(self, builder, value):
      # effectively returns self.meminfo in IR form
      udf_str_and_meminfo = numba.core.cgutils.create_struct_proxy(ManagedUDFString())(
          cuda_target.target_context, builder, value=value
      )
      return udf_str_and_meminfo.meminfo
```

The actual NRT APIs for adjusting the reference count of an object expect to operate
on this ``MemInfo`` object itself rather than the instance:

```c++
extern "C"
struct MemInfo {
cuda::atomic<size_t, cuda::thread_scope_device> refct;
NRT_dtor_function dtor;
void* dtor_info;
void* data;
size_t size;
};
typedef struct MemInfo NRT_MemInfo;
```

Every instance of a reference counted type within the scope of a CUDA thread executing
the UDF is associated with a separate instance of this ``MemInfo`` struct. An INCREF or
DECREF on the instance in numba's intermediate representation formed during compilation
will resolve to an increase or decrease of the `refct` of the ``MemInfo`` associated
with that instance. The NRT_decref implementation calls the ``dtor`` on the ``data`` if
the ``refct`` is found to be zero:


```c++
extern "C" __device__ void NRT_decref(NRT_MemInfo* mi)
{
  if (mi != NULL) {
    mi->refct--;
    if (mi->refct == 0) { NRT_MemInfo_call_dtor(mi); }
  }
}
```

## NRT Requirements

For a type to participate in Numba's reference counting correctly, the following must be
true:

1. The datamodel for the type needs to report that it has a meminfo. This is done by
   returning `True` from `has_nrt_meminfo`.
2. The datamodel must expose the location of the meminfo for that instance to numba's
   lowering phase. This means implementing `get_nrt_meminfo()` such that it returns the
   meminfo in a predictable location in heap memory.
3. Operators or functions that return the type must initialize the meminfo and place it
   at the location numba will report it exists at through (2). This is done in the lowering
   for the operations we support, such as `concat`.

``ManagedUDFString`` fulfills (2) by tying the MemInfo and the string instance that it owns
together into a parent struct. This allows (2) to be implemented by just returning its own
`.meminfo` member, effectively relating the meminfo location to `self` via an offset.
Lowering for operations like `concat` populate this member before returning.


### cuDF string data structures

On the C++ side, libcudf permits storing entire columns of strings. The
``cudf::string_view`` class is a non-owning view of a string --- usually a
single row in a libcudf column --- that provides a convenient abstraction
over working with individual strings in device code, for example in custom
kernels. cuDF Python introduces the ``cudf::strings::udf::udf_string`` class,
an owning container around a single string. This class is used by the numba UDF
code to create new strings in device code. All libcudf string functions are made
available in cuDF Python UDFs by constructing ``cudf::string_view`` instances
that view the strings owned by ``udf_string`` instances.

The cuDF extensions to Numba generate code to manipulate instances of these
classes, so we outline the members of these classes to aid in understanding
them. These classes also have various methods; consult the {ref}`cuDF C++ Developer
Documentation <md_developer_guide>` for further details of these structures.

```c++
class string_view {
  // A pointer to the underlying string data
  char const* p{};
  // The length of the underlying string data in bytes
  size_type bytes{};
  // The offset into the underlying string data in characters
  size_type char_pos{};
  // The offset into the underlying string data in bytes
  size_type byte_pos{};
};
```

```c++
class udf_string {
  // A pointer to the underlying string data
  char* m_data{};
  // The length of the string data in bytes
  cudf::size_type m_bytes{};
  // The size of the underlying allocation in bytes
  cudf::size_type m_capacity{};
};
```

```{note}
A ``udf_string`` has a destructor that frees the underlying string data. This is
important, because the C++ destructor is invoked during destruction of a
Python-side Managed UDF String object.
```


## Implementation

The cuDF implementations for Managed UDF Strings is required to provide:

- Typing and lowering for Managed UDF String operations. The typing has no special properties; it is similar to any other typing implementation in a Numba extension. The lowering is required to ensure that ``NRT_MemInfo`` objects for each managed object are created and initialized correctly.
- C++ implementations of string functions, some of which use libcudf's C++ string functionality. Other functions are provided by the ``strings_udf`` C++ library in cuDF Python. These help with the allocation of data and implement the required destructors.
- Numba shim functions to adapt calls to C++ code for use in Numba code and
  Numba extensions are also required.
- Conversion from String UDF data to and from `cudf::column`.

Use of C++ code for string functionality is not a hard requirement for
implementing string support in a Numba extension - it is instead a pragmatic
choice so that the Python and C++ sides of cuDF can share a single implementation
for string operations instead of trying to keep two separately-maintained
implementations in sync.


The majority of the complexity in the implementation comes from two areas:

- Combining the requirement to use C++ implementations, with the need to provide
correct initialization of `NRT_MemInfo` object, and
- Conversion of Managed UDF String objects back into cuDF columns when a UDF
returns strings.


## String Lifecycle Details

Let's trace the complete lifecycle of a string created by `result = str1 + str2` in a UDF:

### Phase 1: Compilation

**1.1 Numba Analysis**
```python
# User UDF
def my_udf(str1, str2):
    result = str1 + str2
    return result
```

- Typing phase identifies `str1 + str2` as returning a `ManagedUDFString`
- Lowering phase begins for the `+` operator

**1.2 Stack Allocation**
```python
managed_ptr = builder.alloca(
    context.data_model_manager[managed_udf_string].get_value_type()
)
```
- Allocates stack space for the complete `ManagedUDFString` instance
- At this point, both fields are uninitialized

**1.3 Member Pointer Extraction**
```python
udf_str_ptr = builder.gep(managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)])
```
- Gets pointer to the `udf_string` member within the allocated struct

### Phase 2: String Creation via Shim Function

**2.1 Shim Function Call**
```python
meminfo = context.compile_internal(
    builder, call_concat_string_view,
    types.voidptr(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR),
    (udf_str_ptr, lhs_ptr, rhs_ptr)
)
```

**2.2 Inside the Shim Function**
```c++
extern "C" __device__ int concat_shim(void** out_meminfo,
                                      void* output_udf_str,
                                      void* const* lhs,
                                      void* const* rhs) {
    auto lhs_sv = reinterpret_cast<cudf::string_view const*>(lhs);
    auto rhs_sv = reinterpret_cast<cudf::string_view const*>(rhs);

    // Perform actual concat- allocates GPU memory for result
    auto result_str = cudf::strings::udf::concat(*lhs_sv, *rhs_sv);

    // Place result into pre-allocated stack space using placement new
    auto udf_str_ptr = new (output_udf_str) udf_string(std::move(result_str));

    // Create and return the meminfo
    *out_meminfo = make_meminfo_for_new_udf_string(udf_str_ptr);

    return 0;
}
```

In the above, critically the final string is constructed through placement
new which relieves the compiler of the responsibility for cleaning up the
`cudf::udf_string` created there.

**2.3 MemInfo Creation Details**
```c++
__device__ NRT_MemInfo* make_meminfo_for_new_udf_string(udf_string* udf_str) {
    struct mi_str_allocation {
        NRT_MemInfo mi;
        udf_string st;
    };

    // Single heap allocation for both structures
    mi_str_allocation* heap_allocation = (mi_str_allocation*)NRT_Allocate(sizeof(mi_str_allocation));

    NRT_MemInfo* mi_ptr = &(heap_allocation->mi);
    udf_string* heap_str_ptr = &(heap_allocation->st);

    // Initialize MemInfo pointing to co-allocated string
    NRT_MemInfo_init(mi_ptr, heap_str_ptr, 0, udf_str_dtor, NULL);

    // Copy string data to heap location
    memcpy(heap_str_ptr, udf_str, sizeof(udf_string));

    return mi_ptr;
}
```

`mi_str_allocation` is similar in structure to `ManagedUDFString` but has
a `MemInfo` struct value as its first member rather than a pointer.

### Phase 3: Object Assembly and Return

**3.1 Final Assembly**
```python
managed = cgutils.create_struct_proxy(managed_udf_string)(context, builder)
managed.meminfo = meminfo  # Points to heap MemInfo
return managed._getvalue()
```

**3.2 Current Memory State**
- **Stack**: `ManagedUDFString` struct with valid `meminfo` pointer and `udf_string` data
- **Heap**: Co-allocated MemInfo and udf_string structures
- **GPU Memory**: String data owned by heap-allocated udf_string
- **Reference Count**: 1 (object just created)

### Phase 4: Runtime Usage and Reference Management

**4.1 Assignment Operations**

Within the broader kernel being launched, the result of the overall UDF is
assigned:

```python
result = my_udf(input_string)
```

At this point, `result` is a fully initialized `ManagedUDFString`:

- Numba detects assignment of reference counted return value
- Automatically inserts `NRT_incref(managed.meminfo)`
- `heap_allocation->mi.refct` becomes 2
- passed_udf exits, causing an `NRT_decref(managed.meminfo)`.

**4.2 Setitem into the final array**

The final line of the containing kernel sets the result into the output
array:

```
output_string_ary[tid] = result
```

- Adds an incref, bumping the refcount back up to 2.


### Phase 5: Destruction Sequence

**5.1 Final Reference Release**

The kernel being launched is ultimately overall a `void` function. Any
variables contained locally therein will be decref'd at function's exit,
like any other function.

- `result` variable decref'd, but still referred to by the output array
- `heap_allocation->mi.refct` becomes 1


**5.2 Destructor Execution**

The function `column_from_managed_udf_string_array` creates a `cudf::column`
from the output buffer containing the strings. cuDF launches a freeing kernel
that decrefs all the result strings one last time:

```python
def free_managed_udf_string_array(ary, size):
    gid = cuda.grid(1)
    if gid < size:
        NRT_decref(ary[gid])
```

- `NRT_MemInfo_call_dtor` invokes the destructor for the object

```c++
__device__ void udf_str_dtor(void* udf_str, size_t size, void* dtor_info) {
    auto ptr = reinterpret_cast<udf_string*>(udf_str);
    ptr->~udf_string();
}
```

- A `MemInfo` dies after invoking its destructor - the NRT API ensures that
once this is done, the originally `NRT_Allocat`ed pointer is freed. This has
the effect of freeing the entire `mi_str_allocation`.


**5.3 Final Memory State**
- **GPU String Memory**: Freed
- **Heap MemInfo Block**: Freed
- **Stack**: Original `ManagedUDFString` becomes invalid/out-of-scope
- **Reference Count**: N/A (object destroyed)
- **cuDF** A `cudf::column` of string type containing the result of the UDF
