(pylibcudf-developer-docs)=

# Developer Documentation

pylibcudf is a lightweight Cython wrapper around libcudf.
It aims to provide a near-zero overhead interface to accessing libcudf in Python.
It should be possible to achieve near-native C++ performance using Cythonized code calling pylibcudf, while also allowing fairly performant usage from Python.
In addition to these requirements, pylibcudf must also integrate naturally with other Python libraries.
In other words, it should interoperate fairly transparently with standard Python containers, community protocols like `__cuda_array_interface__`, and common vocabulary types like CuPy arrays.


## General Design Principles

To satisfy the goals of pylibcudf, we impose the following set of design principles:
- Every public function or method should be `cpdef`ed. This allows it to be used in both Cython and Python code. This incurs some slight overhead over `cdef` functions, but we assume that this is acceptable because 1) the vast majority of users will be using pure Python rather than Cython, and 2) the overhead of a `cpdef` function over a `cdef` function is on the order of a nanosecond, while CUDA kernel launch overhead is on the order of a microsecond, so these function overheads should be washed out by typical usage of pylibcudf.
- Every variable used should be strongly typed and either be a primitive type (int, float, etc) or a cdef class. Any enums in C++ should be mirrored using `cpdef enum`, which will create both a C-style enum in Cython and a PEP 435-style Python enum that will automatically be used in Python.
- All typing in code should be written using Cython syntax, not PEP 484 Python typing syntax. Not only does this ensure compatibility with Cython < 3, but even with Cython 3 PEP 484 support remains incomplete as of this writing.
- All cudf code should interact only with pylibcudf, never with libcudf directly. This is not currently the case, but is the direction that the library is moving towards.
- Ideally, pylibcudf should depend on no RAPIDS component other than rmm, and should in general have minimal runtime dependencies.
- Type stubs are provided and generated manually. When adding new
  functionality, ensure that the matching type stub is appropriately updated.

## Relationship to libcudf

In general, the relationship between pylibcudf and libcudf can be understood in terms of two components, data structures and algorithms.

(data-structures)=

### Data Structures

Typically, every type in libcudf should have a mirror Cython `cdef` class with an attribute `self.c_obj: unique_ptr[${underlying_type}]` that owns an instance of the underlying libcudf type.
Each type should also implement a corresponding method `cdef ${cython_type} from_libcudf(${underlying_type} dt)` to enable constructing the Cython object from an underlying libcudf instance.
Depending on the nature of the type, the function may need to accept a `unique_ptr` and take ownership e.g. `cdef ${cython_type} from_libcudf(unique_ptr[${underlying_type}] obj)`.
This will typically be the case for types that own GPU data, may want to codify further.

For example, `libcudf::data_type` maps to `pylibcudf.DataType`, which looks like this (implementation omitted):
```cython

cdef class DataType:
    cdef data_type c_obj

    cpdef TypeId id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)
```

This allows pylibcudf functions to accept a typed `DataType` parameter and then trivially call underlying libcudf algorithms by accessing the argument's `c_obj`.

#### pylibcudf Tables and Columns

The primary exception to the above set of rules are libcudf's core data owning types, `cudf::table` and `cudf::column`.
libcudf uses modern C++ idioms based on smart pointers to avoid resource leaks and make code exception-safe.
To avoid passing around raw pointers, and to ensure that ownership semantics are clear, libcudf has separate `view` types corresponding to data owning types.
For example, `cudf::column` owns data, while `cudf::column_view` represents an view on a column of data and `cudf::mutable_column_view` represents a mutable view.
A `column_view` need not actually reference data owned by a `cudf::column`; any memory buffer will do.
This separation allows libcudf algorithms to clearly communicate ownership expectations and allows multiple views into the same data to coexist.

While libcudf algorithms accept views as inputs, any algorithms that allocate data must return `cudf::column` and `cudf::table` objects.
libcudf's ownership model is problematic for pylibcudf, which must be able to seamlessly interoperate with data provided by other Python libraries like PyTorch or Numba.
Therefore, pylibcudf employs the following strategy:
- pylibcudf defines the `gpumemoryview` type, which (analogous to the [Python `memoryview` type](https://docs.python.org/3/library/stdtypes.html#memoryview)) represents a view into memory owned by another object that it keeps alive using Python's standard reference counting machinery. A `gpumemoryview` is constructible from any object implementing the [CUDA Array Interface protocol](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html).
  - This type will eventually be generalized for reuse outside of pylibcudf.
- pylibcudf defines its own Table and Column classes.
  - A Table maintains Python references to the Columns it contains, so multiple Tables may share the same Column.
  - A Column consists of `gpumemoryview`s of its data buffers (which may include children for nested types) and its null mask.
- `pylibcudf.Table` and `pylibcudf.Column` provide easy access to `cudf::table_view` and `cudf::column_view` objects viewing the same columns/memory. These can be then be used when implementing any pylibcudf algorithm in terms of the underlying libcudf algorithm. Specifically, each of these classes owns an instance of the libcudf view type and provides a method `view` that may be used to access a pointer to that object to be passed to libcudf.


### Algorithms

pylibcudf algorithms should look almost exactly like libcudf algorithms.
Any libcudf function should be mirrored in pylibcudf with an identical signature and libcudf types mapped to corresponding pylibcudf types.
All calls to libcudf algorithms should perform any requisite Python preprocessing early, then release the GIL prior to calling libcudf.
For example, here is the implementation of `gather`:
```cython

cpdef Table gather(
    Table source_table,
    Column gather_map,
    OutOfBoundsPolicy bounds_policy
):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table.view(),
                gather_map.view(),
                bounds_policy
            )
        )
    return Table.from_libcudf(move(c_result))
```

There are a couple of notable points from the snippet above:
- The object returned from libcudf is immediately converted to a pylibcudf type.
- `cudf::gather` accepts a `cudf::out_of_bounds_policy` enum parameter. `OutOfBoundsPolicy` is an alias for this type in pylibcudf that matches our Python naming conventions (CapsCase instead of snake\_case).

## Testing

When writing pylibcudf tests, it is important to remember that all the APIs should be tested in the C++ layer in libcudf already.
The primary purpose of pylibcudf tests is to ensure the correctness of the _bindings_; the correctness of the underlying implementation should generally be validated in libcudf.
If pylibcudf tests uncover a libcudf bug, a suitable libcudf test should be added to cover this case rather than relying solely on pylibcudf testing.

pylibcudf's ``conftest.py`` contains some standard parametrized dtype fixture lists that may in turn be used to parametrize other fixtures.
Fixtures allocating data should leverage these dtype lists wherever possible to simplify testing across the matrix of important types.
Where appropriate, new fixture lists may be added.

To run tests as efficiently as possible, the test suite should make generous use of fixtures.
The simplest general structure to follow is for pyarrow array/table/scalar fixtures to be parametrized by one of the dtype list.
Then, a corresponding pylibcudf fixture may be created using a simple `from_arrow` call.
This approach ensures consistent global coverage across types for various tests.

In general, pylibcudf tests should prefer validating against a corresponding pyarrow implementation rather than hardcoding data.
If there is no pyarrow implementation, another alternative is to write a pure Python implementation that loops over the values
of the Table/Column, if a scalar Python equivalent of the pylibcudf implementation exists (this is especially relevant for string methods).

This approach is more resilient to changes to input data, particularly given the fixture strategy outlined above.
Standard tools for comparing between pylibcudf and pyarrow types are provided in the utils module.

Here is an example demonstrating the above points:

```python
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from cudf._lib import pylibcudf as plc
from utils import assert_column_eq

# The pa_dtype fixture is defined in conftest.py.
@pytest.fixture(scope="module")
def pa_column(pa_dtype):
    pa.array([1, 2, 3])


@pytest.fixture(scope="module")
def column(pa_column):
    return plc.Column.from_arrow(pa_column)


def test_foo(pa_column, column):
    index = 1
    got = plc.foo(column)
    expect = pa.foo(pa_column)

    assert_column_eq(expect, got)
```

Some guidelines on what should be tested:
- Tests SHOULD comprehensively cover the API, including all possible combinations of arguments required to ensure good test coverage.
- pylibcudf SHOULD NOT attempt to stress test large data sizes, and SHOULD instead defer to libcudf tests.
  - Exception: In special cases where constructing suitable large tests is difficult in C++ (such as creating suitable input data for I/O testing), tests may be added to pylibcudf instead.
- Nullable data should always be tested.
- Expected exceptions should be tested. Tests should be written from the user's perspective in mind, and if the API is not currently throwing the appropriate exception it should be updated.
  - Important note: If the exception should be produced by libcudf, the underlying libcudf API should be updated to throw the desired exception in C++. Such changes may require consultation with libcudf devs in nontrivial cases. [This issue](https://github.com/NVIDIA/cudf/issues/12885) provides an overview and an indication of acceptable exception types that should cover most use cases. In rare cases a new C++ exception may need to be introduced in [`error.hpp`](https://github.com/NVIDIA/cudf/blob/main/cpp/include/cudf/utilities/error.hpp). If so, this exception will also need to be mapped to a suitable Python exception in `exception_handler.pxd`.

Some guidelines on how best to use pytests.
- By default, fixtures producing device data containers should be of module scope and treated as immutable by tests. Allocating data on the GPU is expensive and slows tests. Almost all pylibcudf operations are out of place operations, so module-scoped fixtures should not typically be problematic to work with. Session-scoped fixtures would also work, but they are harder to reason about since they live in a different module, and if they need to change for any reason they could affect an arbitrarily large number of tests. Module scope is a good balance.
- Where necessary, mutable fixtures should be named as such (e.g. `mutable_col`) and be of function scope. If possible, they can be implemented as simply making a copy of a corresponding module-scope immutable fixture to avoid duplicating the generation logic.

Tests should be organized corresponding to pylibcudf modules, i.e. one test module for each pylibcudf module.

The following sections of the cuDF Python testing guide also generally apply to pylibcudf unless superseded by any statements above:
- [](#test_parametrization)
- [](#xfailing_tests)
- [](#testing_warnings)

## Miscellaneous Notes

### Cython Scoped Enums
Cython 3 introduced support for scoped enumerations.
However, this support has some bugs as well as some easy pitfalls.
Our usage of enums is intended to minimize the complexity of our code while also working around Cython's limitations.

```{warning}
The guidance in this section may change often as Cython is updated and our understanding of best practices evolves.
```

- All pxd files that declare a C++ enum should use `cpdef enum class` declarations.
  -  Reason: This declaration makes the C++ enum available in Cython code while also transparently creating a Python enum.
- Any pxd file containing only C++ declarations must still have a corresponding pyx file if any of the declarations are scoped enums.
  - Reason: The creation of the Python enum requires that Cython actually generate the necessary Python C API code, which will not happen if only a pxd file is present.
-  If a C++ enum will be part of a pylibcudf module's public API, then it should be imported (not cimported) directly into the pyx file and aliased with a name that matches our Python class naming conventions (CapsCase) instead of our C++ naming convention (snake\_case).
  - Reason: We want to expose the enum to both Python and Cython consumers of the module. As a side effect, this aliasing avoids [this Cython bug](https://github.com/cython/cython/issues/5609).
  - Note: Once the above Cython bug is resolved, the enum should also be aliased into the pylibcudf pxd file when it is cimported so that Python and Cython usage will match.

Here is an example of appropriate enum usage.


```cython
# pylibcudf/libcudf/copying.pxd
cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
    # cpdef here so that we export both a cdef enum class and a Python enum.Enum.
    cpdef enum class out_of_bounds_policy(bool):
        NULLIFY
        DONT_CHECK


# pylibcudf/libcudf/copying.pyx
# This file is empty, but is required to compile the Python enum in pylibcudf/libcudf/copying.pxd
# Ensure this file is included in pylibcudf/libcudf/CMakeLists.txt


# pylibcudf/copying.pxd

# cimport the enum using the exact name
# Once https://github.com/cython/cython/issues/5609 is resolved,
# this import should instead be
# from pylibcudf.libcudf.copying cimport out_of_bounds_policy as OutOfBoundsPolicy
from pylibcudf.libcudf.copying cimport out_of_bounds_policy


# pylibcudf/copying.pyx
# Access cpp.copying members that aren't part of this module's public API via
# this module alias
from pylibcudf.libcudf cimport copying as cpp_copying
from pylibcudf.libcudf.copying cimport out_of_bounds_policy

# This import exposes the enum in the public API of this module.
# It requires a no-cython-lint tag because it will be unused: all typing of
# parameters etc will need to use the Cython name `out_of_bounds_policy` until
# the Cython bug is resolved.
from pylibcudf.libcudf.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint
```

### Enum string representations

By default, Cython's `cpdef enum class` generates a valid Python `Enum` type for
each C++ enum. However, the default `__str__` implementation for these enums is not
very informative. It returns the underlying value (e.g., `11` instead of `<type_id.BOOL8: 11>`).

To improve developer experience, we manually set `__str__ = __repr__` for all public
enums. This ensures that printing an enum from Python returns a meaningful name like:

```python
>>> from pylibcudf.types import TypeId
>>> print(TypeId.INT32)
<type_id.INT32>
```


### Handling overloaded functions in libcudf
As a C++ library, libcudf makes extensive use of function overloading.
For example, both of the following functions exist in libcudf:
```cpp
std::unique_ptr<table> empty_like(table_view const& input_table);
std::unique_ptr<column> empty_like(column_view const& input);
```

However, Cython does not directly support overloading in this way, instead following Pythonic semantics where every function name must uniquely identify the function.
Therefore, Cython's [fused types](https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html) should be used when implementing pylibcudf wrappers of overloaded functions like the above.
Fused types are Cython's version of generic programming and in this case amount to writing templated functions that compile into separate copies corresponding to the different C++ overloads.
For the above functions, the equivalent Cython function is
```cython
ctypedef fused ColumnOrTable:
    Table
    Column

cpdef ColumnOrTable empty_like(ColumnOrTable input)
```

[Cython supports specializing the contents of fused-type functions based on the argument types](https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html#type-checking-specializations), so any type-specific logic may be encoded using the appropriate conditionals.
See the pylibcudf source for examples of how to implement such functions.

In the event that libcudf provides multiple overloads for the same function with differing numbers of arguments, specify the maximum number of arguments in the Cython definition,
and set arguments not shared between overloads to `None`. If a user tries to pass in an unsupported argument for a specific overload type, you should raise `ValueError`.

Finally, consider making an libcudf issue if you think this inconsistency can be addressed on the libcudf side.

### Type stubs

Since static type checkers like `mypy` and `pyright` cannot parse
Cython code, we provide type stubs for the pylibcudf package. These
are currently maintained manually, alongside the matching pylibcudf
files.

Every `pyx` file should have a matching `pyi` file that provides the
type stubs. Most functions can be exposed straightforwardly. Some
guiding principles:

- For typed integer arguments in libcudf, use `int` as a type
  annotation.
- For functions which are annotated as a `list` in Cython, but the
  function body does more detailed checking, try and encode the
  detailed information in the type.
- For Cython fused types there are two options:
    1. If the fused type appears only once in the function signature,
       use a `Union` type;
    2. If the fused type appears more than once (or as both an input
       and output type), use a `TypeVar` with
       the variants in the fused type provided as constraints.


As an example, `pylibcudf.copying.split` is typed in Cython as:

```cython
ctypedef fused ColumnOrTable:
    Table
    Column

cpdef list split(ColumnOrTable input, list splits): ...
```

Here we only have a single use of the fused type, and the `list`
arguments do not specify their values. Here, if we provide a `Column`
as input, we receive a `list[Column]` as output, and if we provide a
`Table` we receive `list[Table]` as output.

In the type stub, we can encode this with a `TypeVar`, we can also
provide typing for the `splits` argument that indicates that the split
values must be integers:

```python
ColumnOrTable = TypeVar("ColumnOrTable", Column, Table)

def split(input: ColumnOrTable, splits: list[int]) -> list[ColumnOrTable]: ...
```

Conversely, `pylibcudf.copying.scatter` uses a fused type only once in
its input:

```cython
ctypedef fused TableOrListOfScalars:
    Table
    list

cpdef Table scatter(
    TableOrListOfScalars source, Column scatter_map, Table target
)
```

In the type stub, we can use a normal union in this case

```python
def scatter(
    source: Table | list[Scalar], scatter_map: Column, target: Table
) -> Table: ...
```
