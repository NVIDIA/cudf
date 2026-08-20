# Unit Testing in libcudf

Unit tests in libcudf are written using
[Google Test](https://github.com/google/googletest/blob/master/docs/primer.md).

**Important:** Instead of including `gtest/gtest.h` directly, use
`#include <cudf_test/cudf_gtest.hpp>`.

Also, write test code in the global namespace. That is,
do not write test code in the `cudf` or the `cudf::test` namespace or their
sub-namespaces.
Likewise, do not use `using namespace cudf;` or `using namespace cudf::test;`
in the global namespace.


## Best Practices: What Should We Test?

In general we should test to make sure all code paths are covered. This is not always easy or
possible. But generally this means we test all supported combinations of algorithms and data types,
and all operators supported by algorithms that support multiple operators (e.g. reductions,
groupby).  Here are some other guidelines.

 * In general empty input is not an error in libcudf. Typically empty input results in empty output.
   Tests should verify this.

 * Anything that involves manipulating bitmasks (especially hand-rolled kernels) should have tests
   that check varying number of rows, especially around boundaries like the warp size (32). So, test
   fewer than 32 rows, more than 32 rows, exactly 32 rows, and greater than 64 rows.

 * Most algorithms should have one or more tests exercising inputs with a large enough number of
   rows to require launching multiple thread blocks, especially when values are ultimately
   communicated between blocks (e.g. reductions). This is especially important for custom kernels
   but also applies to Thrust and CUB algorithm calls with lambdas / functors.

 * For anything involving strings or lists, test exhaustive combinations of empty strings/lists,
   null strings/lists and strings/lists with null elements.

 * Strings tests should include a mixture of non-ASCII UTF-8 characters like `é` in test data.

 * Test sliced columns as input (that is, columns that have a nonzero `offset`). This is an easy to
   forget case.

 * Tests that verify various forms of "degenerate" column inputs, for example: empty
   string columns that have no children (not many paths in cudf can generate these but it
   does happen); columns with zero size but that somehow have non-null data pointers; and struct
   columns with no children.

 * Decimal types are not included in the `cudf::test::NumericTypes` type list, but are included in
   `cudf::test::FixedWidthTypes`, so be careful that tests either include or exclude decimal types as
   appropriate.


## Directory and File Naming

The naming of unit test directories and source files should be consistent with the feature being
tested. For example, the tests for APIs in `copying.hpp` should live in `cudf/cpp/tests/copying`.
Each feature (or set of related features) should have its own test source file named
`<feature>_tests.cu/cpp`. For example, `cudf/cpp/src/copying/scatter.cu` has tests in
`cudf/cpp/tests/copying/scatter_tests.cu`.

In the interest of improving compile time, whenever possible, test source files should be `.cpp`
files because `nvcc` is slower than `gcc` in compiling host code. Note that `thrust::device_vector`
includes device code, and so must only be used in `.cu` files. `rmm::device_uvector`,
`rmm::device_buffer` and the various `column_wrapper` types described later can be used in `.cpp`
files, and are therefore preferred in test code over `thrust::device_vector`.

## Base Fixture

All libcudf unit tests should make use of a GTest ["Test Fixture"](https://github.com/google/googletest/blob/master/docs/primer.md#test-fixtures-using-the-same-data-configuration-for-multiple-tests-same-data-multiple-tests).
Even if the fixture is empty, it should inherit from the base fixture `cudf::test::BaseFixture`
found in `include/cudf_test/base_fixture.hpp`. This ensures that RMM is properly initialized and
finalized. `cudf::test::BaseFixture` already inherits from `testing::Test` and therefore it is
not necessary for your test fixtures to inherit from it.

Example:

    class MyTestFixture : public cudf::test::BaseFixture {...};

## Typed Tests

In general, libcudf features must work across all of the supported types (there are exceptions e.g.
not all binary operations are supported for all types). In order to automate the process of running
the same tests across multiple types, we use GTest's
[Typed Tests](https://github.com/google/googletest/blob/master/docs/advanced.md#typed-tests).
Typed tests allow you to write a test once and run it across a list of types.

For example:

```c++
// Fixture must be a template
template <typename T>
class TypedTestFixture : cudf::test::BaseFixture {...};
using TestTypes = cudf::test:Types<int,float,double>; // Notice custom cudf type list type
TYPED_TEST_SUITE(TypedTestFixture, TestTypes);
TYPED_TEST(TypedTestFixture, FirstTest){
    // Access the current type using `TypeParam`
    using T = TypeParam;
}
```

To specify the list of types to use, instead of GTest's `testing::Types<...>`, libcudf provides `cudf::test::Types<...>` which is a custom, drop-in replacement for `testing::Types`.
In this example, all tests using the `TypedTestFixture` fixture will run once for each type in the
list defined in `TestTypes` (`int, float, double`).

### Type Lists

The list of types that are used in tests should be consistent across all tests. To ensure
consistency, several sets of common type lists are provided in
`include/cudf_test/type_lists.hpp`. For example, `cudf::test::NumericTypes` is a type list of all numeric types,
`FixedWidthTypes` is a list of all fixed-width element types, and `cudf::test::AllTypes` is a list of every
element type that libcudf supports.

```c++
#include <cudf_test/type_lists.hpp>

// All tests using TypeTestFixture will be invoked once for each numeric type
TYPED_TEST_SUITE(TypedTestFixture, cudf::test::NumericTypes);
```

Whenever possible, use one of the type list provided in `include/utilities/test/type_lists.hpp`
rather than creating new custom lists.

#### Advanced Type Lists

Sometimes it is necessary to generate more advanced type lists than the simple lists of single types
in the `TypeList` example above. libcudf provides a set of meta-programming utilities in
`include/cudf_test/type_list_utilities.hpp` for generating and composing more advanced type lists.

For example, it may be useful to generate a *nested* type list where each element in the list is two
types. In a nested type list, each element in the list is itself another list. In order to access
the `N`th type within the nested list, use `GetType<NestedList, N>`.

Imagine testing all possible two-type combinations of `<int,float>`. This could be done manually:

```c++
template <typename TwoTypes>
TwoTypesFixture : cudf::test::BaseFixture{...};
using TwoTypesList = Types< Types<int, int>, Types<int, float>,
                            Types<float, int>, Types<float, float> >;
TYPED_TEST_SUITE(TwoTypesFixture, TwoTypesList);
TYPED_TEST(TwoTypesFixture, FirstTest){
    // TypeParam is a list of two types, i.e., a "nested" type list
    // Use `cudf::test::GetType` to retrieve the individual types
    using FirstType = GetType<TypeParam,0>;
    using SecondType = GetType<TypeParam,1>;
}
```

The above example manually specifies all pairs composed of `int` and `float`. `CrossProduct` is a
utility in `type_list_utilities.hpp` which materializes this cross product automatically.

```c++
using TwoTypesList = Types< Types<int, int>, Types<int, float>,
                            Types<float, int>, Types<float, float> >;
using CrossProductTypeList = CrossProduct< Types<int, float>, Types<int, float> >;
// TwoTypesList and CrossProductTypeList are identical
```

`CrossProduct` can be used with an arbitrary number of type lists to generate nested type lists of
two or more types. **However**, overuse of `CrossProduct` can dramatically inflate compile time.
The cross product of two type lists of size `n` and `m` will result in a new list with
`n*m` nested type lists. This means `n*m` templates will be instantiated; `n` and `m` need not be
large before compile time becomes unreasonable.

There are a number of other utilities in `type_list_utilities.hpp`. For more details, see the
documentation in that file and their associated tests in
`cudf/cpp/tests/utilities_tests/type_list_tests.cpp`.

## Utilities

libcudf provides a number of utilities in `include/cudf_test` to make common testing operations more
convenient. Before creating your own test utilities, look to see if one already exists that does
what you need. If not, consider adding a new utility to do what you need. However, make sure that
the utility is generic enough to be useful for other tests and is not overly tailored to your
specific testing need.

### Column Wrappers

In order to make generating input columns easier, libcudf provides the `*_column_wrapper` classes in
`include/cudf_test/column_wrapper.hpp`. These classes wrap a `cudf::column` and provide constructors
for initializing a `cudf::column` object usable with libcudf APIs. Any `*_column_wrapper` class is
implicitly convertible to a `column_view` or `mutable_column_view` and therefore may be
transparently passed to any API expecting a `column_view` or `mutable_column_view` argument.

#### fixed_width_column_wrapper

The `cudf::test::fixed_width_column_wrapper` class should be used for constructing and initializing columns of
any fixed-width element type, e.g., numeric types, timestamp types, Boolean, etc.
`cudf::test::fixed_width_column_wrapper` provides constructors that accept an iterator range to generate each
element in the column. For nullable columns, an additional iterator can be provided to indicate the
validity of each element. There are also constructors that accept a `std::initializer_list<T>` for
the column elements and optionally for the validity of each element.

Example:

```c++
// Creates a non-nullable column of INT32 elements with 5 elements: {0, 1, 2, 3, 4}
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i;});
cudf::test::fixed_width_column_wrapper<int32_t> w(elements, elements + 5);

// Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i;});
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;})
cudf::test::fixed_width_column_wrapper<int32_t> w(elements, elements + 5, validity);

// Creates a non-nullable INT32 column with 4 elements: {1, 2, 3, 4}
cudf::test::fixed_width_column_wrapper<int32_t> w{{1, 2, 3, 4}};

// Creates a nullable INT32 column with 4 elements: {1, NULL, 3, NULL}
cudf::test::fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {1, 0, 1, 0}};
```

#### fixed_point_column_wrapper

The `cudf::test::fixed_point_column_wrapper` class should be used for constructing and initializing columns of
any fixed-point element type (DECIMAL32 or DECIMAL64). `cudf::test::fixed_point_column_wrapper` provides
constructors that accept an iterator range to generate each element in the column. For nullable
columns, an additional iterator can be provided to indicate the validity of each element.
Constructors also take the scale of the fixed-point values to create.

Example:

```c++
// Creates a non-nullable column of 4 DECIMAL32 elements of scale 3: {1000, 2000, 3000, 4000}
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){ return i; });
cudf::test::fixed_point_column_wrapper<int32_t> w(elements, elements + 4, 3);

// Creates a nullable column of 5 DECIMAL32 elements of scale 2: {null, 100, null, 300, null}
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){ return i; });
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){ return i % 2; });
cudf::test::fixed_point_column_wrapper<int32_t> w(elements, elements + 5, validity, 2);
```

#### dictionary_column_wrapper

The `cudf::test::dictionary_column_wrapper` class should be used to create dictionary columns.
`cudf::test::dictionary_column_wrapper` provides constructors that accept an iterator range to generate each
element in the column. For nullable columns, an additional iterator can be provided to indicate the
validity of each element. There are also constructors that accept a `std::initializer_list<T>` for
the column elements and optionally for the validity of each element.

Example:

```c++
// Creates a non-nullable dictionary column of INT32 elements with 5 elements
// keys = {0, 2, 6}, indices = {0, 1, 1, 2, 2}
std::vector<int32_t> elements{0, 2, 2, 6, 6};
cudf::test::dictionary_column_wrapper<int32_t> w(element.begin(), elements.end());

// Creates a nullable dictionary column with 5 elements and a validity iterator.
std::vector<int32_t> elements{0, 2, 0, 6, 0};
// Validity iterator here sets even rows to null.
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;})
// keys = {2, 6}, indices = {NULL, 0, NULL, 1, NULL}
cudf::test::dictionary_column_wrapper<int32_t> w(elements, elements + 5, validity);

// Creates a non-nullable dictionary column with 4 elements.
// keys = {1, 2, 3}, indices = {0, 1, 2, 0}
cudf::test::dictionary_column_wrapper<int32_t> w{{1, 2, 3, 1}};

// Creates a nullable dictionary column with 4 elements and validity initializer.
// keys = {1, 3}, indices = {0, NULL, 1, NULL}
cudf::test::dictionary_column_wrapper<int32_t> w{ {1, 0, 3, 0}, {1, 0, 1, 0}};

// Creates a nullable column of dictionary elements with 5 elements and validity initializer.
std::vector<int32_t> elements{0, 2, 2, 6, 6};
// keys = {2, 6}, indices = {NULL, 0, NULL, 1, NULL}
cudf::test::dictionary_width_column_wrapper<int32_t> w(elements, elements + 5, {0, 1, 0, 1, 0});

// Creates a non-nullable dictionary column with 7 string elements
std::vector<std::string> strings{"", "aaa", "bbb", "aaa", "bbb", "ccc", "bbb"};
// keys = {"","aaa","bbb","ccc"}, indices = {0, 1, 2, 1, 2, 3, 2}
cudf::test::dictionary_column_wrapper<std::string> d(strings.begin(), strings.end());

// Creates a nullable dictionary column with 7 string elements and a validity iterator.
// Validity iterator here sets even rows to null.
// keys = {"a", "bb"}, indices = {NULL, 1, NULL, 1, NULL, 0, NULL}
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::dictionary_column_wrapper<std::string> d({"", "bb", "", "bb", "", "a", ""}, validity);
```

#### strings_column_wrapper

The `cudf::test::strings_column_wrapper` class should be used to create columns of strings. It provides
constructors that accept an iterator range to generate each string in the column. For nullable
columns, an additional iterator can be provided to indicate the validity of each string. There are
also constructors that accept a `std::initializer_list<std::string>` for the column's strings and
optionally for the validity of each element.

Example:

```c++
// Creates a non-nullable STRING column with 7 string elements:
// {"", "this", "is", "a", "column", "of", "strings"}
std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
cudf::test::strings_column_wrapper s(strings.begin(), strings.end());

// Creates a nullable STRING column with 7 string elements:
// {NULL, "this", NULL, "a", NULL, "of", NULL}
std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), validity);

// Creates a non-nullable STRING column with 7 string elements:
// {"", "this", "is", "a", "column", "of", "strings"}
cudf::test::strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"});

// Creates a nullable STRING column with 7 string elements:
// {NULL, "this", NULL, "a", NULL, "of", NULL}
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"}, validity);
```

#### lists_column_wrapper

The `cudf::test::lists_column_wrapper` class should be used to create columns of lists. It provides
constructors that accept an iterator range to generate each list in the column. For nullable
columns, an additional iterator can be provided to indicate the validity of each list. There are
also constructors that accept a `std::initializer_list<T>` for the column's lists and
optionally for the validity of each element. A number of other constructors are available.

Example:

```c++
// Creates an empty LIST column
// []
cudf::test::lists_column_wrapper l{};

// Creates a LIST column with 1 list composed of 2 total integers
// [{0, 1}]
cudf::test::lists_column_wrapper l{0, 1};

// Creates a LIST column with 3 lists
// [{0, 1}, {2, 3}, {4, 5}]
cudf::test::lists_column_wrapper l{ {0, 1}, {2, 3}, {4, 5} };

// Creates a LIST of LIST columns with 2 lists on the top level and
// 4 below
// [ {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}} ]
cudf::test::lists_column_wrapper l{ {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}} };

// Creates a LIST column with 1 list composed of 5 total integers
// [{0, 1, 2, 3, 4}]
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i*2;});
cudf::test::lists_column_wrapper l(elements, elements+5);

// Creates a LIST column with 1 lists composed of 2 total integers
// [{0, NULL}]
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::lists_column_wrapper l{{0, 1}, validity};

// Creates a LIST column with 1 lists composed of 5 total integers
// [{0, NULL, 2, NULL, 4}]
auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i*2;});
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::lists_column_wrapper l(elements, elements+5, validity);

// Creates a LIST column with 1 list composed of 2 total strings
// [{"abc", "def"}]
cudf::test::lists_column_wrapper l{"abc", "def"};

// Creates a LIST of LIST columns with 2 lists on the top level and 4 below
// [ {{0, 1}, NULL}, {{4, 5}, NULL} ]
auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i){return i % 2;});
cudf::test::lists_column_wrapper l{ {{{0, 1}, {2, 3}}, validity}, {{{4, 5}, {6, 7}}, validity} };
```

#### structs_column_wrapper

The `cudf::test::structs_column_wrapper` class should be used to create columns of structs. It provides
constructors that accept a vector or initializer list of pre-constructed columns or column wrappers
for child columns. For nullable columns, an additional iterator can be provided to indicate the
validity of each struct.

Examples:

```c++
// The following constructs a column for struct< int, string >.
auto child_int_col = cudf::test::fixed_width_column_wrapper<int32_t>{ 1, 2, 3, 4, 5 }.release();
auto child_string_col = cudf::test::string_column_wrapper {"All", "the", "leaves", "are", "brown"}.release();

std::vector<std::unique_ptr<cudf::column>> child_columns;
child_columns.push_back(std::move(child_int_col));
child_columns.push_back(std::move(child_string_col));

cudf::test::struct_col wrapper wrapper{
  child_cols,
  {1,0,1,0,1} // Validity
};

auto struct_col {wrapper.release()};

// The following constructs a column for struct< int, string >.
cudf::test::fixed_width_column_wrapper<int32_t> child_int_col_wrapper{ 1, 2, 3, 4, 5 };
cudf::test::string_column_wrapper child_string_col_wrapper {"All", "the", "leaves", "are", "brown"};

cudf::test::structs_column_wrapper wrapper{
  {child_int_col_wrapper, child_string_col_wrapper}
  {1,0,1,0,1} // Validity
};

auto struct_col {wrapper.release()};

// The following constructs a column for struct< int, string >.
cudf::test::fixed_width_column_wrapper<int32_t> child_int_col_wrapper{ 1, 2, 3, 4, 5 };
cudf::test::string_column_wrapper child_string_col_wrapper {"All", "the", "leaves", "are", "brown"};

cudf::test::structs_column_wrapper wrapper{
  {child_int_col_wrapper, child_string_col_wrapper}
  cudf::detail::make_counting_transform_iterator(0, [](auto i){ return i % 2; }) // Validity
};

auto struct_col {wrapper.release()};
```

### Column Comparison Utilities

A common operation in testing is verifying that two columns are equal, or equivalent, or that they
have the same metadata.

#### CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL

Verifies that two columns have the same type, size, and nullability. For nested types, recursively
verifies the equality of type, size and nullability of all nested children.

#### CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT

Verifies that two columns have equivalent type and equal size, ignoring nullability. For nested
types, recursively verifies the equivalence of type, and equality of size of all nested children,
ignoring nullability.

Note "equivalent type". Most types are equivalent if and only they are equal. `fixed_point` types
are one exception. They are equivalent if the representation type is equal, even if they have
different scales. Nested type columns can be equivalent in the case where they both have zero size,
but one has children (also empty) and the other does not. For columns with nonzero size, both equals
and equivalent expect equal number of children.

#### CUDF_TEST_EXPECT_COLUMNS_EQUAL

Verifies that two columns have equal properties and verifies elementwise equality of the column
data. Null elements are treated as equal.

#### CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT

Verifies that two columns have equivalent properties and verifies elementwise equivalence of the
column data. Null elements are treated as equivalent.

#### CUDF_TEST_EXPECT_EQUAL_BUFFERS

Verifies the bitwise equality of two device memory buffers.

#### Caveats

Column comparison functions in the `cudf::test::detail` namespace should **NOT** be used directly.

### Printing and accessing column data

The `<cudf_test/debug_utilities.hpp>` header defines various functions and overloads for printing
columns (`print`), converting column data to string (`to_string`, `to_strings`), and copying data to
the host (`to_host`). For example, to print a `cudf::column_view` contents or `column_wrapper` instance
to the console use the `cudf::test::print()`:
```cpp
  cudf::test::fixed_width_column_wrapper<int32_t> input({1,2,3,4});
  auto splits = cudf::split(input,{2});
  cudf::test::print(input);
  cudf::test::print(splits.front());
```
Fixed-width and strings columns output as comma-separated entries including null rows.
Nested columns are also supported and output includes the offsets and data children as well as
the null mask bits.

## Validating Stream Usage

### Background

libcudf employs a custom-built [preload
library](https://man7.org/linux/man-pages/man8/ld.so.8.html) to validate its internal stream usage
(the code may be found
[`here`](https://github.com/NVIDIA/cudf/blob/main/cpp/tests/utilities/identify_stream_usage.cpp)).
This library wraps every asynchronous CUDA runtime API call that accepts a stream with a check to
ensure that the passed CUDA stream is a valid one, immediately throwing an exception if an invalid
stream is detected. Running tests with this library loaded immediately triggers errors if any test
accidentally runs code on an invalid stream.

Stream validity is determined by overloading the definition of libcudf's default stream. Normally, in
libcudf `cudf::get_default_stream` returns one of `rmm`'s default stream values (depending on
whether or not libcudf is compiled with per thread default stream enabled). In the preload library,
this function is redefined to instead return a new user-created stream managed using a
function-local static `rmm::cuda_stream`. An invalid stream in this situation is defined as any of
CUDA's default stream values (cudaStreamLegacy, cudaStreamDefault, or cudaStreamPerThread), since
any kernel that properly uses `cudf::get_default_stream` will now instead be using the custom stream
created by the preload library.

The preload library supports two different modes, `cudf` mode and `testing` mode. The previous
paragraph describes the behavior of `cudf` mode, where `cudf::get_default_stream` is overloaded. In
`cudf` mode, the preload library ensures that all CUDA runtime APIs are being provided cudf's
default stream. This will detect oversights where, for example, a Thrust call has no stream specified, or
when one of CUDA's default stream values is explicitly specified to a kernel. However, it will not
detect cases where a stream is not correctly forwarded down the call stack, for instance if
some `detail` function that accepts a stream parameter fails to forward it along and instead
erroneously calls `cudf::get_default_stream` instead.

In `testing` mode, the library instead overloads `cudf::test::get_default_stream`. This function
defined in the `cudf::test` namespace enables a more stringent mode of testing. In `testing` mode,
the preload library instead verifies that all CUDA runtime APIs are instead called using the test
namespace's default stream. This distinction is important because cudf internals never use
`cudf::test::get_default_stream`, so this stream value can only appear internally if it was provided
to a public API and forwarded properly all the way down the call stack. While `testing` mode is more
strict than `cudf` mode, it is also more intrusive. `cudf` mode can operate with no changes to the
library or the tests because the preload library overwrites the relevant APIs in place. `testing`
mode, however, can only be used to validate tests that are correctly passing
`cudf::test::get_default_stream` to public libcudf APIs.

In addition to the preload library, the test suite also implements a [custom memory
resource](https://github.com/NVIDIA/cudf/blob/main/cpp/include/cudf_test/stream_checking_resource_adaptor.hpp)
that performs analogous stream verification when its `do_allocate` method is called. During testing
this rmm's default memory resource is set to use this adaptor for additional stream validation.

### Usage

When writing tests for a libcudf API, a special set of additional tests should be added to validate
the API's stream usage. These tests should be placed in the `cpp/tests/streams` directory in a file
corresponding to the header containing the tested APIs, e.g. `cpp/tests/streams/copying_test.cpp`
for all APIs declared in `cpp/include/cudf/copying.hpp`. These tests should contain a minimal
invocation of the tested API with no additional assertions since they are solely designed to check
stream usage. When adding these tests to `cpp/tests/CMakeLists.txt`, the `ConfigureTest` CMake
function should be provided the arguments `STREAM_MODE testing`. This change is sufficient for
CTest to set up the test to automatically load the preload library compiled in `testing` mode when
running the test.

The rest of the test suite is configured to run with the preload library in `cudf` mode. As a
result, all test runs with `ctest` will always include stream validation. Since this configuration
is managed via CMake and CTest, direct execution of the test executables will not use the preload
library at all. Tests will still run and pass normally in this situation, however (with the
exception of the test of the preload library itself).
