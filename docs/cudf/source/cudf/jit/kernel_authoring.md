# CUDA Kernel Authoring for JIT Compilation

This document presents the mental model for developing CUDA kernels that libcudf compiles just in time (JIT). It is intended for developers familiar with CUDA programming who want to write custom kernels that can be compiled at runtime.

This page follows the source-based JIT lifecycle before introducing LTO JIT and the hybrid model.

Across these compilation models, cuDF supports two types of kernel specialization:

- **Kernel configuration specialization** determines the kernel's physical shape, including its memory layout, nullability, vectorization, and input and output types. It directly affects bandwidth usage, memory access patterns, and register usage.
- **Kernel user-defined function (UDF) specialization** determines the operations and computational logic of the kernel. It directly affects instruction count, instruction-level parallelism, register pressure, and compiler optimization opportunities.

## Source-Based JIT Model

The source-based workflow moves through five stages: construct the kernel template, embed and register the source, reflect the runtime types, specialize the UDF, and launch the compiled kernel.

### 1. Construct the Kernel Template

A kernel source template is CUDA source code parameterized to support runtime instantiation and specialization.

The following example shows a binary-operation kernel source template that accepts input columns and produces a single output column:

```cpp
// file: binary_op_kernel.cu
#include <cudf/detail/kernel_instance.cuh>  // runtime-specializes the `CUDF_KERNEL_INSTANCE` macro
#include <cudf/detail/operation_udf.cuh>    // runtime-specializes the `GENERIC_TRANSFORM_OP` macro

template <typename LhsType, typename RhsType, bool RhsIsScalar, typename OutputType>
__device__ void binary_op_kernel(column_device_view const* inputs,
                                 mutable_column_device_view const* output,
                                 size_type n)
{
  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto i = start; i < n; i += stride) {
    GENERIC_TRANSFORM_OP(&output->element<OutputType>(i),
                         inputs[0]->element<LhsType>(i),
                         inputs[1]->element<RhsType>(i * RhsIsScalar));
  }
}

extern "C" __device__ void cudf_kernel_entry(column_device_view const* inputs,
                                             mutable_column_device_view const* outputs,
                                             size_type n)
{
  CUDF_KERNEL_INSTANCE(inputs, outputs, n);
}
```

Key points:

- The `binary_op_kernel` template is parameterized by the input and output types, enabling it to be instantiated and specialized for each type combination.
- The `cudf_kernel_entry` function provides a scope-independent kernel entry point. Because its name does not depend on template parameters, callers can reference it without relying on complex name mangling.
- The `CUDF_KERNEL_INSTANCE` macro selects the input and output type specialization at runtime and represents kernel configuration specialization.
- The `GENERIC_TRANSFORM_OP` macro selects the UDF specialization at runtime and represents kernel UDF specialization.

### 2. Embed and Register the Source

The kernel source template above is embedded in the libcudf binary as a compressed string literal. Embedding the source allows it to be specialized at runtime and compiled into a kernel for GPU execution. The `rtcx_embed_includes` CMake function provided by librtcx registers and embeds kernel source templates in libcudf.

For example:

```cmake
rtcx_embed_includes(
  # ID of the target that embeds the source template
  cudf_cuda_embed
  # Directory containing the kernel source template
  SOURCE_DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sample_unary_op_kernel/jit"
  # Host installation directory for the source template
  DEST_DIRECTORY
  cudf/cpp/src/sample_unary_op_kernel/jit
  # Include directory added to the target
  INCLUDE_DIRECTORIES
  cudf/cpp/src
)
```

### 3. Reflect the Runtime Types

At runtime, users instantiate and specialize the kernel by reflecting on the runtime arguments. The argument values determine the kernel's types and configuration. This process maps the runtime input and output types to the corresponding template parameters, producing a specialization for the specific type combination.

For example:

```cpp
// reflection based on runtime parameters
std::string reflect_binary_op_kernel(data_type lhs_type /* = type_id::INT32 */,
                                     data_type rhs_type /* = type_id::FLOAT32 */,
                                     bool rhs_is_scalar /* = false */,
                                     data_type output_type /* = type_id::FLOAT32 */);

// Kernel instance specialization (substituted during call to `cudf::get_udf_kernel`)
// #define CUDF_KERNEL_INSTANCE binary_op_kernel<int32_t, float, false, float>
```

The mapping can also incorporate policies for kernel configuration, memory layout, vectorization, and architecture-specific optimization. These policies can be selected at runtime based on the properties of the input and output types.

### 4. Specialize the UDF

At runtime, the user-defined function (UDF) is resolved to CUDA code and used to specialize the `GENERIC_TRANSFORM_OP` macro in the kernel template. This specialization allows the compiler to optimize the generated binary for the specific UDF, including inlining and register- and instruction-level optimizations.

For example:

```cpp
// runtime UDF
__device__ int transform(float* out, int lhs, float rhs) { 
    *out = (lhs + rhs) * 0.5F;
    return 0;
}

// Kernel UDF specialization (substituted during call to `cudf::get_udf_kernel`)
// #define GENERIC_TRANSFORM_OP(...) transform(__VA_ARGS__)
```

### 5. Launch the Kernel

The compiled kernel can be launched on the GPU using its handle and arguments. Because the kernel uses a static entry point, its arguments should be type-erased and independent of its template parameters. This design allows the kernel to be launched without knowing those parameters.

```cpp
auto kernel_instance = "binary_op_kernel<int32_t, float, false, float>";
auto udf =
  "__device__ void transform(float * out, int lhs, float rhs){ *out = (lhs + rhs) * 0.5F; }";
auto kernel = cudf::get_udf_kernel("binary_op_kernel.cu", kernel_instance, udf);
column_device_view const* inputs          = ...;
mutable_column_device_view const* outputs = ...;
size_type n                               = ...;
kernel.launch_with({grid_size}, {block_size}, 0, stream, inputs, outputs, n);
```

`cudf::get_udf_kernel` generates the specialized CUDA source, compiles and caches the kernel, and returns a handle (`cudf::kernel`) to the compiled kernel.

As of version 26.06, PTX UDFs are supported by converting them to CUDA C++ with the `asm` directive. This approach still incurs the full cost of CUDA C++ frontend compilation.

## LTO JIT Model

The model described so far uses source-based CUDA JIT compilation. Although convenient, it has several drawbacks:

- **High compilation time**: Most JIT compilation time is spent in the CUDA C++ frontend. The kernel's translation unit contains a substantial amount of code from cuDF and its dependencies that must be processed for every compilation request, even though much of the preprocessed translation unit is redundant.
- **Correctness**: Because source-based JIT code is not compiled until runtime, compilation correctness cannot be verified ahead of time.

Link-time optimization (LTO) JIT addresses both issues and resembles the C++ translation-unit linking process.

LTO JIT allows users to define functions in separate translation units and link them at runtime. For example, the translation unit above can be compiled and distributed without defining the `transform` function. If both the kernel and the UDF are compiled ahead of time (AOT), runtime work is limited to JIT linking, bypassing the CUDA C++ frontend. Like source-based CUDA JIT, LTO JIT allows users to extend cuDF kernels and implement new functionality while providing shorter JIT compilation times and stronger AOT correctness checks. CUDA source is compiled into LTO IR, a binary program representation.

### 1. Construct the Kernel Template

An LTO-compatible binary-operation kernel template can be defined as follows:

```cpp
// file: binary_op_kernel.cu

using transform_type = CUDF_UDF_TYPE;

extern "C" transform_type transform;

template <typename LhsType, typename RhsType, bool RhsIsScalar, typename OutputType>
__device__ void binary_op_kernel(column_device_view const* inputs,
                                 mutable_column_device_view const* output,
                                 size_type n)
{
  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto i = start; i < n; i += stride) {
    transform(&output->element<OutputType>(i),
              inputs[0]->element<LhsType>(i),
              inputs[1]->element<RhsType>(i * RhsIsScalar));
  }
}

extern "C" __device__ void cudf_kernel_entry(column_device_view const* inputs,
                                             mutable_column_device_view const* outputs,
                                             size_type n)
{
  CUDF_KERNEL_INSTANCE(inputs, outputs, n);
}
```

Key points:

- The kernel template is the same as the source-based CUDA JIT template, except that it declares `transform` with external linkage.
- `CUDF_UDF_TYPE` specifies the function signature that `transform` must match.

### 2. Compile, Embed, and Register the Fragments

A compiled unit of CUDA code is called a *fragment*. A kernel compiled for later linking is called a *kernel fragment* (for example, `binary_op_kernel`). A UDF that will later be linked with a kernel is called a *UDF fragment* (for example, `transform`).

You can compile and embed an instance of the kernel above with the `add_fragment` helper function:

```cmake
add_fragment(
  # ID of cuDF's fragments
  cudf_fragments
  FRAGMENT
  # Fragment instance identifier
  binary_op_kernel_float_int32_float
  SOURCE
  # Location of the CUDA source code
  binary_op_kernel.cu
  KERNEL_INSTANCE
  # Templated kernel instance to precompile
  "binary_op_kernel<int32_t, float, false, float>"
  UDF_TYPE
  # Matching UDF type
  "int(float *, int32_t, float)"
)
```

Users can compile and embed their UDFs with the same helper function. For example, consider the following UDF:

```cpp
// file: user_udf.cu
extern "C" __device__ int transform(float* out, int32_t lhs, float rhs)
{
  *out = (lhs + rhs) * 0.5F;
  return 0;
}
```

Users can then register this UDF in their translation unit as follows:

```cmake
add_fragment(user_fragments FRAGMENT user_udf SOURCE user_udf.cu)
```

### 3. Link and Launch the Kernel

The precompiled fragments can be linked and executed at runtime with `cudf::get_lto_linked_kernel`:

```cpp
#include <cudf_fragments.hpp>
#include <user_fragments.hpp>

auto kernel_range =
  cudf_fragments::file_ranges[cudf_fragments::binary_op_kernel_float_int32_float];
std::span kernel_fragment = cudf_fragments::files.subspan(kernel_range[0], kernel_range[1]);
auto udf_range            = user_fragments::file_ranges[user_fragments::user_udf];
auto udf_fragment         = user_fragments::files.subspan(udf_range[0], udf_range[1]);
auto kernel = cudf::get_lto_linked_kernel("binary_op_kernel", {}, {kernel_fragment, udf_fragment});
column_device_view const* inputs          = ...;
mutable_column_device_view const* outputs = ...;
size_type n                               = ...;
kernel.launch_with({grid_size}, {block_size}, 0, stream, inputs, outputs, n);
```

## Hybrid JIT Model

cuDF combines CUDA JIT and LTO JIT for generic transform kernels that support multiple inputs and outputs.

The following is an example cuDF transform kernel:

```cpp
template <template <typename... I> class Inputs,
          template <typename... O> class Outputs>
extern "C" __device__ void transform_kernel(column_device_view const* inputs,
                                            mutable_column_device_view const* output,
                                            size_type n);
```

This kernel has M inputs and N outputs, each of which can support approximately 30 element types (`int32`, `float32`, `decimal32`, and others). Precompiling every kernel configuration would severely increase compilation time and binary size. This combinatorial space requires a hybrid approach: commonly used kernel fragments are precompiled ahead of time (AOT), while the remaining fragments are compiled and cached at runtime.
