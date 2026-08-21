# JIT Performance Guidelines

## When to Use JIT

### 1. Dynamic Expressions

Kernels generated from dynamic expressions, such as SQL expressions and UDFs, can benefit from JIT compilation when those expressions are not known at compile time. JIT compilation can specialize the kernel for the specific expression and input types, potentially improving performance. This benefit is especially important for expressions that run repeatedly because the compilation overhead can be amortized across many executions.

The JIT compiler can optimize for the arithmetic intensity and memory access patterns of each expression. Static expressions known at compile time benefit most from standard AOT kernel compilation.

libcudf also provides a precompiled AST interpreter that evaluates expressions without JIT compilation. Although the interpreter is generally slower than JIT-compiled code, it provides lower cold-start latency for one-time expression evaluation. Its limitations include:

- The expression types it supports
- The input types it supports
- The input dimensionality it supports (scalar versus columnar)
- Register pressure and memory bandwidth usage for complex expressions
- Materializing intermediate results in shared memory instead of registers, which can reduce performance for complex expressions
- Longer AOT compilation and libcudf build times because the interpreter must support all possible expressions and input types, creating a combinatorial number of code paths for the compiler to optimize

### 2. Architecture-Specific Optimizations and Policies

JIT compilation is useful for workloads whose optimization policies—such as memory layouts, vectorization policies, kernel configurations, and GPU-specific tuning—cannot be determined until runtime. Precompiling kernels for every possible configuration is infeasible because of the combinatorial number of code paths and the resulting AOT compilation time. Instead, JIT compilation generates kernels specialized for the current workload, potentially improving code generation and performance.

### 3. Long-Running High-Throughput Workloads

JIT compilation is most beneficial for long-running, high-throughput workloads that can amortize compilation overhead across many executions. The specialized code generated for a specific workload can provide significant performance improvements.

Compilation overhead can still be substantial, especially for complex expressions. We continue to optimize the JIT compilation path, but workload design should account for the trade-off between compilation latency and execution performance. The next section explains how cache state affects that latency.

Workloads that require low cold-start latency may favor precompiled kernels over JIT compilation.

## How Caching Affects JIT Performance

JIT latency depends on whether compiled kernels and precompiled headers are already cached. Establish the cache state before comparing compilation or execution times.

### Cold and Hot Runs

A cold run compiles a kernel that has not yet been cached, whereas a hot run reuses a cached kernel. Cold runs can be significantly slower because JIT compilation includes parsing and resolving C++ code, generating PTX, and compiling the PTX into machine code. Determine whether the workload can tolerate cold-run latency or run long enough to amortize it. Benchmarks should include both cold and hot runs to characterize the workload accurately.

### Precompiled Header Cache

libcudf uses precompiled headers (PCHs) to speed up CUDA JIT compilation. PCHs are fully resolved ASTs of the CUDA C++ headers. They eliminate the need to parse and resolve that code during each JIT compilation. The details of this representation are internal to NVRTC; they are neither exposed to users nor portable across compilers.

PCHs are currently generated during the first cold JIT compilation of a kernel, so the first request always uses a cold PCH cache. The generated PCHs are cached and reused for subsequent kernel compilations. Consequently, the first cold compilation is slower because it incurs the full cost of CUDA C++ frontend compilation.

NVRTC's PCH cache is **process-local** and is not shared across processes. Therefore, PCHs generated while compiling a kernel in one process are unavailable to other processes. This limitation can reduce performance in multiprocess environments that compile the same kernel in several processes. Preparing NVRTC PCHs ahead of time would eliminate this cold-cache overhead.

## Benchmarking JIT Performance

After establishing the cache state, consider the workload's latency requirements, arithmetic intensity, and reuse pattern:

- **Latency requirements**: Workloads with strict latency requirements may need precompiled kernels or the AST interpreter instead of JIT compilation. Prewarming a JIT kernel can reduce cold-start latency, but it may be infeasible when the workload has a combinatorial number of possible code paths or when AOT compilation is expensive.

- **Arithmetic intensity and intermediate count**: Workloads with a high ratio of arithmetic operations to memory operations can benefit more from JIT compilation because they can amortize compilation overhead across more computation. The number of intermediate results also affects performance. To evaluate arbitrary expressions without materializing intermediates in global memory, cuDF provides both a precompiled AST interpreter and a JIT compiler. The AST interpreter materializes intermediates in shared memory to reduce global memory bandwidth. This cost is small for expressions with few intermediates, but it increases as the number of intermediates grows. For complex expressions, the JIT compiler can generate specialized code that keeps intermediates in registers and potentially improves performance.

- **Amortization of JIT compilation overhead**: Determine the break-even point at which repeated execution offsets the compilation cost. This point depends primarily on the number of rows processed and how often the kernel is reused.

## JIT Performance Tuning Options

Use the following environment variables and APIs to control cache behavior, select JIT execution, and collect diagnostics.

### Kernel Cache Controls

#### `LIBCUDF_KERNEL_CACHE_PATH:path`

This environment variable sets the location of the libcudf JIT kernel cache. In a multiprocess container environment, such as a Presto multi-container executor, set it to a shared location so that processes can reuse compiled kernels. If the variable is unset, libcudf attempts to find a suitable, accessible filesystem path for compilation artifacts and kernels.

#### `LIBCUDF_KERNEL_CACHE_PRELOAD:bool`

This environment variable controls whether libcudf preloads previously compiled kernels from the kernel cache into memory (`cuLibrary`). Preloading eliminates kernel-loading overhead from hot benchmark measurements.

#### `LIBCUDF_KERNEL_CACHE_DISABLED:bool`

This environment variable disables caching of libcudf kernel artifacts, ensuring that benchmarked JIT compilations use the cold-cache path. For fully cold-cache measurements, use it with `LIBCUDF_JIT_DISABLE_CUDA_CACHE` to disable cuDF's CUDA code-generation cache.

#### `LIBCUDF_KERNEL_CACHE_CLEAR:bool`

This environment variable removes all kernel artifacts from libcudf's JIT kernel cache.

#### `LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS:integer`

This environment variable limits the number of kernels cached by each process. Use it to control the cache memory footprint in multiprocess environments or to disable kernel caching for benchmarks.

### JIT and CUDA Code-Generation Controls

#### `LIBCUDF_JIT_ENABLED:bool`

This environment variable tells libcudf subsystems to prefer JIT compilation for expression evaluation. For example, it applies to the Parquet reader post-filter subsystem, `compute_column`, and other functions that accept an `AST` tree. JIT evaluation can significantly improve the performance of expressions with high arithmetic intensity.

#### `LIBCUDF_JIT_DISABLE_CUDA_CACHE:bool`

This environment variable disables cuDF's CUDA code-generation cache, including PTX and CUBIN generation after CUDA frontend processing.

### Diagnostics

#### `LIBCUDF_JIT_VERBOSE:bool`

This environment variable makes libcudf print verbose JIT compilation information for debugging and benchmarking, including:

- NVRTC compilation times (CUDA C++ frontend, PTXAS, and code generation)
- NVRTC include headers
- NVRTC compilation flags

#### `LIBCUDF_JIT_DUMP_TRACE:bool`

This environment variable writes a text JIT compilation trace to the command line for debugging and performance analysis. The trace includes:

- NVRTC compilation times (CUDA C++ frontend, PTXAS, and code generation)
- Optimization passes and their times
- NVRTC passes and their times

#### `LIBCUDF_JIT_DUMP_TIME_PROFILE:bool`

This environment variable produces a Perfetto JSON trace instead of a text trace. The output is named `libcudf_kernel_${kernel_name}_trace.json` (for example, `libcudf_kernel_transform_trace.json` or `libcudf_kernel_filter_join_indices_trace.json`) and is written to the current working directory. Open the file with Chrome's tracing tool or the Perfetto UI.

### Programmatic Cache Controls

#### `cudf::enable_jit_cache(bool)`

This function enables or disables the libcudf JIT kernel cache. Disable the cache for cold-cache benchmarks, and combine it with `LIBCUDF_JIT_DISABLE_CUDA_CACHE` for fully cold-cache measurements. It is the programmatic equivalent of `LIBCUDF_KERNEL_CACHE_DISABLED`.

#### `cudf::enable_cuda_cache(bool)`

This function enables or disables cuDF's CUDA code-generation cache. Disable the cache to ensure that benchmarked JIT compilations use the cold code-generation path. It is the programmatic equivalent of `LIBCUDF_JIT_DISABLE_CUDA_CACHE`.

#### `cudf::clear_jit_cache()`

This function removes all artifacts from libcudf's JIT kernel cache. Use it with `cudf::enable_cuda_cache(false)` for fully cold-cache benchmark measurements. It is the programmatic equivalent of `LIBCUDF_KERNEL_CACHE_CLEAR`.
