# Unit Benchmarking in libcudf

Unit benchmarks in libcudf are written using [NVBench](https://github.com/NVIDIA/nvbench).
While many existing benchmarks are written using
[Google Benchmark](https://github.com/google/benchmark), new benchmarks should use NVBench.

The NVBench library is similar to Google Benchmark, but has several quality of life improvements
when doing GPU benchmarking such as displaying the fraction of peak memory bandwidth achieved and
details about the GPU hardware.

Both NVBench and Google Benchmark provide many options for specifying ranges of parameters to
benchmark, as well as to control the time unit reported, among other options. Refer to existing
benchmarks in `cpp/benchmarks` to understand the options.

## Directory and File Naming

The naming of unit benchmark directories and source files should be consistent with the feature
being benchmarked. For example, the benchmarks for APIs in `copying.hpp` should live in
`cpp/benchmarks/copying`. Each feature (or set of related features) should have its own
benchmark source file named `<feature>.cu/cpp`. For example, `cpp/src/copying/scatter.cu` has
benchmarks in `cpp/benchmarks/copying/scatter.cu`.

In the interest of improving compile time, whenever possible, test source files should be `.cpp`
files because `nvcc` is slower than `gcc` in compiling host code. Note that `thrust::device_vector`
includes device code, and so must only be used in `.cu` files. `rmm::device_uvector`,
`rmm::device_buffer` and the various `column_wrapper` types described in [Testing](TESTING.md)
can be used in `.cpp` files, and are therefore preferred in test code over `thrust::device_vector`.

## CUDA Asynchrony and benchmark accuracy

CUDA computations and operations like copies are typically asynchronous with respect to host code,
so it is important to carefully synchronize in order to ensure the benchmark timing is not stopped
before the feature you are benchmarking has completed. An RAII helper class `cuda_event_timer` is
provided in `cpp/benchmarks/synchronization/synchronization.hpp` to help with this. This class
can also optionally clear the GPU L2 cache in order to ensure cache hits do not artificially inflate
performance in repeated iterations.

## Data generation

For generating benchmark input data, helper functions are available at [cpp/benchmarks/common/generate_input.hpp](https://github.com/NVIDIA/cudf/blob/main/cpp/benchmarks/common/generate_input.hpp). The input data generation happens on device, in contrast to any `column_wrapper` where data generation happens on the host.
* `create_sequence_table` can generate sequence columns starting with value 0 in first row and increasing by 1 in subsequent rows.
* `create_random_column` can generate a column filled with random data. The random data parameters are configurable.
* `create_random_table` can generate a table of columns filled with random data. The random data parameters are configurable.

## NVTX Ranges

In order to aid in performance measurement, add NVTX ranges to the data generation and/or
the function being measured. Use either the `CUDF_BENCHMARK_RANGE` or `cudf::benchmark::scoped_range`
for declaring NVTX ranges in the current scope as appropriate:
- Use the `CUDF_BENCHMARK_RANGE()` macro if you want to use the name of the function as the name of the
NVTX range
- Use `cudf::benchmark::scoped_range rng{"custom_name"};` to provide a custom name for the current scope's
NVTX range

For more information about NVTX, see [here](https://github.com/NVIDIA/NVTX/tree/dev/c).

Do not use the libcudf nvtx range classes in benchmark source code to help separate the execution
scope properly in the NSight Compute tools.

## What should we benchmark?

In general, we should benchmark all features over a range of data sizes and types, so that we can
catch regressions across libcudf changes. However, running many benchmarks is expensive, so ideally
we should sample the parameter space in such a way to get good coverage without having to test
exhaustively.

A rule of thumb is that we should benchmark with enough data to reach the point where the algorithm
reaches its saturation bottleneck, whether that bottleneck is bandwidth or computation. Using data
sets larger than this point is generally not helpful, except in specific cases where doing so
exercises different code and can therefore uncover regressions that smaller benchmarks will not
(this should be rare).

## Running and Comparing NVBench Benchmarks in libcudf

### Running Benchmarks

By default, benchmarks are **not** built as part of the libcudf build process. To enable them, pass the `BUILD_BENCHMARKS=ON` flag to CMake such as:

```bash
cmake -DBUILD_BENCHMARKS=ON cpp/build/latest
```
This will build the NVBench benchmark executables under the `cpp/build/latest/benchmarks`
directory. Each benchmark is compiled into its own binary with a `_NVBENCH` suffix.

To list available benchmarks:

```bash
ls cpp/build/latest/benchmarks/*_NVBENCH
```

To view benchmark options:

```bash
./cpp/build/latest/benchmarks/<NAME>_NVBENCH --help
```

### Comparing Benchmarks

To compare two benchmark runs, use the [nvbench_compare.py](https://github.com/NVIDIA/nvbench/blob/main/scripts/nvbench_compare.py) script provided by NVBench.
