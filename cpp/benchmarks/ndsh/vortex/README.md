# Vortex NDS-H comparisons

Optional local-file comparisons for Q1, Q5, Q6, Q9, and Q10. The GPU reader and
CPU writer are benchmark-private, not installed libcudf APIs. `vortex_io` delegates
writing to `ndsh::write_vortex` in `writer.hpp`/`writer.cpp`; Vortex is fetched as
a library dependency. Shared comparison support lives in `../local_io.hpp`, and
independent CPU references live in `../reference/`.
The existing NDS-H/TPC-H disclaimer in the [parent README](../README.md) applies.

## Build and run

Start with the normal [libcudf build environment](../../../../CONTRIBUTING.md).
This integration additionally requires Linux, Rustup, libclang, and the FlatBuffers
compiler required by the pinned Vortex revision. A complete CUDA toolkit is needed;
Vortex inherits the CUDA compiler, host compiler, and architectures selected by cuDF.
Use a fresh build directory rather than reusing a differently configured cuDF build.

From the repository root, with any usual environment-specific CMake options:

```sh
cmake -S cpp -B build-vortex -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON \
  -DCUDF_WITH_VORTEX=ON -DBUILD_TESTS=OFF
cmake --build build-vortex --target \
  NDSH_VORTEX_IO_TEST NDSH_Q01_NVBENCH NDSH_Q05_NVBENCH \
  NDSH_Q06_NVBENCH NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH

build-vortex/benchmarks/NDSH_VORTEX_IO_TEST
KVIKIO_COMPAT_MODE=ON build-vortex/benchmarks/NDSH_Q05_NVBENCH \
  --benchmark ndsh_q5_local --devices 0 --axis scale_factor=1 \
  --axis "format=[parquet,vortex]" --axis "workload=[read,q5]" \
  --axis "cache=[warm,cold]" --axis io=buffered --json q5-local.json
```

`io=buffered` is the sole default I/O mode for both warm and cold states.
`KVIKIO_COMPAT_MODE=ON` keeps Parquet's configurable KvikIO backend buffered;
selecting `io=buffered` does not override that backend configuration.
NVBench string-axis overrides accept values outside the registered defaults, so
run optional direct Vortex measurements separately:

```sh
KVIKIO_COMPAT_MODE=ON build-vortex/benchmarks/NDSH_Q05_NVBENCH \
  --benchmark ndsh_q5_local --devices 0 --axis scale_factor=1 \
  --axis format=vortex --axis "workload=[read,q5]" \
  --axis cache=cold --axis io=direct --json q5-direct.json
```

`io=direct` bypasses the data-page cache for Vortex; metadata remains buffered.
Direct Parquet states are skipped before fixture generation. Cache control is
independent: cold states evict input pages before timing, while warm states perform
a read-only warmup even for direct I/O, without making direct data reads page-cache
hits. Neither mode flushes every storage cache. Do not mix these direct results
with matched buffered format comparisons.

The adapter test target is explicitly built above; with `BUILD_TESTS=OFF`, it is
excluded from the default build and no Vortex CTest tests are registered. Other
local benchmarks are named `ndsh_q1_local`, `ndsh_q6_local`,
`ndsh_q9_local`, and `ndsh_q10_local`. Q9 additionally has
`engine=[binaryop,ast,transform]`. Select a benchmark and scale factor explicitly:
the default axes include SF10, which requires substantial memory and disk space.
Run queries separately on device 0. Put temporary fixtures on disk, not tmpfs,
for cold-I/O measurements; the fixture directory follows `TMPDIR` when set.

`CUDF_WITH_VORTEX` defaults to `OFF`. Vortex is fetched and configured only when
both `BUILD_BENCHMARKS` and `CUDF_WITH_VORTEX` are `ON`; the integration imposes no
`BUILD_SHARED_LIBS` restriction. With either option disabled, the private Vortex
writer and adapter are not configured; there is no public feature-OFF writer stub.
A local Vortex workspace can be selected with
`-DFETCHCONTENT_SOURCE_DIR_VORTEX=/absolute/path/to/vortex`; it must contain the
full workspace, not just `lang/cpp`.

With `CUDF_WITH_VORTEX=ON`, the five `NDSH_Q*_NVBENCH` executables above are
**build-tree-only** and have no install rules. The pinned Vortex revision loads
CUB/nvcomp shared libraries (`.so` files) using absolute Cargo build paths, so
these executables are not relocatable. Run them from the original build tree and
retain the Vortex/Cargo build artifacts at their original paths. This restriction
also applies to the legacy benchmark registrations that share these executables,
not just the local-file comparisons. With `CUDF_WITH_VORTEX=OFF`, the existing
NDSH install rules remain available; unrelated benchmarks retain their install
rules in either mode.

### Benchmark-private writer

The private writer requires current device 0 and one local-file sink. It writes
flat integers, floats, booleans, strings, decimals and day-resolution timestamps,
including nullable, sliced and zero-row inputs. Other timestamp units, durations,
nested/dictionary columns, buffer/custom sinks and remote URIs are rejected.
Names default to `_col0`, `_col1`, etc. CPU compression follows row-bounded DtoH
staging; this is not GPU-native encoding. `mr` controls staging buffers, not all
cuDF scratch, host or Vortex allocations. The call blocks through staging and
file finalization, not durable disk flush, and errors may leave a partial file.
It does not change the CUDA memory-pool retention policy.

The output uses Vortex's experimental CUDA-flat layout, whose cross-version
compatibility is not guaranteed. The benchmark-private integration links the
static Vortex FFI; this does not provide installed runtime packaging or remove
the GPU reader's build-tree `.so` dependency.

`NDSH_VORTEX_WRITER_TEST` is a private GTest executable for `writer_test.cpp`,
covering options, unsupported inputs, file finalization, slices, types and
explicit staging resources. It is separate from `NDSH_VORTEX_IO_TEST`, whose
`vortex_io_test.cpp` has a custom `main` and checks decoded round trips through
the adapter and private writer. The writer GTest requires `BUILD_TESTS`,
`BUILD_BENCHMARKS`, and `CUDF_WITH_VORTEX`.

### Automated correctness smoke tests

Enable `BUILD_TESTS`, `BUILD_BENCHMARKS`, and `CUDF_WITH_VORTEX` together to
register the private writer GTest, the adapter test, and all five local query
smoke tests with CTest. In this mode the adapter test is included in the default
build, alongside the query benchmarks. For example, from the repository root:

```sh
cmake -S cpp -B build-vortex-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON -DCUDF_WITH_VORTEX=ON
cmake --build build-vortex-tests
ctest --test-dir build-vortex-tests -R '^NDSH_VORTEX_WRITER_TEST$' --output-on-failure
ctest --test-dir build-vortex-tests -L vortex --output-on-failure
ctest --test-dir build-vortex-tests \
  -R '^(MEMORY_STATS_LOGGER_TEST|NDSH_FIXTURE_CACHE_TEST)$' --output-on-failure
```

`MEMORY_STATS_LOGGER_TEST` remains benchmark-local, alongside the existing
`NDSH_FIXTURE_CACHE_TEST`. Both require `BUILD_BENCHMARKS` and `BUILD_TESTS`,
but neither requires Vortex. The cache test checks fixture reuse and eviction;
it does not validate OS page-cache behavior for cold-I/O measurements.

The adapter/query smoke tests are `NDSH_VORTEX_IO_TEST` and
`NDSH_Q{01,05,06,09,10}_VORTEX_SMOKE`. They carry the `ndsh`, `vortex`, and `smoke`
labels, run serially, and each has a 600-second timeout. RAPIDS CTest resource accounting reserves one whole GPU per
test and maps it to logical device 0; the tests remain build-tree-only. The query
tests select device 0, SF0.01, both Parquet and
Vortex, the corresponding query workload, warm cache, and `io=buffered` only. Q9 explicitly
covers `binaryop`, `ast`, and `transform`. These are ordinary NVBench invocations
using the existing CLI, not a measurement-skipping mode; setup runs the query
correctness checks before timing. NVBench `Fail:` or `Skip:` output fails the
CTest smoke test as well as a nonzero process exit status.

A CUDA-capable device 0 and writable fixture storage are required. Cold-I/O
measurements remain manual and disk-dependent, using the invocation above; they
are not part of automated correctness coverage. These small-scale smoke runs
are not performance validation. The build-tree-only restriction still applies.

### Dependency pin

The loader pins `d196f6010777ba55658133782ad77151307e733a`, the latest upstream
Vortex `develop` commit fetched on 2026-09-23. This merged revision contains the
projected-scan, bitmap-correctness, embedding, and pipelined-read prerequisites.
The immutable pin keeps builds reproducible; it is not a release or a claim of
build/runtime validation of this cuDF branch.

The cuDF Arrow host-transfer cleanup fix is a prerequisite: export failure paths
must drain outstanding transfers before releasing host buffers.

Local comparisons use layout-derived scan batches (`batch_rows=0`). At this
revision, nonzero scan sizes specify fixed row ranges rather than layout-preserving
maximums. Ranges crossing physical blocks may require unsupported CUDA `Chunked`
concatenation, so explicit-size adapter tests use compatible physical boundaries.

## Comparison contract

See the shared [Parquet/Vortex comparison contract](../README.md#comparison-contract).

## Correctness and limitations

The adapter test covers sliced and empty tables, bitmap boundaries, ordered
projection, metadata, host string staging, owning results, and stream completion.
Query setup checks projected tables and independent CPU references, with synthetic
boundary/null/duplicate-join cases. These checks are outside benchmark timing.

The original cuDF data generator is unchanged. It can produce empty Q6/Q10 results
and sparse low-scale joins; report match counts and do not treat empty queries as
representative full-query performance. Generator corrections are a separate change.

The adapter currently supports local files, device 0, and flat typed columns. It
retains up to 8 GiB in the CUDA default memory pool; peak memory also includes
retained Vortex batches and the owning cuDF result. Concurrent reads increase the
number of live tables. No performance results from the earlier external harness
validate this upstream branch; fresh builds and GPU runs are still required.
