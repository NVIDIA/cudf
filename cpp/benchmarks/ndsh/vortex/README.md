# Vortex NDS-H comparisons

Optional local-file comparisons for Q1, Q5, Q6, Q9, and Q10. The GPU reader remains
benchmark-private; writes delegate to the public `cudf::io::write_vortex` API in
`<cudf/io/vortex.hpp>`. The adapter and benchmark wiring live in cuDF; Vortex is fetched
as a library dependency. Shared comparison support lives in `../local_io.hpp`, and format-independent
CPU references live in `../reference/`.
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

`CUDF_WITH_VORTEX` defaults to `OFF`. Enabling it now configures the public writer
in libcudf even with `BUILD_BENCHMARKS=OFF`; `BUILD_SHARED_LIBS=ON` is required.
With Vortex disabled, `write_vortex` throws `cudf::logic_error` without opening the
output. A local Vortex workspace can be selected
with `-DFETCHCONTENT_SOURCE_DIR_VORTEX=/absolute/path/to/vortex`; it must contain
the full workspace, not just `lang/cpp`.

With `CUDF_WITH_VORTEX=ON`, the five `NDSH_Q*_NVBENCH` executables above are
**build-tree-only** and have no install rules. The pinned Vortex revision loads
CUB/nvcomp shared libraries (`.so` files) using absolute Cargo build paths, so
these executables are not relocatable. Run them from the original build tree and
retain the Vortex/Cargo build artifacts at their original paths. This restriction
also applies to the legacy benchmark registrations that share these executables,
not just the local-file comparisons. With `CUDF_WITH_VORTEX=OFF`, the existing
NDSH install rules remain available; unrelated benchmarks retain their install
rules in either mode.

### Public writer without benchmarks

```sh
cmake -S cpp -B build-vortex-writer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCUDF_WITH_VORTEX=ON -DBUILD_BENCHMARKS=OFF -DBUILD_TESTS=ON
cmake --build build-vortex-writer --target VORTEX_WRITER_TEST
ctest --test-dir build-vortex-writer -R '^VORTEX_WRITER_TEST$' --output-on-failure
```

Clients include `<cudf/io/vortex.hpp>` and link `cudf::cudf`. For example:

```cpp
auto options = cudf::io::vortex_writer_options::builder(
                 cudf::io::sink_info{"table.vortex"}, table.view())
                 .names({"id", "value"})
                 .rows_per_chunk(1 << 20)
                 .build();
cudf::io::write_vortex(options, stream, mr);
```

The public writer requires current device 0 and one local-file sink. It writes
flat integers, floats, booleans, strings, decimals and day-resolution timestamps,
including nullable, sliced and zero-row inputs. Other timestamp units, durations,
nested/dictionary columns, buffer/custom sinks and remote URIs are rejected.
Names default to `_col0`, `_col1`, etc. CPU compression follows row-bounded DtoH
staging; this is not GPU-native encoding. `mr` controls staging buffers, not all
cuDF scratch, host or Vortex allocations. The call blocks through staging and
file finalization, not durable disk flush, and errors may leave a partial file.
It does not change the CUDA memory-pool retention policy.

The output uses Vortex's experimental CUDA-flat layout, whose cross-version
compatibility is not guaranteed. The writer embeds the static FFI privately;
installed clients need no Vortex headers or CMake targets. The host write path
does not execute the GPU reader's CUB/nvcomp kernels, but installed-writer
relocation has not yet been validated. Do not treat this as validated runtime
packaging for the GPU reader or as a stable-format compatibility promise.

`VORTEX_WRITER_TEST` covers options, unsupported inputs, feature-OFF behavior,
file finalization, slices, types and explicit staging resources. Decoded round-trip
coverage remains in `NDSH_VORTEX_IO_TEST`, which now writes via the public API.

### Automated correctness smoke tests

Enable `BUILD_TESTS`, `BUILD_BENCHMARKS`, and `CUDF_WITH_VORTEX` together to
register the adapter test and all five local query smoke tests with CTest. In
this mode the adapter test is included in the default build, alongside the query
benchmarks. For example, from the repository root:

```sh
cmake -S cpp -B build-vortex-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON -DCUDF_WITH_VORTEX=ON
cmake --build build-vortex-tests
ctest --test-dir build-vortex-tests -L vortex --output-on-failure
```

The tests are `NDSH_VORTEX_IO_TEST` and `NDSH_Q{01,05,06,09,10}_VORTEX_SMOKE`.
They carry the `ndsh`, `vortex`, and `smoke` labels, run serially, and each has a
600-second timeout. RAPIDS CTest resource accounting reserves one whole GPU per
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
