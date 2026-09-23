# Optional Vortex writer

The public `cudf::io::write_vortex` API is enabled by `CUDF_WITH_VORTEX=ON`; benchmarks are not required. The option defaults to OFF.

Start with the normal [libcudf build environment](../../../../CONTRIBUTING.md).
This integration additionally requires Linux, Rustup, libclang, and the FlatBuffers
compiler required by the pinned Vortex revision. A complete CUDA toolkit is needed;
Vortex inherits the CUDA compiler, host compiler, and architectures selected by cuDF.
Use a fresh build directory rather than reusing a differently configured cuDF build.

A local full Vortex workspace can be selected with
`-DFETCHCONTENT_SOURCE_DIR_VORTEX=/absolute/path/to/vortex`.

## Build and use the public writer

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
file finalization, slices, types and explicit staging resources.

## Dependency pin

The loader pins `d196f6010777ba55658133782ad77151307e733a`, the latest upstream
Vortex `develop` commit fetched on 2026-09-23. This merged revision contains the
projected-scan, bitmap-correctness, embedding, and pipelined-read prerequisites.
The immutable pin keeps builds reproducible; it is not a release or a claim of
build/runtime validation of this cuDF branch.


## Benchmark-private GPU reader adapter

Enable `BUILD_BENCHMARKS` and `CUDF_WITH_VORTEX` to build `NDSH_VORTEX_IO` and
`NDSH_VORTEX_IO_TEST`. The adapter delegates writes to the public writer and
supports local GPU reads on device 0, ordered projections, owning cuDF results,
and explicit producer/consumer lifetime handling.

```sh
cmake -S cpp -B build-vortex -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON \
  -DCUDF_WITH_VORTEX=ON -DBUILD_TESTS=OFF
cmake --build build-vortex --target NDSH_VORTEX_IO_TEST
build-vortex/benchmarks/NDSH_VORTEX_IO_TEST
```

The adapter test covers sliced and empty tables, null/bitmap boundaries, ordered
projection, default names, host staging, owning results, stream completion,
failure/recovery and concurrent reads. With `BUILD_TESTS=ON` it is part of the
default build and registered with RAPIDS CTest GPU accounting (one whole GPU,
serial execution, 600-second timeout). Otherwise it is explicitly built.

The reader is not installed as a public cuDF API. Its targets remain build-tree-only:
the pinned Vortex revision loads CUB/nvcomp shared libraries from Cargo build paths.
Retain those artifacts at their original locations. RMM does not account for
Vortex allocations, and the adapter retains up to 8 GiB in CUDA's default memory pool.

Use `batch_rows=0` for layout-derived splitting. Nonzero scan sizes request fixed
row ranges; crossing physical blocks can require unsupported CUDA Chunked
concatenation. Direct I/O is optional for data reads; metadata remains buffered.
