# Vortex NDS-H comparisons

Compare local Parquet and Vortex files for Q1, Q5, Q6, Q9, and Q10. The Vortex GPU
reader and CPU writer are private benchmark helpers, not libcudf APIs. The
[NDS-H disclaimer](../README.md#disclaimer) applies.

## Build

Use the [libcudf build environment](../../../../CONTRIBUTING.md), plus Linux,
Rustup, libclang, and the FlatBuffers compiler required by the pinned Vortex
revision. From the repository root:

```sh
cmake -S cpp -B build-vortex -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON \
  -DBUILD_TESTS=ON -DCUDF_WITH_VORTEX=ON
cmake --build build-vortex --target \
  NDSH_Q01_NVBENCH NDSH_Q05_NVBENCH NDSH_Q06_NVBENCH \
  NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH \
  NDSH_QUERY_TEST NDSH_VORTEX_IO_TEST NDSH_VORTEX_WRITER_TEST
```

`CUDF_WITH_VORTEX` defaults to `OFF`. To use a local Vortex workspace, set
`FETCHCONTENT_SOURCE_DIR_VORTEX` to its root directory.

Vortex-enabled executables are **build-tree-only**: Vortex loads CUB/nvcomp
libraries from absolute Cargo build paths. Keep those artifacts in place;
the executables are not relocatable.

## Run

```sh
KVIKIO_COMPAT_MODE=ON build-vortex/benchmarks/NDSH_Q05_NVBENCH \
  --benchmark ndsh_q5_local --devices 0 --axis scale_factor=1 \
  --axis "format=[parquet,vortex]" --axis "workload=[read,q5]" \
  --axis "cache=[warm,cold]" --axis io=buffered --json q5-local.json
```

Use `ndsh_q{1,5,6,9,10}_local` with the corresponding executable and query
workload. Q9 also accepts `--axis "engine=[binaryop,ast,transform]"`.
Select the scale explicitly; defaults include SF10. Run queries separately on
device 0. Cold measurements require disk-backed fixture storage (`TMPDIR`),
not tmpfs.

## Tests

```sh
ctest --test-dir build-vortex -R '^NDSH_QUERY_TEST$' --output-on-failure
KVIKIO_COMPAT_MODE=ON ctest --test-dir build-vortex -L vortex --output-on-failure
```

Unit tests require `BUILD_TESTS`; reader/writer tests additionally require
`CUDF_WITH_VORTEX`. They can be built with `BUILD_BENCHMARKS=OFF`. Query regression
tests cover synthetic boundary and join cases. Query smoke tests require
benchmarks and run both formats at SF0.01 with warm, buffered I/O, including all
Q9 engines. Cold measurements are manual.

## Comparison contract

- Both formats use the same logical full-table data and ordered projections.
  Filters run in cuDF after reading; native Parquet-pushdown benchmarks are separate.
- Q5/Q9/Q10 read independent Vortex tables concurrently; Parquet reads are sequential.
- Compare **CPU wall time**, including reads, optional query execution, owning
  cuDF materialization, cleanup, and device synchronization. Fixture generation,
  writes, correctness checks, and cache eviction are outside timing.
- `cache=warm` performs a read-only warmup. `cache=cold` requires zero resident
  input pages before each timed iteration. This controls the OS page cache,
  not hardware or storage caches.
- `io=buffered` is the default. Set `KVIKIO_COMPAT_MODE=ON` to keep Parquet's
  KvikIO backend buffered. Optional `--axis format=vortex --axis io=direct`
  bypasses the data-page cache, but not metadata caching; report it separately.
  Direct Parquet states are skipped.
- Fixtures retain one scale per query; changing scale regenerates them outside timing.
  Report matched-row counts: generated Q6/Q10 results can be empty, and low-scale
  joins can be sparse.

## Limitations

- Local files, device 0, and flat typed columns only. The writer uses CPU
  compression after bounded DtoH staging; it is not a GPU encoder. See
  [`vortex_writer_options`](writer.hpp) for supported types and sink restrictions.
- Host Arrow export failures can free buffers before pending DtoH transfers
  finish. An outer writer stream drain cannot protect buffers freed inside export.
- The CUDA-flat file layout is experimental, without cross-version compatibility
  guarantees. Reads use layout-derived batches; explicit batch ranges crossing
  physical blocks can require unsupported CUDA concatenation.
- Vortex can retain up to 8 GiB in the CUDA memory pool. RMM statistics exclude
  Vortex allocations; concurrent reads also increase peak memory.
