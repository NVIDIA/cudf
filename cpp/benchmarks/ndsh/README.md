# NDS-H Benchmarks for `libcudf`

## Disclaimer

NDS-H is derived from the TPC-H Benchmarks and as such any results obtained using NDS-H are not
comparable to published TPC-H Benchmark results, as the results obtained from using NDS-H do not
comply with the TPC-H Benchmarks.

## Current Status

For now, only Q1, Q5, Q6, Q9, and Q10 have been implemented

## Code layout

- `q*.cpp`: query execution and benchmark registrations, shared across input formats.
- `utilities.hpp/.cpp`: named tables, query operations, schemas, and format-independent
  data generation.
- `parquet/parquet_io.hpp/.cpp`: Parquet reading, writing, and generated Parquet sources.
- `vortex/vortex_io.hpp/.cpp`: optional Vortex adapter; its focused test and build wiring
  live alongside it.
- `local_io.hpp`: shared local-file fixtures, format dispatch, cache control, and timing.
- `reference/`: format-independent CPU query references and validation support.

These are benchmark helpers. Vortex writes delegate to the optional installed
`cudf::io::write_vortex` API; the Vortex reader remains benchmark-private.

## Optional Parquet/Vortex comparisons

Enable `CUDF_WITH_VORTEX` together with `BUILD_BENCHMARKS` for local Parquet/Vortex
comparisons of all five queries, including concurrent Vortex table reads in
Q5/Q9/Q10. See the [Vortex build instructions](vortex/README.md).

Fixtures retain only the current scale factor per query. Switching scale factors
removes the preceding fixture's files and host reference results before generating
the replacement; revisiting an evicted scale regenerates it outside timing.

### Comparison contract

- Both formats use matched full-table fixtures and identical ordered projections.
  Local query comparisons apply filters in cuDF after reading. Existing native
  Parquet-pushdown/output benchmarks remain separate.
- Q5/Q9/Q10 overlap independent Vortex table reads, join all workers before query
  execution, and check against sequential reads during setup. Parquet table reads
  remain sequential. This concurrency difference is part of the comparison.
- Vortex reads include GPU decoding, Arrow Device import, and an owning cuDF
  materialization. Writes use CPU compression via host Arrow, outside timing.
- Cache state and I/O mode are independent axes. `cache=warm` performs a read-only
  warmup; `cache=cold` evicts the selected files from the OS page cache and requires
  zero resident pages before each timed iteration. This does not flush every cache.
- `io=buffered` is the default for both formats, including cold states. Set
  `KVIKIO_COMPAT_MODE=ON` for matched buffered comparisons: Parquet uses libcudf's
  configurable KvikIO backend, which the `io` axis does not override.
- Explicit `io=direct` runs support Vortex only; Parquet states are skipped before
  fixture generation. Vortex direct I/O bypasses the OS page cache for data reads;
  metadata remains buffered. `cache=warm,io=direct` still performs a warmup, but
  does not make subsequent data reads page-cache hits. Report direct Vortex runs
  separately from matched buffered comparisons.
- Compare CPU wall times, including reads, optional query work, owner destruction,
  and device synchronization. Generation, writes, correctness checks, and eviction
  are outside timing. RMM statistics do not cover Vortex CUDA allocations.
