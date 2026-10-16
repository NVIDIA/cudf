# NDS-H Benchmarks for `libcudf`

## Disclaimer

NDS-H is derived from the TPC-H Benchmarks and as such any results obtained
using NDS-H are not comparable to published TPC-H Benchmark results, as the
results obtained from using NDS-H do not comply with the TPC-H Benchmarks.

## Current Status

For now, only Q1, Q5, Q6, Q9, and Q10 have been implemented

## Vortex comparisons

Enable `CUDF_WITH_VORTEX` and `BUILD_BENCHMARKS` to compare Parquet and Vortex
for all five queries. See the [Vortex README](vortex/README.md) for build and
run instructions, tests, and the comparison contract.
