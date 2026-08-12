# NDS-H Benchmarks for `libcudf`

## Disclaimer

NDS-H is derived from the TPC-H Benchmarks and as such any results obtained using NDS-H are not
comparable to published TPC-H Benchmark results, as the results obtained from using NDS-H do not
comply with the TPC-H Benchmarks.

## Current Status

All 22 NDS-H queries are implemented.

The standard benchmark modes are `end_to_end`, which includes Parquet reads and query execution,
and `compute_only`, which measures query execution using preloaded data. Q9 retains its additional
engine and expression variants.

Pass `--output_directory <path>` to write the generated input tables and one query result to
Parquet for validation. Input export, the validation query execution, and result output are not
timed.
