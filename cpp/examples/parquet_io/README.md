# libcudf C++ examples for Parquet I/O

This C++ example demonstrates using libcudf APIs to read and write Parquet
files with different encodings and compression types.

Blog post that uses this code: https://developer.nvidia.com/blog/encoding-and-compression-guide-for-parquet-string-data-using-rapids/

The following encoding and compression types are demonstrated:
* Encoding types: `DEFAULT`, `DICTIONARY`, `PLAIN`, `DELTA_BINARY_PACKED`,
  `DELTA_LENGTH_BYTE_ARRAY`, `DELTA_BYTE_ARRAY`
* Compression types: `NONE`, `AUTO`, `SNAPPY`, `LZ4`, `ZSTD`

There are two examples included:
1. `parquet_io.cpp`
   Reads an input Parquet file, writes it back out using the specified
   encoding and compression (optionally with page statistics), reads the
   transcoded file, and validates that the data round-trips correctly. The
   write and read steps are timed.
2. `parquet_io_multithreaded.cpp`
   Reads one or more Parquet files (or directories of files) using multiple
   threads and a configurable I/O source type (`FILEPATH`, `HOST_BUFFER`,
   `PINNED_BUFFER`, `DEVICE_BUFFER`), optionally writing and validating the
   output.

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute using the included example.parquet and default encoding/compression
build/parquet_io
# Execute with explicit arguments:
#   <input file> <output file> <encoding type> <compression type> <write page stats: yes/no>
build/parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD
# Execute the multithreaded example:
#   <comma delimited list of dirs and/or files> <input multiplier> <io source type>
#   <number of times to read> <thread count> <write to temp output files and validate: yes/no>
build/parquet_io_multithreaded example.parquet
```

Pass `-h` or `--help` to either executable to print full usage information.

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
