# Libcudf Examples

This folder contains examples to demonstrate libcudf use cases. Running `build.sh` builds all
libcudf examples.

Current examples:

- Basic: demonstrates a basic use case with libcudf and building a custom application with libcudf
- Strings: demonstrates using libcudf for accessing and creating strings columns and for building custom kernels for strings
- Nested Types: demonstrates using libcudf for some operations on nested types
- Pack/Unpack: demonstrates packing and unpacking cuDF tables to and from device and host memory

The CI runner, `ci/run_cudf_examples.sh`, executes examples serially and avoids large upfront GPU memory
reservations: basic and nested-types default to CUDA allocation, and Parquet IO defaults to CUDA
async allocation. Pass `--pool` to basic or either Parquet IO executable to opt into a pool, or
select `pool` with the existing nested-types allocator argument.
