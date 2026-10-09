# Libcudf Examples

This folder contains examples to demonstrate libcudf use cases. Running `build.sh` builds all
libcudf examples.

Current examples:

- Basic: demonstrates a basic use case with libcudf and building a custom application with libcudf
- Strings: demonstrates using libcudf for accessing and creating strings columns and for building custom kernels for strings
- Nested Types: demonstrates using libcudf for some operations on nested types
- Pack/Unpack: demonstrates packing and unpacking cuDF tables to and from device and host memory

The CI runner, `ci/run_cudf_examples.sh`, defaults to two concurrent example groups and supports
`EXAMPLES_PARALLEL_LEVEL=1` for serial execution or `EXAMPLES_PARALLEL_LEVEL=2`. Invocations within
each group remain serial because they can share output files. Basic and nested-types examples
always run serially before the other groups because they reserve large memory pools.
