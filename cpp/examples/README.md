# Libcudf Examples

This folder contains examples to demonstrate libcudf use cases. Running `build.sh` builds all
libcudf examples.

Current examples:

- Basic: demonstrates a basic use case with libcudf and building a custom application with libcudf
- Strings: demonstrates using libcudf for accessing and creating strings columns and for building custom kernels for strings
- Nested Types: demonstrates using libcudf for some operations on nested types
- Prepared Pack: demonstrates transient shuffle and spill workflows using caller-owned device and
  mapped pinned-host destinations, compressed output policies, restoration, and zero-copy unpacking
