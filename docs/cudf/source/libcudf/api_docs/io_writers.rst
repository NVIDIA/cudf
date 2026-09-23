Io Writers
==========

Optional Vortex writer
----------------------

Build libcudf with ``-DCUDF_WITH_VORTEX=ON -DBUILD_SHARED_LIBS=ON`` to enable
``cudf::io::write_vortex``. ``BUILD_BENCHMARKS`` is not required. The option defaults
to OFF; in that configuration the function throws ``cudf::logic_error`` without
opening the destination. Vortex-enabled static libcudf builds are not yet supported.

Consumers include ``<cudf/io/vortex.hpp>`` and link ``cudf::cudf``; they do not need
to include or link Vortex directly:

.. code-block:: cpp

   auto options = cudf::io::vortex_writer_options::builder(
                    cudf::io::sink_info{"table.vortex"}, table.view())
                    .names({"id", "value"})
                    .rows_per_chunk(1 << 20)
                    .build();
   cudf::io::write_vortex(options, stream, mr);

This initial implementation requires Linux and current CUDA device 0. It stages
flat GPU columns to host Arrow in bounded row chunks, then compresses and writes
on the CPU. It supports one local file, not host-buffer, custom or remote sinks.
Column names default to ``_col0``, ``_col1``, etc. Nested/dictionary columns,
durations, and timestamps other than day resolution are not supported.

The supplied stream orders cuDF staging. The supplied device resource controls
staging buffers; cuDF slicing/string-compaction scratch uses the current resource,
and host/Vortex allocations are independent. The call completes staging and file
finalization before returning, but does not guarantee durable storage or atomic
replacement. A failed write may leave a partial file.

Output uses the pinned Vortex revision's **experimental CUDA-flat layout**, without
a cross-version compatibility guarantee. Building the dependency requires Rustup,
libclang and FlatBuffers in addition to the libcudf CUDA toolchain. See the
`Vortex integration build instructions <https://github.com/rapidsai/cudf/blob/main/cpp/benchmarks/ndsh/vortex/README.md>`_
for the dependency pin and source override. GPU reader benchmarks still depend on
build-tree runtime libraries; this API does not provide an installed GPU reader.
Installed-writer relocation must be validated before distribution.

With ``BUILD_TESTS=ON``, ``VORTEX_WRITER_TEST`` exercises the public writer without
requiring benchmarks.

API reference
-------------

.. doxygengroup:: io_writers
   :members:
