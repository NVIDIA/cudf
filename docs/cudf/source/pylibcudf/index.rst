pylibcudf
=========

pylibcudf is a lightweight Cython interface to libcudf that provides near-zero overhead for GPU-accelerated data processing in Python.
It aims to provide minimal overhead interfaces to the C++ libcudf library, while integrating seamlessly with community protocols like ``__cuda_array_interface__``, and common libraries such as CuPy and Numba.
Both our zero-code pandas accelerator (``cudf.pandas``) and our polars GPU execution engine (``cudf.polars``) are built on top of pylibcudf.

Ex: Reading data from a parquet file

pylibcudf:

.. code-block:: python

   import pylibcudf as plc

   source = plc.io.SourceInfo(["dataset.parquet"])
   options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
   table = plc.io.parquet.read_parquet(options)

libcudf:

.. code-block:: cpp

   #include <cudf/io/parquet.hpp>

   int main()
   {
      auto source  = cudf::io::source_info("dataset.parquet");
      auto options = cudf::io::parquet_reader_options::builder(source).build();
      auto table  = cudf::io::read_parquet(options);
   }

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api_docs/index.rst
   developer_docs
