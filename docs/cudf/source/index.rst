NVIDIA cuDF Documentation
=========================

**NVIDIA cuDF** (pronounced "KOO-dee-eff") is a GPU-accelerated library for tabular
data processing. It is part of the `RAPIDS <https://rapids.ai/>`_ suite of
libraries and is composed of multiple sub-projects:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Library
     - Description
   * - `cudf <cudf/index.html>`_
     - A Python library providing a `pandas <https://pandas.pydata.org/>`_-like DataFrame API and a zero-code change accelerator, `cudf.pandas <cudf_pandas/index.html>`_, for existing pandas code.
   * - `cudf-polars <cudf_polars/index.html>`_
     - A Python library providing a GPU engine for `Polars <https://pola.rs/>`_.
   * - `dask-cudf <https://docs.rapids.ai/api/dask-cudf/stable/>`_
     - A Python library providing a GPU backend for `Dask <https://www.dask.org/>`_ DataFrames.
   * - `libcudf <libcudf/index.html>`_
     - A CUDA C++ library with `Apache Arrow <https://arrow.apache.org/>`_ compliant data structures and fundamental algorithms for tabular data.
   * - `pylibcudf <pylibcudf/index.html>`_
     - A Python library providing `Cython <https://cython.org/>`_ bindings for libcudf.

Accelerated Data Engines and Tools
----------------------------------

The following data engines and tools integrate with cuDF:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Data engine or tool
     - cuDF integration
     - Technical documentation
   * - Apache Spark
     - cuDF plugin for Apache Spark
     - `cuDF for Apache Spark user guide <https://docs.nvidia.com/spark-rapids/user-guide/latest/overview.html>`_
   * - DuckDB
     - Sirius
     - `Sirius documentation <https://github.com/sirius-db/sirius>`_
   * - pandas
     - cudf.pandas
     - `cudf.pandas documentation <https://docs.rapids.ai/api/cudf/stable/cudf_pandas/>`_
   * - Polars
     - Polars GPU engine
     - `Polars GPU engine documentation <https://docs.rapids.ai/api/cudf/stable/cudf_polars/>`_
   * - Presto
     - Presto-GPU
     - `Presto on GPU tutorial <https://github.com/prestodb/prestorials/tree/main/docker-compose-native/gpu>`_
   * - Velox
     - Velox on GPU (experimental)
     - `Velox-cuDF documentation <https://github.com/facebookincubator/velox/blob/main/velox/experimental/cudf/README.md>`_

.. toctree::
   :maxdepth: 1
   :caption: Libraries
   :hidden:

   cudf/index
   cudf_pandas/index
   cudf_polars/index
   libcudf/index
   pylibcudf/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer_guide/index
