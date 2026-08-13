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

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   cudf/index
   cudf_pandas/index
   cudf_polars/index
   libcudf/index
   pylibcudf/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer_guide/index
