.. dask-cudf documentation coordinating file, created by
   sphinx-quickstart on Mon Feb  6 18:48:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dask cuDF's documentation!
=====================================

**Dask cuDF** (pronounced "DASK KOO-dee-eff") is an extension
library for the `Dask <https://www.dask.org/>`__ parallel computing
framework. When installed, Dask cuDF is automatically registered
as the ``"cudf"`` dataframe backend for
`Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__.

.. note::
  Neither Dask cuDF nor Dask DataFrame provide support for multi-GPU
  or multi-node execution on their own. You must also deploy a
  `dask.distributed <https://distributed.dask.org/en/stable/>`__ cluster
  to leverage multiple GPUs. We strongly recommend using :doc:`dask-cuda:index`
  to simplify the setup of the cluster, taking advantage of all features
  of the GPU and networking hardware.

If you are familiar with Dask and `pandas <pandas.pydata.org>`__ or
`cuDF <https://docs.rapids.ai/api/cudf/stable/>`__, then Dask cuDF
should feel familiar to you. If not, we recommend starting with `10
minutes to Dask
<https://docs.dask.org/en/stable/10-minutes-to-dask.html>`__ followed
by `10 minutes to cuDF and Dask cuDF
<https://docs.rapids.ai/api/cudf/latest/user_guide/10min/>`__.

After reviewing the sections below, please see the
:ref:`Best Practices <best-practices>` page for further guidance on
using Dask cuDF effectively.


Using Dask cuDF
---------------

The Dask DataFrame API (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply use the `Dask configuration
<https://docs.dask.org/en/stable/how-to/selecting-the-collection-backend.html>`__
system to set the ``"dataframe.backend"`` option to ``"cudf"``.
From Python, this can be achieved like so::

  import dask

  dask.config.set({"dataframe.backend": "cudf"})

Alternatively, you can set ``DASK_DATAFRAME__BACKEND=cudf`` in the
environment before running your code.

Once this is done, the public Dask DataFrame API will leverage
``cudf`` automatically when a new DataFrame collection is created
from an on-disk format using any of the following ``dask.dataframe``
functions:

* :py:func:`dask.dataframe.read_parquet`
* :py:func:`dask.dataframe.read_json`
* :py:func:`dask.dataframe.read_csv`
* :py:func:`dask.dataframe.read_orc`
* :py:func:`dask.dataframe.read_hdf`
* :py:meth:`dask.dataframe.DataFrame.from_dict`


For example::

  import dask.dataframe as dd

  # By default, we obtain a pandas-backed dataframe
  df = dd.read_parquet("data.parquet", ...)

  import dask

  dask.config.set({"dataframe.backend": "cudf"})
  # This now gives us a cuDF-backed dataframe
  df = dd.read_parquet("data.parquet", ...)

When other functions are used to create a new collection
(e.g. :func:`dask.dataframe.from_map`, :func:`dask.dataframe.from_pandas`, :func:`dask.dataframe.from_delayed`,
and :func:`dask.dataframe.from_array`), the backend of the new collection will
depend on the inputs to those functions. For example::

  import pandas as pd
  import cudf

  # This gives us a pandas-backed dataframe
  dd.from_pandas(pd.DataFrame({"a": range(10)}))

  # This gives us a cuDF-backed dataframe
  dd.from_pandas(cudf.DataFrame({"a": range(10)}))

An existing collection can always be moved to a specific backend
using the :meth:`dask.dataframe.DataFrame.to_backend` API::

  # This ensures that we have a cuDF-backed dataframe
  df = df.to_backend("cudf")

  # This ensures that we have a pandas-backed dataframe
  df = df.to_backend("pandas")

The explicit Dask cuDF API
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to providing the ``"cudf"`` backend for Dask DataFrame,
Dask cuDF also provides an explicit ``dask_cudf`` API::

  import dask_cudf

  # This always gives us a cuDF-backed dataframe
  df = dask_cudf.read_parquet("data.parquet", ...)

This API is used implicitly by the Dask DataFrame API when the ``"cudf"``
backend is enabled. Therefore, using it directly will not provide any
performance benefit over the CPU/GPU-portable ``dask.dataframe`` API.
Also, using some parts of the explicit API are incompatible with
automatic query planning (see the next section).

Query Planning
~~~~~~~~~~~~~~

Dask cuDF now provides automatic query planning by default (RAPIDS 24.06+).
As long as the ``"dataframe.query-planning"`` configuration is set to
``True`` (the default) when ``dask.dataframe`` is first imported, `Dask
Expressions <https://github.com/dask/dask-expr>`__ will be used under the hood.

For example, the following code will automatically benefit from predicate
pushdown when the result is computed::

  df = dd.read_parquet("/my/parquet/dataset/")
  result = df.sort_values('B')['A']

Unoptimized expression graph (``df.pprint()``)::

  Projection: columns='A'
    SortValues: by=['B'] shuffle_method='tasks' options={}
      ReadParquetFSSpec: path='/my/parquet/dataset/' ...

Simplified expression graph (``df.simplify().pprint()``)::

  Projection: columns='A'
    SortValues: by=['B'] shuffle_method='tasks' options={}
      ReadParquetFSSpec: path='/my/parquet/dataset/' columns=['A', 'B'] ...

.. note::
  Dask will automatically simplify the expression graph (within
  :func:`dask.optimize`) when the result is converted to a task graph
  (via :func:`dask.compute` or :func:`dask.persist`). You do not need
  to optimize or simplify the graph yourself.

Using Multiple GPUs and Multiple Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever possible, Dask cuDF (i.e. Dask DataFrame) will automatically try
to partition your data into small-enough tasks to fit comfortably in the
memory of a single GPU. This means the necessary compute tasks needed to
compute a query can often be streamed to a single GPU process for
out-of-core computing. This also means that the compute tasks can be
executed in parallel over a multi-GPU cluster.

In order to execute your Dask workflow on multiple GPUs, you will
typically need to use :doc:`dask-cuda:index`
to deploy distributed Dask cluster, and
`Distributed <https://distributed.dask.org/en/stable/client.html>`__
to define a client object. For example::

  from dask_cuda import LocalCUDACluster
  from distributed import Client

  if __name__ == "__main__":

    client = Client(
      LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1",  # Use two workers (on devices 0 and 1)
        rmm_pool_size=0.9,  # Use 90% of GPU memory as a pool for faster allocations
        enable_cudf_spill=True,  # Improve device memory stability
        local_directory="/fast/scratch/",  # Use fast local storage for spilling
      )
    )

    df = dd.read_parquet("/my/parquet/dataset/")
    agg = df.groupby('B').sum()
    agg.compute()  # This will use the cluster defined above

.. note::
  This example uses :func:`dask.compute` to materialize a concrete
  ``cudf.DataFrame`` object in local memory. Never call :func:`dask.compute`
  on a large collection that cannot fit comfortably in the memory of a
  single GPU! See Dask's `documentation on managing computation
  <https://distributed.dask.org/en/stable/manage-computation.html>`__
  for more details.

Please see the :doc:`dask-cuda:index`
documentation for more information about deploying GPU-aware clusters
(including `best practices
<https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/>`__).


API Reference
-------------

Generally speaking, Dask cuDF tries to offer exactly the same API as
Dask DataFrame. There are, however, some minor differences mostly because
cuDF does not `perfectly mirror <https://docs.rapids.ai/api/cudf/stable/user_guide/pandascompat/>`__
the pandas API, or because cuDF provides additional configuration
flags (these mostly occur in data reading and writing interfaces).

As a result, straightforward workflows can be migrated without too
much trouble, but more complex ones that utilise more features may
need a bit of tweaking. The API documentation describes details of the
differences and all functionality that Dask cuDF supports.

.. toctree::
   :maxdepth: 2

   best_practices
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
