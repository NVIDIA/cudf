.. _best-practices:

Dask cuDF Best Practices
========================

This page outlines several important guidelines for using `Dask cuDF
<https://docs.rapids.ai/api/dask-cudf/stable/>`__ effectively.

.. note::
  Since Dask cuDF is a backend extension for
  `Dask DataFrame <https://docs.dask.org/en/stable/dataframe.html>`__,
  the guidelines discussed in the `Dask DataFrames Best Practices
  <https://docs.dask.org/en/stable/dataframe-best-practices.html>`__
  documentation also apply to Dask cuDF (excluding any pandas-specific
  details).


Deployment and Configuration
----------------------------

Use Dask-CUDA
~~~~~~~~~~~~~

To execute a Dask workflow on multiple GPUs, a Dask cluster must
be deployed with `Dask-CUDA <https://docs.rapids.ai/api/dask-cuda/stable/>`__
and `Dask.distributed <https://distributed.dask.org/en/stable/>`__.

When running on a single machine, the `LocalCUDACluster <https://docs.rapids.ai/api/dask-cuda/stable/api/#dask_cuda.LocalCUDACluster>`__
convenience function is strongly recommended. No matter how many GPUs are
available on the machine (even one!), using `Dask-CUDA has many advantages
<https://docs.rapids.ai/api/dask-cuda/stable/#motivation>`__
over default (threaded) execution. Just to list a few:

* Dask-CUDA makes it easy to pin workers to specific devices.
* Dask-CUDA makes it easy to configure memory-spilling options.
* The distributed scheduler collects useful diagnostic information that can be viewed on a dashboard in real time.

Please see `Dask-CUDA's API <https://docs.rapids.ai/api/dask-cuda/stable/>`__
and `Best Practices <https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/>`__
documentation for detailed information. Typical ``LocalCUDACluster`` usage
is also illustrated within the multi-GPU section of `Dask cuDF's
<https://docs.rapids.ai/api/dask-cudf/stable/>`__ documentation.

.. note::
  When running on cloud infrastructure or HPC systems, it is usually best to
  leverage system-specific deployment libraries like `Dask Operator
  <https://docs.dask.org/en/latest/deploying-kubernetes.html>`__ and `Dask-Jobqueue
  <https://jobqueue.dask.org/en/latest/>`__.

  Please see `the RAPIDS deployment documentation <https://docs.rapids.ai/deployment/stable/>`__
  for further details and examples.


Use diagnostic tools
~~~~~~~~~~~~~~~~~~~~

The Dask ecosystem includes several diagnostic tools that you should absolutely use.
These tools include an intuitive `browser dashboard
<https://docs.dask.org/en/stable/dashboard.html>`__ as well as a dedicated
`API for collecting performance profiles
<https://distributed.dask.org/en/latest/diagnosing-performance.html#performance-reports>`__.

No matter the workflow, using the dashboard is strongly recommended.
It provides a visual representation of the worker resources and compute
progress. It also shows basic GPU memory and utilization metrics (under
the ``GPU`` tab). To visualize more detailed GPU metrics in JupyterLab,
use `NVDashboard <https://github.com/rapidsai/jupyterlab-nvdashboard>`__.


Enable cuDF spilling
~~~~~~~~~~~~~~~~~~~~

When using Dask cuDF for classic ETL workloads, it is usually best
to enable `native spilling support in cuDF
<https://docs.rapids.ai/api/cudf/stable/cudf/developer_guide/library_design/#spilling-to-host-memory>`__.
When using :class:`dask_cuda.LocalCUDACluster`, this is easily accomplished by
setting ``enable_cudf_spill=True``.

Use RMM
~~~~~~~

Memory allocations in cuDF are significantly faster and more efficient when
the `RAPIDS Memory Manager (RMM) <https://docs.rapids.ai/api/rmm/stable/>`__
library is configured appropriately on worker processes. In most cases, the best way to manage
memory is by initializing an RMM pool on each worker before executing a
workflow. When using :class:`dask_cuda.LocalCUDACluster`, this is easily accomplished
by setting ``rmm_pool_size`` to a large fraction (e.g. ``0.9``).

See the `Dask-CUDA memory-management documentation
<https://docs.rapids.ai/api/dask-cuda/nightly/examples/best-practices/#gpu-memory-management>`__
for more details.

Use the Dask DataFrame API
~~~~~~~~~~~~~~~~~~~~~~~~~~

Although Dask cuDF provides a public ``dask_cudf`` Python module, we
strongly recommended that you use the CPU/GPU portable ``dask.dataframe``
API instead. Simply `use the Dask configuration system
<https://docs.dask.org/en/stable/how-to/selecting-the-collection-backend.html>`__
to set the ``"dataframe.backend"`` option to ``"cudf"``, and the
``dask_cudf`` module will be imported and used implicitly.

Be sure to use the :meth:`dask.dataframe.DataFrame.to_backend` method if you need to convert
between the different DataFrame backends. For example::

  df = df.to_backend("pandas")  # This gives us a pandas-backed collection

.. note::
  Although :meth:`dask.dataframe.DataFrame.to_backend` makes it easy to move data between pandas
  and cuDF, repetitive CPU-GPU data movement can degrade performance
  significantly. For optimal results, keep your data on the GPU as much
  as possible.

Avoid eager execution
~~~~~~~~~~~~~~~~~~~~~

Although Dask DataFrame collections are lazy by default, there are several
notable methods that will result in the immediate execution of the
underlying task graph:

:meth:`dask.dataframe.DataFrame.compute`: Calling ``ddf.compute()`` will materialize the result of
``ddf`` and return a single cuDF object. This is done by executing the entire
task graph associated with ``ddf`` and concatenating its partitions in
local memory on the client process.

.. note::
  Never call :meth:`dask.dataframe.DataFrame.compute` on a large collection that cannot fit comfortably
  in the memory of a single GPU!

:meth:`dask.dataframe.DataFrame.persist`: Like :meth:`dask.dataframe.DataFrame.compute`, calling ``ddf.persist()`` will
execute the entire task graph associated with ``ddf``. The most important
difference is that the computed partitions will remain in distributed
worker memory instead of being concatenated together on the client process.
Another difference is that :meth:`dask.dataframe.DataFrame.persist` will return immediately when
executing on a distributed cluster. If you need a blocking synchronization
point in your workflow, simply use the :func:`distributed.wait` function::

  ddf = ddf.persist()
  wait(ddf)

.. note::
  Avoid calling :meth:`dask.dataframe.DataFrame.persist` on a large collection that cannot fit comfortably
  in global worker memory. If the total sum of the partition sizes is larger
  than the sum of all GPU memory, calling persist will result in significant
  spilling from device memory. If the individual partition sizes are large, this
  is likely to produce an OOM error.

:func:`len` / :meth:`dask.dataframe.DataFrame.head` / :meth:`dask.dataframe.DataFrame.tail`: Although these operations are used
often within pandas/cuDF code to quickly inspect data, it is best to avoid
them in Dask DataFrame. In most cases, these operations will execute some or all
of the underlying task graph to materialize the collection.

:meth:`dask.dataframe.DataFrame.sort_values` / :meth:`dask.dataframe.DataFrame.set_index` : These operations both require Dask to
eagerly collect quantile information about the column(s) being targeted by the
global sort operation. See the next section for notes on sorting considerations.

.. note::
  When using :meth:`dask.dataframe.DataFrame.set_index`, be sure to pass in ``sort=False`` whenever the
  global collection does not **need** to be sorted by the new index.

Avoid Sorting
~~~~~~~~~~~~~

`The design of Dask DataFrame <https://docs.dask.org/en/stable/dataframe-design.html#dask-dataframe-design>`__
makes it advantageous to work with data that is already sorted along its index at
creation time. For most other cases, it is best to avoid sorting unless the logic
of the workflow makes global ordering absolutely necessary.

If the purpose of a :meth:`dask.dataframe.DataFrame.sort_values` operation is to ensure that all unique
values in ``by`` will be moved to the same output partition, then `shuffle
<https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.shuffle.html>`__
is often the better option.


Reading Data
------------

Tune the partition size
~~~~~~~~~~~~~~~~~~~~~~~

The ideal partition size is usually between 1/32 and 1/8 the memory
capacity of a single GPU. Increasing the partition size will typically
reduce the number of tasks in your workflow and improve the GPU utilization
for each task. However, if the partitions are too large, the risk of OOM
errors can become significant.

.. note::
  As a general rule of thumb, start with 1/32-1/16 for shuffle-intensive workflows
  (e.g. large-scale sorting and joining), and 1/16-1/8 otherwise. For pathologically
  skewed data distributions, it may be necessary to target 1/64 or smaller.
  This rule of thumb comes from anecdotal optimization and OOM-debugging
  experience. Since every workflow is different, choosing the best partition
  size is both an art and a science.

The easiest way to tune the partition size is when the DataFrame collection
is first created by a function like :func:`dask.dataframe.read_parquet`, :func:`dask.dataframe.read_csv`,
or :func:`dask.dataframe.from_map`. For example, both :func:`dask.dataframe.read_parquet` and :func:`dask.dataframe.read_csv`
expose a ``blocksize`` argument for adjusting the maximum partition size.

If the partition size cannot be tuned effectively at creation time, the
`repartition <https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.repartition.html>`__
method can be used as a last resort.


Use Parquet
~~~~~~~~~~~

`Parquet <https://parquet.apache.org/docs/file-format/>`__ is the recommended
file format for Dask cuDF. It provides efficient columnar storage and enables
Dask to perform valuable query optimizations like column projection and
predicate pushdown.

The most important arguments to :func:`dask.dataframe.read_parquet` are ``blocksize`` and
``aggregate_files``:

``blocksize``: Use this argument to specify the maximum partition size.
The default is `"256 MiB"`, but larger values are usually more performant
on GPUs with more than 8 GiB of memory. Dask will use the ``blocksize``
value to map a discrete number of Parquet row-groups (or files) to each
output partition. This mapping will only account for the uncompressed
storage size of each row group, which is usually smaller than the
correspondng ``cudf.DataFrame``.

``aggregate_files``: Use this argument to specify whether Dask should
map multiple files to the same DataFrame partition. The default is
``False``, but ``aggregate_files=True`` is usually more performant when
the dataset contains many files that are smaller than half of ``blocksize``.

If you know that your files correspond to a reasonable partition size
before splitting or aggregation, set ``blocksize=None`` to disallow
file splitting. In the absence of column-projection pushdown, this will
result in a simple 1-to-1 mapping between files and output partitions.

.. note::
  If your workflow requires a strict 1-to-1 mapping between files and
  partitions, use :func:`dask.dataframe.from_map` to manually construct your partitions
  with ``cudf.read_parquet``. When :func:`dask.dataframe.read_parquet` is used,
  query-planning optimizations may automatically aggregate distinct files
  into the same partition (even when ``aggregate_files=False``).

.. note::
  Metadata collection can be extremely slow when reading from remote
  storage (e.g. S3 and GCS). When reading many remote files that all
  correspond to a reasonable partition size, use ``blocksize=None``
  to avoid unnecessary metadata collection.

.. note::
  When reading from remote storage (e.g. S3 and GCS), performance will
  likely improve with ``filesystem="arrow"``. When this option is set,
  PyArrow will be used to perform IO on multiple CPU threads. Please be
  aware that this feature is experimental, and behavior may change in
  the future (without deprecation). Do not pass in ``blocksize`` or
  ``aggregate_files`` when this feature is used. Instead, set the
  ``"dataframe.parquet.minimum-partition-size"`` config to control
  file aggregation.

Use :func:`dask.dataframe.from_map`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement custom DataFrame-creation logic that is not covered by
existing APIs (like :func:`dask.dataframe.read_parquet`), use :func:`dask.dataframe.from_map`
whenever possible. The :func:`dask.dataframe.from_map` API has several advantages
over :func:`dask.dataframe.from_delayed`:

* It allows proper lazy execution of your custom logic
* It enables column projection (as long as the mapped function supports a ``columns`` key-word argument)

See the `from_map API documentation <https://docs.dask.org/en/latest/generated/dask.dataframe.from_map.html>`__
for more details.

.. note::
  Whenever possible, be sure to specify the ``meta`` argument to
  :func:`dask.dataframe.from_map`. If this argument is excluded, Dask will need to
  materialize the first partition eagerly. If a large RMM pool is in
  use on the first visible device, this eager execution on the client
  may lead to an OOM error.


Sorting, Joining, and Grouping
------------------------------

Sorting, joining, and grouping operations all have the potential to
require the global shuffling of data between distinct partitions.
When the initial data fits comfortably in global GPU memory, these
"all-to-all" operations are typically bound by worker-to-worker
communication. When the data is larger than global GPU memory, the
bottleneck is typically device-to-host memory spilling.

Although every workflow is different, the following guidelines
are often recommended:

* Use a distributed cluster with `Dask-CUDA <https://docs.rapids.ai/api/dask-cuda/stable/>`__ workers

* Use native cuDF spilling whenever possible (`Dask-CUDA spilling documentation <https://docs.rapids.ai/api/dask-cuda/stable/spilling/>`__)

* Avoid shuffling whenever possible
    * Use ``split_out=1`` for low-cardinality groupby aggregations
    * Use ``broadcast=True`` for joins when at least one collection comprises a small number of partitions (e.g. ``<=5``)

* `Use UCX <https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx/>`__ if communication is a bottleneck.

.. note::
  UCX enables Dask-CUDA workers to communicate using high-performance
  transport technologies like `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`__
  and Infiniband. Without UCX, inter-process communication will rely
  on TCP sockets.


User-defined functions
----------------------

Most real-world Dask DataFrame workflows use `map_partitions
<https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html>`__
to map user-defined functions across every partition of the underlying data.
This API is a fantastic way to apply custom operations in an intuitive and
scalable way. With that said, the :meth:`dask.dataframe.DataFrame.map_partitions` method will produce
an opaque DataFrame expression that blocks the query-planning `optimizer
<https://docs.dask.org/en/stable/dataframe-optimizer.html>`__ from performing
useful optimizations (like projection and filter pushdown).

Since column-projection pushdown is often the most effective optimization,
it is important to select the necessary columns both before and after calling
:meth:`dask.dataframe.DataFrame.map_partitions`. You can also add explicit filter operations to further
mitigate the loss of filter pushdown.
