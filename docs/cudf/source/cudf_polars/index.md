# Polars GPU engine

cuDF provides GPU-accelerated execution engines for Python users of the Polars Lazy API. The
engines support most of the core expressions and data types as well as a growing set of more
advanced dataframe manipulations and data file formats. When a GPU engine is selected, Polars
converts expressions into an optimized query plan and determines whether the plan is supported
on the GPU. If it is not, the execution transparently falls back to the standard Polars engine
and runs on the CPU.

## Install

Follow the [RAPIDS installation guide](https://docs.rapids.ai/install/) and pick the
`cudf-polars` package for your CUDA and Python versions. For example, with conda:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cudf-polars
```

Or with pip (CUDA 13 wheels; use `cudf-polars-cu12` for CUDA 12):

```bash
pip install cudf-polars-cu13
```

## Quick start

{class}`~cudf_polars.engine.ray.RayEngine` with no arguments uses
every GPU visible to the process, so the same code runs on one GPU and scales to multi-GPU /
multi-node setups automatically:

```python
import polars as pl
from cudf_polars.engine.ray import RayEngine

query = (
    pl.scan_parquet("/data/dataset/*.parquet")
    .filter(pl.col("amount") > 100)
    .group_by("customer_id")
    .agg(pl.col("amount").sum())
)

with RayEngine() as engine:
    result = query.collect(engine=engine)
```

See {doc}`usage` for the full tutorial, {doc}`engines` for a conceptual overview of the
available engines, {doc}`options` for the
{class}`~cudf_polars.engine.options.StreamingOptions` configuration,
{doc}`execute` for keeping query results GPU-resident across chained queries, and
{doc}`memory_errors` for guidance on out-of-memory errors and memory tuning.

## Benchmark

Polars delivers high performance across a wide range of data scales through multiple execution engines. The default CPU engine is highly optimized for interactive and medium-scale analytics on a single node. The Polars GPU engine lets you move seamlessly to GPU nodes, providing meaningful acceleration when your dataset grows to hundreds of gigabytes or more.

We ran the Polars Decision Support (PDS) benchmarks to compare the Polars GPU engine with the CPU engine at larger scale factors to show how the GPU engine delivers meaningful speedups as dataset size grows:

```{eval-rst}
.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../_static/polars_pdsh_sf1k.png
          :width: 100%
          :alt: PDS-H benchmark at scale factor 1K

          PDS-H (SF1K)

     - .. figure:: ../_static/polars_pdsds_sf1k.png
          :width: 100%
          :alt: PDS-DS benchmark at scale factor 1K

          PDS-DS (SF1K)
```

On a single GPU, you can run TB-scale workloads with significant speedups compared to running on CPU. You can also scale up to run on multiple GPUs for processing even larger workloads:

```{eval-rst}
.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../_static/polars_pdsh_sf3k.png
          :width: 100%
          :alt: PDS-H benchmark at scale factor 3K

          PDS-H (SF3K)

     - .. figure:: ../_static/polars_pdsds_sf3k.png
          :width: 100%
          :alt: PDS-DS benchmark at scale factor 3K

          PDS-DS (SF3K)
```

<!-- TODO: replace this link with {doc}`benchmarks` once the published results are reproducible using those instructions -->
For more information on the benchmarks being run, see the PDS queries in the [cuDF GitHub repository](https://github.com/rapidsai/cudf/tree/main/python/cudf_polars/cudf_polars/streaming/benchmarks).

## Learn More

The GPU engine for Polars is now available in Open Beta and the engine is undergoing rapid development.
To learn more, visit the [GPU Support page](https://docs.pola.rs/user-guide/gpu-support/) on the Polars website.

```{toctree}
:maxdepth: 1
:caption: Contents:

usage
engines
options
execute
io_plugins
profiling
other_engines
memory_errors
benchmarks
api
developer_docs
```

## Launch on Google Colab

```{figure} ../_static/colab.png
:width: 200px
:target: https://nvda.ws/4eKlWZW

Try out the GPU engine for Polars in a free GPU notebook environment.
Sign in with your Google account and [launch the demo on Colab](https://nvda.ws/4eKlWZW).
```
