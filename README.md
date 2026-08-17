# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - A GPU-accelerated DataFrame library for tabular data processing</div>

cuDF (pronounced "KOO-dee-eff") is an [Apache 2.0 licensed](LICENSE), GPU-accelerated DataFrame library
for tabular data processing. The cuDF library is one part of the [RAPIDS](https://rapids.ai/) GPU
Accelerated Data Science suite of libraries.

## About

cuDF is composed of multiple libraries including:

* [libcudf](https://docs.rapids.ai/api/libcudf/stable/): A CUDA C++ library with [Apache Arrow](https://arrow.apache.org/) compliant
data structures and fundamental algorithms for tabular data.
* [pylibcudf](https://docs.rapids.ai/api/cudf/stable/pylibcudf/): A Python library providing [Cython](https://cython.org/) bindings for libcudf.
* [cudf](https://docs.rapids.ai/api/cudf/stable/cudf/): A Python library providing
    - A DataFrame library mirroring the [pandas](https://pandas.pydata.org/) API
    - A zero-code change accelerator, [cudf.pandas](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/), for existing pandas code.
* [cudf-polars](https://docs.rapids.ai/api/cudf/stable/cudf_polars/): A Python library providing a GPU engine for [Polars](https://pola.rs/)
* [dask-cudf](https://docs.rapids.ai/api/dask-cudf/stable/): A Python library providing a GPU backend for [Dask](https://www.dask.org/) DataFrames

Notable projects that use cuDF include:

* [Spark RAPIDS](https://github.com/NVIDIA/spark-rapids): A GPU accelerator plugin for [Apache Spark](https://spark.apache.org/)
* [Velox-cuDF](https://github.com/facebookincubator/velox/blob/main/velox/experimental/cudf/README.md): A [Velox](https://velox-lib.io/)
extension module to execute Velox plans on the GPU
* [Sirius](https://www.sirius-db.com/): A GPU-native SQL engine providing extensions for libraries like [DuckDB](https://duckdb.org/)

## Installation

### System Requirements

Operating System, GPU driver, and supported CUDA version information can be found at the [RAPIDS Installation Guide](https://docs.rapids.ai/install/#system-req)

### pip

A stable release of each cudf library is available on PyPI. You will need to match the major version number of your installed CUDA version with a `-cu##` suffix when installing from PyPI.

A development version of each library is available as a nightly release by including the `-i https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/` index.

```bash
# CUDA 13
pip install libcudf-cu13
pip install pylibcudf-cu13
pip install cudf-cu13
pip install cudf-polars-cu13
pip install dask-cudf-cu13

# CUDA 12
pip install libcudf-cu12
pip install pylibcudf-cu12
pip install cudf-cu12
pip install cudf-polars-cu12
pip install dask-cudf-cu12
```

### conda

A stable release of each cudf library is available to be installed with the conda package manager by specifying the `-c rapidsai` channel.

A development version of each library is available as a nightly release by specifying the `-c rapidsai-nightly` channel instead.

```bash
conda install -c rapidsai libcudf
conda install -c rapidsai pylibcudf
conda install -c rapidsai cudf
conda install -c rapidsai cudf-polars
conda install -c rapidsai dask-cudf
```

### source

To install cuDF from source, please follow [the contribution guide](CONTRIBUTING.md#setting-up-your-build-environment) detailing
how to setup the build environment.

## Examples

The following examples showcase reading a parquet file, dropping missing rows with a null value,
and performing a groupby aggregation on the data.

### cudf

`import cudf` and the APIs are largely similar to pandas.

```python
import cudf

df = cudf.read_parquet("data.parquet")
df.dropna().groupby(["A", "B"]).mean()
```

### cudf.pandas

With a Python file containing pandas code:

```python
import pandas as pd

df = pd.read_parquet("data.parquet")
df.dropna().groupby(["A", "B"]).mean()
```

Use cudf.pandas by invoking `python` with `-m cudf.pandas`

```bash
$ python -m cudf.pandas script.py
```

If running the pandas code in an interactive Jupyter environment, call `%load_ext cudf.pandas` before
importing pandas.

```python
In [1]: %load_ext cudf.pandas

In [2]: import pandas as pd

In [3]: df = pd.read_parquet("data.parquet")

In [4]: df.dropna().groupby(["A", "B"]).mean()
```

### cudf-polars

Using Polars' [lazy API](https://docs.pola.rs/user-guide/lazy/), call `collect` with `engine="gpu"` to run
the operation on the GPU

```python
import polars as pl

lf = pl.scan_parquet("data.parquet")
lf.drop_nulls().group_by(["A", "B"]).mean().collect(engine="gpu")
```

## Questions and Discussion

For bug reports or feature requests, please [file an issue](https://github.com/rapidsai/cudf/issues/new/choose) on the GitHub issue tracker.

For questions or discussion about cuDF and GPU data processing, feel free to post in the [RAPIDS Slack](https://rapids.ai/slack-invite) workspace.

## Contributing

cuDF is open to contributions from the community! Please see our [guide for contributing to cuDF](CONTRIBUTING.md) for more information.
