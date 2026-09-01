# <div align="left"><img src="../../img/rapids_logo.png" width="90px"/>&nbsp;Dask cuDF - A GPU Backend for Dask DataFrame</div>

Dask cuDF (a.k.a. dask-cudf or `dask_cudf`) is an extension library for [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html) that provides a Pandas-like API for parallel and larger-than-memory DataFrame computing on GPUs. When installed, Dask cuDF is automatically registered as the `"cudf"` [dataframe backend](https://docs.dask.org/en/stable/how-to/selecting-the-collection-backend.html) for Dask DataFrame.

> [!IMPORTANT]
> Dask cuDF does not provide support for multi-GPU or multi-node execution on its own. You must also deploy a distributed cluster (ideally with [Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/stable/)) to leverage multiple GPUs efficiently.

## Using Dask cuDF

Please visit [the official documentation page](https://docs.rapids.ai/api/dask-cudf/stable/) for detailed information about using Dask cuDF.

## Installation

See the [RAPIDS install page](https://docs.rapids.ai/install/) for the most up-to-date information and commands for installing Dask cuDF and other RAPIDS packages.

## Resources

- [Dask cuDF documentation](https://docs.rapids.ai/api/dask-cudf/stable/)
- [Best practices](https://docs.rapids.ai/api/dask-cudf/stable/best_practices/)
- [cuDF documentation](https://docs.rapids.ai/api/cudf/stable/)
- [10 Minutes to cuDF and Dask cuDF](https://docs.rapids.ai/api/cudf/latest/user_guide/10min/)
- [Dask-CUDA documentation](https://docs.rapids.ai/api/dask-cuda/stable/)
- [Deployment](https://docs.rapids.ai/deployment/stable/)
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help, contribute, and collaborate.

### Quick-start example

A very common Dask cuDF use case is single-node multi-GPU data processing. These workflows typically use the following pattern:

```python
import dask
import dask.dataframe as dd
from dask_cuda import LocalCUDACluster
from distributed import Client

if __name__ == "__main__":

  # Define a GPU-aware cluster to leverage multiple GPUs
  client = Client(
    LocalCUDACluster(
      CUDA_VISIBLE_DEVICES="0,1",  # Use two workers (on devices 0 and 1)
      rmm_pool_size=0.9,  # Use 90% of GPU memory as a pool for faster allocations
      enable_cudf_spill=True,  # Improve device memory stability
      local_directory="/fast/scratch/",  # Use fast local storage for spilling
    )
  )

  # Set the default dataframe backend to "cudf"
  dask.config.set({"dataframe.backend": "cudf"})

  # Create your DataFrame collection from on-disk
  # or in-memory data
  df = dd.read_parquet("/my/parquet/dataset/")

  # Use cudf-like syntax to transform and/or query your data
  query = df.groupby('item')['price'].mean()

  # Compute, persist, or write out the result
  query.head()
```

If you do not have multiple GPUs available, using `LocalCUDACluster` is optional. However, it is still a good idea to [enable cuDF spilling](https://docs.rapids.ai/api/cudf/stable/cudf/developer_guide/library_design/#spilling-to-host-memory).

If you wish to scale across multiple nodes, you will need to use a different mechanism to deploy your Dask-CUDA workers. Please see [the RAPIDS deployment documentation](https://docs.rapids.ai/deployment/stable/) for more instructions.
