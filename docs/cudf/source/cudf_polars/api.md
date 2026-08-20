(cudf-polars-api)=
# API Reference

For the most part, the public API of `cudf-polars` is the Polars API itself. This page
documents the additional classes and functions that `cudf-polars` exposes for the streaming
multi-GPU engines.

## Streaming engines

```{eval-rst}
.. autoclass:: cudf_polars.engine.ray.RayEngine
   :members: from_options, gather_cluster_info, gather_statistics, global_statistics, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.engine.dask.DaskEngine
   :members: from_options, gather_cluster_info, gather_statistics, global_statistics, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.engine.spmd.SPMDEngine
   :members: from_options, gather_cluster_info, gather_statistics, global_statistics, shutdown, nranks, rank, comm, context
   :show-inheritance:

.. autoclass:: cudf_polars.engine.default_singleton_engine.DefaultSingletonEngine
   :members: get_or_create, shutdown
   :show-inheritance:
```

The engine classes share a common base class:

```{eval-rst}
.. autoclass:: cudf_polars.engine.core.StreamingEngine
   :members: gather_cluster_info, gather_statistics, global_statistics, shutdown, nranks
   :show-inheritance:

.. autoclass:: cudf_polars.engine.core.ClusterInfo
   :members:
```

## Persisted results

Returned by `engine.execute()` to keep query results GPU-resident (see {doc}`execute`).

```{eval-rst}
.. autoclass:: cudf_polars.engine.persisted_result.PersistedQueryResult
   :members: lazy, release
```

## Configuration

```{eval-rst}
.. autoclass:: cudf_polars.engine.options.StreamingOptions
   :members: from_dict, to_dict, to_rapidsmpf_options, to_executor_options, to_engine_options

.. autodata:: cudf_polars.engine.options.UNSPECIFIED

.. autoclass:: cudf_polars.engine.hardware_binding.HardwareBindingPolicy
   :members:

.. autofunction:: cudf_polars.engine.hardware_binding.bind_to_gpu
```

## SPMD helpers

```{eval-rst}
.. autofunction:: cudf_polars.engine.spmd.allgather_polars_dataframe

.. autofunction:: cudf_polars.streaming.actor_graph.collectives.common.reserve_op_id
```

## Internal configuration objects

These dataclasses back the `engine_options` surfaced by `pl.GPUEngine` and `StreamingOptions`.
Most users interact with them through `StreamingOptions` fields rather than directly.

```{eval-rst}
.. automodule:: cudf_polars.utils.config
   :members:
      DynamicPlanningOptions,
      JoinFilterPushdownOptions,
      MemoryResourceConfig,
      ParquetOptions,
      StreamingExecutor,
      StreamingFallbackMode,
```

## Quent Integration

cudf-polars can emit [Quent events](https://github.com/rapidsai/quent), which can be
used to profile your queries.

```{eval-rst}
.. automodule:: cudf_polars.quent
   :members:
      Attribute,
      Engine,
      HomogeneousListValue,
      Implementation,
      QuentContext,
      Query,
      QueryGroup,
      ScalarValue,
      Value,
```
