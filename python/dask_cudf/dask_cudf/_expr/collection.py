# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from functools import cached_property

from dask import config
from dask.dataframe import get_collection_type
from dask.dataframe.core import is_dataframe_like
from dask.dataframe.dispatch import get_parallel_type
from dask.typing import no_default

import cudf

from dask_cudf._expr import (
    DXDataFrame,
    DXIndex,
    DXSeries,
    FrameBase,
    _raise_if_object_series,
    new_collection,
)

##
## Custom collection classes
##


class CudfFrameBase(FrameBase):
    def _prepare_cov_corr(self, min_periods, numeric_only):
        # Upstream version of this method sets min_periods
        # to 2 by default (which is not supported by cudf)
        # TODO: Remove when cudf supports both min_periods
        # and numeric_only
        # See: https://github.com/NVIDIA/cudf/issues/12626
        # See: https://github.com/NVIDIA/cudf/issues/9009
        self._meta.cov(min_periods=min_periods)

        frame = self
        if numeric_only:
            numerics = self._meta._get_numeric_data()
            if len(numerics.columns) != len(self.columns):
                frame = frame[list(numerics.columns)]
        return frame, min_periods

    # var can be removed if cudf#15179 is addressed.
    # See: https://github.com/NVIDIA/cudf/issues/14935
    def var(
        self,
        axis=0,
        skipna=True,
        ddof=1,
        numeric_only=False,
        split_every=False,
        **kwargs,
    ):
        _raise_if_object_series(self, "var")
        axis = self._validate_axis(axis)
        self._meta.var(axis=axis, skipna=skipna, numeric_only=numeric_only)
        frame = self
        if is_dataframe_like(self._meta) and numeric_only:
            # Convert to pandas - cudf does something weird here
            index = self._meta.to_pandas().var(numeric_only=True).index
            frame = frame[list(index)]
        return new_collection(
            frame.expr.var(
                axis, skipna, ddof, numeric_only, split_every=split_every
            )
        )

    def rename_axis(
        self, mapper=no_default, index=no_default, columns=no_default, axis=0
    ):
        from dask_cudf._expr.expr import RenameAxisCudf

        return new_collection(
            RenameAxisCudf(
                self, mapper=mapper, index=index, columns=columns, axis=axis
            )
        )


class DataFrame(DXDataFrame, CudfFrameBase):
    @classmethod
    def from_dict(cls, *args, **kwargs):
        with config.set({"dataframe.backend": "cudf"}):
            return DXDataFrame.from_dict(*args, **kwargs)

    def set_index(
        self,
        *args,
        divisions=None,
        **kwargs,
    ):
        if divisions == "quantile":
            divisions = None
            warnings.warn(
                "Ignoring divisions='quantile'. This option is now "
                "deprecated. Please raise an issue on github if this "
                "feature is necessary.",
                FutureWarning,
            )

        return super().set_index(*args, divisions=divisions, **kwargs)

    def groupby(
        self,
        by,
        group_keys=True,
        sort=None,
        observed=None,
        dropna=None,
        **kwargs,
    ):
        from dask_cudf._expr.groupby import GroupBy

        if isinstance(by, FrameBase) and not isinstance(by, DXSeries):
            raise ValueError(
                f"`by` must be a column name or list of columns, got {by}."
            )

        if "as_index" in kwargs:
            msg = (
                "The `as_index` argument is now deprecated. All groupby "
                "results will be consistent with `as_index=True`."
            )

            if kwargs.pop("as_index") is not True:
                raise NotImplementedError(
                    f"{msg} Please reset the index after aggregating."
                )
            else:
                warnings.warn(msg, FutureWarning)

        return GroupBy(
            self,
            by,
            group_keys=group_keys,
            sort=sort,
            observed=observed,
            dropna=dropna,
            **kwargs,
        )

    def to_orc(self, *args, **kwargs):
        from dask_cudf.io.orc import to_orc as to_orc_impl

        return to_orc_impl(self, *args, **kwargs)

    @staticmethod
    def read_text(*args, **kwargs):
        from dask_cudf.io.text import read_text as read_text_impl

        return read_text_impl(*args, **kwargs)

    def clip(self, lower=None, upper=None, axis=1):
        if axis not in (None, 1):
            raise NotImplementedError("axis not yet supported in clip.")
        return new_collection(self.expr.clip(lower, upper, 1))


class Series(DXSeries, CudfFrameBase):
    def groupby(self, by, **kwargs):
        from dask_cudf._expr.groupby import SeriesGroupBy

        return SeriesGroupBy(self, by, **kwargs)

    @cached_property
    def list(self):
        from dask_cudf._expr.accessors import ListMethods

        return ListMethods(self)

    @cached_property
    def struct(self):
        from dask_cudf._expr.accessors import StructMethods

        return StructMethods(self)

    def clip(self, lower=None, upper=None, axis=1):
        if axis not in (None, 1):
            raise NotImplementedError("axis not yet supported in clip.")
        return new_collection(self.expr.clip(lower, upper, 1))


class Index(DXIndex, CudfFrameBase):
    pass  # Same as pandas (for now)


# dask.dataframe dispatch
get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.Index, lambda _: Index)


# dask_expr dispatch (might go away?)
get_collection_type.register(cudf.DataFrame, lambda _: DataFrame)
get_collection_type.register(cudf.Series, lambda _: Series)
get_collection_type.register(cudf.Index, lambda _: Index)


##
## Support conversion to GPU-backed Array collections
##


def _create_array_collection_with_meta(expr):
    # NOTE: This is the GPU compatible version of
    # `create_array_collection` for cupy arrays.
    import numpy as np

    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
    from dask.layers import Blockwise

    result = expr.optimize()
    dsk = result.__dask_graph__()
    name = result._name
    meta = result._meta
    divisions = result.divisions
    chunks = (
        (np.nan,) * (len(divisions) - 1),
        *tuple((d,) for d in meta.shape[1:]),
    )
    if len(chunks) > 1:
        if isinstance(dsk, HighLevelGraph):
            layer = dsk.layers[name]
        else:
            # dask-expr provides a dict only
            layer = dsk
        if isinstance(layer, Blockwise):
            layer.new_axes["j"] = chunks[1][0]
            layer.output_indices = (*layer.output_indices, "j")
        else:
            from dask._task_spec import Alias, Task

            suffix = (0,) * (len(chunks) - 1)
            for i in range(len(chunks[0])):
                task = layer.get((name, i))
                new_key = (name, i, *suffix)
                if isinstance(task, Task):
                    task = Alias(new_key, task.key)
                layer[new_key] = task
    return da.Array(dsk, name=name, chunks=chunks, meta=meta)


@get_collection_type.register_lazy("cupy")
def _register_cupy():
    import cupy

    get_collection_type.register(
        cupy.ndarray,
        lambda _: _create_array_collection_with_meta,
    )


@get_collection_type.register_lazy("cupyx")
def _register_cupyx():
    # Needed for cuml
    from cupyx.scipy.sparse import spmatrix

    get_collection_type.register(
        spmatrix,
        lambda _: _create_array_collection_with_meta,
    )
