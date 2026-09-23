# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools

from dask.dataframe.dispatch import (
    is_categorical_dtype,
    make_meta,
    meta_nonempty,
)
from dask.typing import no_default
from dask.utils import is_dataframe_like

import cudf

from dask_cudf._expr import (
    CumulativeBlockwise,
    Elemwise,
    EnforceRuntimeDivisions,
    Expr,
    Reduction,
    RenameAxis,
    Var,
    VarColumns,
    _shuffle_module,
    new_collection,
)

##
## Custom expressions
##


class RenameAxisCudf(RenameAxis):
    # TODO: Remove this after rename_axis is supported in cudf
    # (See: https://github.com/NVIDIA/cudf/issues/16895)
    @staticmethod
    def operation(df, index=no_default, **kwargs):
        if index != no_default:
            df.index.name = index
            return df
        raise NotImplementedError(
            "Only `index` is supported for the cudf backend"
        )


class ToCudfBackend(Elemwise):
    # TODO: Inherit from ToBackend when rapids-dask-dependency
    # is pinned to dask>=2024.8.1
    _parameters = ["frame", "options"]
    _projection_passthrough = True
    _filter_passthrough = True
    _preserves_partitioning_information = True

    @staticmethod
    def operation(df, options):
        from dask_cudf.backends import to_cudf_dispatch

        return to_cudf_dispatch(df, **options)

    def _simplify_down(self):
        if isinstance(
            self.frame._meta, (cudf.DataFrame, cudf.Series, cudf.Index)
        ):
            # We already have cudf data
            return self.frame


##
## Custom expression patching
##


# This can be removed after cudf#15176 is addressed.
# See: https://github.com/NVIDIA/cudf/issues/15176
class PatchCumulativeBlockwise(CumulativeBlockwise):
    @property
    def _args(self) -> list:
        return self.operands[:1]

    @property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


# The upstream Var code uses `Series.values`, and relies on numpy
# for most of the logic. Unfortunately, cudf -> cupy conversion
# is not supported for data containing null values. Therefore,
# we must implement our own version of Var for now. This logic
# is mostly copied from dask-cudf.


class VarCudf(Reduction):
    # Uses the parallel version of Welford's online algorithm (Chan '79)
    # (http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
    _parameters = [
        "frame",
        "skipna",
        "ddof",
        "numeric_only",
        "split_every",
    ]
    _defaults = {
        "skipna": True,
        "ddof": 1,
        "numeric_only": False,
        "split_every": False,
    }

    @functools.cached_property
    def _meta(self):
        return make_meta(
            meta_nonempty(self.frame._meta).var(
                skipna=self.skipna, numeric_only=self.numeric_only
            )
        )

    @property
    def chunk_kwargs(self):
        return dict(skipna=self.skipna, numeric_only=self.numeric_only)

    @property
    def combine_kwargs(self):
        return {}

    @property
    def aggregate_kwargs(self):
        return dict(ddof=self.ddof)

    @classmethod
    def reduction_chunk(cls, x, skipna=True, numeric_only=False):
        kwargs = {"numeric_only": numeric_only} if is_dataframe_like(x) else {}
        if skipna or numeric_only:
            n = x.count(**kwargs)
            kwargs["skipna"] = skipna
            avg = x.mean(**kwargs)
        else:
            # Not skipping nulls, so might as well
            # avoid the full `count` operation
            n = len(x)
            kwargs["skipna"] = skipna
            avg = x.sum(**kwargs) / n
        if numeric_only:
            # Workaround for cudf bug
            # (see: https://github.com/NVIDIA/cudf/issues/13731)
            x = x[n.index]
        m2 = ((x - avg) ** 2).sum(**kwargs)
        return n, avg, m2

    @classmethod
    def reduction_combine(cls, parts):
        n, avg, m2 = parts[0]
        for i in range(1, len(parts)):
            n_a, avg_a, m2_a = n, avg, m2
            n_b, avg_b, m2_b = parts[i]
            n = n_a + n_b
            avg = (n_a * avg_a + n_b * avg_b) / n
            delta = avg_b - avg_a
            m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
        return n, avg, m2

    @classmethod
    def reduction_aggregate(cls, vals, ddof=1):
        vals = cls.reduction_combine(vals)
        n, _, m2 = vals
        return m2 / (n - ddof)


def _patched_var(
    self,
    axis=0,
    skipna=True,
    ddof=1,
    numeric_only=False,
    split_every=False,
):
    if axis == 0:
        if hasattr(self._meta, "to_pandas"):
            return VarCudf(self, skipna, ddof, numeric_only, split_every)
        else:
            return Var(self, skipna, ddof, numeric_only, split_every)
    elif axis == 1:
        return VarColumns(self, skipna, ddof, numeric_only)
    else:
        raise ValueError(f"axis={axis} not supported. Please specify 0 or 1")


# Temporary work-around for missing cudf + categorical support
# See: https://github.com/NVIDIA/cudf/issues/11795
# TODO: Fix RepartitionQuantiles and remove this in cudf>24.06

_original_get_divisions = _shuffle_module._get_divisions


def _patched_get_divisions(frame, other, *args, **kwargs):
    # NOTE: The following two lines contains the "patch"
    # (we simply convert the partitioning column to pandas)
    if is_categorical_dtype(other._meta.dtype) and hasattr(
        other.frame._meta, "to_pandas"
    ):
        other = new_collection(other).to_backend("pandas")._expr

    # Call "original" function
    return _original_get_divisions(frame, other, *args, **kwargs)


_original_erd_divisions = EnforceRuntimeDivisions._divisions


def _patched_erd_divisions(self):
    # This patch is needed for upstream dask testing
    # (dask/dataframe/tests/test_indexing.py::test_gpu_loc).
    # Without this patch, an individual element of divisions
    # may end up as a 0-dim cupy array.
    # TODO: Find long-term fix.
    # Maybe update `LocList._layer_information`?
    divs = _original_erd_divisions(self)
    return tuple(div.item() if hasattr(div, "item") else div for div in divs)


_PATCHED = False


def _patch_dask_expr():
    global _PATCHED

    if not _PATCHED:
        CumulativeBlockwise._args = PatchCumulativeBlockwise._args
        CumulativeBlockwise._kwargs = PatchCumulativeBlockwise._kwargs
        Expr.var = _patched_var
        _shuffle_module._get_divisions = _patched_get_divisions
        EnforceRuntimeDivisions._divisions = _patched_erd_divisions
        _PATCHED = True
