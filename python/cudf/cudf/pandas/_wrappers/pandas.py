# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import contextlib
import copyreg
import datetime
import functools
import inspect
import os
import pickle

import pandas as pd

# cuGraph third party integration test, test_cugraph_from_pandas_adjacency,
# fails without this pyarrow.dataset import
# I suspect it relates to pyarrow's pandas-shim that gets imported
# with this module https://github.com/rapidsai/cudf/issues/14521#issue-2015198786
import pyarrow.dataset as ds  # noqa: F401
from pandas._libs.tslibs import offsets as liboffsets
from pandas._testing import at, getitem, iat, iloc, loc, setitem
from pandas.compat._optional import import_optional_dependency
from pandas.io.excel._openpyxl import OpenpyxlWriter as pd_OpenpyxlWriter
from pandas.io.excel._xlsxwriter import XlsxWriter as pd_XlsxWriter
from pandas.tseries.holiday import (
    AbstractHolidayCalendar as pd_AbstractHolidayCalendar,
    EasterMonday as pd_EasterMonday,
    GoodFriday as pd_GoodFriday,
    Holiday as pd_Holiday,
    HolidayCalendarFactory as pd_HolidayCalendarFactory,
    HolidayCalendarMetaClass as pd_HolidayCalendarMetaClass,
    USColumbusDay as pd_USColumbusDay,
    USFederalHolidayCalendar as pd_USFederalHolidayCalendar,
    USLaborDay as pd_USLaborDay,
    USMartinLutherKingJr as pd_USMartinLutherKingJr,
    USMemorialDay as pd_USMemorialDay,
    USPresidentsDay as pd_USPresidentsDay,
    USThanksgivingDay as pd_USThanksgivingDay,
)

import cudf

from ..annotation import nvtx
from ..fast_slow_proxy import (
    _CUDF_PANDAS_NVTX_COLORS,
    _DELETE,
    _fast_slow_function_call,
    _FastSlowAttribute,
    _FunctionProxy,
    _maybe_wrap_result,
    _setattr_fsproxy_no_mirror,
    _State,
    _Unusable,
    is_proxy_object,
    make_final_proxy_type as _make_final_proxy_type,
    make_intermediate_proxy_type as _make_intermediate_proxy_type,
    register_proxy_func,
)
from .common import (
    array_function_method,
    array_method,
    arrow_array_method,
    cuda_array_interface,
    custom_iter,
)

from pandas.io.sas.sas7bdat import (  # isort: skip
    SAS7BDATReader as pd_SAS7BDATReader,
)
from pandas.io.sas.sas_xport import (  # isort: skip
    XportReader as pd_XportReader,
)

try:
    from IPython import get_ipython

    ipython_shell = get_ipython()
except ImportError:
    ipython_shell = None


def make_final_proxy_type(
    name,
    fast_type,
    slow_type,
    **kwargs,
):
    assert "module" not in kwargs
    return _make_final_proxy_type(
        name, fast_type, slow_type, module=slow_type.__module__, **kwargs
    )


def make_intermediate_proxy_type(
    name,
    fast_type,
    slow_type,
    **kwargs,
):
    assert "module" not in kwargs
    return _make_intermediate_proxy_type(
        name, fast_type, slow_type, module=slow_type.__module__, **kwargs
    )


class _AccessorAttr:
    """
    Descriptor that ensures that accessors like `.dt` and `.str`
    return the corresponding accessor types when accessed on `Series`
    and `Index` _types_ (not instances).n

    Attribute access for _instances_ uses the regular fast-then-slow
    lookup defined in `__getattr__`.
    """

    def __init__(self, typ):
        self._typ = typ

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, cls=None):
        if obj is None:
            return self._typ
        else:
            return _FastSlowAttribute(self._name).__get__(obj, type(obj))


def Timestamp_Timedelta__new__(cls, *args, **kwargs):
    # Call fast/slow constructor
    # This takes care of running __init__ as well, but must be paired
    # with a removal of the defaulted __init__ that
    # make_final_proxy_type provides.
    # Timestamp & Timedelta don't always return same types as self,
    # hence this method is needed.
    self, _ = _fast_slow_function_call(
        lambda cls, args, kwargs: cls(*args, **kwargs),
        None,
        cls,
        args,
        kwargs,
    )
    return self


Timedelta = make_final_proxy_type(
    "Timedelta",
    _Unusable,
    pd.Timedelta,
    bases=(datetime.timedelta,),
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__new__": Timestamp_Timedelta__new__,
        "__init__": _DELETE,
        "__array_priority__": 100,
    },
)


Timestamp = make_final_proxy_type(
    "Timestamp",
    _Unusable,
    pd.Timestamp,
    bases=(datetime.datetime,),
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__new__": Timestamp_Timedelta__new__,
        "__init__": _DELETE,
        "__array_priority__": 100,
    },
)

DatetimeProperties = make_intermediate_proxy_type(
    "DatetimeProperties",
    cudf.core.series.DatetimeProperties,
    pd.core.indexes.accessors.DatetimeProperties,
)

TimedeltaProperties = make_intermediate_proxy_type(
    "TimedeltaProperties",
    cudf.core.series.TimedeltaProperties,
    pd.core.indexes.accessors.TimedeltaProperties,
)

CombinedDatetimelikeProperties = make_intermediate_proxy_type(
    "CombinedDatetimelikeProperties",
    cudf.core.series.DatetimeProperties,
    pd.core.indexes.accessors.CombinedDatetimelikeProperties,
)

StringMethods = make_intermediate_proxy_type(
    "StringMethods",
    cudf.core.accessors.string.StringMethods,
    pd.core.strings.accessor.StringMethods,
)

ListMethods = make_intermediate_proxy_type(
    "ListMethods",
    cudf.core.accessors.lists.ListMethods,
    pd.core.arrays.arrow.accessors.ListAccessor,
)

SparseAccessor = make_intermediate_proxy_type(
    "SparseAccessor",
    _Unusable,
    pd.core.arrays.sparse.accessor.SparseAccessor,
)

StructAccessor = make_intermediate_proxy_type(
    "StructAccessor",
    cudf.core.accessors.struct.StructMethods,
    pd.core.arrays.arrow.accessors.StructAccessor,
)
_CategoricalAccessor = make_intermediate_proxy_type(
    "CategoricalAccessor",
    cudf.core.accessors.categorical.CategoricalAccessor,
    pd.core.arrays.categorical.CategoricalAccessor,
)


def _DataFrame__dir__(self):
    # Column names that are string identifiers are added to the dir of the
    # DataFrame
    # See https://github.com/pandas-dev/pandas/blob/43691a2f5d235b08f0f3aa813d8fdcb7c4ce1e47/pandas/core/indexes/base.py#L878
    _pd_df_dir = dir(pd.DataFrame)
    return _pd_df_dir + [
        colname
        for colname in self.columns
        if isinstance(colname, str) and colname.isidentifier()
    ]


def ignore_ipython_canary_check(self, **kwargs):
    raise AttributeError(
        "_ipython_canary_method_should_not_exist_ doesn't exist"
    )


def _DataFrame_dtypes_apply_func(value):
    if isinstance(value, (cudf.CategoricalDtype, cudf.IntervalDtype)):
        return value.to_pandas()
    return value


def _DataFrame__dtypes(self):
    result = _fast_slow_function_call(
        lambda self: self.dtypes,
        None,
        self,
    )[0]
    result = _maybe_wrap_result(
        result._fsproxy_slow.apply(_DataFrame_dtypes_apply_func), None
    )
    result.force_state(_State.SLOW)
    return result


@functools.wraps(pd.DataFrame.columns)
def _DataFrame_columns(self):
    result = _fast_slow_function_call(
        lambda self: self.columns,
        None,
        self,
    )[0]
    result.force_state(_State.SLOW)
    return result


def _to_xarray(self):
    # Call xarray conversion functions directly with self (the proxy object).
    # We must pass the proxy (self), not the slow pandas object, because xarray
    # does isinstance checks against pd.MultiIndex and pd.api.extensions.ExtensionArray.
    # After cudf.pandas.install(), these refer to proxy classes. The slow object
    # contains real pandas types that don't pass isinstance checks against the proxy
    # classes.
    xr = import_optional_dependency("xarray")
    if self.ndim == 1:
        return xr.DataArray.from_series(self)
    else:
        return xr.Dataset.from_dataframe(self)


# pandas.ExcelWriter uses __new__ to dispatch to the engine-specific subclass
# (OpenpyxlWriter, XlsxWriter, etc.) based on the `engine` kwarg.  The proxy
# must replicate this: construct the real writer with the accelerator disabled
# (so we get the actual pandas writer, not a recursive proxy) then wrap the
# result.  __init__ is a no-op because construction is fully handled in __new__.
def _ExcelWriter__new__(cls, *args, **kwargs):
    if cls is not ExcelWriter:
        return object.__new__(cls)

    from ..module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        writer = pd.ExcelWriter(*args, **kwargs)
    return _maybe_wrap_result(writer, pd.ExcelWriter, *args, **kwargs)


def _ExcelWriter__init__(self, *args, **kwargs):
    pass


DataFrame = make_final_proxy_type(
    "DataFrame",
    cudf.DataFrame,
    pd.DataFrame,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__dir__": _DataFrame__dir__,
        "__arrow_c_stream__": _FastSlowAttribute("__arrow_c_stream__"),
        "_constructor": _FastSlowAttribute("_constructor"),
        "_constructor_sliced": _FastSlowAttribute("_constructor_sliced"),
        "_accessors": set(),
        "_ipython_canary_method_should_not_exist_": ignore_ipython_canary_check,
        "columns": property(_DataFrame_columns),
        "dtypes": property(_DataFrame__dtypes),
        "__iter__": custom_iter,
        "attrs": _FastSlowAttribute("attrs"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "style": _FastSlowAttribute("style", private=True),
        "_mgr": _FastSlowAttribute("_mgr", private=True),
        "plot": _FastSlowAttribute("plot", private=True),
        "sparse": _FastSlowAttribute("sparse", private=True),
        "expanding": _FastSlowAttribute("expanding", private=True),
        "_AXIS_LEN": _FastSlowAttribute("_AXIS_LEN", private=True),
        "_AXIS_TO_AXIS_NUMBER": _FastSlowAttribute(
            "_AXIS_TO_AXIS_NUMBER", private=True
        ),
        "_AXIS_ORDERS": _FastSlowAttribute("_AXIS_ORDERS", private=True),
        "flags": _FastSlowAttribute("flags", private=True),
        "memory_usage": _FastSlowAttribute("memory_usage"),
        "__sizeof__": _FastSlowAttribute("__sizeof__"),
        "to_xarray": _to_xarray,
    },
)


def custom_repr_html(obj):
    # This custom method is need to register a html format
    # for ipython
    return _fast_slow_function_call(
        lambda obj: obj._repr_html_(),
        None,
        obj,
    )[0]


if ipython_shell:
    # See: https://ipython.readthedocs.io/en/stable/config/integrating.html#formatters-for-third-party-types
    html_formatter = ipython_shell.display_formatter.formatters["text/html"]
    html_formatter.for_type(DataFrame, custom_repr_html)


def _Series_dtype(self):
    dtype = self._fsproxy_wrapped.dtype
    if isinstance(
        dtype,
        (
            cudf.ListDtype,
            cudf.StructDtype,
            cudf.Decimal32Dtype,
            cudf.Decimal64Dtype,
            cudf.Decimal128Dtype,
        ),
    ):
        dtype = self._fsproxy_slow.dtype
    return _maybe_wrap_result(dtype, None)


_SeriesAtIndexer = make_intermediate_proxy_type(
    "_SeriesAtIndexer",
    cudf.core.series._SeriesAtIndexer,
    pd.core.indexing._AtIndexer,
)


_SeriesiAtIndexer = make_intermediate_proxy_type(
    "_SeriesiAtIndexer",
    cudf.core.series._SeriesiAtIndexer,
    pd.core.indexing._iAtIndexer,
)


Series = make_final_proxy_type(
    "Series",
    cudf.Series,
    pd.Series,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__array_function__": array_function_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        "__iter__": custom_iter,
        "memory_usage": _FastSlowAttribute("memory_usage"),
        "__sizeof__": _FastSlowAttribute("__sizeof__"),
        "dt": _AccessorAttr(CombinedDatetimelikeProperties),
        "at": _FastSlowAttribute("at"),
        "iat": _FastSlowAttribute("iat"),
        "str": _AccessorAttr(StringMethods),
        "list": _AccessorAttr(ListMethods),
        "struct": _AccessorAttr(StructAccessor),
        "cat": _AccessorAttr(_CategoricalAccessor),
        "_constructor": _FastSlowAttribute("_constructor"),
        "_constructor_expanddim": _FastSlowAttribute("_constructor_expanddim"),
        "_accessors": set(),
        "dtype": property(_Series_dtype),
        "to_xarray": _to_xarray,
        "attrs": _FastSlowAttribute("attrs"),
        "_mgr": _FastSlowAttribute("_mgr", private=True),
        "array": _FastSlowAttribute("array", private=True),
        "_values": _FastSlowAttribute("_values", private=True),
        "values": _FastSlowAttribute("values"),
        "sparse": _FastSlowAttribute("sparse", private=True),
        "_AXIS_LEN": _FastSlowAttribute("_AXIS_LEN", private=True),
        "_AXIS_TO_AXIS_NUMBER": _FastSlowAttribute(
            "_AXIS_TO_AXIS_NUMBER", private=True
        ),
        "_AXIS_ORDERS": _FastSlowAttribute("_AXIS_ORDERS", private=True),
        "flags": _FastSlowAttribute("flags", private=True),
    },
)


def _Index_values(self):
    from cudf.utils.dtypes import is_pandas_nullable_extension_dtype

    # For nullable integer dtypes (Int8/Int16/Int32/Int64/UInt8/.../UInt64),
    # cudf's .values returns a plain ndarray, but pandas returns an
    # IntegerArray (ExtensionArray) that supports NA values.  Using the cudf
    # fast path here produces the wrong result type, so we always use the
    # slow (pandas) path for these dtypes.
    # Access dtype directly from the wrapped object to stay out of the proxy
    # dispatch path and to avoid mutating the proxy state.
    dtype = self._fsproxy_wrapped.dtype
    if is_pandas_nullable_extension_dtype(dtype):
        slow = self._fsproxy_fast_to_slow()
        return _maybe_wrap_result(slow.values, None)
    return _fast_slow_function_call(getattr, None, self, "values")[0]


def Index__new__(cls, *args, **kwargs):
    # Call fast/slow constructor
    # This takes care of running __init__ as well, but must be paired
    # with a removal of the defaulted __init__ that
    # make_final_proxy_type provides.
    self, _ = _fast_slow_function_call(
        lambda cls, args, kwargs: cls(*args, **kwargs),
        None,
        cls,
        args,
        kwargs,
    )
    return self


def Index__setattr__(self, name, value):
    if name.startswith("_"):
        object.__setattr__(self, name, value)
        return
    if name == "name":
        setattr(self._fsproxy_wrapped, "name", value)
    if name == "names":
        setattr(self._fsproxy_wrapped, "names", value)
    return _FastSlowAttribute("__setattr__").__get__(self, type(self))(
        name, value
    )


Index = make_final_proxy_type(
    "Index",
    cudf.Index,
    pd.Index,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    additional_attributes={
        "__array__": array_method,
        "__array_function__": array_function_method,
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        # matches pd.Index (1000) so wrapped-ndarray proxies defer
        # binops/ufuncs to the Index's reflected op
        "__array_priority__": pd.Index.__array_priority__,
        "dt": _AccessorAttr(CombinedDatetimelikeProperties),
        "str": _AccessorAttr(StringMethods),
        "cat": _AccessorAttr(_CategoricalAccessor),
        "__iter__": custom_iter,
        "memory_usage": _FastSlowAttribute("memory_usage"),
        "__sizeof__": _FastSlowAttribute("__sizeof__"),
        "__init__": _DELETE,
        "__new__": Index__new__,
        "__setattr__": Index__setattr__,
        "_constructor": _FastSlowAttribute("_constructor"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_accessors": set(),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
        "name": _FastSlowAttribute("name"),
        "nbytes": _FastSlowAttribute("nbytes", private=True),
        "array": _FastSlowAttribute("array", private=True),
        "values": property(_Index_values),
    },
)

RangeIndex = make_final_proxy_type(
    "RangeIndex",
    cudf.RangeIndex,
    pd.RangeIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "name": _FastSlowAttribute("name"),
        "nbytes": _FastSlowAttribute("nbytes", private=True),
        "array": _FastSlowAttribute("array", private=True),
        "_range": _FastSlowAttribute("_range"),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
    },
)

SparseDtype = make_final_proxy_type(
    "SparseDtype",
    _Unusable,
    pd.SparseDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

# Special caseing `ArrowDtype` as it is not yet added to `cudf` namespace
# both fast and slow paths are `pd.ArrowDtype`
ArrowDtype = make_final_proxy_type(
    "ArrowDtype",
    pd.ArrowDtype,
    pd.ArrowDtype,
    bases=(pd.api.extensions.ExtensionDtype,),
    fast_to_slow=lambda fast: fast,
    slow_to_fast=lambda slow: slow,
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "pyarrow_dtype": _FastSlowAttribute("pyarrow_dtype"),
    },
)

SparseArray = make_final_proxy_type(
    "SparseDtype",
    _Unusable,
    pd.arrays.SparseArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_readonly": _FastSlowAttribute("_readonly", private=True),
    },
)

CategoricalIndex = make_final_proxy_type(
    "CategoricalIndex",
    cudf.CategoricalIndex,
    pd.CategoricalIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "name": _FastSlowAttribute("name"),
        "nbytes": _FastSlowAttribute("nbytes", private=True),
        "array": _FastSlowAttribute("array", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
    },
)

Categorical = make_final_proxy_type(
    "Categorical",
    _Unusable,
    pd.Categorical,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

CategoricalDtype = make_final_proxy_type(
    "CategoricalDtype",
    cudf.CategoricalDtype,
    pd.CategoricalDtype,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

DatetimeIndex = make_final_proxy_type(
    "DatetimeIndex",
    cudf.DatetimeIndex,
    pd.DatetimeIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
        "name": _FastSlowAttribute("name"),
        "nbytes": _FastSlowAttribute("nbytes", private=True),
        "array": _FastSlowAttribute("array", private=True),
    },
)

DatetimeArray = make_final_proxy_type(
    "DatetimeArray",
    _Unusable,
    pd.arrays.DatetimeArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
    },
)

DatetimeTZDtype = make_final_proxy_type(
    "DatetimeTZDtype",
    _Unusable,
    pd.DatetimeTZDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
    },
)

TimedeltaIndex = make_final_proxy_type(
    "TimedeltaIndex",
    cudf.TimedeltaIndex,
    pd.TimedeltaIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
        "name": _FastSlowAttribute("name"),
        "nbytes": _FastSlowAttribute("nbytes", private=True),
        "array": _FastSlowAttribute("array", private=True),
    },
)

try:
    from pandas.arrays import NumpyExtensionArray as pd_NumpyExtensionArray

    NumpyExtensionArray = make_final_proxy_type(
        "NumpyExtensionArray",
        _Unusable,
        pd_NumpyExtensionArray,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
            "_ndarray": _FastSlowAttribute("_ndarray"),
            "_dtype": _FastSlowAttribute("_dtype"),
            "_readonly": _FastSlowAttribute("_readonly", private=True),
        },
    )

except ImportError:
    from pandas.arrays import PandasArray as pd_PandasArray

    PandasArray = make_final_proxy_type(
        "PandasArray",
        _Unusable,
        pd_PandasArray,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
            "_ndarray": _FastSlowAttribute("_ndarray"),
            "_dtype": _FastSlowAttribute("_dtype"),
            "_readonly": _FastSlowAttribute("_readonly", private=True),
        },
    )

TimedeltaArray = make_final_proxy_type(
    "TimedeltaArray",
    _Unusable,
    pd.arrays.TimedeltaArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
    },
)

PeriodIndex = make_final_proxy_type(
    "PeriodIndex",
    _Unusable,
    pd.PeriodIndex,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

PeriodArray = make_final_proxy_type(
    "PeriodArray",
    _Unusable,
    pd.arrays.PeriodArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

PeriodDtype = make_final_proxy_type(
    "PeriodDtype",
    _Unusable,
    pd.PeriodDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Period = make_final_proxy_type(
    "Period",
    _Unusable,
    pd.Period,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__array_priority__": 100,
    },
)


MultiIndex = make_final_proxy_type(
    "MultiIndex",
    cudf.MultiIndex,
    pd.MultiIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "names": _FastSlowAttribute("names"),
        "_lexsort_depth": _FastSlowAttribute("_lexsort_depth", private=True),
        "_should_fallback_to_positional": _FastSlowAttribute(
            "_should_fallback_to_positional", private=True
        ),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
    },
)

TimeGrouper = make_intermediate_proxy_type(
    "TimeGrouper",
    _Unusable,
    pd.api.typing.TimeGrouper,
)

Grouper = make_final_proxy_type(
    "Grouper",
    cudf.Grouper,
    pd.Grouper,
    fast_to_slow=lambda fast: pd.Grouper(
        **{
            k: getattr(fast, k)
            for k in {"key", "level", "freq", "closed", "label"}
            if getattr(fast, k, None) is not None
        }
    ),
    slow_to_fast=lambda slow: cudf.Grouper(
        **{
            k: getattr(slow, k)
            for k in {"key", "level", "freq", "closed", "label"}
            if getattr(slow, k, None) is not None
        }
    ),
    additional_attributes={
        "binner": _FastSlowAttribute("binner"),
        "dropna": _FastSlowAttribute("dropna"),
        "freq": _FastSlowAttribute("freq"),
        "key": _FastSlowAttribute("key"),
        "level": _FastSlowAttribute("level"),
        "sort": _FastSlowAttribute("sort"),
    },
)

StringArray = make_final_proxy_type(
    "StringArray",
    _Unusable,
    pd.arrays.StringArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(NumpyExtensionArray, pd.api.extensions.ExtensionArray),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "__array__": _FastSlowAttribute("__array__"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__arrow_array__": _FastSlowAttribute("__arrow_array__"),
    },
)


ArrowStringArray = make_final_proxy_type(
    "ArrowStringArray",
    _Unusable,
    pd.core.arrays.string_arrow.ArrowStringArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "_pa_array": _FastSlowAttribute("_pa_array", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "__array__": _FastSlowAttribute("__array__", private=True),
        "__invert__": _FastSlowAttribute("__invert__"),
        "__neg__": _FastSlowAttribute("__neg__"),
        "__pos__": _FastSlowAttribute("__pos__", private=True),
        "__abs__": _FastSlowAttribute("__abs__"),
        "__contains__": _FastSlowAttribute("__contains__"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__arrow_array__": _FastSlowAttribute("__arrow_array__"),
    },
)


StorageExtensionDtype = make_final_proxy_type(
    "StorageExtensionDtype",
    _Unusable,
    pd.core.dtypes.base.StorageExtensionDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
)

StringDtype = make_final_proxy_type(
    "StringDtype",
    _Unusable,
    pd.StringDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "storage": _FastSlowAttribute("storage"),
    },
)

BooleanArray = make_final_proxy_type(
    "BooleanArray",
    _Unusable,
    pd.arrays.BooleanArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.core.arrays.BaseMaskedArray,),
    additional_attributes={
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

BooleanDtype = make_final_proxy_type(
    "BooleanDtype",
    _Unusable,
    pd.BooleanDtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        # Expose the slow type as ``__class__`` so that
        # ``real_dtype == proxy_dtype`` is True: pandas' ExtensionDtype.__eq__
        # checks ``isinstance(other, type(self))``, which otherwise fails for a
        # proxy that only subclasses the abstract ExtensionDtype. These masked
        # dtypes have empty ``_metadata`` so equality reduces to this check.
        "__class__": pd.BooleanDtype,
    },
)

IntegerArray = make_final_proxy_type(
    "IntegerArray",
    _Unusable,
    pd.arrays.IntegerArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.core.arrays.BaseMaskedArray,),
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
    },
)

Int8Dtype = make_final_proxy_type(
    "Int8Dtype",
    _Unusable,
    pd.Int8Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Int8Dtype,
    },
)


Int16Dtype = make_final_proxy_type(
    "Int16Dtype",
    _Unusable,
    pd.Int16Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Int16Dtype,
    },
)

Int32Dtype = make_final_proxy_type(
    "Int32Dtype",
    _Unusable,
    pd.Int32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Int32Dtype,
    },
)

Int64Dtype = make_final_proxy_type(
    "Int64Dtype",
    _Unusable,
    pd.Int64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Int64Dtype,
    },
)

UInt8Dtype = make_final_proxy_type(
    "UInt8Dtype",
    _Unusable,
    pd.UInt8Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.UInt8Dtype,
    },
)

UInt16Dtype = make_final_proxy_type(
    "UInt16Dtype",
    _Unusable,
    pd.UInt16Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.UInt16Dtype,
    },
)

UInt32Dtype = make_final_proxy_type(
    "UInt32Dtype",
    _Unusable,
    pd.UInt32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.UInt32Dtype,
    },
)

UInt64Dtype = make_final_proxy_type(
    "UInt64Dtype",
    _Unusable,
    pd.UInt64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.UInt64Dtype,
    },
)

IntervalIndex = make_final_proxy_type(
    "IntervalIndex",
    cudf.IntervalIndex,
    pd.IntervalIndex,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(Index,),
    additional_attributes={
        "__init__": _DELETE,
        "__setattr__": Index__setattr__,
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "_cache": _FastSlowAttribute("_cache", private=True),
        "name": _FastSlowAttribute("name"),
    },
)

IntervalArray = make_final_proxy_type(
    "IntervalArray",
    _Unusable,
    pd.arrays.IntervalArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionArray,),
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "_left": _FastSlowAttribute("_left", private=True),
        "_right": _FastSlowAttribute("_right", private=True),
    },
)

IntervalDtype = make_final_proxy_type(
    "IntervalDtype",
    cudf.IntervalDtype,
    pd.IntervalDtype,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        # ``_closed`` is a pandas-private attribute; source it from the slow
        # (pandas) object so cudf's IntervalDtype need not define it.
        "_closed": _FastSlowAttribute("_closed", private=True),
    },
)

Interval = make_final_proxy_type(
    "Interval",
    _Unusable,
    pd.Interval,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__array_priority__": 1000,
    },
)

FloatingArray = make_final_proxy_type(
    "FloatingArray",
    _Unusable,
    pd.arrays.FloatingArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.core.arrays.BaseMaskedArray,),
    additional_attributes={
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "_data": _FastSlowAttribute("_data", private=True),
        "_mask": _FastSlowAttribute("_mask", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
    },
)

Float32Dtype = make_final_proxy_type(
    "Float32Dtype",
    _Unusable,
    pd.Float32Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Float32Dtype,
    },
)

Float64Dtype = make_final_proxy_type(
    "Float64Dtype",
    _Unusable,
    pd.Float64Dtype,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    bases=(pd.api.extensions.ExtensionDtype,),
    additional_attributes={
        "__from_arrow__": _FastSlowAttribute("__from_arrow__"),
        "__hash__": _FastSlowAttribute("__hash__"),
        "__class__": pd.Float64Dtype,
    },
)


# ``GroupBy.nth`` is a property returning a selector object on both
# sides; registering the selector pair as an intermediate proxy gives
# ``gb.nth`` a proxied result whose ``__call__``/``__getitem__`` get the
# usual call-time fast/slow dispatch (with the slow side re-derived from
# the recorded ``getattr`` provenance on fallback).
GroupByNthSelector = make_intermediate_proxy_type(
    "GroupByNthSelector",
    cudf.core.groupby.groupby.GroupByNthSelector,
    pd.core.groupby.indexing.GroupByNthSelector,
)

SeriesGroupBy = make_intermediate_proxy_type(
    "SeriesGroupBy",
    cudf.core.groupby.groupby.SeriesGroupBy,
    pd.core.groupby.SeriesGroupBy,
)

DataFrameGroupBy = make_intermediate_proxy_type(
    "DataFrameGroupBy",
    cudf.core.groupby.groupby.DataFrameGroupBy,
    pd.core.groupby.DataFrameGroupBy,
    additional_attributes={
        "_grouper": _FastSlowAttribute("_grouper", private=True),
    },
)

RollingGroupBy = make_intermediate_proxy_type(
    "RollingGroupBy",
    cudf.core.window.rolling.RollingGroupby,
    pd.core.window.rolling.RollingGroupby,
)

_SeriesIlocIndexer = make_intermediate_proxy_type(
    "_SeriesIlocIndexer",
    cudf.core.series._SeriesIlocIndexer,
    pd.core.indexing._iLocIndexer,
)

_DataFrameIlocIndexer = make_intermediate_proxy_type(
    "_SeriesIlocIndexer",
    cudf.core.dataframe._DataFrameIlocIndexer,
    pd.core.indexing._iLocIndexer,
)

_SeriesLocIndexer = make_intermediate_proxy_type(
    "_SeriesLocIndexer",
    cudf.core.series._SeriesLocIndexer,
    pd.core.indexing._LocIndexer,
)

_DataFrameLocIndexer = make_intermediate_proxy_type(
    "_DataFrameLocIndexer",
    cudf.core.dataframe._DataFrameLocIndexer,
    pd.core.indexing._LocIndexer,
)

_LocIndexer = make_intermediate_proxy_type(
    "_LocIndexer",
    cudf.core.dataframe._DataFrameLocIndexer,
    pd.core.indexing._LocIndexer,
)

_AtIndexer = make_intermediate_proxy_type(
    "_AtIndexer",
    cudf.core.dataframe._DataFrameAtIndexer,
    pd.core.indexing._AtIndexer,
)

_iAtIndexer = make_intermediate_proxy_type(
    "_iAtIndexer",
    cudf.core.dataframe._DataFrameiAtIndexer,
    pd.core.indexing._iAtIndexer,
)

FixedForwardWindowIndexer = make_final_proxy_type(
    "FixedForwardWindowIndexer",
    _Unusable,
    pd.api.indexers.FixedForwardWindowIndexer,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "index_array": _FastSlowAttribute("index_array"),
        "window_size": _FastSlowAttribute("window_size"),
    },
)

VariableOffsetWindowIndexer = make_final_proxy_type(
    "VariableOffsetWindowIndexer",
    _Unusable,
    pd.api.indexers.VariableOffsetWindowIndexer,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "index": _FastSlowAttribute("index"),
        "index_array": _FastSlowAttribute("index_array"),
        "offset": _FastSlowAttribute("offset"),
        "window_size": _FastSlowAttribute("window_size"),
    },
)

Window = make_intermediate_proxy_type(
    "Window",
    _Unusable,
    pd.core.window.rolling.Window,
)

Rolling = make_intermediate_proxy_type(
    "Rolling",
    cudf.core.window.Rolling,
    pd.core.window.Rolling,
)

ExponentialMovingWindow = make_intermediate_proxy_type(
    "ExponentialMovingWindow",
    cudf.core.window.ewm.ExponentialMovingWindow,
    pd.core.window.ewm.ExponentialMovingWindow,
)

ExponentialMovingWindowGroupby = make_intermediate_proxy_type(
    "ExponentialMovingWindowGroupby",
    _Unusable,
    pd.core.window.ewm.ExponentialMovingWindowGroupby,
)

OnlineExponentialMovingWindow = make_intermediate_proxy_type(
    "OnlineExponentialMovingWindow",
    _Unusable,
    pd.core.window.ewm.OnlineExponentialMovingWindow,
)

EWMMeanState = make_intermediate_proxy_type(
    "EWMMeanState",
    _Unusable,
    pd.core.window.online.EWMMeanState,
)

Expanding = make_intermediate_proxy_type(
    "Expanding",
    _Unusable,
    pd.core.window.expanding.Expanding,
)

ExpandingGroupby = make_intermediate_proxy_type(
    "ExpandingGroupby",
    _Unusable,
    pd.core.window.expanding.ExpandingGroupby,
)

Resampler = make_intermediate_proxy_type(
    "Resampler", cudf.core.resample._Resampler, pd.api.typing.Resampler
)

DataFrameResampler = make_intermediate_proxy_type(
    "DataFrameResampler",
    cudf.core.resample.DataFrameResampler,
    pd.api.typing.Resampler,
)

SeriesResampler = make_intermediate_proxy_type(
    "SeriesResampler",
    cudf.core.resample.SeriesResampler,
    pd.api.typing.Resampler,
)

DatetimeIndexResampler = make_intermediate_proxy_type(
    "DatetimeIndexResampler",
    _Unusable,
    pd.core.resample.DatetimeIndexResampler,
)

PeriodIndexResampler = make_intermediate_proxy_type(
    "PeriodIndexResampler",
    _Unusable,
    pd.core.resample.PeriodIndexResampler,
)

TimedeltaIndexResampler = make_intermediate_proxy_type(
    "TimedeltaIndexResampler",
    _Unusable,
    pd.core.resample.TimedeltaIndexResampler,
)

DatetimeIndexResamplerGroupby = make_intermediate_proxy_type(
    "DatetimeIndexResamplerGroupby",
    _Unusable,
    pd.core.resample.DatetimeIndexResamplerGroupby,
)

PeriodIndexResamplerGroupby = make_intermediate_proxy_type(
    "PeriodIndexResamplerGroupby",
    _Unusable,
    pd.core.resample.PeriodIndexResamplerGroupby,
)

TimedeltaIndexResamplerGroupby = make_intermediate_proxy_type(
    "TimedeltaIndexResamplerGroupby",
    _Unusable,
    pd.core.resample.TimedeltaIndexResamplerGroupby,
)

StataReader = make_final_proxy_type(
    "StataReader",
    _Unusable,
    pd.io.stata.StataReader,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_nvar": _FastSlowAttribute("_nvar", private=True),
        "_typlist": _FastSlowAttribute("_typlist", private=True),
        "_varlist": _FastSlowAttribute("_varlist", private=True),
        "_fmtlist": _FastSlowAttribute("_fmtlist", private=True),
        "_path_or_buf": _FastSlowAttribute("_path_or_buf", private=True),
        "_chunk": _FastSlowAttribute("_chunk", private=True),
    },
)

HDFStore = make_final_proxy_type(
    "HDFStore",
    _Unusable,
    pd.HDFStore,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={"__hash__": _FastSlowAttribute("__hash__")},
)

ExcelFile = make_final_proxy_type(
    "ExcelFile",
    _Unusable,
    pd.ExcelFile,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "engine": _FastSlowAttribute("engine"),
        "storage_options": _FastSlowAttribute("storage_options"),
    },
)

ExcelWriter = make_final_proxy_type(
    "ExcelWriter",
    _Unusable,
    pd.ExcelWriter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "__fspath__": _FastSlowAttribute("__fspath__"),
        "__init__": _ExcelWriter__init__,
        "__new__": _ExcelWriter__new__,
    },
    bases=(os.PathLike,),
    metaclasses=(abc.ABCMeta,),
)

OpenpyxlWriter = make_final_proxy_type(
    "OpenpyxlWriter",
    _Unusable,
    pd_OpenpyxlWriter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__fspath__": _FastSlowAttribute("__fspath__"),
    },
    bases=(ExcelWriter,),
)

XlsxWriter = make_final_proxy_type(
    "XlsxWriter",
    _Unusable,
    pd_XlsxWriter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__fspath__": _FastSlowAttribute("__fspath__"),
    },
    bases=(ExcelWriter,),
)

try:
    from pandas.io.formats.style import Styler as pd_Styler  # isort: skip
    from pandas.io.formats.style import StylerRenderer as pd_StylerRenderer

    StylerRenderer = make_final_proxy_type(
        "StylerRenderer",
        _Unusable,
        pd_StylerRenderer,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "caption": _FastSlowAttribute("caption"),
            "cell_context": _FastSlowAttribute("cell_context"),
            "cell_ids": _FastSlowAttribute("cell_ids"),
            "cellstyle_map": _FastSlowAttribute("cellstyle_map"),
            "cellstyle_map_columns": _FastSlowAttribute(
                "cellstyle_map_columns"
            ),
            "cellstyle_map_index": _FastSlowAttribute("cellstyle_map_index"),
            "columns": _FastSlowAttribute("columns"),
            "concatenated": _FastSlowAttribute("concatenated"),
            "css": _FastSlowAttribute("css"),
            "ctx": _FastSlowAttribute("ctx"),
            "ctx_columns": _FastSlowAttribute("ctx_columns"),
            "ctx_index": _FastSlowAttribute("ctx_index"),
            "data": _FastSlowAttribute("data"),
            "hidden_columns": _FastSlowAttribute("hidden_columns"),
            "hidden_rows": _FastSlowAttribute("hidden_rows"),
            "hide_column_names": _FastSlowAttribute("hide_column_names"),
            "hide_columns_": _FastSlowAttribute("hide_columns_"),
            "hide_index_": _FastSlowAttribute("hide_index_"),
            "hide_index_names": _FastSlowAttribute("hide_index_names"),
            "index": _FastSlowAttribute("index"),
            "table_attributes": _FastSlowAttribute("table_attributes"),
            "table_styles": _FastSlowAttribute("table_styles"),
            "tooltips": _FastSlowAttribute("tooltips"),
            "uuid": _FastSlowAttribute("uuid"),
            "uuid_len": _FastSlowAttribute("uuid_len"),
        },
    )

    Styler = make_final_proxy_type(
        "Styler",
        _Unusable,
        pd_Styler,
        bases=(StylerRenderer,),
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "css": _FastSlowAttribute("css"),
            "ctx": _FastSlowAttribute("ctx"),
            "index": _FastSlowAttribute("index"),
            "data": _FastSlowAttribute("data"),
            "_display_funcs": _FastSlowAttribute(
                "_display_funcs", private=True
            ),
            "_compute": _FastSlowAttribute("_compute", private=True),
            "table_styles": _FastSlowAttribute("table_styles"),
            "columns": _FastSlowAttribute("columns"),
            "caption": _FastSlowAttribute("caption"),
            "_todo": _FastSlowAttribute("_todo", private=True),
            "ctx_columns": _FastSlowAttribute("ctx_columns"),
            "ctx_index": _FastSlowAttribute("ctx_index"),
            "_display_funcs_index": _FastSlowAttribute(
                "_display_funcs_index", private=True
            ),
            "_display_funcs_column_names": _FastSlowAttribute(
                "_display_funcs_column_names", private=True
            ),
            "_display_funcs_index_names": _FastSlowAttribute(
                "_display_funcs_index_names", private=True
            ),
            "uuid": _FastSlowAttribute("uuid"),
            "hide_index_": _FastSlowAttribute("hide_index_"),
            "hide_index_names": _FastSlowAttribute("hide_index_names"),
            "hide_columns_": _FastSlowAttribute("hide_columns_"),
            "hide_column_names": _FastSlowAttribute("hide_column_names"),
            "table_attributes": _FastSlowAttribute("table_attributes"),
            "cell_context": _FastSlowAttribute("cell_context"),
            "cell_ids": _FastSlowAttribute("cell_ids"),
            "cellstyle_map": _FastSlowAttribute("cellstyle_map"),
            "cellstyle_map_columns": _FastSlowAttribute(
                "cellstyle_map_columns"
            ),
            "cellstyle_map_index": _FastSlowAttribute("cellstyle_map_index"),
            "concatenated": _FastSlowAttribute("concatenated"),
            "hidden_columns": _FastSlowAttribute("hidden_columns"),
            "hidden_rows": _FastSlowAttribute("hidden_rows"),
            "tooltips": _FastSlowAttribute("tooltips"),
            "uuid_len": _FastSlowAttribute("uuid_len"),
        },
    )
except ImportError:
    # Styler requires Jinja to be installed
    pass


def _find_user_frame(level=0):
    frame = inspect.currentframe()
    skip = 0
    while frame:
        modname = frame.f_globals.get("__name__", "")
        # TODO: Remove "nvtx." entry once we cross nvtx-0.2.11 as minimum version
        if modname == "__main__" or not modname.startswith(("cudf.", "nvtx.")):
            if skip == level:
                return frame
            skip += 1
        frame = frame.f_back
    raise RuntimeError("Could not find the user's frame.")


_eval_func = _FunctionProxy(_Unusable(), pd.eval)

register_proxy_func(pd.read_pickle)(
    _FunctionProxy(_Unusable(), pd.read_pickle)
)

register_proxy_func(pd.to_pickle)(_FunctionProxy(_Unusable(), pd.to_pickle))
register_proxy_func(pd.api.types.is_list_like)(  # noqa: TID251
    _FunctionProxy(_Unusable(), pd.api.types.is_list_like)  # noqa: TID251
)
register_proxy_func(loc)(loc)
register_proxy_func(iloc)(iloc)
register_proxy_func(at)(at)
register_proxy_func(iat)(iat)
register_proxy_func(setitem)(setitem)
register_proxy_func(getitem)(getitem)


def _get_eval_locals_and_globals(level, local_dict=None, global_dict=None):
    frame = _find_user_frame(level=level)
    local_dict = dict(frame.f_locals) if local_dict is None else local_dict
    global_dict = dict(frame.f_globals) if global_dict is None else global_dict
    return local_dict, global_dict


@register_proxy_func(pd.core.computation.eval.eval)
@nvtx.annotate(
    "CUDF_PANDAS_EVAL",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _eval(
    *args,
    parser="pandas",
    engine=None,
    local_dict=None,
    global_dict=None,
    **kwargs,
):
    # Custom implementation of to pre-process globals and
    # locals before calling pd.eval.
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _eval_func(
        *args,
        parser=parser,
        engine=engine,
        local_dict=local_dict,
        global_dict=global_dict,
        **kwargs,
    )


_orig_df_eval_method = DataFrame.eval


@register_proxy_func(pd.core.accessor.register_dataframe_accessor)
def _register_dataframe_accessor(name):
    return pd.core.accessor._register_accessor(name, DataFrame)


@register_proxy_func(pd.core.accessor.register_series_accessor)
def _register_series_accessor(name):
    return pd.core.accessor._register_accessor(name, Series)


@register_proxy_func(pd.core.accessor.register_index_accessor)
def _register_index_accessor(name):
    return pd.core.accessor._register_accessor(name, Index)


@contextlib.contextmanager
def null_assert_produces_warning(*args, **kwargs):
    # We do not want pandas unit tests to fail because
    # assert_produces_warning doesn't see a warning.
    # No an explicit public API
    try:
        yield []
    finally:
        pass


@register_proxy_func(pd._testing.assert_produces_warning)
def _register_assert_produces_warning(*args, **kwargs):
    return null_assert_produces_warning(*args, **kwargs)


def null_raises_chained_assignment_error(*args, **kwargs):
    # This assertion function also uses assert_produces_warning
    # we want to ignore in pandas unit tests.
    return null_assert_produces_warning(*args, **kwargs)


@register_proxy_func(pd._testing.raises_chained_assignment_error)
def _register_raises_chained_assignment_error(*args, **kwargs):
    return null_raises_chained_assignment_error(*args, **kwargs)


@nvtx.annotate(
    "CUDF_PANDAS_DATAFRAME_EVAL",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _df_eval_method(self, *args, local_dict=None, global_dict=None, **kwargs):
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _orig_df_eval_method(
        self, *args, local_dict=local_dict, global_dict=global_dict, **kwargs
    )


_orig_query_eval_method = DataFrame.query


@nvtx.annotate(
    "CUDF_PANDAS_DATAFRAME_QUERY",
    color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
    domain="cudf_pandas",
)
def _df_query_method(self, *args, local_dict=None, global_dict=None, **kwargs):
    # `query` API internally calls `eval`, hence we are making use of
    # helps of `eval` to populate locals and globals dict.
    level = kwargs.get("level", 0)
    local_dict, global_dict = _get_eval_locals_and_globals(
        level, local_dict, global_dict
    )
    return _orig_query_eval_method(
        self, *args, local_dict=local_dict, global_dict=global_dict, **kwargs
    )


# These custom implementations are installed by cudf.pandas itself and must
# not be mirrored onto (and clobber) the real ``pandas.DataFrame``.
_setattr_fsproxy_no_mirror(DataFrame, "eval", _df_eval_method)
_setattr_fsproxy_no_mirror(DataFrame, "query", _df_query_method)

_JsonReader = make_intermediate_proxy_type(
    "_JsonReader",
    _Unusable,
    pd.io.json._json.JsonReader,
)

_TextFileReader = make_intermediate_proxy_type(
    "_TextFileReader", _Unusable, pd.io.parsers.readers.TextFileReader
)

_XportReader = make_intermediate_proxy_type(
    "_XportReader", _Unusable, pd_XportReader
)

_SAS7BDATReader = make_intermediate_proxy_type(
    "_SAS7BDATReader", _Unusable, pd_SAS7BDATReader
)

USFederalHolidayCalendar = make_final_proxy_type(
    "USFederalHolidayCalendar",
    _Unusable,
    pd_USFederalHolidayCalendar,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

HolidayCalendarMetaClass = make_final_proxy_type(
    "HolidayCalendarMetaClass",
    _Unusable,
    pd_HolidayCalendarMetaClass,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)


@register_proxy_func(pd_HolidayCalendarFactory)
def holiday_calendar_factory_wrapper(*args, **kwargs):
    # Call the original HolidayCalendarFactory
    result = _FunctionProxy(_Unusable(), pd_HolidayCalendarFactory)(
        *args, **kwargs
    )
    # Return the slow proxy of the result
    return result._fsproxy_slow


AbstractHolidayCalendar = make_final_proxy_type(
    "AbstractHolidayCalendar",
    _Unusable,
    pd_AbstractHolidayCalendar,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
    metaclasses=(pd_HolidayCalendarMetaClass,),
)

Holiday = make_final_proxy_type(
    "Holiday",
    _Unusable,
    pd_Holiday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "day": _FastSlowAttribute("day"),
        "days_of_week": _FastSlowAttribute("days_of_week"),
        "end_date": _FastSlowAttribute("end_date"),
        "exclude_dates": _FastSlowAttribute("exclude_dates"),
        "month": _FastSlowAttribute("month"),
        "name": _FastSlowAttribute("name"),
        "observance": _FastSlowAttribute("observance"),
        "offset": _FastSlowAttribute("offset"),
        "start_date": _FastSlowAttribute("start_date"),
        "year": _FastSlowAttribute("year"),
    },
)
USThanksgivingDay = make_final_proxy_type(
    "USThanksgivingDay",
    _Unusable,
    pd_USThanksgivingDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

USColumbusDay = make_final_proxy_type(
    "USColumbusDay",
    _Unusable,
    pd_USColumbusDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

USLaborDay = make_final_proxy_type(
    "USLaborDay",
    _Unusable,
    pd_USLaborDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

USMemorialDay = make_final_proxy_type(
    "USMemorialDay",
    _Unusable,
    pd_USMemorialDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

USMartinLutherKingJr = make_final_proxy_type(
    "USMartinLutherKingJr",
    _Unusable,
    pd_USMartinLutherKingJr,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

USPresidentsDay = make_final_proxy_type(
    "USPresidentsDay",
    _Unusable,
    pd_USPresidentsDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)


GoodFriday = make_final_proxy_type(
    "GoodFriday",
    _Unusable,
    pd_GoodFriday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

EasterMonday = make_final_proxy_type(
    "EasterMonday",
    _Unusable,
    pd_EasterMonday,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

FY5253 = make_final_proxy_type(
    "FY5253",
    _Unusable,
    pd.offsets.FY5253,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BDay = make_final_proxy_type(
    "BDay",
    _Unusable,
    pd.offsets.BDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BHalfYearBegin = make_final_proxy_type(
    "BHalfYearBegin",
    _Unusable,
    pd.offsets.BHalfYearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BHalfYearEnd = make_final_proxy_type(
    "BHalfYearEnd",
    _Unusable,
    pd.offsets.BHalfYearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BMonthBegin = make_final_proxy_type(
    "BMonthBegin",
    _Unusable,
    pd.offsets.BMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BMonthEnd = make_final_proxy_type(
    "BMonthEnd",
    _Unusable,
    pd.offsets.BMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BQuarterBegin = make_final_proxy_type(
    "BQuarterBegin",
    _Unusable,
    pd.offsets.BQuarterBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BQuarterEnd = make_final_proxy_type(
    "BQuarterEnd",
    _Unusable,
    pd.offsets.BQuarterEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BusinessDay = make_final_proxy_type(
    "BusinessDay",
    _Unusable,
    pd.offsets.BusinessDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BusinessHour = make_final_proxy_type(
    "BusinessHour",
    _Unusable,
    pd.offsets.BusinessHour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BusinessMonthBegin = make_final_proxy_type(
    "BusinessMonthBegin",
    _Unusable,
    pd.offsets.BusinessMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BusinessMonthEnd = make_final_proxy_type(
    "BusinessMonthEnd",
    _Unusable,
    pd.offsets.BusinessMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BYearBegin = make_final_proxy_type(
    "BYearBegin",
    _Unusable,
    pd.offsets.BYearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

BYearEnd = make_final_proxy_type(
    "BYearEnd",
    _Unusable,
    pd.offsets.BYearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CBMonthBegin = make_final_proxy_type(
    "CBMonthBegin",
    _Unusable,
    pd.offsets.CBMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CBMonthEnd = make_final_proxy_type(
    "CBMonthEnd",
    _Unusable,
    pd.offsets.CBMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CDay = make_final_proxy_type(
    "CDay",
    _Unusable,
    pd.offsets.CDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CustomBusinessDay = make_final_proxy_type(
    "CustomBusinessDay",
    _Unusable,
    pd.offsets.CustomBusinessDay,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CustomBusinessHour = make_final_proxy_type(
    "CustomBusinessHour",
    _Unusable,
    pd.offsets.CustomBusinessHour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CustomBusinessMonthBegin = make_final_proxy_type(
    "CustomBusinessMonthBegin",
    _Unusable,
    pd.offsets.CustomBusinessMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

CustomBusinessMonthEnd = make_final_proxy_type(
    "CustomBusinessMonthEnd",
    _Unusable,
    pd.offsets.CustomBusinessMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

# Pandas stores DateOffset kwargs (e.g., `day`, `months`) and the resolved
# `_offset` directly on the instance __dict__, so they don't appear in
# dir(pd.DateOffset) and aren't picked up by the proxy machinery. Register
# each one explicitly so attribute lookup falls back to the wrapped instance.
_DATEOFFSET_INSTANCE_ATTRS = {
    name: _FastSlowAttribute(name, private=name.startswith("_"))
    for name in (
        *liboffsets._relativedelta_kwds,
        "_offset",
        "_use_relativedelta",
    )
}

DateOffset = make_final_proxy_type(
    "DateOffset",
    _Unusable,
    pd.offsets.DateOffset,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        **_DATEOFFSET_INSTANCE_ATTRS,
    },
)

BaseOffset = make_final_proxy_type(
    "BaseOffset",
    _Unusable,
    pd.offsets.BaseOffset,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        **_DATEOFFSET_INSTANCE_ATTRS,
    },
)

Day = make_final_proxy_type(
    "Day",
    _Unusable,
    pd.offsets.Day,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Easter = make_final_proxy_type(
    "Easter",
    _Unusable,
    pd.offsets.Easter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

FY5253Quarter = make_final_proxy_type(
    "FY5253Quarter",
    _Unusable,
    pd.offsets.FY5253Quarter,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

HalfYearBegin = make_final_proxy_type(
    "HalfYearBegin",
    _Unusable,
    pd.offsets.HalfYearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

HalfYearEnd = make_final_proxy_type(
    "HalfYearEnd",
    _Unusable,
    pd.offsets.HalfYearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Hour = make_final_proxy_type(
    "Hour",
    _Unusable,
    pd.offsets.Hour,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

LastWeekOfMonth = make_final_proxy_type(
    "LastWeekOfMonth",
    _Unusable,
    pd.offsets.LastWeekOfMonth,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Micro = make_final_proxy_type(
    "Micro",
    _Unusable,
    pd.offsets.Micro,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Milli = make_final_proxy_type(
    "Milli",
    _Unusable,
    pd.offsets.Milli,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Minute = make_final_proxy_type(
    "Minute",
    _Unusable,
    pd.offsets.Minute,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)


MonthBegin = make_final_proxy_type(
    "MonthBegin",
    _Unusable,
    pd.offsets.MonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

MonthEnd = make_final_proxy_type(
    "MonthEnd",
    _Unusable,
    pd.offsets.MonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Nano = make_final_proxy_type(
    "Nano",
    _Unusable,
    pd.offsets.Nano,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

QuarterBegin = make_final_proxy_type(
    "QuarterBegin",
    _Unusable,
    pd.offsets.QuarterBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

QuarterEnd = make_final_proxy_type(
    "QuarterEnd",
    _Unusable,
    pd.offsets.QuarterEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Second = make_final_proxy_type(
    "Second",
    _Unusable,
    pd.offsets.Second,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

SemiMonthBegin = make_final_proxy_type(
    "SemiMonthBegin",
    _Unusable,
    pd.offsets.SemiMonthBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

SemiMonthEnd = make_final_proxy_type(
    "SemiMonthEnd",
    _Unusable,
    pd.offsets.SemiMonthEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Tick = make_final_proxy_type(
    "Tick",
    _Unusable,
    pd.offsets.Tick,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Week = make_final_proxy_type(
    "Week",
    _Unusable,
    pd.offsets.Week,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

WeekOfMonth = make_final_proxy_type(
    "WeekOfMonth",
    _Unusable,
    pd.offsets.WeekOfMonth,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

YearBegin = make_final_proxy_type(
    "YearBegin",
    _Unusable,
    pd.offsets.YearBegin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

YearEnd = make_final_proxy_type(
    "YearEnd",
    _Unusable,
    pd.offsets.YearEnd,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

Flags = make_final_proxy_type(
    "Flags",
    _Unusable,
    pd.Flags,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

NamedAgg = make_final_proxy_type(
    "NamedAgg",
    pd.NamedAgg,
    pd.NamedAgg,
    fast_to_slow=lambda fast: fast,
    slow_to_fast=lambda slow: slow,
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        "aggfunc": _FastSlowAttribute("aggfunc"),
        "column": _FastSlowAttribute("column"),
        "kwargs": _FastSlowAttribute("kwargs"),
    },
)


def _arrow_extension_array_astype(self, dtype=None, copy: bool = False):
    if copy is False and cudf.api.types.is_dtype_equal(self.dtype, dtype):
        return self
    return _maybe_wrap_result(
        self._fsproxy_wrapped.astype(dtype=dtype, copy=copy), self
    )


ArrowExtensionArray = make_final_proxy_type(
    "ArrowExtensionArray",
    _Unusable,
    pd.arrays.ArrowExtensionArray,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_pa_array": _FastSlowAttribute("_pa_array", private=True),
        "_readonly": _FastSlowAttribute("_readonly", private=True),
        "__array__": _FastSlowAttribute("__array__", private=True),
        "__invert__": _FastSlowAttribute("__invert__"),
        "__neg__": _FastSlowAttribute("__neg__"),
        "__pos__": _FastSlowAttribute("__pos__", private=True),
        "__abs__": _FastSlowAttribute("__abs__"),
        "__contains__": _FastSlowAttribute("__contains__"),
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
        "__arrow_array__": arrow_array_method,
        "astype": _arrow_extension_array_astype,
    },
)

FrozenList = make_final_proxy_type(
    "FrozenList",
    _Unusable,
    pd.core.indexes.frozen.FrozenList,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
    },
)

# The following are subclasses of `pandas.core.base.PandasObj`,
# excluding subclasses defined in `pandas.core.internals`.  These are
# not strictly part of the Pandas public API, but they do appear as
# return types.

NDFrame = make_final_proxy_type(
    "NDFrame",
    _Unusable,
    pd.core.generic.NDFrame,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

DatetimeTimedeltaMixin = make_final_proxy_type(
    "DatetimeTimedeltaMixin",
    _Unusable,
    pd.core.indexes.datetimelike.DatetimeTimedeltaMixin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

DatetimeIndexOpsMixin = make_final_proxy_type(
    "DatetimeIndexOpsMixin",
    _Unusable,
    pd.core.indexes.datetimelike.DatetimeIndexOpsMixin,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

NDArrayBackedExtensionIndex = make_final_proxy_type(
    "NDArrayBackedExtensionIndex",
    _Unusable,
    pd.core.indexes.extension.NDArrayBackedExtensionIndex,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "_cache": _FastSlowAttribute("_cache", private=True),
        "_engine": _FastSlowAttribute("_engine", private=True),
        "__array__": array_method,
        "__array_ufunc__": _FastSlowAttribute("__array_ufunc__"),
    },
)

# SQLTable and SQLiteTable set their instance attributes in __init__,
# so they aren't visible via dir(cls). Forward them explicitly because
# the proxy's __getattr__ fallback is only installed when the slow type
# defines __getattr__.
_SQL_TABLE_ATTRS = {
    name: _FastSlowAttribute(name)
    for name in (
        "dtype",
        "frame",
        "if_exists",
        "index",
        "keys",
        "name",
        "pd_sql",
        "prefix",
        "schema",
        "table",
    )
}

SQLTable = make_final_proxy_type(
    "SQLTable",
    _Unusable,
    pd.io.sql.SQLTable,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        **_SQL_TABLE_ATTRS,
    },
)

SQLiteTable = make_final_proxy_type(
    "SQLiteTable",
    _Unusable,
    pd.io.sql.SQLiteTable,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        **_SQL_TABLE_ATTRS,
    },
)

SQLDatabase = make_final_proxy_type(
    "SQLDatabase",
    _Unusable,
    pd.io.sql.SQLDatabase,
    fast_to_slow=_Unusable(),
    slow_to_fast=_Unusable(),
    additional_attributes={
        "__hash__": _FastSlowAttribute("__hash__"),
        **{
            name: _FastSlowAttribute(name)
            for name in ("con", "exit_stack", "meta", "returns_generator")
        },
    },
)

_PANDAS_OBJ_FINAL_TYPES = [
    pd.core.indexes.accessors.PeriodProperties,
    pd.core.indexes.accessors.Properties,
    pd.plotting._core.PlotAccessor,
    pd.io.sql.SQLiteDatabase,
    pd.io.sql.PandasSQL,
]

_PANDAS_OBJ_INTERMEDIATE_TYPES = [
    pd.core.groupby.groupby.GroupByPlot,
    pd.core.groupby.groupby.GroupBy,
    pd.core.groupby.groupby.BaseGroupBy,
]

for typ in _PANDAS_OBJ_FINAL_TYPES:
    if typ.__name__ in globals():
        # if we already defined a proxy type
        # corresponding to this type, use that.
        continue
    globals()[typ.__name__] = make_final_proxy_type(
        typ.__name__,
        _Unusable,
        typ,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__hash__": _FastSlowAttribute("__hash__"),
        },
    )


for typ in _PANDAS_OBJ_INTERMEDIATE_TYPES:
    if typ.__name__ in globals():
        # if we already defined a proxy type
        # corresponding to this type, use that.
        continue
    globals()[typ.__name__] = make_intermediate_proxy_type(
        typ.__name__,
        _Unusable,
        typ,
    )


# Save the original __init__ methods
_original_Series_init = cudf.Series.__init__
_original_DataFrame_init = cudf.DataFrame.__init__
_original_Index_init = cudf.Index.__init__
_original_from_pandas = cudf.from_pandas


def wrap_init(original_init):
    @functools.wraps(original_init)
    def wrapped_init(self, data=None, *args, **kwargs):
        if is_proxy_object(data):
            data = data.as_gpu_object()
            if (
                isinstance(data, type(self))
                and len(args) == 0
                and len(kwargs) == 0
            ):
                # This short-circuits the constructor to avoid
                # unnecessary work when the data is already a
                # proxy object of the same type.
                # It is a common case in `cuml` and `xgboost`.
                # For perf impact see:
                # https://github.com/rapidsai/cudf/pull/17878/files#r1936469215
                self.__dict__.update(data.__dict__)
                return
        original_init(self, data, *args, **kwargs)

    return wrapped_init


def wrap_call(original_call):
    @functools.wraps(original_call)
    def wrapped_call(cls, data, *args, **kwargs):
        if is_proxy_object(data):
            data = data.as_gpu_object()
        return original_call(cls, data, *args, **kwargs)

    return wrapped_call


def wrap_from_pandas(original_call):
    @functools.wraps(original_call)
    def wrapped_from_pandas(obj, *args, **kwargs):
        if is_proxy_object(obj):
            obj = obj.as_gpu_object()
            return obj
        return original_call(obj, *args, **kwargs)

    return wrapped_from_pandas


def wrap_from_pandas_dataframe(original_call):
    @functools.wraps(original_call)
    def wrapped_from_pandas_dataframe(dataframe, *args, **kwargs):
        if is_proxy_object(dataframe):
            dataframe = dataframe.as_gpu_object()
            if isinstance(dataframe, cudf.DataFrame):
                return dataframe
        return original_call(dataframe, *args, **kwargs)

    return wrapped_from_pandas_dataframe


def wrap_from_pandas_series(original_call):
    @functools.wraps(original_call)
    def wrapped_from_pandas_series(s, *args, **kwargs):
        if is_proxy_object(s):
            s = s.as_gpu_object()
            if isinstance(s, cudf.Series):
                return s
        return original_call(s, *args, **kwargs)

    return wrapped_from_pandas_series


def wrap_from_pandas_index(original_call):
    @functools.wraps(original_call)
    def wrapped_from_pandas_index(index, *args, **kwargs):
        if is_proxy_object(index):
            index = index.as_gpu_object()
            if isinstance(index, cudf.Index):
                return index
        return original_call(index, *args, **kwargs)

    return wrapped_from_pandas_index


def wrap_from_pandas_multiindex(original_call):
    @functools.wraps(original_call)
    def wrapped_from_pandas_multiindex(multiindex, *args, **kwargs):
        if is_proxy_object(multiindex):
            multiindex = multiindex.as_gpu_object()
            if isinstance(multiindex, cudf.MultiIndex):
                return multiindex
        return original_call(multiindex, *args, **kwargs)

    return wrapped_from_pandas_multiindex


@functools.wraps(_original_DataFrame_init)
def DataFrame_init_(
    self, data=None, index=None, columns=None, *args, **kwargs
):
    data_is_proxy = is_proxy_object(data)

    if data_is_proxy:
        data = data.as_gpu_object()
    if is_proxy_object(index):
        index = index.as_gpu_object()
    if is_proxy_object(columns):
        columns = columns.as_cpu_object()
    if (
        (
            (data_is_proxy and isinstance(data, type(self)))
            and (index is None)
            and (columns is None)
        )
        and len(args) == 0
        and len(kwargs) == 0
    ):
        self.__dict__.update(data.__dict__)
        return
    _original_DataFrame_init(self, data, index, columns, *args, **kwargs)


def initial_setup():
    """
    This is a one-time setup function that can contain
    any initialization code that needs to be run once
    when the module is imported. Currently, it is used
    to wrap the __init__ methods and enable pandas compatibility mode.
    """
    cudf.Series.__init__ = wrap_init(_original_Series_init)
    cudf.Index.__init__ = wrap_init(_original_Index_init)
    cudf.DataFrame.__init__ = DataFrame_init_
    cudf.from_pandas = wrap_from_pandas(_original_from_pandas)
    cudf.set_option("mode.pandas_compatible", True)


def _unpickle_obj(pickled_args):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        unpickler, args = pickle.loads(pickled_args)
    obj = unpickler(*args)
    return obj


def _reduce_proxied_td_obj(obj):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        pickled_args = pickle.dumps(obj._fsproxy_wrapped.__reduce__())

    return _unpickle_obj, (pickled_args,)


def _reduce_obj(obj):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        pickled_args = pickle.dumps(obj.__reduce__())

    return _unpickle_obj, (pickled_args,)


def _generic_reduce_obj(obj, unpickle_func):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        pickled_args = pickle.dumps(obj.__reduce__())

    return unpickle_func, (pickled_args,)


def _frame_unpickle_obj(pickled_args):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        unpickled_intermediate = pickle.loads(pickled_args)
        reconstructor_func = unpickled_intermediate[0]
        obj = reconstructor_func(*unpickled_intermediate[1])
        obj.__setstate__(unpickled_intermediate[2])
    return obj


def _index_unpickle_obj(pickled_args):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        unpickled_intermediate = pickle.loads(pickled_args)
        reconstructor_func = unpickled_intermediate[0]
        obj = reconstructor_func(*unpickled_intermediate[1])

    return obj


def _reduce_offset_obj(obj):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        pickled_args = pickle.dumps(obj.__getstate__())

    return _unpickle_offset_obj, (pickled_args,)


def _unpickle_offset_obj(pickled_args):
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        data = pickle.loads(pickled_args)
        data.pop("_offset")
        data.pop("_use_relativedelta")
    obj = pd.DateOffset(**data)
    return obj


copyreg.dispatch_table[Timestamp] = _reduce_proxied_td_obj
copyreg.dispatch_table[pd.Timestamp] = _reduce_obj
# same reducer/unpickler can be used for Timedelta:
copyreg.dispatch_table[Timedelta] = _reduce_proxied_td_obj
copyreg.dispatch_table[pd.Timedelta] = _reduce_obj
# Period has no fast (cudf) representation; pickle the wrapped/real pandas
# object directly (e.g. matplotlib figures store pandas.Period x-data).
copyreg.dispatch_table[Period] = _reduce_proxied_td_obj
copyreg.dispatch_table[pd.Period] = _reduce_obj

# TODO: Need to find a way to unpickle cross-version(old) pickled objects.
# Register custom reducer/unpickler functions for pandas objects
# so that they can be pickled/unpickled correctly:
copyreg.dispatch_table[pd.Series] = lambda obj: _generic_reduce_obj(
    obj, _frame_unpickle_obj
)
copyreg.dispatch_table[pd.DataFrame] = lambda obj: _generic_reduce_obj(
    obj, _frame_unpickle_obj
)

copyreg.dispatch_table[pd.Index] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)
copyreg.dispatch_table[pd.RangeIndex] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)
copyreg.dispatch_table[pd.DatetimeIndex] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)
copyreg.dispatch_table[pd.TimedeltaIndex] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)
copyreg.dispatch_table[pd.CategoricalIndex] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)
copyreg.dispatch_table[pd.MultiIndex] = lambda obj: _generic_reduce_obj(
    obj, _index_unpickle_obj
)

copyreg.dispatch_table[pd.DateOffset] = _reduce_offset_obj
# Concrete pandas offsets (Day, Week, Minute, ...) are stored e.g. as the
# ``freq`` of matplotlib's date converters. The module accelerator makes the
# offset module attributes resolve to proxies, so pickle's class-identity
# check fails; reduce each via its real ``__reduce__`` (with the accelerator
# disabled). ``DateOffset`` keeps its dedicated reducer above.
for _offset_cls in vars(pd.tseries.offsets).values():
    if (
        isinstance(_offset_cls, type)
        and issubclass(_offset_cls, pd.tseries.offsets.BaseOffset)
        and _offset_cls is not pd.DateOffset
    ):
        copyreg.dispatch_table.setdefault(_offset_cls, _reduce_obj)


def _unpickle_NaT():
    return pd.NaT


def _reduce_NaT(obj):
    # NaT.__reduce__ returns the cyfunction ``_nat_unpickle`` directly,
    # but cudf.pandas replaces ``pandas._libs.tslibs.nattype._nat_unpickle``
    # with a function proxy. Pickle's identity check between the cyfunction
    # and the module-level proxy then raises ``PicklingError``. Round-trip
    # NaT via our own helper to avoid the lookup.
    return _unpickle_NaT, ()


copyreg.dispatch_table[type(pd.NaT)] = _reduce_NaT
