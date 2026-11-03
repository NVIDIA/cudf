# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from pandas.api import types as pd_types  # noqa: TID251

import cudf
from cudf.api import types


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, True),
        (pd.CategoricalDtype, True),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), True),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, True),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), True),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), True),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        # TODO: Currently creating an empty Series of list type ignores the
        # provided type and instead makes a float64 Series.
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        # TODO: Currently creating an empty Series of struct type fails because
        # it uses a numpy utility that doesn't understand StructDtype.
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_categorical_dtype(obj, expect):
    assert types._is_categorical_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, True),
        (int, True),
        (float, True),
        (complex, True),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, True),
        (np.int_, True),
        (np.float64, True),
        (np.complex128, True),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), True),
        (np.int_(), True),
        (np.float64(), True),
        (np.complex128(), True),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), True),
        (np.dtype("int"), True),
        (np.dtype("float"), True),
        (np.dtype("complex"), True),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), True),
        (np.array([], dtype=np.int_), True),
        (np.array([], dtype=np.float64), True),
        (np.array([], dtype=np.complex128), True),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), True),
        (pd.Series(dtype="int"), True),
        (pd.Series(dtype="float"), True),
        (pd.Series(dtype="complex"), True),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, True),
        (cudf.Decimal64Dtype, True),
        (cudf.Decimal32Dtype, True),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), True),
        (cudf.Decimal64Dtype(5, 2), True),
        (cudf.Decimal32Dtype(5, 2), True),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), True),
        (cudf.Series(dtype="int"), True),
        (cudf.Series(dtype="float"), True),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), True),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), True),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), True),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
        (cudf.Index([], dtype="bool"), True),
        (cudf.Index([], dtype="int64"), True),
        (cudf.Index([], dtype="uint8"), True),
        (cudf.Index([], dtype="float64"), True),
        (cudf.Index([], dtype="boolean"), True),
        (cudf.Index([], dtype="Int64"), True),
        (cudf.Index([], dtype="UInt16"), True),
        (cudf.Index([], dtype="Float32"), True),
        (cudf.Index([], dtype="str"), False),
        (cudf.Index([], dtype="datetime64[s]"), False),
        (cudf.Index([], dtype="timedelta64[s]"), False),
    ),
)
def test_is_numeric_dtype(obj, expect):
    assert types.is_numeric_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, True),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, True),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), True),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), True),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), True),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), True),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), True),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_integer_dtype(obj, expect):
    assert types.is_integer_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), True),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), True),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_integer(obj, expect):
    assert types.is_integer(obj) == expect


# TODO: Temporarily ignoring all cases of "object" until we decide what to do.
@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, True),
        # (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, True),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), True),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), True),
        (np.dtype("unicode"), True),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        # (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), True),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        # (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), True),
        (pd.Series(dtype="unicode"), True),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        # (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), True),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_string_dtype(obj, expect):
    assert types.is_string_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, True),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), True),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), True),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), True),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), True),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), True),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_datetime_dtype(obj, expect):
    assert types.is_datetime_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, True),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), True),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), True),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_list_dtype(obj, expect):
    assert types.is_list_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, True),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        # (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), True),
        (cudf.Decimal128Dtype(5, 2), False),
        (cudf.Decimal64Dtype(5, 2), False),
        (cudf.Decimal32Dtype(5, 2), False),
        # (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), False),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), False),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), True),
        # (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_struct_dtype(obj, expect):
    # TODO: All inputs of interval types are currently disabled due to
    # inconsistent behavior of is_struct_dtype for interval types that will be
    # fixed as part of the array refactor.
    assert types.is_struct_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj, expect",
    (
        # Base Python objects.
        (bool(), False),
        (int(), False),
        (float(), False),
        (complex(), False),
        ("", False),
        (object(), False),
        # Base Python types.
        (bool, False),
        (int, False),
        (float, False),
        (complex, False),
        (str, False),
        (object, False),
        # NumPy types.
        (np.bool_, False),
        (np.int_, False),
        (np.float64, False),
        (np.complex128, False),
        (np.str_, False),
        (np.datetime64, False),
        (np.timedelta64, False),
        # NumPy scalars.
        (np.bool_(), False),
        (np.int_(), False),
        (np.float64(), False),
        (np.complex128(), False),
        (np.str_(), False),
        (np.datetime64(), False),
        (np.timedelta64(), False),
        # NumPy dtype objects.
        (np.dtype("bool"), False),
        (np.dtype("int"), False),
        (np.dtype("float"), False),
        (np.dtype("complex"), False),
        (np.dtype("str"), False),
        (np.dtype("unicode"), False),
        (np.dtype("datetime64"), False),
        (np.dtype("timedelta64"), False),
        (np.dtype("object"), False),
        # NumPy arrays.
        (np.array([], dtype=np.bool_), False),
        (np.array([], dtype=np.int_), False),
        (np.array([], dtype=np.float64), False),
        (np.array([], dtype=np.complex128), False),
        (np.array([], dtype=np.str_), False),
        (np.array([], dtype=np.datetime64), False),
        (np.array([], dtype=np.timedelta64), False),
        (np.array([], dtype=object), False),
        # Pandas dtypes.
        (pd.CategoricalDtype.type, False),
        (pd.CategoricalDtype, False),
        # Pandas objects.
        (pd.Series(dtype="bool"), False),
        (pd.Series(dtype="int"), False),
        (pd.Series(dtype="float"), False),
        (pd.Series(dtype="complex"), False),
        (pd.Series(dtype="str"), False),
        (pd.Series(dtype="unicode"), False),
        (pd.Series(dtype="datetime64[s]"), False),
        (pd.Series(dtype="timedelta64[s]"), False),
        (pd.Series(dtype="category"), False),
        (pd.Series(dtype="object"), False),
        # cuDF dtypes.
        (cudf.CategoricalDtype, False),
        (cudf.ListDtype, False),
        (cudf.StructDtype, False),
        (cudf.Decimal128Dtype, False),
        (cudf.Decimal64Dtype, False),
        (cudf.Decimal32Dtype, False),
        (cudf.IntervalDtype, False),
        # cuDF dtype instances.
        (cudf.CategoricalDtype(["a"]), False),
        (cudf.ListDtype(int), False),
        (cudf.StructDtype({"a": int}), False),
        (cudf.Decimal128Dtype(5, 2), True),
        (cudf.Decimal64Dtype(5, 2), True),
        (cudf.Decimal32Dtype(5, 2), True),
        (cudf.IntervalDtype(int), False),
        # cuDF objects
        (cudf.Series(dtype="bool"), False),
        (cudf.Series(dtype="int"), False),
        (cudf.Series(dtype="float"), False),
        (cudf.Series(dtype="str"), False),
        (cudf.Series(dtype="datetime64[s]"), False),
        (cudf.Series(dtype="timedelta64[s]"), False),
        (cudf.Series(dtype="category"), False),
        (cudf.Series(dtype=cudf.Decimal128Dtype(5, 2)), True),
        (cudf.Series(dtype=cudf.Decimal64Dtype(5, 2)), True),
        (cudf.Series(dtype=cudf.Decimal32Dtype(5, 2)), True),
        (cudf.Series([[1, 2], [3, 4, 5]]), False),
        (cudf.Series([{"a": 1, "b": 2}, {"c": 3}]), False),
        (cudf.Series(dtype=cudf.IntervalDtype(int)), False),
    ),
)
def test_is_decimal_dtype(obj, expect):
    assert types.is_decimal_dtype(obj) == expect


@pytest.mark.parametrize(
    "obj",
    (
        # Base Python objects.
        bool(),
        int(),
        float(),
        complex(),
        "",
        object(),
        # Base Python types.
        bool,
        int,
        float,
        complex,
        str,
        object,
        # NumPy types.
        np.bool_,
        np.int_,
        np.float64,
        np.complex128,
        np.str_,
        np.datetime64,
        np.timedelta64,
        # NumPy scalars.
        np.bool_(),
        np.int_(),
        np.float64(),
        np.complex128(),
        np.str_(),
        np.datetime64(),
        np.timedelta64(),
        # NumPy dtype objects.
        np.dtype("bool"),
        np.dtype("int"),
        np.dtype("float"),
        np.dtype("complex"),
        np.dtype("str"),
        np.dtype("unicode"),
        np.dtype("datetime64"),
        np.dtype("timedelta64"),
        np.dtype("object"),
        # NumPy arrays.
        np.array([], dtype=np.bool_),
        np.array([], dtype=np.int_),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.complex128),
        np.array([], dtype=np.str_),
        np.array([], dtype=np.datetime64),
        np.array([], dtype=np.timedelta64),
        np.array([], dtype=object),
        # Pandas dtypes.
        # TODO: pandas does not consider these to be categoricals.
        # pd.CategoricalDtype.type,
        # pd.CategoricalDtype,
        # Pandas objects.
        pd.Series(dtype="bool"),
        pd.Series(dtype="int"),
        pd.Series(dtype="float"),
        pd.Series(dtype="complex"),
        pd.Series(dtype="str"),
        pd.Series(dtype="unicode"),
        pd.Series(dtype="datetime64[s]"),
        pd.Series(dtype="timedelta64[s]"),
        pd.Series(dtype="category"),
        pd.Series(dtype="object"),
    ),
)
def test_pandas_agreement(obj):
    with pytest.warns(DeprecationWarning):
        expected = pd_types.is_categorical_dtype(obj)
    with pytest.warns(DeprecationWarning):
        actual = types.is_categorical_dtype(obj)
    assert expected == actual
    assert types.is_numeric_dtype(obj) == pd_types.is_numeric_dtype(obj)
    assert types.is_integer_dtype(obj) == pd_types.is_integer_dtype(obj)
    assert types.is_integer(obj) == pd_types.is_integer(obj)
    assert types.is_string_dtype(obj) == pd_types.is_string_dtype(obj)


@pytest.mark.parametrize(
    "obj",
    (
        # Base Python objects.
        bool(),
        int(),
        float(),
        complex(),
        "",
        object(),
        # Base Python types.
        bool,
        int,
        float,
        complex,
        str,
        object,
        # NumPy types.
        np.bool_,
        np.int_,
        np.float64,
        np.complex128,
        np.str_,
        np.datetime64,
        np.timedelta64,
        # NumPy scalars.
        np.bool_(),
        np.int_(),
        np.float64(),
        np.complex128(),
        np.str_(),
        np.datetime64(),
        np.timedelta64(),
        # NumPy dtype objects.
        np.dtype("bool"),
        np.dtype("int"),
        np.dtype("float"),
        np.dtype("complex"),
        np.dtype("str"),
        np.dtype("unicode"),
        np.dtype("datetime64"),
        np.dtype("timedelta64"),
        np.dtype("object"),
        # NumPy arrays.
        np.array([], dtype=np.bool_),
        np.array([], dtype=np.int_),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.complex128),
        np.array([], dtype=np.str_),
        np.array([], dtype=np.datetime64),
        np.array([], dtype=np.timedelta64),
        np.array([], dtype=object),
        # Pandas dtypes.
        # TODO: pandas does not consider these to be categoricals.
        # pd.CategoricalDtype.type,
        # pd.CategoricalDtype,
        # Pandas objects.
        pd.Series(dtype="bool"),
        pd.Series(dtype="int"),
        pd.Series(dtype="float"),
        pd.Series(dtype="complex"),
        pd.Series(dtype="str"),
        pd.Series(dtype="unicode"),
        pd.Series(dtype="datetime64[s]"),
        pd.Series(dtype="timedelta64[s]"),
        pd.Series(dtype="category"),
        pd.Series(dtype="object"),
    ),
)
def test_pandas_agreement_scalar(obj):
    assert types.is_scalar(obj) == pd_types.is_scalar(obj)


# TODO: Add test of interval.
