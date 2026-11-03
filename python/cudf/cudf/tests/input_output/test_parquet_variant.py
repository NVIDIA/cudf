# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from packaging import version

import cudf

pytestmark = pytest.mark.skipif(
    version.parse(pa.__version__) < version.parse("21"),
    reason="pyarrow < 21 does not recognize the Parquet VARIANT logical type",
)


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "parquet"


def _expect_variant_metadata_value_pair(dtype):
    assert isinstance(dtype, cudf.StructDtype)
    fields = dtype.fields
    for name in ("metadata", "value"):
        assert name in fields, f"missing VARIANT child {name!r}"
        child = fields[name]
        assert isinstance(child, cudf.ListDtype)
        assert child.element_type == np.dtype("uint8")


@pytest.mark.parametrize(
    "filename",
    ["variant_minimal.parquet", "variant_multirow.parquet"],
)
def test_read_unshredded_variant_parquet_shape(datadir, filename):
    df = cudf.read_parquet(datadir / filename)
    assert df.shape[1] == 1
    variant_dtype = df.dtypes.iloc[0]
    _expect_variant_metadata_value_pair(variant_dtype)
    assert set(variant_dtype.fields) == {"metadata", "value"}


def test_read_variant_parquet_zero_rows(datadir):
    df = cudf.read_parquet(datadir / "variant_multirow.parquet", nrows=0)
    assert len(df) == 0
    _expect_variant_metadata_value_pair(df.dtypes.iloc[0])


def test_read_shredded_variant_parquet_shape(datadir):
    df = cudf.read_parquet(datadir / "duckdb_variant_sample.parquet")
    assert df.shape == (2, 2)
    assert df["id"].dtype == np.dtype("int32")

    variant_dtype = df["v"].dtype
    _expect_variant_metadata_value_pair(variant_dtype)
    assert "typed_value" in variant_dtype.fields, (
        "shredded VARIANT must surface typed children alongside metadata/value"
    )


_NESTED_FIXTURES = {
    "unshredded": "variant_nested_unshredded.parquet",
    "shredded": "variant_nested_shredded.parquet",
}


def _variant_child_of_list(list_dtype):
    assert isinstance(list_dtype, cudf.ListDtype)
    return list_dtype.element_type


def _check_variant_struct(variant_dtype, shredded):
    _expect_variant_metadata_value_pair(variant_dtype)
    expected = {"metadata", "value"}
    if shredded:
        expected.add("typed_value")
        assert variant_dtype.fields["typed_value"] == np.dtype("int32"), (
            "shredded fixture writes typed_value as int32"
        )
    assert set(variant_dtype.fields) == expected, (
        f"unexpected variant child fields: got {set(variant_dtype.fields)}, "
        f"expected {expected}"
    )


def _check_top(dtype, shredded):
    _check_variant_struct(dtype, shredded)


def _check_in_list(dtype, shredded):
    _check_variant_struct(_variant_child_of_list(dtype), shredded)


def _check_in_struct(dtype, shredded):
    assert isinstance(dtype, cudf.StructDtype)
    assert set(dtype.fields) == {"a", "b"}
    for name in ("a", "b"):
        _check_variant_struct(dtype.fields[name], shredded)


def _check_in_list_of_struct(dtype, shredded):
    inner = _variant_child_of_list(dtype)
    assert isinstance(inner, cudf.StructDtype)
    assert set(inner.fields) == {"v"}
    _check_variant_struct(inner.fields["v"], shredded)


_COLUMN_CHECKERS = [
    ("v_top", _check_top),
    ("v_list", _check_in_list),
    ("v_struct", _check_in_struct),
    ("v_list_struct", _check_in_list_of_struct),
]


@pytest.fixture(scope="module", params=sorted(_NESTED_FIXTURES))
def nested_variant_df(request, datadir):
    df = cudf.read_parquet(datadir / _NESTED_FIXTURES[request.param])
    return request.param, df


def test_nested_variant_shape(nested_variant_df):
    _, df = nested_variant_df
    assert df.shape == (3, 4)
    assert list(df.columns) == ["v_top", "v_list", "v_struct", "v_list_struct"]


def test_nested_variant_top_level(nested_variant_df):
    shred_state, df = nested_variant_df
    _check_top(df["v_top"].dtype, shredded=shred_state == "shredded")


def test_nested_variant_in_list(nested_variant_df):
    shred_state, df = nested_variant_df
    _check_in_list(df["v_list"].dtype, shredded=shred_state == "shredded")


def test_nested_variant_in_struct(nested_variant_df):
    shred_state, df = nested_variant_df
    _check_in_struct(df["v_struct"].dtype, shredded=shred_state == "shredded")


def test_nested_variant_in_list_of_struct(nested_variant_df):
    shred_state, df = nested_variant_df
    _check_in_list_of_struct(
        df["v_list_struct"].dtype, shredded=shred_state == "shredded"
    )


@pytest.mark.parametrize("shred_state", sorted(_NESTED_FIXTURES))
@pytest.mark.parametrize(
    "column,checker", _COLUMN_CHECKERS, ids=[c for c, _ in _COLUMN_CHECKERS]
)
def test_nested_variant_zero_rows(datadir, shred_state, column, checker):
    df = cudf.read_parquet(
        datadir / _NESTED_FIXTURES[shred_state], nrows=0, columns=[column]
    )
    assert len(df) == 0
    assert list(df.columns) == [column]
    checker(df[column].dtype, shredded=shred_state == "shredded")


@pytest.mark.parametrize("shred_state", sorted(_NESTED_FIXTURES))
@pytest.mark.parametrize(
    "column,checker", _COLUMN_CHECKERS, ids=[c for c, _ in _COLUMN_CHECKERS]
)
def test_nested_variant_column_projection(
    datadir, shred_state, column, checker
):
    df = cudf.read_parquet(
        datadir / _NESTED_FIXTURES[shred_state], columns=[column]
    )
    assert df.shape == (3, 1)
    assert list(df.columns) == [column]
    checker(df[column].dtype, shredded=shred_state == "shredded")
