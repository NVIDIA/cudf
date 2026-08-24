# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os
from string import ascii_letters

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import NUMERIC_TYPES, UNSIGNED_TYPES

pytest.importorskip("tables")


@pytest.fixture(params=[0, 10])
def pdf(request):
    types = set([*NUMERIC_TYPES, "datetime64[ns]", "bool"]) - set(
        UNSIGNED_TYPES
    )
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    rng = np.random.default_rng(1)
    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        rng.integers(0, 50, size=(nrows, ncols)),
        columns=pd.Index([f"col_{typ}" for typ in types]),
        index=pd.RangeIndex(nrows, name="test_index"),
    )
    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype(typer).rename(
        {"col_datetime64[ns]": "col_datetime64"}, axis=1
    )

    # Create non-numeric categorical data otherwise may be typecasted
    data = rng.choice(list(ascii_letters), size=nrows)
    test_pdf["col_category"] = pd.Series(data, dtype="category")

    return (test_pdf, nrows)


@pytest.fixture
def gdf(pdf):
    pdf, nrows = pdf
    return (cudf.DataFrame(pdf), nrows)


@pytest.fixture(params=["fixed", "table"])
def hdf_files(request, tmp_path, pdf):
    pdf, nrows = pdf
    if request.param == "fixed":
        pdf = pdf.drop("col_category", axis=1)

    fname_df = tmp_path / "test_df.hdf"
    pdf.to_hdf(fname_df, key="hdf_df_tests", format=request.param)

    fname_series = {}
    for column in pdf.columns:
        fname_series[column] = tmp_path / "test_series.hdf"
        pdf[column].to_hdf(
            fname_series[column], key="hdf_series_tests", format=request.param
        )
    return (fname_df, fname_series, request.param, nrows)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize(
    "columns",
    [["col_int8"], ["col_category"], ["col_int32", "col_float32"], None],
)
def test_hdf_reader(hdf_files, columns):
    hdf_df_file, hdf_series, format, nrows = hdf_files
    if format == "fixed" and columns is not None:
        pytest.skip("Can't use columns with format 'fixed'")
    if format == "table" and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")
    expect_df = pd.read_hdf(hdf_df_file, columns=columns)
    got_df = cudf.read_hdf(hdf_df_file, columns=columns)

    assert_eq(
        expect_df, got_df, check_categorical=False, check_index_type=False
    )

    for column in hdf_series.keys():
        expect_series = pd.read_hdf(hdf_series[column])
        got_series = cudf.read_hdf(hdf_series[column])

        assert_eq(expect_series, got_series, check_index_type=False)


@pytest.mark.parametrize("format", ["fixed", "table"])
@pytest.mark.parametrize("complib", ["zlib", "bzip2", "lzo", "blosc"])
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_hdf_writer(tmp_path, pdf, gdf, complib, format):
    pdf, nrows = pdf
    if format == "table" and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")
    gdf, _ = gdf

    if format == "fixed":
        pdf = pdf.drop("col_category", axis=1)
        gdf = gdf.drop("col_category", axis=1)

    pdf_df_fname = tmp_path / "pdf_df.hdf"
    gdf_df_fname = tmp_path / "gdf_df.hdf"

    pdf.to_hdf(pdf_df_fname, key="hdf_tests", format=format, complib=complib)
    gdf.to_hdf(gdf_df_fname, key="hdf_tests", format=format, complib=complib)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect = pd.read_hdf(pdf_df_fname)
    got = pd.read_hdf(gdf_df_fname)

    assert_eq(expect, got, check_index_type=False)

    for column in pdf.columns:
        pdf_series_fname = tmp_path / (column + "_" + "pdf_series.hdf")
        gdf_series_fname = tmp_path / (column + "_" + "gdf_series.hdf")

        pdf[column].to_hdf(
            pdf_series_fname, key="hdf_tests", format=format, complib=complib
        )
        gdf[column].to_hdf(
            gdf_series_fname, key="hdf_tests", format=format, complib=complib
        )

        assert os.path.exists(pdf_series_fname)
        assert os.path.exists(gdf_series_fname)

        expect_series = pd.read_hdf(pdf_series_fname)
        got_series = pd.read_hdf(gdf_series_fname)

        assert_eq(expect_series, got_series, check_index_type=False)
