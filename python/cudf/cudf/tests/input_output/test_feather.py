# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import NUMERIC_TYPES


@pytest.fixture(params=[0, 10])
def pdf(request):
    rng = np.random.default_rng(seed=0)
    types = [*NUMERIC_TYPES, "bool"]
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        }
    )
    # Create non-numeric categorical data otherwise may get typecasted
    data = [ascii_letters[rng.integers(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")
    return test_pdf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize(
    "columns",
    [["col_int8"], ["col_category"], ["col_int32", "col_float32"], None],
)
def test_feather_reader(pdf, columns, tmp_path):
    feather_file = tmp_path / "test.feather"
    pdf.to_feather(feather_file)
    expect = pa.feather.read_table(feather_file, columns=columns).to_pandas()
    got = (
        cudf.read_feather(feather_file, columns=columns)
        .to_arrow(preserve_index=False)
        .to_pandas()
    )

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_feather_writer(tmp_path, pdf):
    gdf = cudf.DataFrame(pdf)
    pdf_fname = tmp_path / "pdf.feather"
    gdf_fname = tmp_path / "gdf.feather"

    pdf.to_feather(pdf_fname)
    gdf.to_feather(gdf_fname)

    expect = pa.feather.read_table(pdf_fname)
    got = pa.feather.read_table(gdf_fname)
    # For empty dataframes, pandas writes category dict with null values type
    # while cudf writes it with string values type. Cast to match.
    if len(expect) == 0 and "col_category" in expect.column_names:
        idx = expect.schema.get_field_index("col_category")
        expect = expect.set_column(
            idx,
            got.schema.field(idx),
            expect.column(idx).cast(got.schema.field(idx).type),
        )
    assert pa.Table.equals(expect, got)
