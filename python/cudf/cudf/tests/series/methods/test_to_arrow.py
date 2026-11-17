# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa

import cudf


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical.from_codes([0, 1, -1], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert pa_cat.equals(gd_cat.to_arrow())
