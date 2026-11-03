# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cudf
from cudf.testing import assert_eq


def test_value_counts_no_subset():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    with pytest.raises(KeyError):
        gdf.value_counts(subset=["not_a_column_name"])


@pytest.mark.parametrize(
    "data",
    [
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", None, None, "Louise"],
        },
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("use_subset", [True, False])
def test_value_counts(
    data,
    sort,
    ascending,
    normalize,
    dropna,
    use_subset,
):
    subset = [next(iter(data.keys()))]
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    got = gdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )
    expected = pdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )

    # With dropna=False, cuDF and pandas index null representations differ
    # (<NA> vs NaN), so index-type equality is not meaningful.
    assert_eq(
        got.sort_index(),
        expected.sort_index(),
        check_index_type=dropna,
    )
