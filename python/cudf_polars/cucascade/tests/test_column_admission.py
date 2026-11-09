# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf_polars_cucascade import ColumnStats


def test_column_stats_not_constructible():
    with pytest.raises(ValueError, match=r"cannot be constructed directly"):
        ColumnStats()
