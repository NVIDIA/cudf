# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pylibcudf.io.datasource import Datasource

from cudf_polars_cucascade import AdvisingDatasource


def test_advising_datasource_is_datasource():
    assert issubclass(AdvisingDatasource, Datasource)


def test_advising_datasource_not_constructible():
    with pytest.raises(ValueError, match=r"cannot be constructed directly"):
        AdvisingDatasource()
