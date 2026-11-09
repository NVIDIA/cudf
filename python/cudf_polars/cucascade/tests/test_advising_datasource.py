# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pylibcudf.io.datasource import Datasource

from cudf_polars_cucascade import AdvisingDatasource


def test_an_advising_datasource_is_a_pylibcudf_datasource():
    assert issubclass(AdvisingDatasource, Datasource)


def test_an_advising_datasource_is_not_constructible():
    with pytest.raises(ValueError, match=r"ScanCache\.open"):
        AdvisingDatasource()
