# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(scope="module")
def row_group_size():
    """Row group size used by hybrid scan parquet test fixtures."""
    return 250
