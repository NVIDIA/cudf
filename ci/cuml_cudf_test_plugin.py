# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.nodeid.endswith(
            "test_text_feature_extraction.py::test_countvectorizer_custom_vocabulary"
        ):
            # cuml.feature_extraction.text._check_vocabulary still calls
            # Series.sort_index(inplace=True); leave other warnings visible.
            item.add_marker(
                pytest.mark.filterwarnings(
                    r"ignore:^The inplace keyword in Series\.sort_index is "
                    r"deprecated and will be removed in a future version "
                    r"\(PDEP-8\)\.:pandas.errors.Pandas4Warning"
                )
            )
