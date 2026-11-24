# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf._version import __version__

from cudf_polars_cucascade import built_against_cudf_version


def test_built_against_cudf_version():
    assert len(built_against_cudf_version) == 3
    assert all(isinstance(part, int) for part in built_against_cudf_version)
    major, minor = (int(part) for part in __version__.split(".")[:2])
    assert built_against_cudf_version[:2] == (major, minor)
