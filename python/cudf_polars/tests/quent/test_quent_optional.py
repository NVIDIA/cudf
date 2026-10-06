# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for using cudf-polars without the optional Quent extension."""

from __future__ import annotations

import subprocess
import sys


def test_imports_without_quent_extension() -> None:
    code = """
import sys
sys.modules["cudf_polars_quent"] = None

import cudf_polars.quent
import polars as pl

pl.LazyFrame({"a": [1, 2]}).collect(engine="gpu")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
