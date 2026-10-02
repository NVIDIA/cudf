# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for using cudf-polars without the optional Quent extension."""

from __future__ import annotations

import subprocess
import sys


def test_imports_without_quent_extension() -> None:
    code = """
import builtins

original_import = builtins.__import__

def without_quent(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cudf_polars_quent":
        raise ImportError("blocked optional Quent extension")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_quent

import cudf_polars.quent
from cudf_polars.engine.spmd import SPMDEngine

assert cudf_polars.quent.QuentContext()
assert SPMDEngine
"""
    subprocess.run([sys.executable, "-c", code], check=True)
