# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version utilities so that cudf_polars supports a range of polars versions."""

from __future__ import annotations

from packaging.version import parse

from polars import __version__

POLARS_VERSION = parse(__version__)
POLARS_LOWER_BOUND = parse("2.0")
POLARS_VERSION_LT_21 = parse("2.1") > POLARS_VERSION


def _ensure_polars_version() -> None:
    if POLARS_VERSION < POLARS_LOWER_BOUND:
        raise ImportError(
            f"cudf_polars requires py-polars v{POLARS_LOWER_BOUND} or greater."
        )  # pragma: no cover
