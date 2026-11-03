# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PDS-DS query parameters for all scale factors."""

from __future__ import annotations

from typing import Any

from cudf_polars.streaming.benchmarks.pdsds_parameters.parameter_substitutions import (
    PARAMETERS,
)


def load_parameters(
    scale_factor: int | str, query_id: int, *, qualification: bool = False
) -> dict[str, Any]:
    """
    Load parameters for a specific query and scale factor.

    Parameters
    ----------
    scale_factor : int | str
        The PDS-DS scale factor (e.g., 1, 10, 100, 1000, 10000) or "qualification"
        for TPC-DS qualification parameters. Ignored if qualification=True.
    query_id : int
        The PDS-DS query number
    qualification : bool
        If True, load TPC-DS qualification parameters from specification Appendix B
        instead of randomly generated parameters for the given scale factor.

    Returns
    -------
    dict
        Dictionary of parameter names to values

    Raises
    ------
    KeyError
        If parameters are not found for the given scale factor or query ID
    """
    sf_key = "qualification" if qualification else str(scale_factor)

    try:
        scale_params = PARAMETERS["scale_factors"][sf_key]
    except KeyError as e:
        if qualification:
            msg = "No qualification parameters found"
        else:
            msg = f"No parameters found for scale factor {scale_factor}"
        raise KeyError(msg) from e

    try:
        return scale_params[str(query_id)]
    except KeyError as e:
        param_type = (
            "qualification" if qualification else f"scale factor {scale_factor}"
        )
        msg = f"No parameters found for query {query_id} at {param_type}"
        raise KeyError(msg) from e


__all__ = ["load_parameters"]
