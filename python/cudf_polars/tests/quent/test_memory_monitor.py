# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Quent memory resource sampler."""

from __future__ import annotations

import pytest

from cudf_polars.quent._context import QuentConfig


@pytest.mark.parametrize("interval_s", [0, -1.0])
def test_quent_config_rejects_non_positive_poll_interval(interval_s: float) -> None:
    with pytest.raises(ValueError, match="device_memory_poll_interval_s"):
        QuentConfig(device_memory_poll_interval_s=interval_s)


@pytest.mark.parametrize("interval_s", [None, 0.25])
def test_quent_config_poll_interval_roundtrips(tmp_path, interval_s) -> None:
    config = QuentConfig(
        output_root=str(tmp_path), device_memory_poll_interval_s=interval_s
    )
    assert QuentConfig._deserialize(config._serialize()) == config


def test_quent_config_polls_once_per_second_by_default() -> None:
    assert QuentConfig().device_memory_poll_interval_s == 0.1
