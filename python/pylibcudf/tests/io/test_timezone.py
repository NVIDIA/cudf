# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import zoneinfo

import pytest

import pylibcudf as plc


@pytest.fixture
def los_angeles_tzpath():
    tz_path = next(
        (
            tz_path
            for tz_path in zoneinfo.TZPATH
            if os.path.isfile(os.path.join(tz_path, "America/Los_Angeles"))
        ),
        None,
    )
    if tz_path is None:
        pytest.skip("No TZif directory with America/Los_Angeles available.")
    return tz_path


def test_make_timezone_transition_table(los_angeles_tzpath):
    result = plc.io.timezone.make_timezone_transition_table(
        los_angeles_tzpath, "America/Los_Angeles"
    )
    assert isinstance(result, plc.Table)
    assert result.num_rows() > 0


def test_make_tz_transition_table_resolves_backward_alias(
    tmp_path, los_angeles_tzpath
):
    os.makedirs(tmp_path / "America", exist_ok=True)
    shutil.copyfile(
        os.path.join(los_angeles_tzpath, "America", "Los_Angeles"),
        tmp_path / "America" / "Los_Angeles",
    )
    (tmp_path / "tzdata.zi").write_text("L America/Los_Angeles US/Pacific\n")

    canonical = plc.io.timezone.make_timezone_transition_table(
        str(tmp_path), "America/Los_Angeles"
    )
    via_alias = plc.io.timezone.make_timezone_transition_table(
        str(tmp_path), "US/Pacific"
    )
    assert canonical.num_rows() == via_alias.num_rows()
    assert canonical.num_columns() == via_alias.num_columns()
