# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the sharded pandas-tests result merge.

The merged file is the baseline every PR diffs against, so a silent error here
shows up as phantom failures in unrelated PRs rather than as a broken job.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_SCRIPT = (
    pathlib.Path(__file__).parents[3]
    / "ci"
    / "cudf_pandas_scripts"
    / "pandas-tests"
    / "merge-results.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("merge_results", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_results


@pytest.fixture(scope="module")
def merge_results():
    if not _SCRIPT.is_file():
        pytest.skip(f"{_SCRIPT} not found")
    return _load()


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_sums_numeric_fields_for_a_shared_module(merge_results, tmp_path):
    # Sharding is per test, not per module, so the same module appears in
    # several shards and its counts have to add up.
    a = _write(
        tmp_path, "a.json", {"m.py": {"total": 10, "passed": 8, "failed": 2}}
    )
    b = _write(
        tmp_path, "b.json", {"m.py": {"total": 5, "passed": 5, "failed": 0}}
    )
    assert merge_results([a, b]) == {
        "m.py": {"total": 15, "passed": 13, "failed": 2}
    }


def test_unions_modules_seen_in_only_one_shard(merge_results, tmp_path):
    a = _write(tmp_path, "a.json", {"m.py": {"total": 1}})
    b = _write(tmp_path, "b.json", {"n.py": {"total": 2}})
    assert merge_results([a, b]) == {
        "m.py": {"total": 1},
        "n.py": {"total": 2},
    }


def test_keeps_first_value_for_nonnumeric_fields(merge_results, tmp_path):
    a = _write(tmp_path, "a.json", {"m.py": {"note": "first", "total": 1}})
    b = _write(tmp_path, "b.json", {"m.py": {"note": "second", "total": 1}})
    assert merge_results([a, b])["m.py"] == {"note": "first", "total": 2}


def test_does_not_sum_booleans(merge_results, tmp_path):
    # bool is a subclass of int; summing it would turn two True into 2.
    a = _write(tmp_path, "a.json", {"m.py": {"flag": True}})
    b = _write(tmp_path, "b.json", {"m.py": {"flag": True}})
    assert merge_results([a, b])["m.py"]["flag"] is True


def test_sums_floats(merge_results, tmp_path):
    a = _write(tmp_path, "a.json", {"m.py": {"seconds": 1.5}})
    b = _write(tmp_path, "b.json", {"m.py": {"seconds": 2.25}})
    assert merge_results([a, b])["m.py"]["seconds"] == pytest.approx(3.75)


def test_no_inputs_yields_an_empty_summary(merge_results):
    assert merge_results([]) == {}


def test_single_shard_is_passed_through_unchanged(merge_results, tmp_path):
    payload = {"m.py": {"total": 3, "passed": 3, "name": "x"}}
    assert merge_results([_write(tmp_path, "a.json", payload)]) == payload
