# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for packaging collector-produced Quent contexts."""

from __future__ import annotations

import json
import uuid
import zipfile
from typing import TYPE_CHECKING
from unittest.mock import ANY

import pytest

pytest.importorskip("cudf_polars_quent")

from cudf_polars.quent._export import (
    SIDECAR_FILE_NAME,
    write_quent_export,
)
from cudf_polars.quent._runtime import QuentCollector, QuentSession

if TYPE_CHECKING:
    from pathlib import Path


def _collect_engine_events(root: Path) -> Path:
    collector = QuentCollector(root)
    session = QuentSession(collector.address)
    identifier = uuid.uuid4()
    session._engines[identifier] = (
        session.context.engine_observer()
        .handle(identifier)
        .init(
            instance_name="test",
            implementation={
                "name": "cudf-polars",
                "version": "test",
                "backend": "spmd",
                "custom_attributes": {"backend": "spmd"},
            },
        )
    )
    session._engines.pop(identifier).exit()
    session.close()
    collector.close()
    return next(path for path in root.iterdir() if path.is_dir())


def test_collector_export_writes_sidecar_and_generated_streams(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quent"
    context = _collect_engine_events(root)

    sidecar = json.loads((context / SIDECAR_FILE_NAME).read_text())
    assert sidecar["model"]["name"] == "CudfPolars"
    assert sidecar["model"]["analyzer_package"] == "cudf-polars-quent-analyzer"
    engine_file = next((context / "Engine").glob("*.ndjson"))
    assert [
        json.loads(line)["data"] for line in engine_file.read_text().splitlines()
    ] == [
        {"Init": ANY},
        {"Exit": {"seq": 1}},
    ]


def test_export_packages_generated_context(tmp_path: Path) -> None:
    root = tmp_path / "quent"
    context = _collect_engine_events(root)
    archive_path = tmp_path / "trace.zip"
    write_quent_export(root, archive_path)
    assert not root.exists()

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert f"{context.name}/{SIDECAR_FILE_NAME}" in names
        streams = {name.split("/")[1] for name in names if "/" in name}
        assert "Engine" in streams
        assert "engine" not in streams


def test_benchmark_writer_packages_collector_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cudf_polars.streaming.benchmarks import utils as benchmark_utils

    run_id = uuid.uuid4()
    root = tmp_path / str(run_id)
    _collect_engine_events(root)

    class FakeEngine:
        _quent_output_root = root

    archive_path = tmp_path / "logs" / f"{run_id}.zip"
    monkeypatch.chdir(tmp_path)
    assert (
        benchmark_utils._write_quent_traces(
            FakeEngine(),  # type: ignore[arg-type]
            run_id,
            collect_traces=True,
            quent_archive=archive_path,
        )
        == archive_path
    )
    assert not root.exists()
    with zipfile.ZipFile(archive_path) as archive:
        assert any(name.endswith(SIDECAR_FILE_NAME) for name in archive.namelist())
