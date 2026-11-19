# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit status and error reporting for Quent artifact writes in the benchmark CLI."""

from __future__ import annotations

import dataclasses
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cudf_polars_quent")

from cudf_polars.quent._export import BrowserBundleInfo
from cudf_polars.streaming.benchmarks import utils
from cudf_polars.streaming.benchmarks.utils import (
    EXIT_QUERY_FAILURE,
    EXIT_SUCCESS,
)

if TYPE_CHECKING:
    from pathlib import Path


def _finalize(
    *,
    query_failures: list[tuple[int, int]] | None = None,
    artifact_error: BaseException | None = None,
) -> int:
    """Run ``_finalize_benchmark_run`` and return its exit status."""
    run_options = MagicMock(summarize=False, output=None)
    run_config = MagicMock(validation_method=None)
    with pytest.raises(SystemExit) as excinfo:
        utils._finalize_benchmark_run(
            run_options,
            run_config,
            validation_failures=[],
            query_failures=query_failures or [],
            serializable_engine_config={"roles": [], "extra_info": {}},
            startup_duration_ms=None,
            shutdown_duration_ms=None,
            artifact_error=artifact_error,
        )
    assert isinstance(excinfo.value.code, int)
    return excinfo.value.code


def test_finalize_exits_zero_without_artifact_error() -> None:
    assert _finalize() == EXIT_SUCCESS


def test_finalize_exits_nonzero_on_artifact_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _finalize(artifact_error=OSError("disk full")) == 1
    captured = capsys.readouterr()
    assert "Failed to write Quent artifacts" in captured.err
    assert "disk full" in captured.err


def test_finalize_keeps_query_failure_code_over_artifact_error() -> None:
    code = _finalize(
        query_failures=[(1, 0)],
        artifact_error=OSError("disk full"),
    )
    assert code == EXIT_QUERY_FAILURE


def _bundle_info(path: Path) -> BrowserBundleInfo:
    return BrowserBundleInfo(
        path=path,
        engine_id="00000000-0000-0000-0000-000000000000",
        context_count=1,
        event_count=2,
        schema_digest="sha256:" + "0" * 64,
        format_version=1,
    )


def test_bundle_failure_is_returned_and_zip_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = uuid.uuid4()
    export_root = tmp_path / str(run_id)
    export_root.mkdir()
    archive = tmp_path / "logs" / f"{run_id}.zip"
    engine = SimpleNamespace(_quent_output_root=export_root)
    config: dict = {"roles": [], "extra_info": {}}

    def fake_export(root: Path, archive_path: Path) -> Path:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"PK")
        return archive_path

    def broken_bundle(source: Path, output: Path) -> BrowserBundleInfo:
        assert source == archive
        raise RuntimeError("bundle failed")

    monkeypatch.setattr(utils, "write_quent_export", fake_export)
    monkeypatch.setattr(utils, "write_quent_browser_bundle", broken_bundle)

    error = utils._write_quent_traces(
        engine=engine,  # type: ignore[arg-type]
        run_id=run_id,
        quent_archive=archive,
        serializable_engine_config=config,
    )

    assert isinstance(error, RuntimeError)
    assert str(error) == "bundle failed"
    assert archive.is_file()
    assert config == {"roles": [], "extra_info": {}}


def test_bundle_success_records_role_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = uuid.uuid4()
    export_root = tmp_path / str(run_id)
    export_root.mkdir()
    archive = tmp_path / "logs" / f"{run_id}.zip"
    engine = SimpleNamespace(_quent_output_root=export_root)
    config: dict = {"roles": [], "extra_info": {}}
    bundle_path = archive.with_suffix(".quent-browser")

    def fake_export(root: Path, archive_path: Path) -> Path:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"PK")
        return archive_path

    def fake_bundle(source: Path, output: Path) -> BrowserBundleInfo:
        output.write_bytes(b"QNTBR001")
        return _bundle_info(output)

    monkeypatch.setattr(utils, "write_quent_export", fake_export)
    monkeypatch.setattr(utils, "write_quent_browser_bundle", fake_bundle)

    error = utils._write_quent_traces(
        engine=engine,  # type: ignore[arg-type]
        run_id=run_id,
        quent_archive=archive,
        serializable_engine_config=config,
    )

    assert error is None
    [role] = config["roles"]
    assert role["type"] == "quent-browser"
    assert role["filename"] == bundle_path.name
    assert role["model_name"] == "cudf-polars"
    assert config["extra_info"]["quent-browser-bundle"] == str(bundle_path.absolute())


def test_run_id_mismatch_is_reported_not_raised(tmp_path: Path) -> None:
    export_root = tmp_path / "not-the-run-id"
    export_root.mkdir()
    engine = SimpleNamespace(_quent_output_root=export_root)

    error = utils._write_quent_traces(
        engine=engine,  # type: ignore[arg-type]
        run_id=uuid.uuid4(),
        quent_archive=tmp_path / "logs" / "x.zip",
        serializable_engine_config={"roles": [], "extra_info": {}},
    )

    assert isinstance(error, ValueError)
    assert "does not match run" in str(error)


def test_bundle_info_fields_match_role() -> None:
    """Every field of the bundle summary must be accepted by the role schema."""
    fields = {f.name for f in dataclasses.fields(BrowserBundleInfo)} - {"path"}
    role_fields = {f.name for f in dataclasses.fields(utils.QuentBrowserRole)}
    assert fields <= role_fields
