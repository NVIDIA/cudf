# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package collector-produced cudf-polars Quent contexts."""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SIDECAR_FILE_NAME = "model.qmi"


@dataclasses.dataclass(frozen=True)
class BrowserBundleInfo:
    """Metadata returned by the typed browser-bundle converter."""

    path: Path
    engine_id: str
    context_count: int
    event_count: int
    schema_digest: str
    format_version: int


def write_quent_browser_bundle(export_root: Path, output: Path) -> BrowserBundleInfo:
    """Convert a collector directory or Quent ZIP to a relocatable browser bundle."""
    import cudf_polars_quent

    summary = cudf_polars_quent.write_browser_bundle(export_root, output)
    return BrowserBundleInfo(
        path=output,
        engine_id=summary["engine_id"],
        context_count=summary["context_count"],
        event_count=summary["event_count"],
        schema_digest=summary["schema_digest"],
        format_version=summary["format_version"],
    )


def write_quent_export(export_root: Path, quent_archive: Path) -> Path:
    """Package collector-produced contexts into one Quent ZIP archive."""
    quent_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = quent_archive.with_name(f".{quent_archive.name}.tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(export_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(export_root)
            archive.write(path, relative)
    temporary.replace(quent_archive)
    shutil.rmtree(export_root, ignore_errors=True)
    with contextlib.suppress(OSError):
        export_root.parent.rmdir()

    return quent_archive


__all__ = [
    "SIDECAR_FILE_NAME",
    "BrowserBundleInfo",
    "write_quent_browser_bundle",
    "write_quent_export",
]
