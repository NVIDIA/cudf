# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package collector-produced cudf-polars Quent contexts."""

from __future__ import annotations

import contextlib
import shutil
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SIDECAR_FILE_NAME = "model.qmi"


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
    "write_quent_export",
]
