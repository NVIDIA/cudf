# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path("python/pylibcudf/pylibcudf")


def _git_ls_files(pattern: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", pattern],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines()]


def _spdx_header(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    header = []
    for line in lines:
        if line.startswith("# SPDX-"):
            header.append(line)
            continue
        if header and line == "":
            break
        if header:
            break
    return "\n".join(header) + "\n\n" if header else ""


def _stored_spdx_header(path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return ""
    lines = result.stdout.splitlines()
    header = []
    for line in lines:
        if line.startswith("# SPDX-"):
            header.append(line)
            continue
        if header and line == "":
            break
        if header:
            break
    return "\n".join(header) + "\n\n" if header else ""


def _replace_header(path: Path, header: str) -> None:
    lines = path.read_text().splitlines()
    while lines and lines[0].startswith("# SPDX-"):
        lines.pop(0)
    if lines and lines[0] == "":
        lines.pop(0)
    path.write_text(header + "\n".join(lines) + "\n")


def _generate_stub(pyx_file: Path, pyi_file: Path) -> int:
    header = _spdx_header(pyi_file) or _stored_spdx_header(pyi_file) or _spdx_header(pyx_file)
    result = subprocess.run(
        [
            "stubgen-pyx",
            str(PACKAGE_DIR),
            "--file",
            pyx_file.relative_to(PACKAGE_DIR).as_posix(),
            "--output-file",
            str(pyi_file),
            "--continue-on-error",
            "--include-private",
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        _replace_header(pyi_file, header)
    else:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def main() -> int:
    failures = []
    for pyi_file in sorted(
        path for path in _git_ls_files(str(PACKAGE_DIR)) if path.suffix == ".pyi"
    ):
        pyx_file = pyi_file.with_suffix(".pyx")
        if not pyx_file.exists():
            continue
        if _generate_stub(pyx_file, pyi_file) != 0:
            failures.append(pyx_file)

    if failures:
        print("Failed to generate pylibcudf stubs for:", file=sys.stderr)
        for path in failures:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
