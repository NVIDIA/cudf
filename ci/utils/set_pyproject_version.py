# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path

from packaging.version import Version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("version", type=Version)
    args = parser.parse_args()

    contents, replacements = re.subn(
        r'^version = "[^"]+"$',
        f'version = "{args.version}"',
        args.pyproject.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError(f"Expected one project version in {args.pyproject}")
    args.pyproject.write_text(contents)


if __name__ == "__main__":
    main()
