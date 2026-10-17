# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import email
import tomllib
import zipfile
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("expected_version", type=Version)
    parser.add_argument("cargo_manifest", type=Path)
    args = parser.parse_args()

    _, filename_version, _, _ = parse_wheel_filename(args.wheel.name)
    if filename_version != args.expected_version:
        raise RuntimeError(
            f"Wheel filename version {filename_version} does not match "
            f"{args.expected_version}"
        )

    with zipfile.ZipFile(args.wheel) as archive:
        metadata_files = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"Expected one METADATA file in {args.wheel}, "
                f"found {len(metadata_files)}"
            )
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
    metadata_version = Version(metadata["Version"])

    if metadata_version != args.expected_version:
        raise RuntimeError(
            f"Wheel METADATA version {metadata_version} does not match "
            f"{args.expected_version}"
        )

    with args.cargo_manifest.open("rb") as file:
        cargo_version = Version(tomllib.load(file)["package"]["version"])
    if cargo_version.release != args.expected_version.release:
        raise RuntimeError(
            f"Cargo release {cargo_version.release} does not match "
            f"wheel release {args.expected_version.release}"
        )


if __name__ == "__main__":
    main()
