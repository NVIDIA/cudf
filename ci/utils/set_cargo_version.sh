#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: set_cargo_version.sh <PEP 440 version> <path to Cargo.toml>
#
# maturin reads the package version from Cargo.toml, which needs SemVer: strip leading zeros
# rapids-pre-commit-hooks: disable-next-line
# (26.12.00a62 -> 26.12.0a62) and mark pre-releases (26.12.0a62 -> 26.12.0-a62). maturin maps
# the result back to the original PEP 440 version.

set -euo pipefail

if (($# != 2)); then
  echo "Usage: $0 <PEP 440 version> <path to Cargo.toml>" >&2
  exit 1
fi

pep440_version=$1
cargo_toml=$2

cargo_version=$(sed -E 's/(^|\.)0+([0-9])/\1\2/g; s/([0-9])(a|rc)([0-9])/\1-\2\3/' <<< "${pep440_version}")
sed -i "s/^version = .*/version = \"${cargo_version}\"/" "${cargo_toml}"
