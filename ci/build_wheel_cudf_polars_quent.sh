#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.97}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
# sccache-dist drops directories that are only traversed via `..` in `#[path]`
# attributes (e.g. libc's glibc sysdeps), so remote rustc can't find the file.
# TODO: Remove this restriction once sccache-dist is able to properly
# traverse these directories.
export SCCACHE_NO_DIST_COMPILE=1

rapids-retry rustup toolchain install "${RUSTUP_TOOLCHAIN}" \
  --profile minimal \
  --component rustfmt \
  --component clippy

export RAPIDS_PACKAGE_VERSION
RAPIDS_PACKAGE_VERSION=$(<VERSION)
export CUDF_POLARS_QUENT_USE_RAPIDS_BUILD=true
# shellcheck source=ci/run_cudf_polars_quent_tests.sh
source ./ci/run_cudf_polars_quent_tests.sh

mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
python -m auditwheel repair \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/cudf_polars/quent/bridge/dist/*

shopt -s nullglob
generated_wheels=("${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*.whl)
shopt -u nullglob
if ((${#generated_wheels[@]} != 1)); then
  echo "Expected exactly one generated Quent wheel, found ${#generated_wheels[@]}" >&2
  exit 1
fi
if [[ "$(basename "${generated_wheels[0]}")" != *-cp312-abi3-*.whl ]]; then
  echo "Expected a cp312-abi3 wheel, found ${generated_wheels[0]}" >&2
  exit 1
fi

./ci/validate_wheel.sh \
  python/cudf_polars/quent/bridge \
  "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  10M

export RAPIDS_PACKAGE_NAME
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cudf-polars-quent cudf --stable)"
