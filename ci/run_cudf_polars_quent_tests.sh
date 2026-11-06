#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking or sourcing this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

BRIDGE_DIR="${PWD}/python/cudf_polars/quent/bridge"
TRACKED_STUB="${BRIDGE_DIR}/cudf_polars_quent.pyi"

pushd "${BRIDGE_DIR}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked
cargo clean -p cudf-polars-quent
rm -f dist/*.whl target/wheels/*.whl

# maturin reads the wheel version from Cargo.toml, which needs SemVer: strip leading zeros
# rapids-pre-commit-hooks: disable-next-line
# (26.12.00a62 -> 26.12.0a62) and mark pre-releases (26.12.0a62 -> 26.12.0-a62). maturin maps
# the result back to the original PEP 440 version. Local runs keep the checked-in version.
if [[ -n "${RAPIDS_PACKAGE_VERSION:-}" ]]; then
  cargo_version=$(sed -E 's/(^|\.)0+([0-9])/\1\2/g; s/([0-9])(a|rc)([0-9])/\1-\2\3/' <<< "${RAPIDS_PACKAGE_VERSION}")
  sed -i "s/^version = .*/version = \"${cargo_version}\"/" Cargo.toml
fi
popd

if [[ "${CUDF_POLARS_QUENT_USE_RAPIDS_BUILD:-false}" == "true" ]]; then
  # Do not pass --locked: changing the version requires cargo to update the crate's own
  # Cargo.lock entry. The cargo check above already verified the checked-in lockfile.
  build_wheel cudf-polars-quent python/cudf_polars/quent/bridge
  shopt -s nullglob
  generated_wheels=("${BRIDGE_DIR}"/dist/*.whl)
  shopt -u nullglob
else
  pushd "${BRIDGE_DIR}"
  python -m maturin build --release
  popd
  shopt -s nullglob
  generated_wheels=("${BRIDGE_DIR}"/target/wheels/*.whl)
  shopt -u nullglob
fi

if ((${#generated_wheels[@]} != 1)); then
  echo "Expected exactly one generated Quent wheel, found ${#generated_wheels[@]}" >&2
  exit 1
fi
if [[ "$(basename "${generated_wheels[0]}")" != *-cp312-abi3-*.whl ]]; then
  echo "Expected a cp312-abi3 wheel, found ${generated_wheels[0]}" >&2
  exit 1
fi

shopt -s nullglob
generated_stubs=("${BRIDGE_DIR}"/target/*/build/cudf-polars-quent-*/out/cudf_polars_quent.pyi)
shopt -u nullglob
if ((${#generated_stubs[@]} == 0)); then
  echo "No generated Quent stub found" >&2
  exit 1
fi

generated_stub="${generated_stubs[0]}"
for candidate in "${generated_stubs[@]:1}"; do
  if [[ "${candidate}" -nt "${generated_stub}" ]]; then
    generated_stub="${candidate}"
  fi
done

cp "${generated_stub}" "${TRACKED_STUB}"
git diff --exit-code -- "${TRACKED_STUB}"

pushd python/cudf_polars/quent/analyzer
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
popd
