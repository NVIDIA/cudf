#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking or sourcing this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

REPO_ROOT="${PWD}"
BRIDGE_DIR="${REPO_ROOT}/python/cudf_polars/quent/bridge"
TRACKED_STUB="${BRIDGE_DIR}/cudf_polars_quent.pyi"

pushd "${BRIDGE_DIR}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked
cargo clean -p cudf-polars-quent
rm -f dist/*.whl target/wheels/*.whl

# maturin reads the wheel version from Cargo.toml. Local runs keep the checked-in version.
if [[ -n "${RAPIDS_PACKAGE_VERSION:-}" ]]; then
  "${REPO_ROOT}/ci/utils/set_cargo_version.sh" "${RAPIDS_PACKAGE_VERSION}" Cargo.toml
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
cargo check --locked --no-default-features
popd

pushd python/cudf_polars/quent/store
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
popd

pushd python/cudf_polars/quent/browser
cargo fmt --all -- --check
cargo clippy --locked --features generate --all-targets -- -D warnings
cargo test --locked --features generate --all-targets
cargo build --locked --release --target wasm32-unknown-unknown
target_dir="$(
  cargo metadata --no-deps --format-version 1 |
    python -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])'
)"
rm -rf generated
wasm-bindgen \
  "${target_dir}/wasm32-unknown-unknown/release/cudf_polars_quent_browser.wasm" \
  --target web \
  --out-dir generated
rg -q "export class BrowserServer" generated/cudf_polars_quent_browser.d.ts
test "$(stat -c%s generated/cudf_polars_quent_browser_bg.wasm)" -lt 5000000
popd
