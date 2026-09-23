#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

BRIDGE_DIR="${PWD}/python/cudf_polars/quent/bridge"
TRACKED_STUB="${PWD}/python/cudf_polars/cudf_polars/_quent.pyi"

pushd "${BRIDGE_DIR}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked
python -m maturin build --locked

shopt -s nullglob
generated_stubs=(target/*/build/cudf-polars-quent-*/out/_quent/__init__.pyi)
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
popd

git diff --exit-code -- "${TRACKED_STUB}"
