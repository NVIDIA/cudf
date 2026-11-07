#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export LC_ALL=C

manifests="${SRC_DIR}/staged-manifests"
staged="${SRC_DIR}/staged-files"

# Compare sorted inventories so dependency files stay out of package manifests.
# NUL delimiters keep whitespace in installed filenames from changing ownership.
prefix_files() {
  find "$PREFIX" \( -type f -o -type l \) -printf '%P\0' | sort -z
}

# Merge payloads into existing dependency directories without recopying file contents.
# Batch files by parent directory to limit subprocess overhead and argument size.
move_files() {
  local source=$1 destination=$2 file directory previous=
  local -a batch=()
  while IFS= read -r -d '' file; do
    directory=.
    if [[ "$file" == */* ]]; then
      directory=${file%/*}
    fi
    if (( ${#batch[@]} )) && { [[ "$directory" != "$previous" ]] || (( ${#batch[@]} >= 128 )); }; then
      mkdir -p "${destination}/${previous}"
      mv -t "${destination}/${previous}" -- "${batch[@]}"
      batch=()
    fi
    previous=$directory
    batch+=("${source}/${file}")
  done
  if (( ${#batch[@]} )); then
    mkdir -p "${destination}/${previous}"
    mv -t "${destination}/${previous}" -- "${batch[@]}"
  fi
}

case "$1" in
  record)
    package=$2
    shift 2
    mkdir -p "$manifests"
    # Outputs must introduce distinct paths so dependency files and earlier
    # outputs stay out of this manifest. Keep earlier installs available for
    # dependent installs and for compiling the examples.
    prefix_files > "${manifests}/before"
    "$@"
    prefix_files > "${manifests}/after"
    comm -z -13 "${manifests}/before" "${manifests}/after" >> "${manifests}/${package}.files"
    ;;
  finish)
    # Keep earlier libraries available until examples and all installs finish.
    # Moving only newly installed files leaves the dependency environment intact.
    for manifest in "${manifests}"/*.files; do
      package=$(basename "$manifest" .files)
      move_files "$PREFIX" "${staged}/${package}" < "$manifest"
    done
    # Rattler captures the entire workdir; only installed payloads are needed now.
    find "$SRC_DIR" -mindepth 1 -maxdepth 1 \
      ! -name staged-files ! -name conda_build.sh ! -name build_env.sh \
      -exec rm -rf -- {} +
    ;;
  restore)
    package=$2
    # Merge into the output's dependency prefix without another file-content copy.
    find "${staged}/${package}" \( -type f -o -type l \) -printf '%P\0' |
      move_files "${staged}/${package}" "$PREFIX"
    ;;
  *)
    echo "Unknown staging operation: $1" >&2
    exit 1
    ;;
esac
