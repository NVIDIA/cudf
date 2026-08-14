#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download a java-gather GitHub Actions artifact into:
#   <output-dir>/runs/<run_id>/artifacts/<artifact_id>/
# Default <output-dir>: java/smoke-tests/.cache/downloads

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TESTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARTIFACT_URL=""
OUTPUT_DIR="${CUDF_JAVA_SMOKE_OUTPUT_DIR:-${SMOKE_TESTS_ROOT}/.cache/downloads}"

info() { printf '==> %s\n' "$*" >&2; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "${flag} requires a value"
  fi
}

print_help() {
  cat << EOF
Usage: $(basename "$0") --artifact-url <url> [--output-dir DIR]

Download the cudf_java_maven_repo GitHub Actions artifact into:
  <output-dir>/runs/<run_id>/artifacts/<artifact_id>/

  --output-dir DIR   Default: ${SMOKE_TESTS_ROOT}/.cache/downloads
                     (or \$CUDF_JAVA_SMOKE_OUTPUT_DIR)

Requires gh. If the destination already exists and looks valid, warns and reuses it.
EOF
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) print_help; exit 0 ;;
      --artifact-url)
        require_value "$1" "${2:-}"
        ARTIFACT_URL="$2"
        shift 2
        ;;
      --output-dir)
        require_value "$1" "${2:-}"
        OUTPUT_DIR="$2"
        shift 2
        ;;
      *) die "Unknown argument: $1 (try --help)" ;;
    esac
  done

  if [[ -z "${ARTIFACT_URL}" ]]; then
    die "--artifact-url is required (try --help)"
  fi
  if ! command -v gh >/dev/null 2>&1; then
    die "'gh' (GitHub CLI) is required"
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    die "'unzip' is required"
  fi

  local repo run_id artifact_id
  if [[ "${ARTIFACT_URL}" =~ github\.com/([^/]+/[^/]+)/actions/runs/([A-Za-z0-9_]+)/artifacts/([A-Za-z0-9_]+)/?$ ]]; then
    repo="${BASH_REMATCH[1]}"
    run_id="${BASH_REMATCH[2]}"
    artifact_id="${BASH_REMATCH[3]}"
  else
    die "Could not parse artifact URL: ${ARTIFACT_URL}"
  fi

  local dest="${OUTPUT_DIR}/runs/${run_id}/artifacts/${artifact_id}"
  if [[ -d "${dest}/ai/rapids/cudf" ]]; then
    warn "Destination already exists; reusing: ${dest}"
    printf '%s\n' "${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    die "Destination exists but is incomplete (no ai/rapids/cudf/): ${dest}"
  fi

  mkdir -p "${dest}"
  info "Downloading artifact ${artifact_id} from ${repo} run ${run_id}"
  gh api \
    -H "Accept: application/vnd.github+json" \
    "/repos/${repo}/actions/artifacts/${artifact_id}/zip" \
    > "${dest}/artifact.zip"
  unzip -q -o "${dest}/artifact.zip" -d "${dest}"

  if [[ -d "${dest}/cudf_java_maven_repo/ai" ]]; then
    mv "${dest}/cudf_java_maven_repo/ai" "${dest}/"
    rm -rf "${dest:?}/cudf_java_maven_repo"
  fi
  if [[ ! -d "${dest}/ai/rapids/cudf" ]]; then
    die "No ai/rapids/cudf/ under ${dest} after download"
  fi

  printf 'artifact-url:%s\n' "${ARTIFACT_URL}" > "${dest}/.source"
  info "Destination: ${dest}"
  printf '%s\n' "${dest}"
}

main "$@"
