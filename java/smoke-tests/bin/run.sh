#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-test cudf-java classifier JARs (cudf + slf4j classpath) in Docker.
#
# Exactly one source mode:
#   --artifact-url URL | --maven-repo-dir DIR | --use-maven-home | --use-maven-central

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TESTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${SMOKE_TESTS_ROOT}/.cache"

MODE=""
MAVEN_REPO=""
VERSION=""
ARTIFACT_URL=""
CUDA_VERSION=""
USE_CENTRAL=0

PARQUET_WARN_NEEDLE='Chunked Parquet reader: a chunk_read_limit'

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

set_mode() {
  local next="$1"
  if [[ -n "${MODE}" ]]; then
    die "Conflicting source modes: already using --${MODE}, cannot also use --${next}"
  fi
  MODE="${next}"
}

print_help() {
  cat << EOF
Usage: $(basename "$0") --artifact-url URL [options]
       $(basename "$0") --maven-repo-dir DIR [options]
       $(basename "$0") --use-maven-home [options]
       $(basename "$0") --use-maven-central --version VER [options]

Always runs smoke tests in Docker (Ubuntu 24.04 CUDA runtime + JDK 17 + Maven).

Source modes (exactly one required):
  --artifact-url URL      Fetch gather artifact via fetch_bundle.sh, then test
  --maven-repo-dir DIR    Local Maven tree (must contain ai/rapids/cudf/)
  --use-maven-home        Use ~/.m2/repository
  --use-maven-central     Resolve from Maven Central (--version required)

Options:
  --version VER           ai.rapids:cudf version
                          (required for --use-maven-central; otherwise if
                          omitted, exactly one version must exist under
                          ai/rapids/cudf/)
  --cuda-version 12|13    Narrow classifiers to this CUDA major
  -h, --help
EOF
}

discover_version() {
  local cudf_dir="${MAVEN_REPO}/ai/rapids/cudf"
  if [[ ! -d "${cudf_dir}" ]]; then
    die "No ai/rapids/cudf/ under ${MAVEN_REPO}"
  fi
  local -a versions=()
  local d
  for d in "${cudf_dir}"/*/; do
    if [[ ! -d "${d}" ]]; then
      continue
    fi
    versions+=("$(basename "${d}")")
  done
  if [[ "${#versions[@]}" -eq 0 ]]; then
    die "No version directories under ${cudf_dir}"
  fi
  if [[ "${#versions[@]}" -gt 1 ]]; then
    die "Multiple versions under ${cudf_dir}; pass --version. Found: ${versions[*]}"
  fi
  printf '%s\n' "${versions[0]}"
}

classifiers_for_arch() {
  local cuda="${1:-}"
  case "$(uname -m)" in
    x86_64)
      case "${cuda}" in
        "")  echo "unclassified,cuda12,cuda13" ;;
        12)  echo "unclassified,cuda12" ;;
        13)  echo "cuda13" ;;
        *)   die "--cuda-version must be 12 or 13 (got: ${cuda})" ;;
      esac
      ;;
    aarch64|arm64)
      case "${cuda}" in
        "")  echo "cuda12-arm64,cuda13-arm64" ;;
        12)  echo "cuda12-arm64" ;;
        13)  echo "cuda13-arm64" ;;
        *)   die "--cuda-version must be 12 or 13 (got: ${cuda})" ;;
      esac
      ;;
    *) die "Unsupported arch $(uname -m)" ;;
  esac
}

jar_for_classifier() {
  local c="$1"
  if [[ "${c}" == "unclassified" ]]; then
    echo "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}/cudf-${VERSION}.jar"
  else
    echo "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}/cudf-${VERSION}-${c}.jar"
  fi
}

ensure_image() {
  local major="$1"
  local tag="cudf-java-smoke:cuda${major}"
  local dockerfile="${SMOKE_TESTS_ROOT}/docker/Dockerfile.cuda${major}"
  if ! docker image inspect "${tag}" >/dev/null 2>&1; then
    info "Building ${tag}"
    docker build -t "${tag}" -f "${dockerfile}" "${SMOKE_TESTS_ROOT}/docker" >&2
  fi
  echo "${tag}"
}

run_smoke() {
  local classifier="$1"
  local log_file="${CACHE_DIR}/logs/smoke-${classifier}.log"
  mkdir -p "${CACHE_DIR}/logs"

  local -a mvn_cmd=(mvn -B "-Dcudf.version=${VERSION}")
  if [[ "${classifier}" == "unclassified" ]]; then
    mvn_cmd+=(-Dcudf.unclassified=true)
  else
    mvn_cmd+=("-Dcuda.classifier=${classifier}")
  fi
  mvn_cmd+=(package exec:java)

  local major=12
  case "${classifier}" in
    *cuda13*) major=13 ;;
  esac
  local tag
  tag="$(ensure_image "${major}")"

  # Maven writes compile output under target/; keep the tree writable.
  mkdir -p "${CACHE_DIR}/m2-container"

  local -a docker_cmd=(
    docker run --rm --gpus all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -e MAVEN_OPTS=-Duser.home=/tmp
    -v "${SMOKE_TESTS_ROOT}:/smoke"
    -v "${CACHE_DIR}/m2-container:/tmp/.m2"
    -w /smoke
  )

  info "Smoke classifier=${classifier} version=${VERSION}"
  set +e
  if [[ "${USE_CENTRAL}" -eq 1 ]]; then
    "${docker_cmd[@]}" \
      "${tag}" \
      "${mvn_cmd[@]}" \
        -Dcudf.smoke.central=true \
        -Dmaven.repo.local=/tmp/.m2/repository \
      >"${log_file}" 2>&1
  else
    "${docker_cmd[@]}" \
      -v "${MAVEN_REPO}:/maven-repo:ro" \
      "${tag}" \
      "${mvn_cmd[@]}" \
        -Dcudf.maven.repo=/maven-repo \
        -Dmaven.repo.local=/tmp/.m2/repository \
      >"${log_file}" 2>&1
  fi
  local rc=$?
  set -e
  cat "${log_file}"

  if [[ "${rc}" -ne 0 ]]; then
    return 1
  fi
  if ! grep -F -q "${PARQUET_WARN_NEEDLE}" "${log_file}"; then
    warn "Missing expected CUDF_LOG_WARN in ${log_file}"
    warn "  needle: ${PARQUET_WARN_NEEDLE}"
    return 1
  fi
  return 0
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) print_help; exit 0 ;;
      --artifact-url)
        require_value "$1" "${2:-}"
        set_mode "artifact-url"
        ARTIFACT_URL="$2"
        shift 2
        ;;
      --maven-repo-dir)
        require_value "$1" "${2:-}"
        set_mode "maven-repo-dir"
        MAVEN_REPO="$2"
        shift 2
        ;;
      --use-maven-home)
        set_mode "use-maven-home"
        MAVEN_REPO="${HOME}/.m2/repository"
        shift
        ;;
      --use-maven-central)
        set_mode "use-maven-central"
        USE_CENTRAL=1
        shift
        ;;
      --version)
        require_value "$1" "${2:-}"
        VERSION="$2"
        shift 2
        ;;
      --cuda-version)
        require_value "$1" "${2:-}"
        CUDA_VERSION="$2"
        shift 2
        ;;
      *) die "Unknown argument: $1 (try --help)" ;;
    esac
  done

  if [[ -z "${MODE}" ]]; then
    die "Exactly one source mode is required (try --help)"
  fi
  if ! command -v docker >/dev/null 2>&1; then
    die "docker is required"
  fi

  if [[ "${MODE}" == "artifact-url" ]]; then
    MAVEN_REPO="$("${SCRIPT_DIR}/fetch_bundle.sh" --artifact-url "${ARTIFACT_URL}")"
  fi

  if [[ "${USE_CENTRAL}" -eq 0 ]]; then
    if [[ ! -d "${MAVEN_REPO}/ai/rapids/cudf" ]]; then
      die "No ai/rapids/cudf/ under ${MAVEN_REPO}"
    fi
    MAVEN_REPO="$(cd "${MAVEN_REPO}" && pwd)"
    if [[ -z "${VERSION}" ]]; then
      VERSION="$(discover_version)"
    elif [[ ! -d "${MAVEN_REPO}/ai/rapids/cudf/${VERSION}" ]]; then
      die "Version ${VERSION} not under ${MAVEN_REPO}/ai/rapids/cudf/"
    fi
  else
    if [[ -z "${VERSION}" ]]; then
      die "--use-maven-central requires --version"
    fi
  fi

  local classifiers
  classifiers="$(classifiers_for_arch "${CUDA_VERSION}")"

  info "Mode:        ${MODE}"
  if [[ "${USE_CENTRAL}" -eq 0 ]]; then
    info "Maven repo:  ${MAVEN_REPO}"
  fi
  info "Version:     ${VERSION}"
  info "Classifiers: ${classifiers}"

  # Force Maven to resolve cudf from the source currently under test.
  rm -rf "${CACHE_DIR}/m2-container/repository/ai/rapids/cudf/${VERSION:?}"

  local failed=0
  local ran=0
  local c jar
  local -a classifier_list
  IFS=',' read -r -a classifier_list <<< "${classifiers}"
  for c in "${classifier_list[@]}"; do
    # Trim surrounding whitespace from comma-split tokens (e.g. " cuda12").
    c="$(echo "$c" | xargs)"
    if [[ -z "${c}" ]]; then
      continue
    fi
    if [[ "${USE_CENTRAL}" -eq 0 ]]; then
      jar="$(jar_for_classifier "${c}")"
      if [[ ! -f "${jar}" ]]; then
        warn "Skipping missing classifier '${c}' (${jar})"
        continue
      fi
    fi
    ran=$((ran + 1))
    if run_smoke "${c}"; then
      info "PASS: ${c}"
    else
      warn "FAIL: ${c}"
      failed=1
    fi
  done

  if [[ "${ran}" -eq 0 ]]; then
    die "No classifier JARs were smoke tested (all missing or none selected)"
  fi
  if [[ "${failed}" -ne 0 ]]; then
    die "One or more classifier smoke tests failed"
  fi
  info "All classifier smoke tests passed"
}

main "$@"
