#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Maven classifier helpers for cuDF Java JARs. Source, do not execute.
#
# Derives the cuda<major>[-arm64] classifier, locates the unique classifier JAR
# under a build output dir, and asserts the expected JAR + POM pair.

# Print Maven classifier for a CUDA version + host arch (e.g. cuda12, cuda12-arm64).
cudf_java_maven_classifier() {
  local cuda_ver=${1:?cuda version required}
  local major host_arch
  major="$(echo "${cuda_ver}" | cut -d. -f1)"
  host_arch="$(uname -m)"
  case "${host_arch}" in
    x86_64)
      echo "cuda${major}"
      ;;
    aarch64|arm64)
      echo "cuda${major}-arm64"
      ;;
    *)
      echo "Error: unsupported host arch '${host_arch}'" >&2
      return 1
      ;;
  esac
}

# Echo the unique cudf-*-${classifier}.jar under dir; error if 0 or >1.
cudf_java_find_classifier_jar() {
  local dir=${1:?classifier out dir required}
  local classifier=${2:?classifier required}
  local jar="" candidate
  for candidate in "${dir}"/cudf-*-"${classifier}".jar; do
    if [[ ! -f ${candidate} ]]; then
      continue
    fi
    if [[ -n ${jar} ]]; then
      echo "Error: multiple JARs matching cudf-*-${classifier}.jar in ${dir}" >&2
      ls -1 "${dir}" >&2
      return 1
    fi
    jar=${candidate}
  done
  if [[ -z ${jar} ]]; then
    echo "Error: no cudf-*-${classifier}.jar in ${dir}" >&2
    ls -1 "${dir}" >&2
    return 1
  fi
  echo "${jar}"
}

# Assert unique classifier JAR + unique cudf-*.pom; print basenames.
cudf_java_assert_classifier_artifacts() {
  local dir=${1:?classifier out dir required}
  local classifier=${2:?classifier required}
  local jar pom="" candidate
  if ! jar="$(cudf_java_find_classifier_jar "${dir}" "${classifier}")"; then
    return 1
  fi
  for candidate in "${dir}"/cudf-*.pom; do
    if [[ ! -f ${candidate} ]]; then
      continue
    fi
    if [[ -n ${pom} ]]; then
      echo "Error: multiple POMs in ${dir}" >&2
      ls -1 "${dir}" >&2
      return 1
    fi
    pom=${candidate}
  done
  if [[ -z ${pom} ]]; then
    echo "Error: no cudf-*.pom in ${dir}" >&2
    ls -1 "${dir}" >&2
    return 1
  fi
  echo "cuDF Java JAR build succeeded:"
  echo "  $(basename "${jar}")"
  echo "  $(basename "${pom}")"
}

# Echo the absolute path of the unique classifier JAR under a java-build
# artifact tree. Classifier from $2 or RAPIDS_CUDA_VERSION; errors if 0 or >1.
cudf_java_resolve_artifact_jar() {
  local root=${1:?artifact root required}
  local cuda_ver=${2:-${RAPIDS_CUDA_VERSION:-}}
  local classifier
  local -a matches

  if [[ -z ${cuda_ver} ]]; then
    echo "Error: set RAPIDS_CUDA_VERSION or pass a cuda version" >&2
    return 1
  fi
  if [[ ! -d ${root} ]]; then
    echo "Error: artifact root '${root}' is not a directory" >&2
    return 1
  fi
  if ! classifier="$(cudf_java_maven_classifier "${cuda_ver}")"; then
    return 1
  fi

  # -${classifier}.jar excludes the sources/javadoc/tests JARs.
  mapfile -t matches < <(find "${root}" -type f -name "cudf-*-${classifier}.jar" | sort)
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Error: no cudf-*-${classifier}.jar under ${root}" >&2
    find "${root}" -type f >&2
    return 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "Error: multiple cudf-*-${classifier}.jar under ${root}" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 1
  fi
  realpath "${matches[0]}"
}
