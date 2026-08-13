#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Decoupled gather step: assemble per-classifier cuDF Java JARs into a single
# Maven-repository-layout directory.
#
# Input: --jars-dir contains one subdirectory per classifier, each holding
#   - exactly one cudf-<version>-<classifier>.jar
#   - one cudf-<version>-sources.jar
#   - one cudf-<version>-javadoc.jar
#   - one cudf-<version>.pom
# Subdir names ARE the classifier names, and the artifact version is derived
# from the JAR filenames (all subdirs must agree).
#
# Output layout:
#     <output-dir>/ai/rapids/cudf/<version>/cudf-<version>-<classifier>.jar
#     <output-dir>/ai/rapids/cudf/<version>/cudf-<version>-sources.jar
#     <output-dir>/ai/rapids/cudf/<version>/cudf-<version>-javadoc.jar
#     <output-dir>/ai/rapids/cudf/<version>/cudf-<version>.pom

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/argparse.sh"

GROUP_PATH="ai/rapids"
ARTIFACT_ID="cudf"

JARS_DIR=""
OUTPUT_DIR=""

print_help() {
  cat << EOF

Usage: assemble_maven_repo.sh --jars-dir <path> --output-dir <path>

Gathers per-classifier cuDF Java JARs into a single Maven-repository-layout tree.

REQUIRED:
    -j, --jars-dir       Parent directory containing one subdirectory per
                         classifier. Each subdir must hold:
                             cudf-<version>-<classifier>.jar
                             cudf-<version>-sources.jar
                             cudf-<version>-javadoc.jar
                             cudf-<version>.pom
                         Subdir name is the classifier.
    -o, --output-dir     Directory to receive the combined Maven-repository layout.

OPTIONS:
    -h, --help           Show this help message.

EXAMPLE:
    assemble_maven_repo.sh --jars-dir /tmp/jars --output-dir /tmp/maven-repo
    # given /tmp/jars/{cuda12,cuda13}/ inputs, produces:
    #   /tmp/maven-repo/ai/rapids/cudf/<version>/cudf-<version>-cuda12.jar
    #   /tmp/maven-repo/ai/rapids/cudf/<version>/cudf-<version>-cuda13.jar
    #   /tmp/maven-repo/ai/rapids/cudf/<version>/cudf-<version>-sources.jar
    #   /tmp/maven-repo/ai/rapids/cudf/<version>/cudf-<version>-javadoc.jar
    #   /tmp/maven-repo/ai/rapids/cudf/<version>/cudf-<version>.pom

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        print_help
        exit 0
        ;;
      -j|--jars-dir)
        require_value "$1" "$2"
        JARS_DIR=$2
        shift 2
        ;;
      -o|--output-dir)
        require_value "$1" "$2"
        OUTPUT_DIR=$2
        shift 2
        ;;
      *)
        echo "Error: Unknown argument $1"
        print_help
        exit 1
        ;;
    esac
  done
}

parse_args "$@"

require_arg --jars-dir   "${JARS_DIR}"
require_arg --output-dir "${OUTPUT_DIR}"

if [[ ! -d ${JARS_DIR} ]]; then
  echo "Error: --jars-dir '${JARS_DIR}' does not exist."
  exit 1
fi

if [[ -e ${OUTPUT_DIR} && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "Error: --output-dir '${OUTPUT_DIR}' must be empty or nonexistent" >&2
  exit 1
fi

ASSEMBLE_FINISHED=0
cleanup_partial_output() {
  if [[ ${ASSEMBLE_FINISHED} -eq 0 && -e ${OUTPUT_DIR} ]]; then
    echo "Assembly did not complete; removing partial output at ${OUTPUT_DIR}" >&2
    rm -rf "${OUTPUT_DIR}"
  fi
}
trap cleanup_partial_output EXIT

echo "Assembling Maven repository layout"
echo "  jars dir:   ${JARS_DIR}"
echo "  output dir: ${OUTPUT_DIR}"

# Walk every classifier subdirectory. Each subdir must contain exactly one
# cudf-*-<classifier>.jar. The version is derived from the filename and must
# match across all subdirs.
FIRST_VERSION=""
CLASSIFIERS_SEEN=""

for subdir in "${JARS_DIR}"/*/; do
  classifier=$(basename "${subdir}")

  jar=""
  for candidate in "${subdir}"cudf-*-"${classifier}".jar; do
    if [[ -f "${candidate}" ]]; then
      if [[ -n "${jar}" ]]; then
        echo "Error: multiple JARs in ${subdir} match cudf-*-${classifier}.jar" >&2
        exit 1
      fi
      jar=${candidate}
    fi
  done

  if [[ -z "${jar}" ]]; then
    echo "Error: no cudf-*-${classifier}.jar found in ${subdir}" >&2
    exit 1
  fi

  # Filename is cudf-<version>-<classifier>.jar. Peel prefix and suffix.
  base=$(basename "${jar}" .jar)
  version=$(echo "${base}" | sed -e 's/^cudf-//' -e "s/-${classifier}$//")

  if [[ -z "${FIRST_VERSION}" ]]; then
    FIRST_VERSION=${version}
  elif [[ "${version}" != "${FIRST_VERSION}" ]]; then
    echo "Error: inconsistent versions across subdirs: ${FIRST_VERSION} vs ${version} (${subdir})" >&2
    exit 1
  fi

  DEST_DIR="${OUTPUT_DIR}/${GROUP_PATH}/${ARTIFACT_ID}/${version}"
  mkdir -p "${DEST_DIR}"
  cp -f "${jar}" "${DEST_DIR}/"
  echo "  + $(basename "${jar}")"
  CLASSIFIERS_SEEN="${CLASSIFIERS_SEEN} ${classifier}"
done

if [[ -z "${FIRST_VERSION}" ]]; then
  echo "Error: no classifier subdirs found under ${JARS_DIR}" >&2
  exit 1
fi

# Sources and javadoc jars are classifier-independent (pure Java, no arch or
# cuda variation). Every classifier subdir produces byte-equivalent copies;
# pick the lexicographically first subdir's copy as canonical. Fail fast if
# any subdir is missing either file - that indicates -Prelease or
# -Pjavadoc-jdk17 did not activate for that classifier's build.
DEST_DIR="${OUTPUT_DIR}/${GROUP_PATH}/${ARTIFACT_ID}/${FIRST_VERSION}"
FIRST_CLASSIFIER_SUBDIR=""
for subdir in "${JARS_DIR}"/*/; do
  if [[ -z ${FIRST_CLASSIFIER_SUBDIR} ]]; then
    FIRST_CLASSIFIER_SUBDIR=${subdir}
  fi
  for required in "cudf-${FIRST_VERSION}-sources.jar" \
                  "cudf-${FIRST_VERSION}-javadoc.jar"; do
    if [[ ! -f ${subdir}${required} ]]; then
      echo "Error: ${required} missing from ${subdir} (expected in every classifier subdir)" >&2
      exit 1
    fi
  done
done

for shared in "cudf-${FIRST_VERSION}-sources.jar" \
              "cudf-${FIRST_VERSION}-javadoc.jar"; do
  cp -f "${FIRST_CLASSIFIER_SUBDIR}${shared}" "${DEST_DIR}/"
  echo "  + ${shared}"
done

# POM is identical across subdirs; copy the first one found.
POM_SRC=""
for subdir in "${JARS_DIR}"/*/; do
  candidate=${subdir}cudf-${FIRST_VERSION}.pom
  if [[ -f "${candidate}" ]]; then
    POM_SRC=${candidate}
    break
  fi
done

if [[ -z "${POM_SRC}" ]]; then
  echo "Error: no cudf-${FIRST_VERSION}.pom found under ${JARS_DIR}" >&2
  exit 1
fi

DEST_DIR="${OUTPUT_DIR}/${GROUP_PATH}/${ARTIFACT_ID}/${FIRST_VERSION}"
cp -f "${POM_SRC}" "${DEST_DIR}/cudf-${FIRST_VERSION}.pom"
echo "  + cudf-${FIRST_VERSION}.pom"

echo "Maven repository assembled successfully at ${OUTPUT_DIR}"
echo "Classifiers present:${CLASSIFIERS_SEEN}"

ASSEMBLE_FINISHED=1
