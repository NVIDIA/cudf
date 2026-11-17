#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the Java tests against an already-packaged classifier JAR.
#
# Activates -Ppackaged-jar-tests so the surefire classpath uses the JAR at
# JAVA_JAR instead of a locally compiled target/classes tree.
#
# When JAVA_JAR is unset, download the java-build artifact via
# rapids-download-from-github (pr.yaml for PRs, build.yaml otherwise).
#
# Inputs (environment variables):
#   JAVA_JAR                       Optional. Absolute path to the classifier JAR.
#                                  If unset, the JAR is downloaded from the
#                                  matching java-build artifact.
#   LIBCUDF_LARGE_STRINGS_ENABLED  Optional; defaults to 0.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
. "${REPO_ROOT}/java/ci/java_classifier.sh"

if [[ -z ${JAVA_JAR:-} ]]; then
  if [[ -z ${RAPIDS_CUDA_VERSION:-} ]]; then
    echo "Error: RAPIDS_CUDA_VERSION must be set when JAVA_JAR is unset" >&2
    exit 1
  fi
  cuda_major="${RAPIDS_CUDA_VERSION%%.*}"

  # matrix.ARCH values are amd64/arm64; $(arch) / uname -m return x86_64/aarch64.
  case "$(uname -m)" in
    x86_64)  java_arch=amd64 ;;
    aarch64|arm64) java_arch=arm64 ;;
    *)
      echo "Error: unsupported host arch '$(uname -m)'" >&2
      exit 1
      ;;
  esac

  rapids-logger "Downloading cudf_java_${java_arch}_cu${cuda_major}"
  JAVA_PKG="$(rapids-download-from-github "cudf_java_${java_arch}_cu${cuda_major}")"
  JAVA_JAR="$(cudf_java_resolve_artifact_jar "${JAVA_PKG}")"
  export JAVA_JAR
fi

if [[ ! -f ${JAVA_JAR} ]]; then
  echo "Error: JAVA_JAR='${JAVA_JAR}' does not point to an existing file" >&2
  exit 1
fi

if ! command -v mvn >/dev/null 2>&1 || ! command -v java >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/java/ci/setup_java_env.sh"
fi

# Disable large strings for the Java test suite.
export LIBCUDF_LARGE_STRINGS_ENABLED="${LIBCUDF_LARGE_STRINGS_ENABLED:-0}"

PRODUCT_JAR="$(cd "$(dirname "${JAVA_JAR}")" && pwd)/$(basename "${JAVA_JAR}")"
rapids-logger "Product JAR: ${PRODUCT_JAR}"

rapids-logger "Check GPU usage"
nvidia-smi

rm -rf "${REPO_ROOT}/java/target"

pushd "${REPO_ROOT}/java" >/dev/null
set +e
timeout 30m mvn -B test -Ppackaged-jar-tests \
  "-Dcudf.jar.path=${PRODUCT_JAR}" \
  "-DCUDF_JNI_ENABLE_PROFILING=OFF"
EXITCODE=$?
set -e
popd >/dev/null
rapids-logger "Java tests exit=${EXITCODE}"
exit "${EXITCODE}"
