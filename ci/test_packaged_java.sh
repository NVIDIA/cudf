#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the Java tests against an already-packaged classifier JAR.
#
# Activates -Ppackaged-jar-tests so the surefire classpath uses the JAR at
# JAVA_JAR instead of a locally compiled target/classes tree. Caller places
# the JAR (e.g. via actions/download-artifact) before invoking this script.
#
# Inputs (environment variables):
#   JAVA_JAR                       Absolute path to the classifier JAR (required).
#   LIBCUDF_LARGE_STRINGS_ENABLED  Optional; defaults to 0 (same as ci/test_java.sh).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z ${JAVA_JAR:-} || ! -f ${JAVA_JAR} ]]; then
  echo "Error: JAVA_JAR must point to an existing classifier JAR" >&2
  exit 1
fi

if ! command -v mvn >/dev/null 2>&1 || ! command -v java >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/java/ci/setup_java_env.sh"
fi

# Match the existing conda Java test entrypoint in ci/test_java.sh.
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
