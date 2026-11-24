#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Times partitioned inner join retrieval with and without a JoinMatchContext.
# Everything runs in a RAPIDS ci-wheel container, so the host only needs Docker and a GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUDF_VERSION="$(awk -F. '{printf "%s.%s.%d", $1, $2, $3}' "$REPO_ROOT/VERSION")-SNAPSHOT"
CUDA_VERSION="${RAPIDS_CUDA_VERSION:-12.9.2}"
MAIN_CLASS="ai.rapids.cudf.examples.joincache.JoinMatchContextExample"

SCALE=small
ITERATIONS=5
BUILD_ROWS=""
PROBE_ROWS=""
PARTITIONS=""
CUDA_CLASSIFIER="cuda${CUDA_VERSION%%.*}"
JAR="${MAVEN_REPO:-$HOME/.m2/repository}/ai/rapids/cudf/$CUDF_VERSION/cudf-$CUDF_VERSION-$CUDA_CLASSIFIER.jar"

err() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Joins a large fact table against a smaller dimension table and produces the
results one partition at a time, timing two ways of doing it.

  uncached  Counts the total number of matches first, then joins each
            partition. Every fact row is looked up in the hash table twice.
  cached    Builds a JoinMatchContext, which looks up every fact row once and
            keeps the matches. Each partition then reuses those matches.

Dimension keys are unique. Fact keys are skewed so that a few keys are very
common, like popular products in a sales table.

Options:
  --scale small|medium|large|xlarge   Preset size (default: small)
                                        small    1M dimension,   4M fact,  8 partitions
                                        medium  10M dimension,  40M fact, 16 partitions
                                        large   50M dimension, 200M fact, 32 partitions
                                        xlarge 100M dimension, 400M fact, 64 partitions
  --build-rows N       Dimension (build) rows, overrides the preset
  --probe-rows N       Fact (probe) rows, overrides the preset
  --partitions N       Number of probe partitions, overrides the preset
  --iterations N       Timed iterations per path (default: 5)
  --jar PATH           cudf JAR to use
                       (default: \$MAVEN_REPO or ~/.m2/repository, cudf $CUDF_VERSION $CUDA_CLASSIFIER)
  -h, --help           Show this help
USAGE
}

while (( $# > 0 )); do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --build-rows) BUILD_ROWS="$2"; shift 2 ;;
    --probe-rows) PROBE_ROWS="$2"; shift 2 ;;
    --partitions) PARTITIONS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --jar) JAR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) err "unknown option: $1 (try --help)" ;;
  esac
done

case "$SCALE" in
  small) preset=(1000000 4000000 8) ;;
  medium) preset=(10000000 40000000 16) ;;
  large) preset=(50000000 200000000 32) ;;
  xlarge) preset=(100000000 400000000 64) ;;
  *) err "unknown scale: $SCALE (try --help)" ;;
esac
BUILD_ROWS="${BUILD_ROWS:-${preset[0]}}"
PROBE_ROWS="${PROBE_ROWS:-${preset[1]}}"
PARTITIONS="${PARTITIONS:-${preset[2]}}"

if [[ ! -f "$JAR" ]]; then
  err "cudf JAR not found at $JAR. Build and install cudf Java first, or pass --jar."
fi
JAR="$(cd "$(dirname "$JAR")" && pwd)/$(basename "$JAR")"

# shellcheck disable=SC1091
. "$REPO_ROOT/java/ci/ci_wheel_image.sh"
IMAGE="$(cudf_java_ci_wheel_image "$CUDA_VERSION")"

docker run --rm --gpus all \
  --volume "$SCRIPT_DIR:/example" \
  --volume "$JAR:/cudf.jar:ro" \
  --workdir /example \
  --env HOST_IDS="$(id -u):$(id -g)" \
  "$IMAGE" bash -c '
    set -euo pipefail
    trap "chown -R \$HOST_IDS /example/target 2>/dev/null || true" EXIT
    dnf install -y -q maven java-1.8.0-openjdk-devel >/dev/null 2>&1
    export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
    mvn -q -B -Dmaven.repo.local=/example/target/m2 -Dcudf.jar.path=/cudf.jar package
    java -cp "target/classes:target/dependency/*:/cudf.jar" "$0" "$@"
  ' "$MAIN_CLASS" "$BUILD_ROWS" "$PROBE_ROWS" "$PARTITIONS" "$ITERATIONS"
