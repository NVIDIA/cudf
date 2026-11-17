#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the cuDF Java native library deployment-mode examples.
# Builds the example project once (if needed) and then runs two scenarios
# back-to-back so their startup behavior can be compared:
#   1. Default JAR extraction
#   2. Pre-unpacked library directory (lib-native-dir)
# Every scenario passes -Dai.rapids.cudf.lib-log-load-timing=true so the
# per-library extract/load timings and the aggregate summary always appear
# on stderr.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fixed location for pre-extracted .so files used by scenario 2. A fixed
# path (not mktemp) is required so that -k can keep the directory around
# for reuse on the next invocation.
UNPACKED_DIR="$SCRIPT_DIR/unpacked-libs"

# Maven local repo. Honor the user's MAVEN_REPO env var if set, otherwise
# fall back to the standard ~/.m2/repository location.
M2_REPO="${MAVEN_REPO:-$HOME/.m2/repository}"
CUDF_GROUP_DIR="$M2_REPO/ai/rapids/cudf"

MAIN_CLASS="ai.rapids.cudf.examples.deployment.SimpleWorkload"
EXAMPLE_VERSION="$(awk -F. '{printf "%s.%s.%d", $1, $2, $3}' "$SCRIPT_DIR/../../../VERSION")-SNAPSHOT"
EXAMPLE_JAR="$SCRIPT_DIR/target/cudf-deployment-modes-example-${EXAMPLE_VERSION}.jar"

KEEP=0

err() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Run the cuDF Java native library deployment-mode examples and print timings.
Two scenarios run in order:

  Scenario 1: Default JAR extraction
  Scenario 2: Pre-unpacked library directory (lib-native-dir)

Options:
  -k, --keep-unpacked-libs    After scenario 2, keep the unpacked-libs/
                              directory so the next run can reuse it
                              instead of re-extracting the JAR.
  -h, --help                  Show this help and exit.

Both scenarios pass -Dai.rapids.cudf.lib-log-load-timing=true so the
NativeDepsLoader timing summary is visible. Wall time per scenario is
measured by the script and printed below the JVM output.
USAGE
}

# Locate the non-classifier-test cudf JAR matching EXAMPLE_VERSION.
# Accept any cuda* classifier (e.g. cudf-<version>-SNAPSHOT-cuda12.jar) but
# skip *-tests.jar / *-sources.jar / *-javadoc.jar.
locate_cudf_jar() {
  if [[ ! -d "$CUDF_GROUP_DIR" ]]; then
    err "cudf Maven artifacts not found at $CUDF_GROUP_DIR. Make sure cudf is built first."
  fi
  local version_dir="$CUDF_GROUP_DIR/$EXAMPLE_VERSION"
  if [[ ! -d "$version_dir" ]]; then
    err "cudf Maven artifacts for version $EXAMPLE_VERSION not found at $version_dir." \
        "Make sure the expected version of cudf is built first."
  fi
  local jar=""
  for candidate in "$version_dir"/cudf-"$EXAMPLE_VERSION"*.jar; do
    if [[ ! -f "$candidate" ]]; then
      continue
    fi
    if [[ "$candidate" =~ -(tests|sources|javadoc)\.jar$ ]]; then
      continue
    fi
    jar=$candidate
    break
  done
  if [[ -z "$jar" ]]; then
    err "no cudf-$EXAMPLE_VERSION*.jar found under $version_dir. Make sure cudf is built first."
  fi
  printf '%s\n' "$jar"
}

build_example() {
  if [[ -f "$EXAMPLE_JAR" && -d "$SCRIPT_DIR/target/dependency" ]]; then
    return
  fi
  printf 'Building deployment-modes example (mvn package)...\n' >&2
  (cd "$SCRIPT_DIR" && mvn -q package -Dsnapshot_version="$EXAMPLE_VERSION")
}

build_classpath() {
  printf '%s:%s/*\n' "$EXAMPLE_JAR" "$SCRIPT_DIR/target/dependency"
}

prepare_unpacked_libs() {
  # In a real deployment, the .so files would already exist on the filesystem
  # or in the container image. This step simulates that by unpacking them from
  # the local Maven JAR.
  local cudf_jar="$1"
  if [[ -d "$UNPACKED_DIR" && -n "$(ls -A "$UNPACKED_DIR" 2>/dev/null)" ]]; then
    printf 'Reusing existing unpacked-libs at %s\n' "$UNPACKED_DIR"
    return
  fi
  local jar_arch jar_os jar_prefix
  jar_arch=$(java -XshowSettings:property -version 2>&1 \
               | awk '/os\.arch/{print $NF}')
  jar_os=$(java -XshowSettings:property -version 2>&1 \
             | awk '/os\.name/{print $NF}')
  if [[ -z "$jar_arch" || -z "$jar_os" ]]; then
    err "Could not determine os.arch / os.name from the JVM. Is 'java' on PATH?"
  fi
  jar_prefix="${jar_arch}/${jar_os}"
  mkdir -p "$UNPACKED_DIR"
  printf 'Unpacking native libs from %s into %s (JAR prefix: %s)\n' \
    "$cudf_jar" "$UNPACKED_DIR" "$jar_prefix"
  (cd "$UNPACKED_DIR" && unzip -j -o "$cudf_jar" "${jar_prefix}/*.so" >/dev/null)
}

cleanup_unpacked_libs() {
  if (( KEEP == 0 )); then
    rm -rf "$UNPACKED_DIR"
  fi
}

SECTION_DASH='-----------------------------------------------------------------------'

print_begin() {
  printf '\n=========================== Scenario %d Begin ===========================\n\n' "$1"
}
print_end() {
  printf '\n===========================  Scenario %d End  ===========================\n\n' "$1"
}
print_div() {
  printf '\n%s\n\n' "$SECTION_DASH"
}

run_scenario_default() {
  local cp
  cp=$(build_classpath)

  print_begin 1
  cat <<'DESC'
Default JAR extraction. NativeDepsLoader extracts each bundled .so from
the cudf JAR into a fresh JVM temp file using the read buffer. This is
the cost every cuDF Java application incurs today on
first use.
DESC

  print_div
  printf 'Using default per-process JAR extraction.\n'
  print_div

  local t0_ns elapsed_ms rc=0
  t0_ns=$(date +%s%N)
  java -cp "$cp" \
       -Dai.rapids.cudf.lib-log-load-timing=true \
       "$MAIN_CLASS" || rc=$?
  elapsed_ms=$(( ( $(date +%s%N) - t0_ns ) / 1000000 ))

  printf '\nTotal Wall Time: %d ms (java exit=%d)\n' "$elapsed_ms" "$rc"
  print_end 1
  return "$rc"
}

run_scenario_prepacked() {
  local cp
  cp=$(build_classpath)

  print_begin 2
  cat <<'DESC'
Pre-unpacked library directory. Locates the cudf-*.jar in the local
Maven repository, unpacks the bundled native libraries (.so files) into
a fixed unpacked-libs/ directory, and launches the workload with
-Dai.rapids.cudf.lib-native-dir=<dir>. This skips the per-process JAR
extraction step entirely and simulates a container image that pre-bakes
the native libs at image-build time.
DESC
  print_div
  local cudf_jar
  cudf_jar=$(locate_cudf_jar)
  printf 'Using cudf JAR: %s\n' "$cudf_jar"
  prepare_unpacked_libs "$cudf_jar"
  printf 'Using lib-native-dir=%s\n' "$UNPACKED_DIR"
  print_div

  local t0_ns elapsed_ms rc=0
  t0_ns=$(date +%s%N)
  java -cp "$cp" \
       -Dai.rapids.cudf.lib-log-load-timing=true \
       "-Dai.rapids.cudf.lib-native-dir=$UNPACKED_DIR" \
       "$MAIN_CLASS" || rc=$?
  elapsed_ms=$(( ( $(date +%s%N) - t0_ns ) / 1000000 ))

  # Cleanup runs even if java failed, so that an error here without -k
  # does not leave a stale unpacked-libs/ behind.
  cleanup_unpacked_libs

  printf '\nTotal Wall Time: %d ms (java exit=%d)\n' "$elapsed_ms" "$rc"
  print_end 2
  return "$rc"
}

while (( $# > 0 )); do
  case "$1" in
    -k|--keep-unpacked-libs) KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "unknown option: $1 (try --help)" ;;
  esac
done

build_example

overall_rc=0
run_scenario_default || overall_rc=$?
run_scenario_prepacked || overall_rc=$?

exit "$overall_rc"
