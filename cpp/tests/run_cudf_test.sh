#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Installed CTest commands use paths relative to their launch directory.
test_executable=$(realpath -e "$1")
test_stream_checker="$2"
shift 2
if [[ "${test_stream_checker}" != - ]]; then
    test_stream_checker=$(realpath -e "${test_stream_checker}")
fi

case "${GTEST_OUTPUT:-}" in
    xml|json)
        export GTEST_OUTPUT="${GTEST_OUTPUT}:${PWD}/test_detail.${GTEST_OUTPUT}"
        ;;
    xml:*|json:*)
        output_file="${GTEST_OUTPUT#*:}"
        output_file="${output_file:-test_detail.${GTEST_OUTPUT%%:*}}"
        if [[ "${output_file}" != /* ]]; then
            export GTEST_OUTPUT="${GTEST_OUTPUT%%:*}:${PWD}/${output_file}"
        fi
        ;;
esac

# NVRTC's automatic PCH filenames collide when processes share a working directory.
test_work_dir=$(realpath "$(mktemp -d -t cudf-test.XXXXXXXXXX)")
trap 'rm -rf -- "${test_work_dir}"' EXIT
cd "${test_work_dir}"

if [[ "${test_stream_checker}" == - ]]; then
    "${test_executable}" "$@"
else
    LD_PRELOAD="${test_stream_checker}" "${test_executable}" "$@"
fi
