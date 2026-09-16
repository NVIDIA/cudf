#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf" || exit

# TODO: Temporary workaround for compute-sanitizer bug 5824899 that occurs only on the examples
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')
USE_COMPUTE_SANITIZER=true
if [[ "${GPU_COMPUTE_CAP}" == "12.0" ]]; then
    USE_COMPUTE_SANITIZER=false
    echo "Disabling compute-sanitizer for examples for sm_120 device"
fi

run_example() {
    local parent
    parent=$(basename "${PWD}")
    local cmd=("$@")
    if ${USE_COMPUTE_SANITIZER}; then
        cmd=(compute-sanitizer --tool memcheck "${cmd[@]}")
    fi
    echo "Running ${parent} example: ${cmd[*]}"
    "${cmd[@]}"
}

pushd basic || exit
run_example ./basic_example
popd || exit

pushd hybrid_scan_io || exit
run_example ./hybrid_scan_io example.parquet string_col 0000001 PINNED_BUFFER
run_example ./hybrid_scan_pipeline example.parquet 2 HOST_BUFFER ROW_GROUPS 2
run_example ./hybrid_scan_pipeline example.parquet 2 FILEPATH BYTE_RANGES 2
run_example ./hybrid_scan_multifile_single_step example.parquet 10 2 YES DEVICE_BUFFER 2
run_example ./hybrid_scan_multifile_single_step example.parquet 10 2 NO FILEPATH 1
run_example ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 PINNED_BUFFER 2
run_example ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 HOST_BUFFER 1
popd || exit

pushd nested_types || exit
run_example ./deduplication
popd || exit

pushd parquet_io || exit
run_example ./parquet_io example.parquet
run_example ./parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE
run_example ./parquet_io_multithreaded example.parquet
run_example ./parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2
popd || exit

pushd parquet_inspect || exit
run_example ./parquet_inspect example.parquet
popd || exit

pushd strings || exit
run_example ./custom_optimized names.csv
run_example ./custom_prealloc names.csv
run_example ./custom_with_malloc names.csv
popd || exit

pushd string_transformers || exit
run_example ./compute_checksum_jit info.csv output.csv
run_example ./extract_email_jit info.csv output.csv
run_example ./extract_email_precompiled info.csv output.csv
run_example ./format_phone_jit info.csv output.csv
run_example ./format_phone_precompiled info.csv output.csv
run_example ./localize_phone_jit info.csv output.csv
run_example ./localize_phone_precompiled info.csv output.csv
run_example ./url_log_transforms logs.csv output.csv regex 100
run_example ./url_log_transforms logs.csv output.csv precompiled 100
run_example ./url_log_transforms logs.csv output.csv cuda-jit 100 --cold
run_example ./url_log_transforms logs.csv output.csv lto-jit 100 --cold
popd || exit

exit ${EXITCODE}
