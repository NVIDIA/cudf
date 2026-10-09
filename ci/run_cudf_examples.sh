#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
SECONDS=0

# Keep serial execution available for comparisons and constrained runners.
EXAMPLES_PARALLEL_LEVEL=${EXAMPLES_PARALLEL_LEVEL:-2}
case "${EXAMPLES_PARALLEL_LEVEL}" in
    1|2) ;;
    *) echo "EXAMPLES_PARALLEL_LEVEL must be 1 or 2" >&2; exit 1 ;;
esac

cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf" || exit 1

# TODO: Temporary workaround for compute-sanitizer bug 5824899 that occurs only on the examples
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]') || exit 1
USE_COMPUTE_SANITIZER=true
if [[ "${GPU_COMPUTE_CAP}" == "12.0" ]]; then
    USE_COMPUTE_SANITIZER=false
    echo "Disabling compute-sanitizer for examples for sm_120 device"
fi

log_dir=$(mktemp -d) || exit 1
trap 'rm -r "${log_dir}"' EXIT
EXITCODE=0

run_example() {
    local cmd=("$@")
    if ${USE_COMPUTE_SANITIZER}; then
        cmd=(compute-sanitizer --tool memcheck --port "${sanitizer_port}" --error-exitcode 1 "${cmd[@]}")
    fi
    echo "Running ${group} example: ${cmd[*]}"
    "${cmd[@]}" || {
        echo "Example failed: ${cmd[*]}" >&2
        group_status=1
    }
}

run_group() (
    # The parent owns these logs; a worker must not remove other workers' output.
    trap - EXIT
    local group=$1 sanitizer_port=$2 group_status=0
    SECONDS=0
    if ! cd "${group}"; then
        echo "1 0" > "${log_dir}/${group}.result"
        exit 1
    fi
    case "${group}" in
        basic)
            run_example ./basic_example
            ;;
        hybrid_scan_io)
            run_example ./hybrid_scan_io example.parquet string_col 0000001 PINNED_BUFFER
            run_example ./hybrid_scan_pipeline example.parquet 2 HOST_BUFFER ROW_GROUPS 2
            run_example ./hybrid_scan_pipeline example.parquet 2 FILEPATH BYTE_RANGES 2
            run_example ./hybrid_scan_multifile_single_step example.parquet 10 2 YES DEVICE_BUFFER 2
            run_example ./hybrid_scan_multifile_single_step example.parquet 10 2 NO FILEPATH 1
            run_example ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 PINNED_BUFFER 2
            run_example ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 HOST_BUFFER 1
            ;;
        nested_types)
            run_example ./deduplication
            ;;
        parquet_io)
            run_example ./parquet_io example.parquet
            run_example ./parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE
            run_example ./parquet_io_multithreaded example.parquet
            run_example ./parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2
            ;;
        parquet_inspect)
            run_example ./parquet_inspect example.parquet
            ;;
        strings)
            run_example ./custom_optimized names.csv
            run_example ./custom_prealloc names.csv
            run_example ./custom_with_malloc names.csv
            ;;
        string_transformers)
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
            ;;
    esac
    echo "${group_status} ${SECONDS}" > "${log_dir}/${group}.result"
    exit "${group_status}"
)

# Start the longest parallel groups first to maximize their overlap.
groups=(basic nested_types hybrid_scan_io string_transformers parquet_io strings parquet_inspect)
pids=()
for index in "${!groups[@]}"; do
    group=${groups[index]}
    echo "Starting example group: ${group}"
    # Concurrent sanitizer sessions need disjoint port ranges on the same runner.
    run_group "${group}" "$((49152 + index * 100))" > "${log_dir}/${group}.log" 2>&1 &
    pid=$!
    # These examples intentionally reserve half the available GPU memory.
    if (( index < 2 )); then
        wait "${pid}" || EXITCODE=1
    else
        pids+=("${pid}")
        if (( ${#pids[@]} >= EXAMPLES_PARALLEL_LEVEL )); then
            # Reap by PID below: wait -n can return 127 if both workers already finished.
            wait -n "${pids[@]}" 2>/dev/null || true
            running_pids=$(jobs -pr)
            pending_pids=()
            for pid in "${pids[@]}"; do
                if [[ " ${running_pids//$'\n'/ } " == *" ${pid} "* ]]; then
                    pending_pids+=("${pid}")
                else
                    wait "${pid}" || EXITCODE=1
                fi
            done
            pids=("${pending_pids[@]}")
        fi
    fi
done
for pid in "${pids[@]}"; do
    wait "${pid}" || EXITCODE=1
done

for group in "${groups[@]}"; do
    echo "::group::Examples: ${group}"
    cat "${log_dir}/${group}.log" || EXITCODE=1
    if read -r group_status elapsed < "${log_dir}/${group}.result"; then
        echo "${group}: status=${group_status}, elapsed=${elapsed}s"
    else
        echo "${group}: missing completion record" >&2
        EXITCODE=1
    fi
    echo "::endgroup::"
done
echo "Example stage: elapsed=${SECONDS}s, status=${EXITCODE}"
exit "${EXITCODE}"
