#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

: "${CUDF_IWYU_LOG_FILE:?CUDF_IWYU_LOG_FILE must be set}"
: "${CUDF_IWYU_REAL_EXE:?CUDF_IWYU_REAL_EXE must be set}"
: "${CUDF_IWYU_DIAGNOSTICS_DIR:?CUDF_IWYU_DIAGNOSTICS_DIR must be set}"

wrapper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
iwyu_args=("$@")

log_invocation() {
  local event="$1"
  local status="${2:-}"

  {
    printf '%s event=%s pid=%s ppid=%s status=%s command=' "$(date -u +%FT%TZ)" "${event}" "$$" "${PPID}" "${status}"
    printf '%q ' "${CUDF_IWYU_REAL_EXE}" "${iwyu_args[@]}"
    printf '\n'
  } >> "${CUDF_IWYU_LOG_FILE}"
}

log_invocation start
time_log="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.time"
time_exe="$(dirname "${CUDF_IWYU_REAL_EXE}")/time"
if [[ ! -x "${time_exe}" ]]; then
  printf 'GNU time is unavailable at %s\n' "${time_exe}" >&2
  exit 1
fi
trace_iwyu=0
gdb_iwyu=0
for arg in "${iwyu_args[@]}"; do
  if [[ "${arg}" =~ ${CUDF_IWYU_STRACE_SOURCE_REGEX:-^$} ]]; then
    trace_iwyu=1
  fi
  if [[ "${arg}" =~ ${CUDF_IWYU_GDB_SOURCE_REGEX:-^$} ]]; then
    gdb_iwyu=1
  fi
done
profile_iwyu=0
if (( trace_iwyu )) && [[ "${CUDF_IWYU_PERF_AVAILABLE:-0}" == 1 ]]; then
  profile_iwyu=1
fi

command=("${CUDF_IWYU_REAL_EXE}" "$@")
if (( trace_iwyu || profile_iwyu )); then
  command=(timeout --signal=TERM --kill-after=30s \
    "${CUDF_IWYU_TRACE_TIMEOUT_SECONDS:?CUDF_IWYU_TRACE_TIMEOUT_SECONDS must be set}" \
    "${command[@]}")
fi
if (( trace_iwyu && !gdb_iwyu )); then
  trace_prefix="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.strace"
  command=(strace -ff -ttt -T -s 256 -o "${trace_prefix}" \
    -e 'trace=%file,%process,%network' "${command[@]}")
fi
if (( profile_iwyu )); then
  perf_prefix="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.perf"
  command=(perf record --freq 99 --call-graph 'dwarf,8192' --output "${perf_prefix}.data" -- "${command[@]}")
fi
if (( gdb_iwyu )); then
  gdb_log="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.gdb.log"
  : "${CUDF_IWYU_GDB_TIMEOUT_SECONDS:?CUDF_IWYU_GDB_TIMEOUT_SECONDS must be set}"
  command=(gdb --batch --quiet --ex "set logging file ${gdb_log}" --ex 'set logging enabled on' \
    --command "${wrapper_dir}/iwyu_capture.gdb" \
    --args "${CUDF_IWYU_REAL_EXE}" "$@")
fi
"${time_exe}" -v -o "${time_log}" "${command[@]}"
status=$?
if (( profile_iwyu )); then
  perf report --stdio --input "${perf_prefix}.data" > "${perf_prefix}.report" 2>&1 || true
fi
log_invocation end "${status}"
exit "${status}"
