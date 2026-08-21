#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export GTEST_CUDF_RMM_MODE=cuda
export GTEST_CUDF_MEMORY_PEAK=1
export GTEST_BRIEF=1

# Collect all test results
results=()
for gt in gtests/*_TEST ; do
  test_name=$(basename "${gt}")
  echo -n "$test_name"
  # dependent on the string output from testing_main.hpp
  bytes=$(${gt} 2>/dev/null | grep Peak | cut -d' ' -f4)
  echo ",${bytes}"
  results+=("$test_name,$bytes")
done

unset GTEST_BRIEF
unset GTEST_CUDF_RMM_MODE
unset GTEST_CUDF_MEMORY_PEAK

# Output tests using more than 1GB
echo ""
echo "Tests using more than 1GB of memory:"
threshold=$((1024 * 1024 * 1024))
for result in "${results[@]}" ; do
  test_name=$(echo "$result" | cut -d',' -f1)
  bytes=$(echo "$result" | cut -d',' -f2)
  if [[ -n "$bytes" && "$bytes" -gt "$threshold" ]] ; then
    # Convert bytes to GB with 2 decimal places
    gb=$(awk "BEGIN {printf \"%.2f\", $bytes / (1024 * 1024 * 1024)}")
    echo "$test_name: ${gb} GB"
  fi
done
