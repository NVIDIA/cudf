#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wait_for_builds() {
  local status=0
  local pid
  # A bare wait hides child failures; wait for every child before publishing artifacts.
  for pid in "$@"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}
