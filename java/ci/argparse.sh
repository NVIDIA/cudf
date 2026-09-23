#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared argparse helpers for the java/ci/ host orchestrator scripts.
# Meant to be sourced, not executed:
#   . "${SCRIPT_DIR}/argparse.sh"

# require_value <flag_name> <value>
# Check: exit 1 when <value> is empty (i.e. the flag was passed
# without its argument, or was the last token on the command line).
require_value() {
  local flag=$1
  local value=$2
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} requires a value" >&2
    exit 1
  fi
}

# require_arg <flag_name> <value>
# Check: assert that a required flag was actually supplied by the
# caller. Prints the script's print_help (if defined) then exits 1 on failure.
# Preserves the existing behavior of showing help after a "required flag missing"
# error.
require_arg() {
  local flag=$1
  local value=$2
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} is required." >&2
    if declare -F print_help > /dev/null; then
      print_help
    fi
    exit 1
  fi
}
