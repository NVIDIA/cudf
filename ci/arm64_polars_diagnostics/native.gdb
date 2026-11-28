# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set pagination off
set confirm off
handle SIGPIPE nostop noprint pass
handle SIGSEGV stop print pass
catch signal SIGSEGV
commands
  silent
  shell date -u +%Y-%m-%dT%H:%M:%S.%NZ
  printf "\nCUDF_NATIVE_SIGSEGV\n"
  p $_siginfo
  info symbol $pc
  x/i $pc
  info registers
  thread apply all bt
  info sharedlibrary
  info proc mappings
  continue
end
printf "CUDF_GDB_ATTACHED\n"
continue
