/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file pair.cuh
 * @brief String-pair layout shared by cuDF host checks and generated CUDA kernels.
 *
 * Defines the character pointer and byte length used to describe captured
 * strings before cuDF column materialization.
 */
#pragma once

#include <libregex_ir/executor.cuh>

namespace cudf::experimental::detail::regex_jit::device {
// The host integration checks this JIT ABI against string_index_pair.
struct string_pair {
  char const* first;
  regex_ir::device::i32 second;
};
}  // namespace cudf::experimental::detail::regex_jit::device
