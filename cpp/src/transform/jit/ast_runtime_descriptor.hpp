/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

namespace cudf::jit::ast_runtime {

// Keep this internal ABI independent of column types and schema width so one row-loop fragment
// can serve every supported fixed-width AST expression.
struct input_descriptor {
  void const* data;
  cuda::std::uint32_t const* null_mask;
  cuda::std::int32_t offset;
  bool scalar;
};

struct output_descriptor {
  void* data;
  cuda::std::uint32_t* null_mask;
};

}  // namespace cudf::jit::ast_runtime
