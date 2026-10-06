/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/transform/program_info.hpp>

namespace cudf::detail {
struct filter_join_program_metadata {
  std::string tag                            = {};
  std::vector<transform_input_spec> inputs   = {};
  std::vector<transform_output_spec> outputs = {};
  bool null_aware                            = false;
  bool user_data                             = false;
  rtcx::kernel_ref kernel{nullptr};
  std::size_t left_input_count = 0;
};
}  // namespace cudf::detail
