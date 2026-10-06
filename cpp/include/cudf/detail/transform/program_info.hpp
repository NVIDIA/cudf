/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/transform.hpp>

#include <rtcx/rtcx.hpp>

namespace cudf::detail {

struct transform_program_metadata {
  std::string tag                            = {};
  std::vector<transform_input_spec> inputs   = {};
  std::vector<transform_output_spec> outputs = {};
  bool null_aware                            = false;
  bool user_data                             = false;
  rtcx::kernel_ref kernel{nullptr};
};

template <typename ProgramInfo>
bool matches_transform_program(ProgramInfo const& kernel,
                               bool null_aware,
                               bool user_data,
                               std::span<transform_input_spec const> inputs,
                               std::span<transform_output_spec const> outputs)
{
  auto input_matches =
    [](auto const& self, transform_input_spec const& a, transform_input_spec const& b) -> bool {
    if (a.type != b.type || a.is_scalar != b.is_scalar) { return false; }
    if (a.type != type_id::DICTIONARY32) { return true; }
    if (a.children.size() != b.children.size()) { return false; }
    for (std::size_t i = 0; i < a.children.size(); ++i) {
      if (!self(self, a.children[i], b.children[i])) { return false; }
    }
    return true;
  };
  if (kernel.kernel.get() == nullptr || kernel.null_aware != null_aware ||
      kernel.user_data != user_data || kernel.inputs.size() != inputs.size() ||
      kernel.outputs.size() != outputs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!input_matches(input_matches, kernel.inputs[i], inputs[i])) { return false; }
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (kernel.outputs[i].type != outputs[i].type ||
        kernel.outputs[i].has_string_offsets != outputs[i].has_string_offsets) {
      return false;
    }
  }
  return true;
}
}  // namespace cudf::detail
