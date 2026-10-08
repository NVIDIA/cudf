/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace regex_ir {
enum class matcher_abi : std::uint8_t {
  BOOLEAN       = 0,
  COUNT         = 1,
  BUILTIN_COUNT = 2,
  FIND          = 3,
  CAPTURES      = 4,
  REPLACE       = 5,
  SPLIT         = 6,
  LIMITED_SPLIT = 7
};

struct matcher_layout {
  std::string_view result;
  std::string_view parameters;
  std::string_view arguments;
};

inline matcher_layout layout(matcher_abi abi)
{
  switch (abi) {
    case matcher_abi::BOOLEAN: return {"bool", "", ""};
    case matcher_abi::COUNT: return {"regex_ir::device::i64", "", ""};
    case matcher_abi::BUILTIN_COUNT:
      return {"regex_ir::device::i64", ", char const* flags", ", flags"};
    case matcher_abi::FIND: return {"bool", ", regex_ir::device::i64* spans", ", spans"};
    case matcher_abi::CAPTURES:
      return {
        "bool", ", regex_ir::device::i64 search, regex_ir::device::i64* spans", ", search, spans"};
    case matcher_abi::REPLACE: return {"regex_ir::device::i64", ", char* output", ", output"};
    case matcher_abi::SPLIT:
      return {"regex_ir::device::i64", ", regex_ir::device::i64* spans", ", spans"};
    case matcher_abi::LIMITED_SPLIT:
      return {
        "regex_ir::device::i64",
        R"CUDA(, regex_ir::device::i64 *spans, regex_ir::device::i64 limit, regex_ir::device::i64 capacity,
  char *overflow)CUDA",
        ", spans, limit, capacity, overflow"};
  }
  throw std::invalid_argument("invalid matcher ABI");
}

// One definition of the cross-fragment ABI, shared by both source emitters.
inline std::string matcher_signature(matcher_abi abi, std::string_view name, bool workspace)
{
  auto spec = layout(abi);
  return std::format(
    "extern \"C\" __device__ {} {}({}char const* data, regex_ir::device::i64 size{})",
    spec.result,
    name,
    workspace ? "char* workspace, " : "",
    spec.parameters);
}
}  // namespace regex_ir
