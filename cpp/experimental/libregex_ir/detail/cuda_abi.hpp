/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuda_abi.hpp
 * @brief Shared ABI descriptions for generated matcher and column-wrapper code.
 *
 * Centralizes device signatures
 * so separately compiled matcher and wrapper fragments agree.
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

// Shared by the matcher and column-wrapper source emitters.
inline std::string matcher_signature(matcher_abi abi, std::string_view name)
{
  switch (abi) {
    case matcher_abi::BOOLEAN:
      return std::format(
        R"CUDA(extern "C" __device__ bool {}(char* workspace, char const* data, regex_ir::device::i64 size))CUDA",
        name);
    case matcher_abi::COUNT:
      return std::format(
        R"CUDA(extern "C" __device__ regex_ir::device::i64 {}(char* workspace, char const* data, regex_ir::device::i64 size))CUDA",
        name);
    case matcher_abi::BUILTIN_COUNT:
      return std::format(
        R"CUDA(extern "C" __device__ regex_ir::device::i64 {}(char* workspace, char const* data, regex_ir::device::i64 size, char const* flags))CUDA",
        name);
    case matcher_abi::FIND:
      return std::format(
        R"CUDA(extern "C" __device__ bool {}(char* workspace, char const* data, regex_ir::device::i64 size, regex_ir::device::i64* spans))CUDA",
        name);
    case matcher_abi::CAPTURES:
      return std::format(
        R"CUDA(extern "C" __device__ bool {}(char* workspace, char const* data, regex_ir::device::i64 size, regex_ir::device::i64 search, regex_ir::device::i64* spans))CUDA",
        name);
    case matcher_abi::REPLACE:
      return std::format(
        R"CUDA(extern "C" __device__ regex_ir::device::i64 {}(char* workspace, char const* data, regex_ir::device::i64 size, char* output))CUDA",
        name);
    case matcher_abi::SPLIT:
      return std::format(
        R"CUDA(extern "C" __device__ regex_ir::device::i64 {}(char* workspace, char const* data, regex_ir::device::i64 size, regex_ir::device::i64* spans))CUDA",
        name);
    case matcher_abi::LIMITED_SPLIT:
      return std::format(
        R"CUDA(extern "C" __device__ regex_ir::device::i64 {}(char* workspace, char const* data, regex_ir::device::i64 size, regex_ir::device::i64* spans, regex_ir::device::i64 limit, regex_ir::device::i64 capacity, char* overflow))CUDA",
        name);
  }
  throw std::invalid_argument("invalid matcher ABI");
}
}  // namespace regex_ir
