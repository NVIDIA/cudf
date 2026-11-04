/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file regex_ir.cpp
 * @brief Public compilation entry point and regex compiler pipeline.
 *
 * Validates operation arguments, runs parsing, lowering, optimization, and CUDA
 * emission, and translates internal diagnostics into public errors.
 */

#include <libregex_ir/detail/cuda_codegen.hpp>
#include <libregex_ir/detail/detail.hpp>

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace regex_ir {

instruction_ir compile_instruction_ir(std::string_view pattern,
                                      operation const& selected,
                                      compile_options const& options,
                                      optimization_options const& optimization)
{
  auto automata = compile_automata(pattern, options);
  return optimize(lower(automata, selected), optimization);
}

compile_result compile(std::string_view pattern,
                       operation_kind operation_kind_value,
                       std::optional<std::string> replacement,
                       compile_options const& options)
{
  switch (operation_kind_value) {
    case operation_kind::CONTAINS:
    case operation_kind::MATCHES:
    case operation_kind::COUNT:
    case operation_kind::EXTRACT:
    case operation_kind::FIND:
    case operation_kind::FIND_ALL:
    case operation_kind::SPLIT:
      if (replacement.has_value()) {
        throw std::invalid_argument("replacement is only valid for the REPLACE operation");
      }
      break;
    case operation_kind::REPLACE:
      if (!replacement.has_value()) {
        throw std::invalid_argument("replacement is required for the REPLACE operation");
      }
      break;
    default: throw std::invalid_argument("invalid regex operation");
  }

  try {
    auto compiled = compile_instruction_ir(
      pattern, operation{operation_kind_value, replacement.value_or("")}, options, {});
    return generate_cuda_source(compiled, {});
  } catch (compile_failure const& failure) {
    throw std::invalid_argument(std::format(
      "regex compilation failed at byte {}: {}", failure.source.offset, failure.message));
  }
}

}  // namespace regex_ir
