/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex_ir_nvvm_templates.hpp>

#include <cctype>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace regex_ir {

using nvvm_replacement = std::pair<std::string_view, std::string_view>;

inline std::string_view nvvm_template(std::size_t index)
{
  if (index >= std::size(regex_ir_nvvm_templates::file_ranges)) {
    throw std::out_of_range("invalid embedded NVVM template index");
  }
  auto& range = regex_ir_nvvm_templates::file_ranges[index];
  return {reinterpret_cast<char const*>(regex_ir_nvvm_templates::files.data() + range[0]),
          range[1]};
}

inline std::string_view nvvm_template_section(std::size_t index, std::string_view section)
{
  auto input        = nvvm_template(index);
  auto begin_marker = std::format("; BEGIN {}\n", section);
  auto end_marker   = std::format("; END {}", section);
  auto begin        = input.find(begin_marker);
  if (begin == std::string_view::npos) {
    throw std::invalid_argument(std::format("missing NVVM template section '{}'", section));
  }
  begin += begin_marker.size();
  auto end = input.find(end_marker, begin);
  if (end == std::string_view::npos) {
    throw std::invalid_argument(std::format("unterminated NVVM template section '{}'", section));
  }
  return input.substr(begin, end - begin);
}

inline std::string render_nvvm_template_text(
  std::string_view input, std::initializer_list<nvvm_replacement> replacements = {})
{
  auto result = std::string{input};
  for (auto begin = result.find('@'); begin != std::string::npos;
       begin      = result.find('@', begin + 1)) {
    auto end = result.find('@', begin + 1);
    if (end == std::string::npos) { break; }
    auto candidate   = std::string_view{result}.substr(begin + 1, end - begin - 1);
    auto placeholder = !candidate.empty();
    for (auto character : candidate) {
      placeholder = placeholder &&
                    (std::isupper(static_cast<unsigned char>(character)) != 0 ||
                     std::isdigit(static_cast<unsigned char>(character)) != 0 || character == '_');
    }
    if (placeholder && candidate != "KERNEL_ENTRY") {
      auto supplied = false;
      auto token    = std::string_view{result}.substr(begin, end - begin + 1);
      for (auto& replacement : replacements) {
        supplied = supplied || replacement.first == token;
      }
      if (!supplied) { throw std::invalid_argument("unresolved NVVM template placeholder"); }
    }
    begin = end;
  }

  for (auto& [token, value] : replacements) {
    if (token.empty() || result.find(token) == std::string::npos) {
      throw std::invalid_argument(
        std::format("NVVM template replacement token '{}' is missing", token));
    }
    for (auto position = result.find(token); position != std::string::npos;
         position      = result.find(token, position + value.size())) {
      result.replace(position, token.size(), value);
    }
  }

  return result;
}

inline std::string render_nvvm_template(std::size_t index,
                                        std::initializer_list<nvvm_replacement> replacements = {})
{
  return render_nvvm_template_text(nvvm_template(index), replacements);
}

inline std::string render_nvvm_template_section(
  std::size_t index,
  std::string_view section,
  std::initializer_list<nvvm_replacement> replacements = {})
{
  return render_nvvm_template_text(nvvm_template_section(index, section), replacements);
}

template <typename... Args>
std::string nvvm_template(std::size_t index, std::string_view section, Args&&... args)
{
  return std::vformat(nvvm_template_section(index, section), std::make_format_args(args...));
}

}  // namespace regex_ir
