/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "glushkov_regcomp.hpp"
#include "regex_program_impl.h"

#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/regex/regex_program.hpp>

#include <cstdlib>
#include <memory>
#include <string>

namespace cudf {
namespace strings {

std::unique_ptr<regex_program> regex_program::create(std::string_view pattern,
                                                     regex_flags flags,
                                                     capture_groups capture)
{
  auto p = new regex_program(pattern, flags, capture);
  return std::unique_ptr<regex_program>(p);
}

regex_program::~regex_program()                                         = default;
regex_program::regex_program(regex_program&& other) noexcept            = default;
regex_program& regex_program::operator=(regex_program&& other) noexcept = default;

regex_program::regex_program(std::string_view pattern, regex_flags flags, capture_groups capture)
  : _pattern(pattern),
    _flags(flags),
    _impl(
      std::make_unique<regex_program_impl>(detail::reprog::create_from(pattern, flags, capture)))
{
  auto const* env = std::getenv("LIBCUDF_DISABLE_GLUSHKOV");
  if (env == nullptr) { _impl->glushkov_prog = detail::build_glushkov_program(_impl->prog); }
}

std::string regex_program::pattern() const { return _pattern; }

regex_flags regex_program::flags() const { return _flags; }

capture_groups regex_program::capture() const { return _capture; }

int32_t regex_program::instructions_count() const { return _impl->prog.insts_count(); }

int32_t regex_program::groups_count() const { return _impl->prog.groups_count(); }

std::size_t regex_program::compute_working_memory_size(int32_t num_strings) const
{
  if (_impl->glushkov_prog.get() != nullptr) { return 0; }
  return detail::reprog_device::compute_working_memory_size(num_strings, instructions_count());
}

std::pair<literal_fast_path, std::string> regex_program::get_literal_fast_path() const
{
  return _impl->prog.check_for_literal_fast_path();
}

}  // namespace strings
}  // namespace cudf
