/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/experimental/strings/regex.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/split/split_re.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace cudf::test {

struct interpreter_regex_backend {
  using program = strings::regex_program;

  static std::unique_ptr<program> create(
    experimental::regex_operation,
    std::string_view pattern,
    strings::regex_flags flags                     = strings::regex_flags::DEFAULT,
    strings::capture_groups captures               = strings::capture_groups::EXTRACT,
    experimental::regex_jit_program_options const& = {})
  {
    return program::create(pattern, flags, captures);
  }

  static std::unique_ptr<program> create(
    experimental::regex_operation operation,
    std::string_view pattern,
    experimental::regex_jit_program_options const& options,
    strings::regex_flags flags       = strings::regex_flags::DEFAULT,
    strings::capture_groups captures = strings::capture_groups::EXTRACT)
  {
    return create(operation, pattern, flags, captures, options);
  }

  static std::unique_ptr<column> contains_re(strings_column_view const& input, program const& regex)
  {
    return strings::contains_re(input, regex);
  }

  static std::unique_ptr<column> matches_re(strings_column_view const& input, program const& regex)
  {
    return strings::matches_re(input, regex);
  }

  static std::unique_ptr<column> count_re(strings_column_view const& input, program const& regex)
  {
    return strings::count_re(input, regex);
  }

  static std::unique_ptr<table> extract(strings_column_view const& input, program const& regex)
  {
    return strings::extract(input, regex);
  }

  static std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                                    program const& regex)
  {
    return strings::extract_all_record(input, regex);
  }

  static std::unique_ptr<column> extract_single(strings_column_view const& input,
                                                program const& regex,
                                                size_type group)
  {
    return strings::extract_single(input, regex, group);
  }

  static std::unique_ptr<column> findall(strings_column_view const& input, program const& regex)
  {
    return strings::findall(input, regex);
  }

  static std::unique_ptr<column> find_re(strings_column_view const& input, program const& regex)
  {
    return strings::find_re(input, regex);
  }

  static std::unique_ptr<column> replace_re(
    strings_column_view const& input,
    program const& regex,
    string_scalar const& replacement           = string_scalar(""),
    std::optional<size_type> max_replace_count = std::nullopt)
  {
    return strings::replace_re(input, regex, replacement, max_replace_count);
  }

  static std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                                       program const& regex,
                                                       std::string_view replacement)
  {
    return strings::replace_with_backrefs(input, regex, replacement);
  }

  static std::unique_ptr<table> split_re(strings_column_view const& input,
                                         program const& regex,
                                         size_type maxsplit = -1)
  {
    return strings::split_re(input, regex, maxsplit);
  }

  static std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                          program const& regex,
                                          size_type maxsplit = -1)
  {
    return strings::rsplit_re(input, regex, maxsplit);
  }

  static std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                                 program const& regex,
                                                 size_type maxsplit = -1)
  {
    return strings::split_record_re(input, regex, maxsplit);
  }

  static std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                                  program const& regex,
                                                  size_type maxsplit = -1)
  {
    return strings::rsplit_record_re(input, regex, maxsplit);
  }
};

struct jit_regex_backend {
  using program = experimental::regex_jit_program;

  static std::unique_ptr<program> create(
    experimental::regex_operation operation,
    std::string_view pattern,
    strings::regex_flags flags                             = strings::regex_flags::DEFAULT,
    strings::capture_groups captures                       = strings::capture_groups::EXTRACT,
    experimental::regex_jit_program_options const& options = {})
  {
    return program::create(pattern, operation, options, flags, captures);
  }

  static std::unique_ptr<program> create(
    experimental::regex_operation operation,
    std::string_view pattern,
    experimental::regex_jit_program_options const& options,
    strings::regex_flags flags       = strings::regex_flags::DEFAULT,
    strings::capture_groups captures = strings::capture_groups::EXTRACT)
  {
    return create(operation, pattern, flags, captures, options);
  }

  static std::unique_ptr<column> contains_re(strings_column_view const& input, program const& regex)
  {
    return experimental::contains_re(input, regex);
  }

  static std::unique_ptr<column> matches_re(strings_column_view const& input, program const& regex)
  {
    return experimental::matches_re(input, regex);
  }

  static std::unique_ptr<column> count_re(strings_column_view const& input, program const& regex)
  {
    return experimental::count_re(input, regex);
  }

  static std::unique_ptr<table> extract(strings_column_view const& input, program const& regex)
  {
    return experimental::extract(input, regex);
  }

  static std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                                    program const& regex)
  {
    return experimental::extract_all_record(input, regex);
  }

  static std::unique_ptr<column> extract_single(strings_column_view const& input,
                                                program const& regex,
                                                size_type group)
  {
    return experimental::extract_single(input, regex, group);
  }

  static std::unique_ptr<column> findall(strings_column_view const& input, program const& regex)
  {
    return experimental::findall(input, regex);
  }

  static std::unique_ptr<column> find_re(strings_column_view const& input, program const& regex)
  {
    return experimental::find_re(input, regex);
  }

  static std::unique_ptr<column> replace_re(
    strings_column_view const& input,
    program const& regex,
    string_scalar const& replacement           = string_scalar(""),
    std::optional<size_type> max_replace_count = std::nullopt)
  {
    return experimental::replace_re(input, regex, replacement, max_replace_count);
  }

  static std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                                       program const& regex,
                                                       std::string_view replacement)
  {
    return experimental::replace_with_backrefs(input, regex, replacement);
  }

  static std::unique_ptr<table> split_re(strings_column_view const& input,
                                         program const& regex,
                                         size_type maxsplit = -1)
  {
    return experimental::split_re(input, regex, maxsplit);
  }

  static std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                          program const& regex,
                                          size_type maxsplit = -1)
  {
    return experimental::rsplit_re(input, regex, maxsplit);
  }

  static std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                                 program const& regex,
                                                 size_type maxsplit = -1)
  {
    return experimental::split_record_re(input, regex, maxsplit);
  }

  static std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                                  program const& regex,
                                                  size_type maxsplit = -1)
  {
    return experimental::rsplit_record_re(input, regex, maxsplit);
  }
};

using regex_backends = ::testing::Types<interpreter_regex_backend, jit_regex_backend>;

}  // namespace cudf::test
