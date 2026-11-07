/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/contains.hpp>
#include <cudf/strings/experimental/regex.hpp>
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
#include <type_traits>
#include <utility>

namespace cudf::test {

namespace regex_backend_detail {

#define CUDF_REGEX_DISPATCH(name)                                                   \
  template <typename Program, typename... Args>                                     \
  auto name(strings_column_view const& input, Program const& regex, Args&&... args) \
  {                                                                                 \
    if constexpr (std::is_same_v<Program, strings::regex_program>) {                \
      return strings::name(input, regex, std::forward<Args>(args)...);              \
    } else {                                                                        \
      return experimental::name(input, regex, std::forward<Args>(args)...);         \
    }                                                                               \
  }

CUDF_REGEX_DISPATCH(contains_re)
CUDF_REGEX_DISPATCH(matches_re)
CUDF_REGEX_DISPATCH(count_re)
CUDF_REGEX_DISPATCH(extract)
CUDF_REGEX_DISPATCH(extract_all_record)
CUDF_REGEX_DISPATCH(extract_single)
CUDF_REGEX_DISPATCH(findall)
CUDF_REGEX_DISPATCH(find_re)
CUDF_REGEX_DISPATCH(replace_re)
CUDF_REGEX_DISPATCH(replace_with_backrefs)
CUDF_REGEX_DISPATCH(split_re)
CUDF_REGEX_DISPATCH(rsplit_re)
CUDF_REGEX_DISPATCH(split_record_re)
CUDF_REGEX_DISPATCH(rsplit_record_re)

#undef CUDF_REGEX_DISPATCH

}  // namespace regex_backend_detail

template <typename Program>
struct regex_backend {
  using program = Program;
  static constexpr bool supports_unbounded_nesting =
    std::is_same_v<program, strings::regex_program>;

#define CUDF_REGEX_CREATE(name)                                                                \
  static std::unique_ptr<program> create_##name(                                               \
    std::string_view pattern,                                                                  \
    strings::regex_flags flags                             = strings::regex_flags::DEFAULT,    \
    strings::capture_groups captures                       = strings::capture_groups::EXTRACT, \
    experimental::regex_jit_program_options const& options = {})                               \
  {                                                                                            \
    if constexpr (std::is_same_v<program, strings::regex_program>) {                           \
      return program::create(pattern, flags, captures);                                        \
    } else {                                                                                   \
      return program::create_##name(pattern, options, flags, captures);                        \
    }                                                                                          \
  }                                                                                            \
                                                                                               \
  static std::unique_ptr<program> create_##name(                                               \
    std::string_view pattern,                                                                  \
    experimental::regex_jit_program_options const& options,                                    \
    strings::regex_flags flags       = strings::regex_flags::DEFAULT,                          \
    strings::capture_groups captures = strings::capture_groups::EXTRACT)                       \
  {                                                                                            \
    return create_##name(pattern, flags, captures, options);                                   \
  }

  CUDF_REGEX_CREATE(contains)
  CUDF_REGEX_CREATE(matches)
  CUDF_REGEX_CREATE(count)
  CUDF_REGEX_CREATE(find)
  CUDF_REGEX_CREATE(extract)
  CUDF_REGEX_CREATE(extract_single)
  CUDF_REGEX_CREATE(extract_all_record)
  CUDF_REGEX_CREATE(findall)
  CUDF_REGEX_CREATE(replace)
  CUDF_REGEX_CREATE(replace_with_backrefs)
  CUDF_REGEX_CREATE(split)
  CUDF_REGEX_CREATE(rsplit)
  CUDF_REGEX_CREATE(split_record)
  CUDF_REGEX_CREATE(rsplit_record)

#undef CUDF_REGEX_CREATE

  static std::unique_ptr<column> contains_re(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::contains_re(input, regex);
  }

  static std::unique_ptr<column> matches_re(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::matches_re(input, regex);
  }

  static std::unique_ptr<column> count_re(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::count_re(input, regex);
  }

  static std::unique_ptr<table> extract(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::extract(input, regex);
  }

  static std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                                    program const& regex)
  {
    return regex_backend_detail::extract_all_record(input, regex);
  }

  static std::unique_ptr<column> extract_single(strings_column_view const& input,
                                                program const& regex,
                                                size_type group)
  {
    return regex_backend_detail::extract_single(input, regex, group);
  }

  static std::unique_ptr<column> findall(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::findall(input, regex);
  }

  static std::unique_ptr<column> find_re(strings_column_view const& input, program const& regex)
  {
    return regex_backend_detail::find_re(input, regex);
  }

  static std::unique_ptr<column> replace_re(
    strings_column_view const& input,
    program const& regex,
    string_scalar const& replacement           = string_scalar(""),
    std::optional<size_type> max_replace_count = std::nullopt)
  {
    return regex_backend_detail::replace_re(input, regex, replacement, max_replace_count);
  }

  static std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                                       program const& regex,
                                                       std::string_view replacement)
  {
    return regex_backend_detail::replace_with_backrefs(input, regex, replacement);
  }

  static std::unique_ptr<table> split_re(
    strings_column_view const& input,
    program const& regex,
    size_type maxsplit                = -1,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return regex_backend_detail::split_re(input, regex, maxsplit, stream, mr);
  }

  static std::unique_ptr<table> rsplit_re(
    strings_column_view const& input,
    program const& regex,
    size_type maxsplit                = -1,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return regex_backend_detail::rsplit_re(input, regex, maxsplit, stream, mr);
  }

  static std::unique_ptr<column> split_record_re(
    strings_column_view const& input,
    program const& regex,
    size_type maxsplit                = -1,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return regex_backend_detail::split_record_re(input, regex, maxsplit, stream, mr);
  }

  static std::unique_ptr<column> rsplit_record_re(
    strings_column_view const& input,
    program const& regex,
    size_type maxsplit                = -1,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return regex_backend_detail::rsplit_record_re(input, regex, maxsplit, stream, mr);
  }
};

using interpreter_regex_backend = regex_backend<strings::regex_program>;
using jit_regex_backend         = regex_backend<experimental::regex_jit_program>;

using regex_backends = ::testing::Types<interpreter_regex_backend, jit_regex_backend>;

}  // namespace cudf::test
