/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace experimental {

/**
 * @brief Regex operation represented by a JIT regex program
 *
 * Regex JIT programs are specialized for the result and control flow required
 * by one API. The discriminator prevents a program from being used by a different
 * API.
 */
enum class regex_operation : std::uint8_t {
  CONTAINS = 0,           ///< Used by contains_re
  MATCHES,                ///< Used by matches_re
  COUNT,                  ///< Used by count_re
  FIND,                   ///< Used by find_re
  EXTRACT,                ///< Used by extract
  EXTRACT_SINGLE,         ///< Used by extract_single
  EXTRACT_ALL_RECORD,     ///< Used by extract_all_record
  FINDALL,                ///< Used by findall
  REPLACE,                ///< Used by replace_re
  REPLACE_WITH_BACKREFS,  ///< Used by replace_with_backrefs
  SPLIT,                  ///< Used by split_re
  RSPLIT,                 ///< Used by rsplit_re
  SPLIT_RECORD,           ///< Used by split_record_re
  RSPLIT_RECORD,          ///< Used by rsplit_record_re
};

/**
 * @brief Policy controlling bounded span caching in materializing regex JIT APIs
 */
enum class regex_jit_span_cache_policy : std::uint8_t {
  AUTO,   ///< Enable caching when input sampling predicts a benefit
  OFF,    ///< Disable caching and rematch rows during materialization
  FORCE,  ///< Enable caching whenever the input fits the bounded cache
};

/**
 * @brief Values embedded in an operation-specialized regex JIT program
 */
struct regex_jit_program_options {
  std::optional<size_type> group{};              ///< Capture group for `extract_single`
  std::optional<std::string> replacement{};      ///< Replacement for replace APIs
  std::optional<size_type> max_replace_count{};  ///< Replacement limit for `replace_re`
  size_type maxsplit{-1};                        ///< Split limit for split APIs
  /// Span-cache selection policy
  regex_jit_span_cache_policy span_cache_policy{regex_jit_span_cache_policy::AUTO};
};

/**
 * @brief Operation-specialized regex program for the experimental JIT APIs
 *
 * The pattern is parsed into Regex IR when the object is created. The resulting
 * API-specific kernels are compiled and retained by the program, which is
 * immutable after creation.
 */
struct regex_jit_program {
  struct regex_jit_program_impl;

  static std::unique_ptr<regex_jit_program> create(
    std::string_view pattern,
    regex_operation operation,
    regex_jit_program_options const& options = {},
    strings::regex_flags flags               = strings::regex_flags::DEFAULT,
    strings::capture_groups captures         = strings::capture_groups::EXTRACT);

  regex_jit_program()                                    = delete;
  regex_jit_program(regex_jit_program const&)            = delete;
  regex_jit_program& operator=(regex_jit_program const&) = delete;
  ~regex_jit_program();

  regex_jit_program(regex_jit_program&& other) noexcept;
  regex_jit_program& operator=(regex_jit_program&& other) noexcept;

  [[nodiscard]] std::string pattern() const;
  [[nodiscard]] strings::regex_flags flags() const;
  [[nodiscard]] strings::capture_groups capture() const;
  [[nodiscard]] regex_operation operation() const;
  [[nodiscard]] size_type groups_count() const;

 private:
  regex_jit_program(std::string_view pattern,
                    regex_operation operation,
                    regex_jit_program_options const& options,
                    strings::regex_flags flags,
                    strings::capture_groups captures);

  std::string _pattern;
  regex_operation _operation;
  strings::regex_flags _flags;
  strings::capture_groups _captures;
  regex_jit_program_options _options;
  std::unique_ptr<regex_jit_program_impl> _impl;

  friend struct regex_jit_program_accessor;
};

std::unique_ptr<column> contains_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> matches_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> count_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> extract(strings_column_view const& input,
                               regex_jit_program const& prog,
                               cuda::stream_ref stream   = cudf::get_default_stream(),
                               cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> extract_all_record(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> extract_single(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type group,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> findall(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> find_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> replace_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  string_scalar const& replacement           = string_scalar(""),
  std::optional<size_type> max_replace_count = std::nullopt,
  cuda::stream_ref stream                    = cudf::get_default_stream(),
  cudf::memory_resources mr                  = cudf::get_current_device_resource_ref());

std::unique_ptr<column> replace_with_backrefs(
  strings_column_view const& input,
  regex_jit_program const& prog,
  std::string_view replacement,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> split_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> rsplit_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> split_record_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> rsplit_record_re(
  strings_column_view const& input,
  regex_jit_program const& prog,
  size_type maxsplit        = -1,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
