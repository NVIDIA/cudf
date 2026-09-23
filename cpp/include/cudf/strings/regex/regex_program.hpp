/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/regex/flags.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>

namespace CUDF_EXPORT cudf {
namespace strings {

/**
 * @addtogroup strings_regex
 * @{
 */

/**
 * @brief Regex program class
 *
 * Create an instance from a regex pattern and use it to call the appropriate
 * strings APIs. An instance can be reused.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns and APIs that support regex.
 */
struct regex_program {
  struct regex_program_impl;

  /**
   * @brief Create a program from a pattern
   *
   * @throw cudf::logic_error If pattern is invalid or contains unsupported features
   *
   * @param pattern Regex pattern
   * @param flags Regex flags for interpreting special characters in the pattern
   * @param capture Controls how capture groups in the pattern are used
   * @return Instance of this object
   */
  static std::unique_ptr<regex_program> create(std::string_view pattern,
                                               regex_flags flags      = regex_flags::DEFAULT,
                                               capture_groups capture = capture_groups::EXTRACT);

  regex_program()                                = delete;
  regex_program(regex_program const&)            = delete;
  regex_program& operator=(regex_program const&) = delete;
  ~regex_program();

  /**
   * @brief Move constructor
   *
   * @param other Object to move from
   */
  regex_program(regex_program&& other) noexcept;

  /**
   * @brief Move operator assignment
   *
   * @param other Object to move from
   * @return this object
   */
  regex_program& operator=(regex_program&& other) noexcept;

  /**
   * @brief Return the pattern used to create this instance
   *
   * @return regex pattern as a string
   */
  [[nodiscard]] std::string pattern() const;

  /**
   * @brief Return the regex_flags used to create this instance
   *
   * @return regex flags setting
   */
  [[nodiscard]] regex_flags flags() const;

  /**
   * @brief Return the capture_groups used to create this instance
   *
   * @return capture groups setting
   */
  [[nodiscard]] capture_groups capture() const;

  /**
   * @brief Return the number of instructions in this instance
   *
   * @return Number of instructions
   */
  [[nodiscard]] int32_t instructions_count() const;

  /**
   * @brief Return the number of capture groups in this instance
   *
   * @return Number of groups
   */
  [[nodiscard]] int32_t groups_count() const;

  /**
   * @brief Return the size of the working memory for the regex execution
   *
   * @param num_strings Number of strings for computation
   * @return Size of the working memory in bytes
   */
  [[nodiscard]] std::size_t compute_working_memory_size(int32_t num_strings) const;

  /**
   * @brief Returns literal string if specific fast-path is possible
   *
   * @return Which fast-path is available and the associate literal string
   */
  [[nodiscard]] std::pair<literal_fast_path, std::string> get_literal_fast_path() const;

 private:
  std::string _pattern;
  regex_flags _flags;
  capture_groups _capture;

  std::unique_ptr<regex_program_impl> _impl;

  /**
   * @brief Constructor
   *
   * Called by create()
   */
  regex_program(std::string_view pattern, regex_flags flags, capture_groups capture);

  friend struct regex_device_builder;
};

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
