/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex_ir.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace cudf::experimental::detail::regex_jit {

/**
 * @brief One literal or capture component of a replacement template
 */
struct replacement_piece {
  std::string literal;                  ///< Literal UTF-8 bytes emitted by this component
  std::optional<std::int32_t> capture;  ///< Capture index to emit, if present
};

/**
 * @brief Complete the generated definitions for the column-wrapper compilation entry point
 *
 * @param kernel Operation-specific kernel wrapper
 * @param workspace_bytes Temporary matcher storage per worker, or zero for local storage
 * @return Generated header source included after the stable kernel runtime and PCH boundary
 */
[[nodiscard]] std::string make_module(std::string kernel, std::size_t workspace_bytes = 0);

/**
 * @brief Generate a fixed-width output kernel
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param operation Regex operation implemented by the kernel
 * @param repeated_builtin Built-in repeated predicate adapted by the kernel, if any
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the kernel
 * @throw std::invalid_argument If `kernel_name` is not a valid, non-reserved identifier
 */
[[nodiscard]] std::string make_fixed_kernel(
  bool offset64,
  regex_ir::operation_kind operation,
  std::optional<regex_ir::builtin_character_class> repeated_builtin,
  std::string_view kernel_name);

/**
 * @brief Generate a warp-per-row kernel for an exact ASCII literal contains operation
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param literal Non-empty ASCII literal to search for
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the kernel
 */
[[nodiscard]] std::string make_warp_literal_contains_kernel(bool offset64,
                                                            std::string_view literal,
                                                            std::string_view kernel_name);

/**
 * @brief Generate a kernel that emits capture spans
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param capture_slots Number of capture-boundary values available from the matcher
 * @param first_group First explicit capture group to emit
 * @param output_groups Number of capture groups to emit
 * @param column_major Whether output spans are grouped by capture instead of input row
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the kernel
 */
[[nodiscard]] std::string make_capture_kernel(bool offset64,
                                              std::int32_t capture_slots,
                                              std::int32_t first_group,
                                              std::int32_t output_groups,
                                              bool column_major,
                                              std::string_view kernel_name);

/**
 * @brief Generate the sizing pass for an API that enumerates matches or captures
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param capture_slots Number of capture-boundary values available from the matcher
 * @param multiplier Number of output spans produced per accepted match
 * @param require_match Whether rows without a match produce no output entry
 * @param cache Whether to cache bounded match spans for the emission pass
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the sizing kernel
 */
[[nodiscard]] std::string make_enumeration_size_kernel(bool offset64,
                                                       std::int32_t capture_slots,
                                                       std::int32_t multiplier,
                                                       bool require_match,
                                                       bool cache,
                                                       std::string_view kernel_name);

/**
 * @brief Generate the emission pass for an API that enumerates matches or captures
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param capture_slots Number of capture-boundary values available from the matcher
 * @param groups Number of capture groups emitted per match
 * @param findall Whether to emit whole-match spans instead of capture spans
 * @param overflow_only Whether to process only rows that overflowed the span cache
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the emission kernel
 */
[[nodiscard]] std::string make_enumeration_emit_kernel(bool offset64,
                                                       std::int32_t capture_slots,
                                                       std::int32_t groups,
                                                       bool findall,
                                                       bool overflow_only,
                                                       std::string_view kernel_name);

/**
 * @brief Generate a fused sizing or emission kernel for bounded replacement
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param emit Whether to emit output bytes instead of only computing sizes
 * @param output_offset64 Whether output string offsets use 64-bit integers
 * @param cache Whether to cache bounded match spans between passes
 * @param replacement Parsed replacement template to bake into the kernel
 * @param capture_slots Number of capture-boundary values available from the matcher
 * @param max_replace_count Maximum replacements performed per input row
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the replacement kernel
 */
[[nodiscard]] std::string make_limited_replace_kernel(
  bool offset64,
  bool emit,
  bool output_offset64,
  bool cache,
  std::span<replacement_piece const> replacement,
  std::int32_t capture_slots,
  std::int32_t max_replace_count,
  std::string_view kernel_name);

/**
 * @brief Encode a parsed replacement template as regex replacement text
 *
 * @param replacement Parsed replacement template
 * @return Replacement text with capture references and escaped literal dollar signs
 */
[[nodiscard]] std::string encode_replacement(std::span<replacement_piece const> replacement);

/**
 * @brief Generate a generic replacement sizing or emission kernel
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param emit Whether to emit output bytes instead of only computing sizes
 * @param output_offset64 Whether output string offsets use 64-bit integers
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the replacement kernel
 */
[[nodiscard]] std::string make_replace_kernel(bool offset64,
                                              bool emit,
                                              bool output_offset64,
                                              std::string_view kernel_name);

/**
 * @brief Generate the sizing pass for regex split
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param maxsplit Maximum splits per row; a non-positive value means unlimited
 * @param cache Whether to cache bounded field spans for the emission pass
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the split sizing kernel
 */
[[nodiscard]] std::string make_split_size_kernel(bool offset64,
                                                 std::int32_t maxsplit,
                                                 bool cache,
                                                 std::string_view kernel_name);

/**
 * @brief Generate the emission pass for regex split
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param reverse Whether to apply the split limit from the end of each input
 * @param maxsplit Maximum splits per row; a non-positive value means unlimited
 * @param overflow_only Whether to process only rows that overflowed the span cache
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the split emission kernel
 */
[[nodiscard]] std::string make_split_emit_kernel(bool offset64,
                                                 bool reverse,
                                                 std::int32_t maxsplit,
                                                 bool overflow_only,
                                                 std::string_view kernel_name);

/**
 * @brief Generate a sampling kernel used to select a match-span cache size
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param split Whether to sample split fields instead of enumerated match spans
 * @param capture_slots Number of capture-boundary values available from the matcher
 * @param match_limit Maximum matches sampled from each row; a non-positive value means unlimited
 * @param kernel_name Exported kernel entry-point name
 * @return CUDA C++ source for the sampling kernel
 */
[[nodiscard]] std::string make_span_cache_sample_kernel(bool offset64,
                                                        bool split,
                                                        std::int32_t capture_slots,
                                                        std::int32_t match_limit,
                                                        std::string_view kernel_name);

}  // namespace cudf::experimental::detail::regex_jit
