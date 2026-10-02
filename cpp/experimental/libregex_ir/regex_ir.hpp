/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace regex_ir {

/**
 * @brief Unit used to decode and match input characters
 */
enum class character_mode : std::uint8_t {
  UTF8,   ///< Decode input as UTF-8 code points
  BYTES,  ///< Match individual input bytes
};

/**
 * @brief Resource limits enforced while compiling a regular expression
 */
struct compile_limits {
  std::size_t max_pattern_bytes = 1U << 20U;  ///< Maximum pattern size in bytes
  std::size_t max_nesting       = 256;        ///< Maximum nested expression depth
  std::size_t max_states        = 1U << 18U;  ///< Maximum generated automata states
  std::size_t max_transitions   = 1U << 20U;  ///< Maximum generated automata transitions
  std::size_t max_captures      = 256;        ///< Maximum explicit capture groups
  std::uint32_t max_repeat      = 1000;       ///< Maximum finite repetition bound
};

/**
 * @brief Regex syntax and compilation settings
 */
struct compile_options {
  bool case_insensitive          = false;  ///< Enable case-insensitive matching
  bool multiline                 = false;  ///< Make line anchors recognize internal line boundaries
  bool dot_all                   = false;  ///< Allow dot to match configured newline characters
  bool ascii_classes             = true;   ///< Use ASCII semantics for shorthand character classes
  bool extended_newline          = false;  ///< Recognize the extended Unicode newline set
  bool find_match_end_observable = true;   ///< Require FIND to produce its end offset
  std::optional<std::uint32_t>
    extract_capture_group;                           ///< One-based observable group, or all groups
  character_mode characters = character_mode::UTF8;  ///< Input character decoding mode
  compile_limits limits     = compile_limits{};      ///< Compilation resource limits
};

/**
 * @brief Regex API implemented by generated code
 */
enum class operation_kind : std::uint8_t {
  CONTAINS,  ///< Test whether the input contains a match
  MATCHES,   ///< Test whether a match begins at the start of the input
  COUNT,     ///< Count non-overlapping matches
  EXTRACT,   ///< Extract capture groups from a match
  FIND,      ///< Find the span of a match
  FIND_ALL,  ///< Find successive whole-match spans beginning at a supplied byte offset
  REPLACE,   ///< Replace matching spans
  SPLIT,     ///< Split input around matching spans
};

/**
 * @brief Matching executor selected by the compiler
 */
enum class executor_kind : std::uint8_t {
  ITERATIVE_THOMPSON,                ///< Ordered, bounded-worklist Thompson-NFA executor
  STRING_OPERATIONS,                 ///< Generated composition of specialized string operations
  WORD_RUN,                          ///< Specialized boundary-delimited word-run executor
  SINGLE_BYTE_LITERAL,               ///< Specialized single-byte literal executor
  PACKED_ASCII_LITERAL,              ///< Specialized packed ASCII literal executor
  PACKED_UTF8_LITERAL,               ///< Pivoted packed-byte exact UTF-8 literal executor
  UTF8_KMP_LITERAL,                  ///< Exact UTF-8 literal executor with byte-domain KMP fallback
  GLUSHKOV,                          ///< Position-automaton executor
  STREAMING_PRIORITIZED_GLUSHKOV,    ///< Streaming prioritized position-automaton executor
  DETERMINISTIC,                     ///< Deterministic finite-automaton executor
  ASSERTION_AWARE_DETERMINISTIC,     ///< Deterministic executor with zero-width assertions
  PRIORITIZED_DETERMINISTIC,         ///< Deterministic executor preserving branch priority
  TAGGED_PRIORITIZED_DETERMINISTIC,  ///< Prioritized deterministic executor with captures
  BOOLEAN_ALTERNATION,               ///< Dispatcher over separately compiled boolean branches
};

/**
 * @brief Built-in Unicode character predicate delegated to an embedding adapter
 */
enum class builtin_character_class : std::uint8_t {
  DIGIT,      ///< Unicode digit predicate
  NOT_DIGIT,  ///< Negated Unicode digit predicate
  WORD,       ///< Unicode word-character predicate
  NOT_WORD,   ///< Negated Unicode word-character predicate
  SPACE,      ///< Unicode whitespace predicate
  NOT_SPACE,  ///< Negated Unicode whitespace predicate
};

/**
 * @brief Result of compiling a regular expression
 *
 * If `workspace_bytes` is nonzero, the generated executor takes an additional
 * leading `i8*` argument pointing to that many bytes of eight-byte-aligned
 * writable device storage. Concurrent workers must use disjoint storage;
 * sequential calls may reuse it without initialization. Zero leaves the
 * operation-specific signature unchanged.
 */
struct compile_result {
  std::string nvvm_ir;             ///< Textual operation-specialized NVVM IR
  std::uint32_t capture_count;     ///< Number of explicit capture groups
  executor_kind executor;          ///< Executor selected for the pattern and operation
  std::uint32_t executor_states;   ///< Number of states in the selected executor
  std::uint32_t alphabet_classes;  ///< Number of character classes in its alphabet partition
  std::optional<std::string> exact_ascii_literal;  ///< Exact ASCII literal recognized, if any
  std::optional<std::string> exact_literal_bytes;  ///< Exact encoded literal recognized, if any
  std::optional<builtin_character_class> repeated_builtin;  ///< Adapted repeated predicate, if any
  std::size_t workspace_bytes =
    0;  ///< Temporary bytes per worker; zero means no external workspace
};

/**
 * @brief Compile a regular expression into operation-specialized NVVM IR
 *
 * `replacement` is required for `REPLACE` and rejected for every other
 * operation. Compilation failures are reported as `std::invalid_argument`.
 *
 * @param pattern Regular expression encoded as UTF-8 source bytes
 * @param operation Operation implemented by the generated entry point
 * @param replacement Replacement template for `REPLACE`
 * @param options Regex syntax, character-mode, and resource-limit options
 * @return Generated NVVM IR and executor metadata
 */
[[nodiscard]] compile_result compile(std::string_view pattern,
                                     operation_kind operation,
                                     std::optional<std::string> replacement = std::nullopt,
                                     compile_options const& options         = {});

}  // namespace regex_ir
