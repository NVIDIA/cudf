/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex_ir.hpp>

#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace regex_ir {

/**
 * @brief Byte range in the source regular expression
 */
struct source_span {
  std::size_t offset = 0;  ///< Zero-based byte offset
  std::size_t length = 0;  ///< Length in bytes
};

}  // namespace regex_ir

// compile options and operations

namespace regex_ir {

/**
 * @brief Requested regex operation and its operation-specific data
 */
struct operation {
  operation_kind kind     = operation_kind::MATCHES;  ///< Selected operation
  std::string replacement = "";                       ///< Replacement template for `REPLACE`
};

/**
 * @brief Pass controls for Instruction IR optimization
 */
struct optimization_options {
  bool remove_unreachable        : 1 = true;  ///< Remove blocks unreachable from the entry
  bool fold_epsilon_jumps        : 1 = true;  ///< Bypass trivial jump-only blocks
  bool fuse_literals             : 1 = true;  ///< Fuse linear singleton matches into literals
  bool strip_unobserved_captures : 1 = true;  ///< Remove capture writes not used by the result
  std::size_t literal_fusion_limit   = 64;    ///< Maximum code points in one fused literal
};

/**
 * @brief Sentinel upper bound representing an unbounded repetition
 */
inline constexpr std::uint32_t unbounded_repeat = std::numeric_limits<std::uint32_t>::max();

}  // namespace regex_ir

// automata IR

namespace regex_ir {

/**
 * @brief Numeric identifier of an Automata IR state
 */
using state_id = std::uint32_t;

/**
 * @brief Sentinel that does not identify an Automata IR state
 */
inline constexpr state_id invalid_state = static_cast<state_id>(-1);

/**
 * @brief Inclusive Unicode code-point interval
 */
struct codepoint_range {
  char32_t first = U'\0';  ///< First code point in the interval
  char32_t last  = U'\0';  ///< Last code point in the interval
};

/**
 * @brief Recognized character-class category retained alongside normalized ranges
 */
enum class predicate_class : std::uint8_t {
  NONE      = 0,  ///< Predicate has no recognized shorthand category
  DIGIT     = 1,  ///< Configured digit class
  NOT_DIGIT = 2,  ///< Negated configured digit class
  WORD      = 3,  ///< Configured word-character class
  NOT_WORD  = 4,  ///< Negated configured word-character class
  SPACE     = 5,  ///< Configured whitespace class
  NOT_SPACE = 6,  ///< Negated configured whitespace class
  ANY       = 7,  ///< Dot wildcard
};

/**
 * @brief Normalized predicate evaluated by a consuming automata state
 */
struct character_predicate {
  std::vector<codepoint_range> ranges = std::vector<codepoint_range>{};  ///< Sorted ranges
  predicate_class recognized          = predicate_class::NONE;  ///< Original shorthand category
  bool negated          : 1           = false;                  ///< Invert range membership
  bool matches_newline  : 1           = true;  ///< Whether dot accepts configured line terminators
  bool extended_newline : 1 = false;  ///< Whether dot uses the extended line-terminator set

  /**
   * @brief Check whether a code point satisfies this predicate
   *
   * @param value Code point to test
   * @return true if `value` satisfies this predicate
   */
  [[nodiscard]] bool matches(char32_t value) const noexcept;

  /**
   * @brief Check whether this predicate matches exactly one code point
   *
   * @return true if this predicate is a non-negated singleton range
   */
  [[nodiscard]] bool is_singleton() const noexcept;

  /**
   * @brief Return the code point in a singleton predicate
   *
   * @return Singleton code point, or `U'\0'` when this predicate is not a singleton
   */
  [[nodiscard]] char32_t singleton() const noexcept;
};

/**
 * @brief Select a byte anchor using only the literal's own repeated structure
 *
 * For multi-byte literals, the returned byte completes the least-repeated adjacent byte pair.
 * This avoids assumptions about the input language while making the preceding byte an inexpensive
 * secondary guard. Single-byte literals return offset zero.
 *
 * @param literal Non-empty encoded literal
 * @return Byte offset of the selected anchor
 */
[[nodiscard]] inline std::size_t literal_anchor(std::string_view literal)
{
  if (literal.size() < 2U) return 0U;
  auto best       = std::size_t{0};
  auto best_count = literal.size();
  for (std::size_t candidate = 0; candidate + 1U < literal.size(); ++candidate) {
    auto count = std::size_t{0};
    for (std::size_t position = 0; position + 1U < literal.size(); ++position) {
      count += literal[position] == literal[candidate] &&
               literal[position + 1U] == literal[candidate + 1U];
    }
    if (count < best_count) {
      best       = candidate;
      best_count = count;
    }
  }
  return best + 1U;
}

/**
 * @brief Zero-width assertion evaluated at the current input position
 */
enum class assertion_kind : std::uint8_t {
  BEGIN_INPUT       = 0,  ///< Absolute beginning of input
  END_INPUT         = 1,  ///< Absolute end of input
  WORD_BOUNDARY     = 2,  ///< Transition between configured word and non-word characters
  NOT_WORD_BOUNDARY = 3,  ///< Position that is not a configured word boundary
  BEGIN_LINE        = 4,  ///< Beginning of input or a configured multiline boundary
  END_LINE          = 5,  ///< End of input or a configured line boundary
};

/**
 * @brief Capture-boundary update performed by a tagged automata state
 */
enum class capture_action : std::uint8_t {
  BEGIN = 0,  ///< Record the beginning of a capture
  END   = 1,  ///< Record the end of a capture
};

/**
 * @brief Operation performed by an Automata IR state
 */
enum class automata_state_kind : std::uint8_t {
  JUMP      = 0,  ///< Epsilon transition to one successor
  BRANCH    = 1,  ///< Ordered epsilon transition to multiple successors
  CONSUME   = 2,  ///< Match and consume one character
  ASSERTION = 3,  ///< Evaluate a zero-width assertion
  CAPTURE   = 4,  ///< Record a capture boundary
  ACCEPT    = 5,  ///< Accept the current match
};

/**
 * @brief Ordered transition between Automata IR states
 */
struct automata_edge {
  state_id target        = invalid_state;  ///< Destination state
  std::uint32_t priority = 0;              ///< Lower values are attempted first
};

/**
 * @brief One node in an ordered Thompson automaton
 */
struct automata_state {
  state_id id                      = invalid_state;                 ///< Dense state identifier
  automata_state_kind kind         = automata_state_kind::JUMP;     ///< State operation
  source_span source               = source_span{};                 ///< Related pattern range
  std::vector<automata_edge> edges = std::vector<automata_edge>{};  ///< Ordered outgoing edges
  character_predicate predicate    = character_predicate{};         ///< Predicate for `CONSUME`
  assertion_kind assertion         = assertion_kind::BEGIN_INPUT;   ///< Assertion for `ASSERTION`
  capture_action capture           = capture_action::BEGIN;         ///< Action for `CAPTURE`
  std::uint32_t capture_index      = 0;                             ///< Capture index for `CAPTURE`
};

/**
 * @brief Ordered Thompson automaton produced from a regular expression
 */
struct automata_ir {
  std::string pattern                = "";                             ///< Original UTF-8 pattern
  compile_options options            = compile_options{};              ///< Parse options
  std::vector<automata_state> states = std::vector<automata_state>{};  ///< Dense states
  state_id entry                     = invalid_state;                  ///< Entry state
  state_id accept                    = invalid_state;                  ///< Unique accepting state
  std::uint32_t capture_count        = 0;      ///< Number of explicit capture groups
  bool has_alternation     : 1       = false;  ///< Whether the expression contains alternation
  bool has_lazy_quantifier : 1       = false;  ///< Whether any repetition prefers its exit edge
};

}  // namespace regex_ir

// instruction IR

namespace regex_ir {

/**
 * @brief Numeric identifier of an Instruction IR block
 */
using block_id = std::uint32_t;

/**
 * @brief Sentinel that does not identify an Instruction IR block
 */
inline constexpr block_id invalid_block = static_cast<block_id>(-1);

/**
 * @brief Require a number of characters to remain at the cursor
 */
struct can_peek {
  std::uint32_t characters = 1;  ///< Required number of logical characters
};

/**
 * @brief Decode the character at the current cursor
 */
struct read_character {};

/**
 * @brief Test the current character against a predicate
 */
struct match_character {
  character_predicate predicate = character_predicate{};  ///< Predicate that must match
};

/**
 * @brief Test consecutive characters against a fixed literal
 */
struct match_literal {
  std::u32string value = std::u32string{};  ///< Literal Unicode code points
};

/**
 * @brief Advance the input cursor by logical characters
 */
struct advance_cursor {
  std::uint32_t characters = 1;  ///< Number of characters to advance
};

/**
 * @brief Evaluate a zero-width assertion at the cursor
 */
struct test_assertion {
  assertion_kind kind = assertion_kind::BEGIN_INPUT;  ///< Assertion to evaluate
};

/**
 * @brief Record one boundary of a capture group
 */
struct write_capture {
  capture_action action       = capture_action::BEGIN;  ///< Boundary to record
  std::uint32_t capture_index = 0;                      ///< Capture group index
};

/**
 * @brief Accept the current candidate match
 */
struct emit_accept {};

/**
 * @brief Instruction variant used inside an Instruction IR block
 */
using instruction = std::variant<can_peek,
                                 read_character,
                                 match_character,
                                 match_literal,
                                 advance_cursor,
                                 test_assertion,
                                 write_capture,
                                 emit_accept>;

/**
 * @brief Ordered control-flow edge between Instruction IR blocks
 */
struct block_edge {
  block_id target        = invalid_block;  ///< Destination block
  std::uint32_t priority = 0;              ///< Lower values are attempted first
};

/**
 * @brief Straight-line instruction sequence with ordered successors
 */
struct instruction_block {
  block_id id                           = invalid_block;               ///< Dense block identifier
  source_span source                    = source_span{};               ///< Related pattern range
  std::vector<instruction> instructions = std::vector<instruction>{};  ///< Instructions
  std::vector<block_edge> successors    = std::vector<block_edge>{};   ///< Ordered successors
};

/**
 * @brief Parsed component of a replacement template
 */
struct replacement_token {
  /**
   * @brief Replacement token category
   */
  enum class kind : std::uint8_t {
    LITERAL = 0,  ///< Literal UTF-8 bytes
    CAPTURE = 1,  ///< Captured input span
  };

  kind type                   = kind::LITERAL;  ///< Token category
  std::string literal         = "";             ///< Bytes for a literal token
  std::uint32_t capture_index = 0;              ///< Capture index for a capture token
};

/**
 * @brief Logical result produced when Instruction IR is executed
 */
enum class result_shape : std::uint8_t {
  BOOLEAN      = 0,  ///< Boolean match result
  MATCH_SPAN   = 1,  ///< First match span
  MATCH_COUNT  = 2,  ///< Number of matches
  CAPTURES     = 3,  ///< Capture spans
  REPLACEMENT  = 4,  ///< Replaced string
  SPLIT_FIELDS = 5,  ///< Fields split around matches
};

/**
 * @brief Operation-specific execution policy encoded in Instruction IR
 */
struct operation_control {
  bool scan_input  : 1 = false;                  ///< Try candidates after input position zero
  bool require_end : 1 = false;                  ///< Require acceptance at end of input
  result_shape result  = result_shape::BOOLEAN;  ///< Produced result shape
};

/**
 * @brief Typed operation-specialized control-flow IR
 */
struct instruction_ir {
  std::string pattern                        = "";                   ///< Original UTF-8 pattern
  compile_options options                    = compile_options{};    ///< Compilation options
  operation selected_operation               = operation{};          ///< Encoded operation
  operation_control control                  = operation_control{};  ///< Execution policy
  std::vector<instruction_block> blocks      = std::vector<instruction_block>{};  ///< Dense blocks
  block_id entry                             = invalid_block;                     ///< Entry block
  block_id accept                            = invalid_block;  ///< Block containing acceptance
  std::uint32_t capture_count                = 0;              ///< Explicit capture count
  bool has_alternation     : 1               = false;  ///< Whether the expression has alternation
  bool has_lazy_quantifier : 1               = false;  ///< Whether repetition priority is lazy
  std::vector<replacement_token> replacement = std::vector<replacement_token>{};  ///< Template
};

/**
 * @brief Options controlling CUDA C++ source generation
 */
struct cuda_codegen_options {
  std::string symbol_prefix    = "regex_ir_sym";      ///< Prefix for internal symbols
  std::string execute_function = "regex_ir_execute";  ///< Public matcher function name
};

/** @brief Validate the operation-specialized instruction graph. */
void verify(instruction_ir const& ir);

/** @brief Apply instruction graph optimizations before executor analysis. */
instruction_ir optimize(instruction_ir ir, optimization_options const& options);

}  // namespace regex_ir
