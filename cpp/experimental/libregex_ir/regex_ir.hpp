/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
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
  bool case_insensitive : 1 = false;  ///< Enable case-insensitive matching
  bool multiline        : 1 = false;  ///< Make line anchors recognize internal line boundaries
  bool dot_all          : 1 = false;  ///< Allow dot to match configured newline characters
  bool ascii_classes    : 1 = true;   ///< Use ASCII semantics for shorthand character classes
  bool extended_newline : 1 = false;  ///< Recognize the extended Unicode newline set
  bool find_match_end_observable : 1 = true;  ///< Require FIND to produce its end offset
  character_mode characters          = character_mode::UTF8;  ///< Input character decoding mode
  compile_limits limits              = compile_limits{};      ///< Compilation resource limits
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
  RECURSIVE_THOMPSON,                ///< Ordered recursive Thompson-NFA executor
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
 * @brief Result of compiling a regular expression
 */
struct compile_result {
  std::string nvvm_ir;             ///< Textual operation-specialized NVVM IR
  std::uint32_t capture_count;     ///< Number of explicit capture groups
  executor_kind executor;          ///< Executor selected for the pattern and operation
  std::uint32_t executor_states;   ///< Number of states in the selected executor
  std::uint32_t alphabet_classes;  ///< Number of character classes in its alphabet partition
  std::optional<std::string> exact_ascii_literal;  ///< Exact ASCII literal recognized, if any
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

/**
 * @brief Helpers for constructing cuDF-compatible NVVM kernel modules
 *
 * These functions construct the operation-specific kernel portion of a module.
 * Use `assemble` to combine one with the matcher module returned by `compile`.
 */
namespace nvvm {

/**
 * @brief One literal or capture component of a replacement template
 */
struct replacement_piece {
  std::string literal;                  ///< Literal UTF-8 bytes emitted by this component
  std::optional<std::int32_t> capture;  ///< Capture index to emit, if present
};

/**
 * @brief Combine matcher and kernel NVVM modules into one module
 *
 * @param matcher Matcher module returned by `compile`
 * @param kernel Operation-specific kernel module
 * @return A complete textual NVVM module containing both inputs
 * @throw std::invalid_argument If `matcher` has no NVVM version metadata
 */
[[nodiscard]] std::string assemble(std::string matcher, std::string kernel);

/**
 * @brief Generate a fixed-width output kernel
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param operation Regex operation implemented by the kernel
 * @param kernel_name Exported kernel entry-point name
 * @return Textual NVVM IR for the kernel
 * @throw std::invalid_argument If `kernel_name` is not a valid, non-reserved identifier
 */
[[nodiscard]] std::string make_fixed_kernel(bool offset64,
                                            operation_kind operation,
                                            std::string_view kernel_name);

/**
 * @brief Generate a warp-per-row kernel for an exact ASCII literal contains operation
 *
 * @param offset64 Whether input string offsets use 64-bit integers
 * @param literal Non-empty ASCII literal to search for
 * @param kernel_name Exported kernel entry-point name
 * @return Textual NVVM IR for the kernel
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
 * @return Textual NVVM IR for the kernel
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
 * @return Textual NVVM IR for the sizing kernel
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
 * @return Textual NVVM IR for the emission kernel
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
 * @return Textual NVVM IR for the replacement kernel
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
 * @return Textual NVVM IR for the replacement kernel
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
 * @return Textual NVVM IR for the split sizing kernel
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
 * @return Textual NVVM IR for the split emission kernel
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
 * @return Textual NVVM IR for the sampling kernel
 */
[[nodiscard]] std::string make_span_cache_sample_kernel(bool offset64,
                                                        bool split,
                                                        std::int32_t capture_slots,
                                                        std::int32_t match_limit,
                                                        std::string_view kernel_name);

}  // namespace nvvm

}  // namespace regex_ir

#ifdef REGEX_IR_IMPLEMENTATION

// diagnostics

namespace regex_ir {

/**
 * @brief Byte range in the source regular expression
 */
struct source_span {
  std::size_t offset = 0;  ///< Zero-based byte offset
  std::size_t length = 0;  ///< Length in bytes
};

/**
 * @brief Stable categories for compile and verification diagnostics
 */
enum class diagnostic_code : std::uint8_t {
  UNEXPECTED_END          = 0,   ///< Pattern ended before a construct was complete
  UNEXPECTED_TOKEN        = 1,   ///< Token is not valid at its source position
  INVALID_ESCAPE          = 2,   ///< Escape sequence is malformed or unknown
  INVALID_CHARACTER_CLASS = 3,   ///< Character class is malformed
  INVALID_QUANTIFIER      = 4,   ///< Repetition syntax or bounds are invalid
  UNMATCHED_PARENTHESIS   = 5,   ///< Opening or closing parenthesis has no match
  UNSUPPORTED_FEATURE     = 6,   ///< Pattern uses syntax outside the supported subset
  RESOURCE_LIMIT          = 7,   ///< Configured compilation resource limit was exceeded
  INVALID_REPLACEMENT     = 8,   ///< Replacement template is malformed
  INVALID_AUTOMATA_IR     = 9,   ///< Automata IR invariant was violated
  INVALID_INSTRUCTION_IR  = 10,  ///< Instruction IR invariant was violated
};

/**
 * @brief Structured compiler or verifier diagnostic
 */
struct diagnostic {
  diagnostic_code code = diagnostic_code::UNEXPECTED_END;  ///< Diagnostic category
  source_span span     = source_span{};                    ///< Related source range
  std::string message  = "";                               ///< Human-readable explanation
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
  std::uint32_t capture_count        = 0;  ///< Number of explicit capture groups
  bool has_alternation : 1           = false;  ///< Whether the expression contains alternation
  bool has_lazy_quantifier : 1       = false;  ///< Whether any repetition prefers its exit edge
};

/**
 * @brief Validate an Automata IR graph
 *
 * @param ir Automata IR to validate
 * @return Diagnostics describing every detected invariant violation
 */
[[nodiscard]] std::vector<diagnostic> verify(automata_ir const& ir);

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
  bool scan_input          : 1 = false;  ///< Try candidates after input position zero
  bool require_end         : 1 = false;  ///< Require acceptance at end of input
  bool first_only          : 1 = true;   ///< Stop after the first accepted match
  bool advance_after_empty : 1 = true;   ///< Advance after an empty global match
  result_shape result          = result_shape::BOOLEAN;  ///< Produced result shape
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
  bool has_alternation : 1                   = false;  ///< Whether the expression has alternation
  bool has_lazy_quantifier : 1               = false;  ///< Whether repetition priority is lazy
  std::vector<replacement_token> replacement = std::vector<replacement_token>{};  ///< Template
};

/**
 * @brief Options controlling CUDA-oriented NVVM IR generation
 */
struct nvvm_ir_codegen_options {
  std::string symbol_prefix    = "regex_ir_sym";      ///< Prefix for internal symbols
  std::string execute_function = "regex_ir_execute";  ///< Public matcher function name
};

/**
 * @brief Validate an Instruction IR graph
 *
 * @param ir Instruction IR to validate
 * @return Diagnostics describing every detected invariant violation
 */
[[nodiscard]] std::vector<diagnostic> verify(instruction_ir const& ir);

/**
 * @brief Generate operation-specialized NVVM IR for a regex operation
 *
 * the returned module is self-contained device code with a public function named by
 * `options.execute_function` and prefixed internal symbols. NVVM IR uses LLVM syntax but follows
 * NVIDIA's stricter NVVM target contract. Assertion-free contains and matches graphs are
 * determinized into a bounded Unicode-class transition table; other valid boolean graphs use the
 * ordered fallback executor. The generated ABI is selected by the operation:
 * exact one-byte ASCII global operations use a direct byte scan, and non-scanning deterministic
 * machines reject immediately after entering their dead state.
 *
 * - contains and matches: `i1(i8*, i64)`;
 * - find: `i1(i8*, i64, i64*)`, with one begin/end pair in the final argument. When
 *   `compile_options::find_match_end_observable` is false, an input-anchored FIND may leave the
 *   end element unwritten;
 * - find-all: `i1(i8*, i64, i64, i64*)`, with a search byte offset followed by one whole-match
 *   begin/end pair;
 * - count: `i64(i8*, i64)`;
 * - extract: `i1(i8*, i64, i64, i64*)`, with a search byte followed by storage for the whole
 *   match and explicit capture pairs;
 * - replace: `i64(i8*, i64, i8*)`, returning the result byte count and writing the result when
 *   the final argument is non-null; and
 * - split: `i64(i8*, i64, i64*)`, returning the field count and writing field begin/end pairs
 *   when the final argument is non-null.
 *
 * passing a null output to replace or split performs a sizing pass without materialization.
 * replacement output storage must not overlap the input range.
 *
 * @param ir Optimized operation-specialized Instruction IR to render
 * @param options Symbol names
 * @return Generated NVVM IR and executor metadata
 * @throw std::invalid_argument If the IR or a requested symbol name is invalid
 */
[[nodiscard]] compile_result generate_nvvm_ir(instruction_ir const& ir,
                                              nvvm_ir_codegen_options const& options = {});

}  // namespace regex_ir

// compiler API

namespace regex_ir {

/**
 * @brief Value-or-diagnostics return type used by compiler stages
 *
 * @tparam T Successful value type
 */
template <typename T>
struct result {
  std::optional<T> value              = std::nullopt;               ///< Compiled value
  std::vector<diagnostic> diagnostics = std::vector<diagnostic>{};  ///< Diagnostics

  /**
   * @brief Check whether this result contains a compiled value
   *
   * @return true if compilation succeeded and `value` is populated
   */
  [[nodiscard]] explicit operator bool() const noexcept { return value.has_value(); }
};

/**
 * @brief Result of compiling a pattern to Automata IR
 */
using automata_result = result<automata_ir>;

/**
 * @brief Result of lowering or compiling Instruction IR
 */
using instruction_result = result<instruction_ir>;

/**
 * @brief Parse a regex and construct ordered Thompson Automata IR
 *
 * @param pattern Regex pattern encoded as UTF-8 source bytes
 * @param options Syntax, character-mode, and resource-limit options
 * @return Automata IR on success, otherwise structured diagnostics
 */
[[nodiscard]] automata_result compile_automata(std::string_view pattern,
                                               compile_options const& options = {});

/**
 * @brief Lower Automata IR to operation-specialized Instruction IR
 *
 * @param automata Verified Automata IR
 * @param selected Operation whose control and result shape should be encoded
 * @return Unoptimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result lower(automata_ir const& automata, operation const& selected);

/**
 * @brief Optimize and verify Instruction IR
 *
 * @param ir Instruction IR to consume and optimize
 * @param options Optimization-pass configuration
 * @return Optimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result optimize(instruction_ir ir,
                                          optimization_options const& options = {});

/**
 * @brief Compile a regex directly to optimized operation-specialized Instruction IR
 *
 * this convenience function performs parsing, Thompson construction, lowering,
 * optimization, and verifier checks.
 *
 * @param pattern Regex pattern encoded as UTF-8 source bytes
 * @param selected Operation whose matching and result policy should be encoded
 * @param options Syntax, character-mode, and resource-limit options
 * @param optimization Optimization-pass configuration
 * @return Optimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result compile_instruction_ir(
  std::string_view pattern,
  operation const& selected,
  compile_options const& options           = {},
  optimization_options const& optimization = {});

}  // namespace regex_ir

#endif  // REGEX_IR_IMPLEMENTATION
