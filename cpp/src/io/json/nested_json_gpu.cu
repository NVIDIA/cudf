/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/fst/logical_stack.cuh"
#include "io/fst/lookup_tables.cuh"
#include "io/utilities/parsing_utils.cuh"
#include "nested_json.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <limits>
#include <stack>

// Debug print flag
#ifndef NJP_DEBUG_PRINT
// #define NJP_DEBUG_PRINT
#endif

namespace {

/**
 * @brief While parsing the token stream, we use a stack of tree_nodes to maintain all the
 * information about the data path that is relevant.
 */
struct tree_node {
  // The column that this node is associated with
  cudf::io::json::json_column* column;

  // The row offset that this node belongs to within the given column
  uint32_t row_index;

  // Selected child column
  // E.g., if this is a struct node, and we subsequently encountered the field name "a", then this
  // point's to the struct's "a" child column
  cudf::io::json::json_column* current_selected_col = nullptr;

  std::size_t num_children = 0;
};

/**
 * @brief Verifies that the JSON input can be handled without corrupted data due to offset
 * overflows.
 *
 * @param input_size The JSON inputs size in bytes
 */
void check_input_size(std::size_t input_size)
{
  // Transduce() writes symbol offsets that may be as large input_size-1
  CUDF_EXPECTS(input_size == 0 || (input_size - 1) <= std::numeric_limits<int32_t>::max(),
               "Given JSON input is too large");
}
}  // namespace

namespace cudf::io::json {

// FST to help fixing the stack context of characters that follow the first record on each JSON line
namespace fix_stack_of_excess_chars {

// Type used to represent the target state in the transition table
using StateT = char;

// Type used to represent a symbol group id
using SymbolGroupT = uint8_t;

/**
 * @brief Definition of the DFA's states.
 */
enum class dfa_states : StateT {
  // Before the first record on the JSON line
  BEFORE,
  // Within the first record on the JSON line
  WITHIN,
  // Excess data that follows the first record on the JSON line
  EXCESS,
  // Total number of states
  NUM_STATES
};

/**
 * @brief Definition of the symbol groups
 */
enum class dfa_symbol_group_id : SymbolGroupT {
  ROOT,              ///< Symbol for root stack context
  DELIMITER,         ///< Line delimiter symbol group
  OTHER,             ///< Symbol group that implicitly matches all other tokens
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

/**
 * @brief Function object to map (input_symbol,stack_context) tuples to a symbol group.
 */
struct SymbolPairToSymbolGroupId {
  SymbolT delimiter = '\n';
  CUDF_HOST_DEVICE SymbolGroupT operator()(cuda::std::tuple<SymbolT, StackSymbolT> symbol) const
  {
    auto const input_symbol = cuda::std::get<0>(symbol);
    auto const stack_symbol = cuda::std::get<1>(symbol);
    return static_cast<SymbolGroupT>(
      input_symbol == delimiter
        ? dfa_symbol_group_id::DELIMITER
        : (stack_symbol == '_' ? dfa_symbol_group_id::ROOT : dfa_symbol_group_id::OTHER));
  }
};

/**
 * @brief Translation function object that fixes the stack context of excess data that follows after
 * the first JSON record on each line.
 */
struct TransduceInputOp {
  template <typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE StackSymbolT operator()(StateT const state_id,
                                                     SymbolGroupT const match_id,
                                                     RelativeOffsetT const relative_offset,
                                                     SymbolT const read_symbol) const
  {
    if (state_id == static_cast<StateT>(dfa_states::EXCESS)) { return '_'; }
    return cuda::std::get<1>(read_symbol);
  }

  template <typename SymbolT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                SymbolT const read_symbol) const
  {
    constexpr int32_t single_output_item = 1;
    return single_output_item;
  }
};

// Aliases for readability of the transition table
constexpr auto TT_BEFORE = dfa_states::BEFORE;
constexpr auto TT_INSIDE = dfa_states::WITHIN;
constexpr auto TT_EXCESS = dfa_states::EXCESS;

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> constexpr transition_table{
  {/* IN_STATE            ROOT      NEWLINE     OTHER */
   /* TT_BEFORE    */ {{TT_BEFORE, TT_BEFORE, TT_INSIDE}},
   /* TT_INSIDE    */ {{TT_EXCESS, TT_BEFORE, TT_INSIDE}},
   /* TT_EXCESS    */ {{TT_EXCESS, TT_BEFORE, TT_EXCESS}}}};

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(dfa_states::BEFORE);
}  // namespace fix_stack_of_excess_chars

// FST to prune tokens of invalid lines for recovering JSON lines format
namespace token_filter {

// Type used to represent the target state in the transition table
using StateT = char;

// Type used to represent a symbol group id
using SymbolGroupT = uint8_t;

/**
 * @brief Definition of the DFA's states
 */
enum class dfa_states : StateT { VALID, INVALID, NUM_STATES };

// Aliases for readability of the transition table
constexpr auto TT_INV = dfa_states::INVALID;
constexpr auto TT_VLD = dfa_states::VALID;

/**
 * @brief Definition of the symbol groups
 */
enum class dfa_symbol_group_id : SymbolGroupT {
  ERROR,             ///< Error token symbol group
  DELIMITER,         ///< Record / line delimiter symbol group
  OTHER_SYMBOLS,     ///< Symbol group that implicitly matches all other tokens
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// Lookup table to map an input symbol (i.e., a token) to a symbol group
std::array<std::vector<PdaTokenT>, NUM_SYMBOL_GROUPS - 1> const symbol_groups{{
  {static_cast<PdaTokenT>(token_t::ErrorBegin)},  // Symbols mapping to ERROR
  {static_cast<PdaTokenT>(token_t::LineEnd)}      // Symbols mapping to DELIMITER
}};

/**
 * @brief Function object to map (token,token_index) tuples to a symbol group.
 */
struct UnwrapTokenFromSymbolOp {
  template <typename SymbolGroupLookupTableT>
  CUDF_HOST_DEVICE SymbolGroupT operator()(SymbolGroupLookupTableT const& sgid_lut,
                                           cuda::std::tuple<PdaTokenT, SymbolOffsetT> symbol) const
  {
    PdaTokenT const token_type = cuda::std::get<0>(symbol);
    return sgid_lut.lookup(token_type);
  }
};

/**
 * @brief Translation function object that discards line delimiter tokens and tokens belonging to
 * invalid lines.
 */
struct TransduceToken {
  template <typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE SymbolT operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                RelativeOffsetT const relative_offset,
                                                SymbolT const read_symbol) const
  {
    bool const is_end_of_invalid_line =
      (state_id == static_cast<StateT>(TT_INV) &&
       match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DELIMITER));

    if (is_end_of_invalid_line) {
      return relative_offset == 0 ? SymbolT{token_t::StructEnd, 0}
                                  : SymbolT{token_t::StructBegin, 0};
    } else {
      return read_symbol;
    }
  }

  template <typename SymbolT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                SymbolT const read_symbol) const
  {
    // Number of tokens emitted on invalid lines
    constexpr int32_t num_inv_tokens = 2;

    bool const is_delimiter = match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DELIMITER);

    // If state is either invalid or we're entering an invalid state, we discard tokens
    bool const is_part_of_invalid_line =
      (match_id != static_cast<SymbolGroupT>(dfa_symbol_group_id::ERROR) &&
       state_id == static_cast<StateT>(TT_VLD));

    // Indicates whether we transition from an invalid line to a potentially valid line
    bool const is_end_of_invalid_line = (state_id == static_cast<StateT>(TT_INV) && is_delimiter);

    int32_t const emit_count =
      is_end_of_invalid_line ? num_inv_tokens : (is_part_of_invalid_line && !is_delimiter ? 1 : 0);
    return emit_count;
  }
};

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const transition_table{
  {/* IN_STATE      ERROR   DELIM   OTHER */
   /* VALID    */ {{TT_INV, TT_VLD, TT_VLD}},
   /* INVALID  */ {{TT_INV, TT_VLD, TT_INV}}}};

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(TT_VLD);
}  // namespace token_filter

// JSON to stack operator DFA (Deterministic Finite Automata)
namespace to_stack_op {

// Type used to represent the target state in the transition table
using StateT = char;

/**
 * @brief Definition of the DFA's states
 */
enum class dfa_states : StateT {
  // The active state while outside of a string. When encountering an opening bracket or curly
  // brace, we push it onto the stack. When encountering a closing bracket or brace, we pop from the
  // stack.
  TT_OOS = 0U,

  // The active state while within a string (e.g., field name or a string value). We do not push or
  // pop from the stack while in this state.
  TT_STR,

  // The active state after encountering an escape symbol (e.g., '\'), while in the TT_STR state.
  TT_ESC,

  // Total number of states
  TT_NUM_STATES
};

// Aliases for readability of the transition table
constexpr auto TT_OOS = dfa_states::TT_OOS;
constexpr auto TT_STR = dfa_states::TT_STR;
constexpr auto TT_ESC = dfa_states::TT_ESC;

/**
 * @brief Definition of the symbol groups
 */
enum class dfa_symbol_group_id : uint8_t {
  OPENING_BRACE,     ///< Opening brace SG: {
  OPENING_BRACKET,   ///< Opening bracket SG: [
  CLOSING_BRACE,     ///< Closing brace SG: }
  CLOSING_BRACKET,   ///< Closing bracket SG: ]
  QUOTE_CHAR,        ///< Quote character SG: "
  ESCAPE_CHAR,       ///< Escape character SG: '\'
  DELIMITER_CHAR,    ///< Delimiter character SG
  OTHER_SYMBOLS,     ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(TT_OOS);

template <typename SymbolT>
auto get_sgid_lut(SymbolT delim)
{
  // The i-th string representing all the characters of a symbol group
  std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> symbol_groups{
    {{'{'}, {'['}, {'}'}, {']'}, {'"'}, {'\\'}, {delim}}};

  return symbol_groups;
}

auto get_transition_table(stack_behavior_t stack_behavior)
{
  // Transition table for the default JSON and JSON lines formats
  std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const transition_table{
    {/* IN_STATE          {       [       }       ]       "       \      \n    OTHER */
     /* TT_OOS    */ {{TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS, TT_OOS}},
     /* TT_STR    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_ESC, TT_STR, TT_STR}},
     /* TT_ESC    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR}}}};

  // Transition table for the JSON lines format that recovers from invalid JSON lines
  std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const
    resetting_transition_table{
      {/* IN_STATE          {       [       }       ]       "       \      \n    OTHER */
       /* TT_OOS    */ {{TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS, TT_OOS}},
       /* TT_STR    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_ESC, TT_OOS, TT_STR}},
       /* TT_ESC    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_STR}}}};

  // Transition table specialized on the choice of whether to reset on newlines
  return (stack_behavior == stack_behavior_t::ResetOnDelimiter) ? resetting_transition_table
                                                                : transition_table;
}

auto get_translation_table(stack_behavior_t stack_behavior)
{
  // Translation table for the default JSON and JSON lines formats
  std::array<std::array<std::vector<char>, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const
    translation_table{
      {/* IN_STATE         {      [      }      ]      "      \     <delim>    OTHER */
       /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {}, {}, {}, {}}},
       /* TT_STR    */ {{{}, {}, {}, {}, {}, {}, {}, {}}},
       /* TT_ESC    */ {{{}, {}, {}, {}, {}, {}, {}, {}}}}};

  // Translation table for the JSON lines format that recovers from invalid JSON lines
  std::array<std::array<std::vector<char>, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const
    resetting_translation_table{
      {/* IN_STATE         {      [      }      ]      "      \     <delim>    OTHER */
       /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {}, {}, {'\n'}, {}}},
       /* TT_STR    */ {{{}, {}, {}, {}, {}, {}, {'\n'}, {}}},
       /* TT_ESC    */ {{{}, {}, {}, {}, {}, {}, {'\n'}, {}}}}};

  // Translation table specialized on the choice of whether to reset on newlines
  return stack_behavior == stack_behavior_t::ResetOnDelimiter ? resetting_translation_table
                                                              : translation_table;
}

}  // namespace to_stack_op

// JSON tokenizer pushdown automaton
namespace tokenizer_pda {

// Type used to represent the target state in the transition table
using StateT = char;

/**
 * @brief Symbol groups for the input alphabet for the pushdown automaton
 */
enum class symbol_group_id : PdaSymbolGroupIdT {
  /// Opening brace
  OPENING_BRACE,
  /// Opening bracket
  OPENING_BRACKET,
  /// Closing brace
  CLOSING_BRACE,
  /// Closing bracket
  CLOSING_BRACKET,
  /// Quote
  QUOTE,
  /// Escape
  ESCAPE,
  /// Comma
  COMMA,
  /// Colon
  COLON,
  /// Whitespace
  WHITE_SPACE,
  /// Linebreak
  LINE_BREAK,
  /// Other (any input symbol not assigned to one of the above symbol groups)
  OTHER,
  /// Total number of symbol groups amongst which to differentiate
  NUM_PDA_INPUT_SGS
};

/**
 * @brief Symbols in the stack alphabet
 */
enum class stack_symbol_group_id : PdaStackSymbolGroupIdT {
  /// Symbol representing that we're at the JSON root (nesting level 0)
  STACK_ROOT,

  /// Symbol representing that we're currently within a list object
  STACK_LIST,

  /// Symbol representing that we're currently within a struct object
  STACK_STRUCT,

  /// Total number of symbols in the stack alphabet
  NUM_STACK_SGS
};
constexpr auto NUM_PDA_INPUT_SGS =
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::NUM_PDA_INPUT_SGS);
constexpr auto NUM_STACK_SGS =
  static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::NUM_STACK_SGS);

/// Total number of symbol groups to differentiate amongst (stack alphabet * input alphabet)
constexpr PdaSymbolGroupIdT NUM_PDA_SGIDS = NUM_PDA_INPUT_SGS * NUM_STACK_SGS;

/// Mapping a input symbol to the symbol group id
static __constant__ PdaSymbolGroupIdT tos_sg_to_pda_sgid[] = {
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::LINE_BREAK),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::QUOTE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::COMMA),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::COLON),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OPENING_BRACKET),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::ESCAPE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::CLOSING_BRACKET),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OPENING_BRACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::CLOSING_BRACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER)};

/**
 * @brief Maps a (top-of-stack symbol, input symbol)-pair to a symbol group id of the deterministic
 * visibly pushdown automaton (DVPA)
 */
struct PdaSymbolToSymbolGroupId {
  SymbolT delimiter = '\n';
  template <typename SymbolT, typename StackSymbolT>
  __device__ __forceinline__ PdaSymbolGroupIdT
  operator()(cuda::std::tuple<SymbolT, StackSymbolT> symbol_pair) const
  {
    // The symbol read from the input
    auto symbol = cuda::std::get<0>(symbol_pair);

    // The stack symbol (i.e., what is on top of the stack at the time the input symbol was read)
    // I.e., whether we're reading in something within a struct, a list, or the JSON root
    auto stack_symbol = cuda::std::get<1>(symbol_pair);

    // The stack symbol offset: '_' is the root group (0), '[' is the list group (1), '{' is the
    // struct group (2)
    int32_t stack_idx = static_cast<PdaStackSymbolGroupIdT>(
      (stack_symbol == '_') ? stack_symbol_group_id::STACK_ROOT
                            : ((stack_symbol == '[') ? stack_symbol_group_id::STACK_LIST
                                                     : stack_symbol_group_id::STACK_STRUCT));

    // The relative symbol group id of the current input symbol
    constexpr auto pda_sgid_lookup_size =
      static_cast<int32_t>(sizeof(tos_sg_to_pda_sgid) / sizeof(tos_sg_to_pda_sgid[0]));
    // We map the delimiter character to LINE_BREAK symbol group id, and the newline character
    // to WHITE_SPACE. Note that delimiter cannot be any of opening(closing) brace, bracket, quote,
    // escape, comma, colon or whitespace characters.
    auto constexpr newline    = '\n';
    auto constexpr whitespace = ' ';
    // Cast to unsigned char first so high-bit bytes (>= 0x80) are not sign-extended to negative
    // int32_t values, which would underflow the min() clamp used as the lookup index.
    auto const symbol_position =
      symbol == delimiter
        ? static_cast<int32_t>(newline)
        : (symbol == newline ? static_cast<int32_t>(whitespace)
                             : static_cast<int32_t>(static_cast<unsigned char>(symbol)));
    PdaSymbolGroupIdT symbol_gid =
      tos_sg_to_pda_sgid[min(symbol_position, pda_sgid_lookup_size - 1)];
    return stack_idx * static_cast<PdaSymbolGroupIdT>(symbol_group_id::NUM_PDA_INPUT_SGS) +
           symbol_gid;
  }
};

// The states defined by the pushdown automaton
enum class pda_state_t : StateT {
  // Beginning of value
  PD_BOV,
  // Beginning of array
  PD_BOA,
  // Literal or number
  PD_LON,
  // String
  PD_STR,
  // After escape char when within string
  PD_SCE,
  // After having parsed a value
  PD_PVL,
  // Before the next field name
  PD_BFN,
  // Field name
  PD_FLN,
  // After escape char when within field name
  PD_FNE,
  // After a field name inside a struct
  PD_PFN,
  // Error state (trap state)
  PD_ERR,
  // Total number of PDA states
  PD_NUM_STATES
};

enum class json_format_cfg_t {
  // Format describing regular JSON
  JSON,

  // Format describing permissive newline-delimited JSON
  // I.e., newline characters are only treteated as delimiters at the root stack level
  // E.g., this is treated as a single record:
  // {"a":
  //  123}
  JSON_LINES,

  // Format describing strict newline-delimited JSON
  // I.e., All newlines are delimiting a record, independent of the context they appear in
  JSON_LINES_STRICT,

  // Transition table for parsing newline-delimited JSON that recovers from invalid JSON lines
  // This format also follows `JSON_LINES_STRICT` behaviour
  JSON_LINES_RECOVER

};

// Aliases for readability of the transition table
constexpr auto PD_BOV = pda_state_t::PD_BOV;
constexpr auto PD_BOA = pda_state_t::PD_BOA;
constexpr auto PD_LON = pda_state_t::PD_LON;
constexpr auto PD_STR = pda_state_t::PD_STR;
constexpr auto PD_SCE = pda_state_t::PD_SCE;
constexpr auto PD_PVL = pda_state_t::PD_PVL;
constexpr auto PD_BFN = pda_state_t::PD_BFN;
constexpr auto PD_FLN = pda_state_t::PD_FLN;
constexpr auto PD_FNE = pda_state_t::PD_FNE;
constexpr auto PD_PFN = pda_state_t::PD_PFN;
constexpr auto PD_ERR = pda_state_t::PD_ERR;

constexpr auto PD_NUM_STATES = static_cast<StateT>(pda_state_t::PD_NUM_STATES);

// The starting state of the pushdown automaton
constexpr auto start_state = static_cast<StateT>(pda_state_t::PD_BOV);

/**
 * @brief Getting the transition table
 */
auto get_transition_table(json_format_cfg_t format)
{
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_ROOT) == 0);
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_LIST) == 1);
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_STRUCT) == 2);

  std::array<std::array<pda_state_t, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tt;

  if (format == json_format_cfg_t::JSON || format == json_format_cfg_t::JSON_LINES) {
    // In case of newline-delimited JSON, multiple newlines are ignored, similar to whitespace.
    // Thas is, empty lines are ignored
    // PD_ANL describes the target state after a new line on an empty stack (JSON root level)
    auto const PD_ANL = (format == json_format_cfg_t::JSON) ? PD_PVL : PD_BOV;

    // First row:  empty stack         ("root" level of the JSON)
    // Second row: '[' on top of stack (we're parsing a list value)
    // Third row:  '{' on top of stack (we're parsing a struct value)
    //  {       [       }       ]       "       \       ,       :     space   newline other
    pda_tt[static_cast<StateT>(pda_state_t::PD_BOV)] = {
      PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
      PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
      PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON};
    pda_tt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOA, PD_LON,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOA, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_LON)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_PVL, PD_LON,
      PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_PVL, PD_LON,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_PVL, PD_LON};
    pda_tt[static_cast<StateT>(pda_state_t::PD_STR)] = {
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_SCE)] = {
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ANL, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_PVL, PD_ERR,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_PVL, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_BFN, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_FLN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_PFN, PD_FNE, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
    pda_tt[static_cast<StateT>(pda_state_t::PD_FNE)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
    pda_tt[static_cast<StateT>(pda_state_t::PD_PFN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_PFN, PD_PFN, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_ERR)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR};
  }
  // Transition table for strict JSON lines (including recovery)
  // Newlines are treated as record delimiters
  else {
    // In case of newline-delimited JSON, multiple newlines are ignored, similar to whitespace.
    // Thas is, empty lines are ignored
    // PD_ANL describes the target state after a new line after encountering error state
    auto const PD_ANL = (format == json_format_cfg_t::JSON_LINES_RECOVER) ? PD_BOV : PD_ERR;

    // Target state after having parsed the first JSON value on a JSON line
    // Spark has the special need to ignore everything that comes after the first JSON object
    // on a JSON line instead of marking those as invalid
    auto const PD_AFS = (format == json_format_cfg_t::JSON_LINES_RECOVER) ? PD_PVL : PD_ERR;

    // First row:  empty stack         ("root" level of the JSON)
    // Second row: '[' on top of stack (we're parsing a list value)
    // Third row:  '{' on top of stack (we're parsing a struct value)
    //  {       [       }       ]       "       \       ,       :     space   newline other
    pda_tt[static_cast<StateT>(pda_state_t::PD_BOV)] = {
      PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
      PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
      PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON};
    pda_tt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOV, PD_LON,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOV, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_LON)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_BOV, PD_LON,
      PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_BOV, PD_LON,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_BOV, PD_LON};
    pda_tt[static_cast<StateT>(pda_state_t::PD_STR)] = {
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_SCE)] = {
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR,
      PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_BOV, PD_STR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
      PD_AFS, PD_AFS, PD_AFS, PD_AFS, PD_AFS, PD_AFS, PD_AFS, PD_AFS, PD_PVL, PD_BOV, PD_AFS,
      PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_BOV, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_BOV, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_FLN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_PFN, PD_FNE, PD_FLN, PD_FLN, PD_FLN, PD_BOV, PD_FLN};
    pda_tt[static_cast<StateT>(pda_state_t::PD_FNE)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_BOV, PD_FLN};
    pda_tt[static_cast<StateT>(pda_state_t::PD_PFN)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_PFN, PD_BOV, PD_ERR};
    pda_tt[static_cast<StateT>(pda_state_t::PD_ERR)] = {
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ANL, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ANL, PD_ERR,
      PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ANL, PD_ERR};
  }
  return pda_tt;
}

/**
 * @brief Getting the translation table
 * @param recover_from_error Whether or not the tokenizer should recover from invalid lines. If
 * `recover_from_error` is true, invalid JSON lines end with the token sequence (`ErrorBegin`,
 * `LineEn`) and incomplete JSON lines (e.g., `{"a":123\n`) are treated as invalid lines.
 */
auto get_translation_table(bool recover_from_error)
{
  constexpr auto StructBegin       = token_t::StructBegin;
  constexpr auto StructEnd         = token_t::StructEnd;
  constexpr auto ListBegin         = token_t::ListBegin;
  constexpr auto ListEnd           = token_t::ListEnd;
  constexpr auto StructMemberBegin = token_t::StructMemberBegin;
  constexpr auto StructMemberEnd   = token_t::StructMemberEnd;
  constexpr auto FieldNameBegin    = token_t::FieldNameBegin;
  constexpr auto FieldNameEnd      = token_t::FieldNameEnd;
  constexpr auto StringBegin       = token_t::StringBegin;
  constexpr auto StringEnd         = token_t::StringEnd;
  constexpr auto ValueBegin        = token_t::ValueBegin;
  constexpr auto ValueEnd          = token_t::ValueEnd;
  constexpr auto ErrorBegin        = token_t::ErrorBegin;

  /**
   * @brief Instead of specifying the verbose translation tables twice (i.e., once when
   * `recover_from_error` is true and once when it is false), we use `nl_tokens` to specialize the
   * translation table where it differs depending on the `recover_from_error` option. If and only if
   * `recover_from_error` is true, `recovering_tokens` are returned along with a token_t::LineEnd
   * token, otherwise `regular_tokens` is returned.
   */
  auto nl_tokens = [recover_from_error](std::vector<char> regular_tokens,
                                        std::vector<char> recovering_tokens) {
    if (recover_from_error) {
      recovering_tokens.push_back(token_t::LineEnd);
      return recovering_tokens;
    }
    return regular_tokens;
  };

  /**
   * @brief Helper function that returns `recovering_tokens` if `recover_from_error` is true and
   * returns `regular_tokens` otherwise. This is used to ignore excess characters after the first
   * value in the case of JSON lines that recover from invalid lines, as Spark ignores any excess
   * characters that follow the first record on a JSON line.
   */
  auto alt_tokens = [recover_from_error](std::vector<char> regular_tokens,
                                         std::vector<char> recovering_tokens) {
    if (recover_from_error) { return recovering_tokens; }
    return regular_tokens;
  };

  std::array<std::array<std::vector<char>, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tlt;
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOV)] = {{                    /*ROOT*/
                                                        {StructBegin},      // OPENING_BRACE
                                                        {ListBegin},        // OPENING_BRACKET
                                                        {ErrorBegin},       // CLOSING_BRACE
                                                        {ErrorBegin},       // CLOSING_BRACKET
                                                        {StringBegin},      // QUOTE
                                                        {ErrorBegin},       // ESCAPE
                                                        {ErrorBegin},       // COMMA
                                                        {ErrorBegin},       // COLON
                                                        {},                 // WHITE_SPACE
                                                        nl_tokens({}, {}),  // LINE_BREAK
                                                        {ValueBegin},       // OTHER
                                                        /*LIST*/
                                                        {StructBegin},  // OPENING_BRACE
                                                        {ListBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},   // CLOSING_BRACE
                                                        {ListEnd},      // CLOSING_BRACKET
                                                        {StringBegin},  // QUOTE
                                                        {ErrorBegin},   // ESCAPE
                                                        {ErrorBegin},   // COMMA
                                                        {ErrorBegin},   // COLON
                                                        {},             // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {ValueBegin},                 // OTHER
                                                        /*STRUCT*/
                                                        {StructBegin},  // OPENING_BRACE
                                                        {ListBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},   // CLOSING_BRACE
                                                        {ErrorBegin},   // CLOSING_BRACKET
                                                        {StringBegin},  // QUOTE
                                                        {ErrorBegin},   // ESCAPE
                                                        {ErrorBegin},   // COMMA
                                                        {ErrorBegin},   // COLON
                                                        {},             // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {ValueBegin}}};               // OTHER
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
    {                                        /*ROOT*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*LIST*/
     {StructBegin},                // OPENING_BRACE
     {ListBegin},                  // OPENING_BRACKET
     {ErrorBegin},                 // CLOSING_BRACE
     {ListEnd},                    // CLOSING_BRACKET
     {StringBegin},                // QUOTE
     {ErrorBegin},                 // ESCAPE
     {ErrorBegin},                 // COMMA
     {ErrorBegin},                 // COLON
     {},                           // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
     {ValueBegin},                 // OTHER
     /*STRUCT*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {StructEnd},                          // CLOSING_BRACE
     {ErrorBegin},                         // CLOSING_BRACKET
     {StructMemberBegin, FieldNameBegin},  // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ErrorBegin},                         // COMMA
     {ErrorBegin},                         // COLON
     {},                                   // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),          // LINE_BREAK
     {ErrorBegin}}};                       // OTHER
  pda_tlt[static_cast<StateT>(pda_state_t::PD_LON)] = {
    {                                      /*ROOT*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {ErrorBegin},                         // CLOSING_BRACE
     {ErrorBegin},                         // CLOSING_BRACKET
     {ErrorBegin},                         // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ErrorBegin},                         // COMMA
     {ErrorBegin},                         // COLON
     {ValueEnd},                           // WHITE_SPACE
     nl_tokens({ValueEnd}, {ErrorBegin}),  // LINE_BREAK
     {},                                   // OTHER
     /*LIST*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {ErrorBegin},                         // CLOSING_BRACE
     {ValueEnd, ListEnd},                  // CLOSING_BRACKET
     {ErrorBegin},                         // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ValueEnd},                           // COMMA
     {ErrorBegin},                         // COLON
     {ValueEnd},                           // WHITE_SPACE
     nl_tokens({ValueEnd}, {ErrorBegin}),  // LINE_BREAK
     {},                                   // OTHER
     /*STRUCT*/
     {ErrorBegin},                            // OPENING_BRACE
     {ErrorBegin},                            // OPENING_BRACKET
     {ValueEnd, StructMemberEnd, StructEnd},  // CLOSING_BRACE
     {ErrorBegin},                            // CLOSING_BRACKET
     {ErrorBegin},                            // QUOTE
     {ErrorBegin},                            // ESCAPE
     {ValueEnd, StructMemberEnd},             // COMMA
     {ErrorBegin},                            // COLON
     {ValueEnd},                              // WHITE_SPACE
     nl_tokens({ValueEnd}, {ErrorBegin}),     // LINE_BREAK
     {}}};                                    // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_STR)] = {{              /*ROOT*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {},                           // OTHER
                                                        /*LIST*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {},                           // OTHER
                                                        /*STRUCT*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {}}};                         // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_SCE)] = {{     /*ROOT*/
                                                        {},  // OPENING_BRACE
                                                        {},  // OPENING_BRACKET
                                                        {},  // CLOSING_BRACE
                                                        {},  // CLOSING_BRACKET
                                                        {},  // QUOTE
                                                        {},  // ESCAPE
                                                        {},  // COMMA
                                                        {},  // COLON
                                                        {},  // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {},                           // OTHER
                                                        /*LIST*/
                                                        {},  // OPENING_BRACE
                                                        {},  // OPENING_BRACKET
                                                        {},  // CLOSING_BRACE
                                                        {},  // CLOSING_BRACKET
                                                        {},  // QUOTE
                                                        {},  // ESCAPE
                                                        {},  // COMMA
                                                        {},  // COLON
                                                        {},  // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {},                           // OTHER
                                                        /*STRUCT*/
                                                        {},  // OPENING_BRACE
                                                        {},  // OPENING_BRACKET
                                                        {},  // CLOSING_BRACE
                                                        {},  // CLOSING_BRACKET
                                                        {},  // QUOTE
                                                        {},  // ESCAPE
                                                        {},  // COMMA
                                                        {},  // COLON
                                                        {},  // WHITE_SPACE
                                                        nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
                                                        {}}};                         // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
    {                                 /*ROOT*/
     {alt_tokens({ErrorBegin}, {})},  // OPENING_BRACE
     {alt_tokens({ErrorBegin}, {})},  // OPENING_BRACKET
     {alt_tokens({ErrorBegin}, {})},  // CLOSING_BRACE
     {alt_tokens({ErrorBegin}, {})},  // CLOSING_BRACKET
     {alt_tokens({ErrorBegin}, {})},  // QUOTE
     {alt_tokens({ErrorBegin}, {})},  // ESCAPE
     {alt_tokens({ErrorBegin}, {})},  // COMMA
     {alt_tokens({ErrorBegin}, {})},  // COLON
     {},                              // WHITE_SPACE
     nl_tokens({}, {}),               // LINE_BREAK
     {alt_tokens({ErrorBegin}, {})},  // OTHER
     /*LIST*/
     {ErrorBegin},                 // OPENING_BRACE
     {ErrorBegin},                 // OPENING_BRACKET
     {ErrorBegin},                 // CLOSING_BRACE
     {ListEnd},                    // CLOSING_BRACKET
     {ErrorBegin},                 // QUOTE
     {ErrorBegin},                 // ESCAPE
     {},                           // COMMA
     {ErrorBegin},                 // COLON
     {},                           // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                 // OTHER
     /*STRUCT*/
     {ErrorBegin},                  // OPENING_BRACE
     {ErrorBegin},                  // OPENING_BRACKET
     {StructMemberEnd, StructEnd},  // CLOSING_BRACE
     {ErrorBegin},                  // CLOSING_BRACKET
     {ErrorBegin},                  // QUOTE
     {ErrorBegin},                  // ESCAPE
     {StructMemberEnd},             // COMMA
     {ErrorBegin},                  // COLON
     {},                            // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),   // LINE_BREAK
     {ErrorBegin}}};                // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
    {                                        /*ROOT*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*LIST*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*STRUCT*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {StructEnd},                          // CLOSING_BRACE
     {ErrorBegin},                         // CLOSING_BRACKET
     {StructMemberBegin, FieldNameBegin},  // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ErrorBegin},                         // COMMA
     {ErrorBegin},                         // COLON
     {},                                   // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),          // LINE_BREAK
     {ErrorBegin}}};                       // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_FLN)] = {
    {                                        /*ROOT*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*LIST*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*STRUCT*/
     {},                           // OPENING_BRACE
     {},                           // OPENING_BRACKET
     {},                           // CLOSING_BRACE
     {},                           // CLOSING_BRACKET
     {FieldNameEnd},               // QUOTE
     {},                           // ESCAPE
     {},                           // COMMA
     {},                           // COLON
     {},                           // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
     {}}};                         // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_FNE)] = {
    {                                        /*ROOT*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*LIST*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*STRUCT*/
     {},                           // OPENING_BRACE
     {},                           // OPENING_BRACKET
     {},                           // CLOSING_BRACE
     {},                           // CLOSING_BRACKET
     {},                           // QUOTE
     {},                           // ESCAPE
     {},                           // COMMA
     {},                           // COLON
     {},                           // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
     {}}};                         // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_PFN)] = {
    {                                        /*ROOT*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*LIST*/
     {ErrorBegin},                           // OPENING_BRACE
     {ErrorBegin},                           // OPENING_BRACKET
     {ErrorBegin},                           // CLOSING_BRACE
     {ErrorBegin},                           // CLOSING_BRACKET
     {ErrorBegin},                           // QUOTE
     {ErrorBegin},                           // ESCAPE
     {ErrorBegin},                           // COMMA
     {ErrorBegin},                           // COLON
     {ErrorBegin},                           // WHITE_SPACE
     nl_tokens({ErrorBegin}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin},                           // OTHER
     /*STRUCT*/
     {ErrorBegin},                 // OPENING_BRACE
     {ErrorBegin},                 // OPENING_BRACKET
     {ErrorBegin},                 // CLOSING_BRACE
     {ErrorBegin},                 // CLOSING_BRACKET
     {ErrorBegin},                 // QUOTE
     {ErrorBegin},                 // ESCAPE
     {ErrorBegin},                 // COMMA
     {},                           // COLON
     {},                           // WHITE_SPACE
     nl_tokens({}, {ErrorBegin}),  // LINE_BREAK
     {ErrorBegin}}};               // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_ERR)] = {{                    /*ROOT*/
                                                        {},                 // OPENING_BRACE
                                                        {},                 // OPENING_BRACKET
                                                        {},                 // CLOSING_BRACE
                                                        {},                 // CLOSING_BRACKET
                                                        {},                 // QUOTE
                                                        {},                 // ESCAPE
                                                        {},                 // COMMA
                                                        {},                 // COLON
                                                        {},                 // WHITE_SPACE
                                                        nl_tokens({}, {}),  // LINE_BREAK
                                                        {},                 // OTHER
                                                        /*LIST*/
                                                        {},                 // OPENING_BRACE
                                                        {},                 // OPENING_BRACKET
                                                        {},                 // CLOSING_BRACE
                                                        {},                 // CLOSING_BRACKET
                                                        {},                 // QUOTE
                                                        {},                 // ESCAPE
                                                        {},                 // COMMA
                                                        {},                 // COLON
                                                        {},                 // WHITE_SPACE
                                                        nl_tokens({}, {}),  // LINE_BREAK
                                                        {},                 // OTHER
                                                        /*STRUCT*/
                                                        {},                 // OPENING_BRACE
                                                        {},                 // OPENING_BRACKET
                                                        {},                 // CLOSING_BRACE
                                                        {},                 // CLOSING_BRACKET
                                                        {},                 // QUOTE
                                                        {},                 // ESCAPE
                                                        {},                 // COMMA
                                                        {},                 // COLON
                                                        {},                 // WHITE_SPACE
                                                        nl_tokens({}, {}),  // LINE_BREAK
                                                        {}}};               // OTHER
  return pda_tlt;
}

}  // namespace tokenizer_pda

/**
 * @brief Function object used to filter for brackets and braces that represent push and pop
 * operations
 */
struct JSONToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE fst::stack_op_type operator()(StackSymbolT const& stack_symbol) const
  {
    switch (stack_symbol) {
      case '{':
      case '[': return fst::stack_op_type::PUSH;
      case '}':
      case ']': return fst::stack_op_type::POP;
      default: return fst::stack_op_type::READ;
    }
  }
};

/**
 * @brief Function object used to filter for brackets and braces that represent push and pop
 * operations
 */
struct JSONWithRecoveryToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE fst::stack_op_type operator()(StackSymbolT const& stack_symbol) const
  {
    switch (stack_symbol) {
      case '{':
      case '[': return fst::stack_op_type::PUSH;
      case '}':
      case ']': return fst::stack_op_type::POP;
      case '\n': return fst::stack_op_type::RESET;
      default: return fst::stack_op_type::READ;
    }
  }
};

void json_column::null_fill(row_offset_t up_to_row_offset)
{
  // Fill all the rows up to up_to_row_offset with "empty"/null rows
  validity.resize(word_index(up_to_row_offset) + 1);
  std::fill_n(std::back_inserter(string_offsets),
              up_to_row_offset - string_offsets.size(),
              (string_offsets.size() > 0) ? string_offsets.back() : 0);
  std::fill_n(std::back_inserter(string_lengths), up_to_row_offset - string_lengths.size(), 0);
  std::fill_n(std::back_inserter(child_offsets),
              up_to_row_offset + 1 - child_offsets.size(),
              (child_offsets.size() > 0) ? child_offsets.back() : 0);
  current_offset = up_to_row_offset;
}

void json_column::level_child_cols_recursively(row_offset_t min_row_count)
{
  // Fill this columns with nulls up to the given row count
  null_fill(min_row_count);

  // If this is a struct column, we need to level all its child columns
  if (type == json_col_t::StructColumn) {
    for (auto& child_column : child_columns) {
      child_column.second.level_child_cols_recursively(min_row_count);
    }
  }
  // If this is a list column, we need to make sure that its child column levels its children
  else if (type == json_col_t::ListColumn) {
    auto it = std::begin(child_columns);
    // Make that child column fill its child columns up to its own row count
    if (it != std::end(child_columns)) {
      it->second.level_child_cols_recursively(it->second.current_offset);
    }
  }
};

void json_column::append_row(uint32_t row_index,
                             json_col_t row_type,
                             uint32_t string_offset,
                             uint32_t string_end,
                             uint32_t child_count)
{
  // If, thus far, the column's type couldn't be inferred, we infer it to the given type
  if (type == json_col_t::Unknown) {
    type = row_type;
  }
  // If, at some point within a column, we encounter a nested type (list or struct),
  // we change that column's type to that respective nested type and invalidate all previous rows
  else if (type == json_col_t::StringColumn &&
           (row_type == json_col_t::ListColumn || row_type == json_col_t::StructColumn)) {
    // Change the column type
    type = row_type;

    // Invalidate all previous entries, as they were _not_ of the nested type to which we just
    // converted
    std::fill_n(validity.begin(), validity.size(), 0);
    valid_count = 0U;
  }
  // If this is a nested column but we're trying to insert either (a) a list node into a struct
  // column or (b) a struct node into a list column, we fail
  CUDF_EXPECTS(not((type == json_col_t::ListColumn and row_type == json_col_t::StructColumn) or
                   (type == json_col_t::StructColumn and row_type == json_col_t::ListColumn)),
               "A mix of lists and structs within the same column is not supported");

  // We shouldn't run into this, as we shouldn't be asked to append an "unknown" row type
  CUDF_EXPECTS(type != json_col_t::Unknown, "Encountered invalid JSON token sequence");

  // Fill all the omitted rows with "empty"/null rows (if needed)
  null_fill(row_index);

  // Table listing what we intend to use for a given column type and row type combination
  // col type | row type  => {valid, FAIL, null}
  // -----------------------------------------------
  // List     | List      => valid
  // List     | Struct    => FAIL
  // List     | String    => null
  // Struct   | List      => FAIL
  // Struct   | Struct    => valid
  // Struct   | String    => null
  // String   | List      => valid (we switch col type to list, null'ing all previous rows)
  // String   | Struct    => valid (we switch col type to list, null'ing all previous rows)
  // String   | String    => valid
  bool const is_valid = (type == row_type);
  if (static_cast<size_type>(validity.size()) < word_index(current_offset)) validity.push_back({});
  if (is_valid) { set_bit_unsafe(&validity.back(), intra_word_index(current_offset)); }
  valid_count += (is_valid) ? 1U : 0U;
  string_offsets.push_back(string_offset);
  string_lengths.push_back(string_end - string_offset);
  child_offsets.push_back((child_offsets.size() > 0) ? child_offsets.back() + child_count : 0);
  current_offset++;
};

namespace detail {

void get_stack_context(device_span<SymbolT const> json_in,
                       SymbolT* d_top_of_stack,
                       stack_behavior_t stack_behavior,
                       SymbolT delimiter,
                       rmm::cuda_stream_view stream)
{
  check_input_size(json_in.size());

  // Range of encapsulating function that comprises:
  // -> DFA simulation for filtering out brackets and braces inside of quotes
  // -> Logical stack to infer the stack context
  CUDF_FUNC_RANGE();

  // Symbol representing the JSON-root (i.e., we're at nesting level '0')
  constexpr StackSymbolT root_symbol = '_';
  // This can be any stack symbol from the stack alphabet that does not push onto stack
  constexpr StackSymbolT read_symbol = 'x';

  // Number of stack operations in the input (i.e., number of '{', '}', '[', ']' outside of quotes)
  cudf::detail::device_scalar<SymbolOffsetT> d_num_stack_ops(
    stream, cudf::get_current_device_resource_ref());

  // Prepare finite-state transducer that only selects '{', '}', '[', ']' outside of quotes
  constexpr auto max_translation_table_size =
    to_stack_op::NUM_SYMBOL_GROUPS * to_stack_op::TT_NUM_STATES;

  static constexpr auto min_translated_out = 0;
  static constexpr auto max_translated_out = 1;
  auto json_to_stack_ops_fst               = fst::detail::make_fst(
    fst::detail::make_symbol_group_lut(to_stack_op::get_sgid_lut(delimiter)),
    fst::detail::make_transition_table(to_stack_op::get_transition_table(stack_behavior)),
    fst::detail::
      make_translation_table<max_translation_table_size, min_translated_out, max_translated_out>(
        to_stack_op::get_translation_table(stack_behavior)),
    stream);

  // "Search" for relevant occurrence of brackets and braces that indicate the beginning/end
  // of structs/lists
  // Run FST to estimate the sizes of translated buffers
  json_to_stack_ops_fst.Transduce(json_in.begin(),
                                  static_cast<SymbolOffsetT>(json_in.size()),
                                  cuda::make_discard_iterator(),
                                  cuda::make_discard_iterator(),
                                  d_num_stack_ops.data(),
                                  to_stack_op::start_state,
                                  stream);

  // Copy back to actual number of stack operations
  auto num_stack_ops = d_num_stack_ops.value(stream);
  // Sequence of stack symbols and their position in the original input (sparse representation)
  rmm::device_uvector<StackSymbolT> stack_ops{num_stack_ops, stream};
  rmm::device_uvector<SymbolOffsetT> stack_op_indices{num_stack_ops, stream};

  // Run bracket-brace FST to retrieve starting positions of structs and lists
  json_to_stack_ops_fst.Transduce(json_in.begin(),
                                  static_cast<SymbolOffsetT>(json_in.size()),
                                  stack_ops.data(),
                                  stack_op_indices.data(),
                                  cuda::make_discard_iterator(),
                                  to_stack_op::start_state,
                                  stream);

  // Stack operations with indices are converted to top of the stack for each character in the input
  if (stack_behavior == stack_behavior_t::ResetOnDelimiter) {
    fst::sparse_stack_op_to_top_of_stack<fst::stack_op_support::WITH_RESET_SUPPORT, StackLevelT>(
      stack_ops.data(),
      device_span<SymbolOffsetT>{stack_op_indices.data(), num_stack_ops},
      JSONWithRecoveryToStackOp{},
      d_top_of_stack,
      root_symbol,
      read_symbol,
      json_in.size(),
      stream);
  } else {
    fst::sparse_stack_op_to_top_of_stack<fst::stack_op_support::NO_RESET_SUPPORT, StackLevelT>(
      stack_ops.data(),
      device_span<SymbolOffsetT>{stack_op_indices.data(), num_stack_ops},
      JSONToStackOp{},
      d_top_of_stack,
      root_symbol,
      read_symbol,
      json_in.size(),
      stream);
  }
}

std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> process_token_stream(
  device_span<PdaTokenT const> tokens,
  device_span<SymbolOffsetT const> token_indices,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // Instantiate FST for post-processing the token stream to remove all tokens that belong to an
  // invalid JSON line
  token_filter::UnwrapTokenFromSymbolOp sgid_op{};
  using symbol_t  = cuda::std::tuple<PdaTokenT, SymbolOffsetT>;
  auto filter_fst = fst::detail::make_fst(
    fst::detail::make_symbol_group_lut(token_filter::symbol_groups, sgid_op),
    fst::detail::make_transition_table(token_filter::transition_table),
    fst::detail::make_translation_functor<symbol_t, 0, 2>(token_filter::TransduceToken{}),
    stream);

  auto const mr = cudf::get_current_device_resource_ref();
  cudf::detail::device_scalar<SymbolOffsetT> d_num_selected_tokens(stream, mr);
  rmm::device_uvector<PdaTokenT> filtered_tokens_out{tokens.size(), stream, mr};
  rmm::device_uvector<SymbolOffsetT> filtered_token_indices_out{tokens.size(), stream, mr};

  // The FST is run on the reverse token stream, discarding all tokens between ErrorBegin and the
  // next LineEnd (LineEnd, inv_token_0, inv_token_1, ..., inv_token_n, ErrorBegin, LineEnd, ...),
  // emitting a [StructBegin, StructEnd] pair on the end of such an invalid line. In that example,
  // inv_token_i for i in [0, n] together with the ErrorBegin are removed and replaced with
  // StructBegin, StructEnd. Also, all LineEnd are removed as well, as these are not relevant after
  // this stage anymore
  filter_fst.Transduce(
    cuda::std::make_reverse_iterator(cuda::make_zip_iterator(tokens.data(), token_indices.data()) +
                                     tokens.size()),
    static_cast<SymbolOffsetT>(tokens.size()),
    cuda::std::make_reverse_iterator(
      cuda::make_zip_iterator(filtered_tokens_out.data(), filtered_token_indices_out.data()) +
      tokens.size()),
    cuda::make_discard_iterator(),
    d_num_selected_tokens.data(),
    token_filter::start_state,
    stream);

  auto const num_total_tokens = d_num_selected_tokens.value(stream);
  rmm::device_uvector<PdaTokenT> tokens_out{num_total_tokens, stream, mr};
  rmm::device_uvector<SymbolOffsetT> token_indices_out{num_total_tokens, stream, mr};
  thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               filtered_tokens_out.end() - num_total_tokens,
               filtered_tokens_out.end(),
               tokens_out.data());
  thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               filtered_token_indices_out.end() - num_total_tokens,
               filtered_token_indices_out.end(),
               token_indices_out.data());

  return std::make_pair(std::move(tokens_out), std::move(token_indices_out));
}

std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> get_token_stream(
  device_span<SymbolT const> json_in,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  check_input_size(json_in.size());

  // Range of encapsulating function that parses to internal columnar data representation
  CUDF_FUNC_RANGE();

  auto const delimited_json = options.is_enabled_lines();
  auto const delimiter      = options.get_delimiter();

  // (!delimited_json)                         => JSON
  // (delimited_json and recover_from_error)   => JSON_LINES_RECOVER
  // (delimited_json and !recover_from_error)  => JSON_LINES
  auto format = delimited_json ? (options.recovery_mode() == json_recovery_mode_t::RECOVER_WITH_NULL
                                    ? tokenizer_pda::json_format_cfg_t::JSON_LINES_RECOVER
                                    : tokenizer_pda::json_format_cfg_t::JSON_LINES)
                               : tokenizer_pda::json_format_cfg_t::JSON;

  // Prepare for PDA transducer pass, merging input symbols with stack symbols
  auto const recover_from_error = (format == tokenizer_pda::json_format_cfg_t::JSON_LINES_RECOVER);

  // Memory holding the top-of-stack stack context for the input
  rmm::device_uvector<StackSymbolT> stack_symbols{json_in.size(), stream};

  // Identify what is the stack context for each input character (JSON-root, struct, or list)
  auto const stack_behavior =
    recover_from_error ? stack_behavior_t::ResetOnDelimiter : stack_behavior_t::PushPopWithoutReset;
  get_stack_context(json_in, stack_symbols.data(), stack_behavior, delimiter, stream);

  // Input to the full pushdown automaton finite-state transducer, where a input symbol comprises
  // the combination of a character from the JSON input together with the stack context for that
  // character.
  auto zip_in = cuda::make_zip_iterator(json_in.data(), stack_symbols.data());

  // Spark, as the main stakeholder in the `recover_from_error` option, has the specific need to
  // ignore any characters that follow the first value on each JSON line. This is an FST that
  // fixes the stack context for those excess characters. That is, that all those excess characters
  // will be interpreted in the root stack context
  if (recover_from_error) {
    auto fix_stack_of_excess_chars = fst::detail::make_fst(
      fst::detail::make_symbol_group_lookup_op(
        fix_stack_of_excess_chars::SymbolPairToSymbolGroupId{delimiter}),
      fst::detail::make_transition_table(fix_stack_of_excess_chars::transition_table),
      fst::detail::make_translation_functor<StackSymbolT, 1, 1>(
        fix_stack_of_excess_chars::TransduceInputOp{}),
      stream);
    fix_stack_of_excess_chars.Transduce(zip_in,
                                        static_cast<SymbolOffsetT>(json_in.size()),
                                        stack_symbols.data(),
                                        cuda::make_discard_iterator(),
                                        cuda::make_discard_iterator(),
                                        fix_stack_of_excess_chars::start_state,
                                        stream);

    // Make sure memory of the FST's lookup tables isn't freed before the FST completes
    stream.synchronize();
  }

  constexpr auto max_translation_table_size =
    tokenizer_pda::NUM_PDA_SGIDS *
    static_cast<tokenizer_pda::StateT>(tokenizer_pda::pda_state_t::PD_NUM_STATES);

  auto json_to_tokens_fst = fst::detail::make_fst(
    fst::detail::make_symbol_group_lookup_op(tokenizer_pda::PdaSymbolToSymbolGroupId{delimiter}),
    fst::detail::make_transition_table(tokenizer_pda::get_transition_table(format)),
    fst::detail::make_translation_table<max_translation_table_size, 0, 3>(
      tokenizer_pda::get_translation_table(recover_from_error)),
    stream);

  // In case we're recovering on invalid JSON lines, post-processing the token stream requires to
  // see a JSON-line delimiter as the very first item
  SymbolOffsetT const delimiter_offset =
    (format == tokenizer_pda::json_format_cfg_t::JSON_LINES_RECOVER ? 1 : 0);

  // Perform a PDA-transducer pass
  cudf::detail::device_scalar<std::size_t> num_written_tokens{
    stream, cudf::get_current_device_resource_ref()};

  // Run FST to estimate the size of output buffers
  json_to_tokens_fst.Transduce(zip_in,
                               static_cast<SymbolOffsetT>(json_in.size()),
                               cuda::make_discard_iterator(),
                               cuda::make_discard_iterator(),
                               num_written_tokens.data(),
                               tokenizer_pda::start_state,
                               stream);

  auto const num_total_tokens = num_written_tokens.value(stream) + delimiter_offset;
  rmm::device_uvector<PdaTokenT> tokens{num_total_tokens, stream, mr};
  rmm::device_uvector<SymbolOffsetT> tokens_indices{num_total_tokens, stream, mr};

  // Run FST to translate the input JSON string into tokens and indices at which they occur
  json_to_tokens_fst.Transduce(zip_in,
                               static_cast<SymbolOffsetT>(json_in.size()),
                               tokens.data() + delimiter_offset,
                               tokens_indices.data() + delimiter_offset,
                               cuda::make_discard_iterator(),
                               tokenizer_pda::start_state,
                               stream);

  if (delimiter_offset == 1) {
    tokens.set_element(0, token_t::LineEnd, stream);
    validate_token_stream(json_in, tokens, tokens_indices, options, stream);
    auto [filtered_tokens, filtered_tokens_indices] =
      process_token_stream(tokens, tokens_indices, stream);
    tokens         = std::move(filtered_tokens);
    tokens_indices = std::move(filtered_tokens_indices);
  }

  return std::make_pair(std::move(tokens), std::move(tokens_indices));
}

/**
 * @brief Parses the given JSON string and generates a tree representation of the given input.
 *
 * @param[in,out] root_column The root column of the hierarchy of columns into which data is parsed
 * @param[in,out] current_data_path The stack represents the path from the JSON root node to the
 * first node encountered in \p input
 * @param[in] input The JSON input in host memory
 * @param[in] d_input The JSON input in device memory
 * @param[in] options Parsing options specifying the parsing behaviour
 * @param[in] include_quote_char Whether to include the original quote chars around string values,
 * allowing to distinguish string values from numeric and literal values
 * @param[in] stream The CUDA stream to which kernels are dispatched
 * @param[in] mr Optional, resource with which to allocate
 * @return The columnar representation of the data from the given JSON input
 */
void make_json_column(json_column& root_column,
                      std::stack<tree_node>& current_data_path,
                      host_span<SymbolT const> input,
                      device_span<SymbolT const> d_input,
                      cudf::io::json_reader_options const& options,
                      bool include_quote_char,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  // Range of encapsulating function that parses to internal columnar data representation
  CUDF_FUNC_RANGE();

  // Parse the JSON and get the token stream
  auto const [d_tokens_gpu, d_token_indices_gpu] = get_token_stream(d_input, options, stream, mr);

  // Copy the JSON tokens to the host
  auto tokens            = cudf::detail::make_host_vector_async(d_tokens_gpu, stream);
  auto token_indices_gpu = cudf::detail::make_host_vector_async(d_token_indices_gpu, stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  // Whether this token is the valid token to begin the JSON document with
  auto is_valid_root_token = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin: return true;
      default: return false;
    };
  };

  // Returns the token's corresponding column type
  auto token_to_column_type = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin: return json_col_t::StructColumn;
      case token_t::ListBegin: return json_col_t::ListColumn;
      case token_t::StringBegin: return json_col_t::StringColumn;
      case token_t::ValueBegin: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    };
  };

  // Depending on whether we want to include the quotes of strings or not, respectively, we:
  // (a) strip off the beginning quote included in StringBegin and FieldNameBegin or
  // (b) include of the end quote excluded from in StringEnd and strip off the beginning quote
  // included FieldNameBegin
  auto get_token_index = [include_quote_char](PdaTokenT const token,
                                              SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT quote_char_size = 1;
    switch (token) {
      // Optionally strip off quote char included for StringBegin
      case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
      // Optionally include trailing quote char for string values excluded for StringEnd
      case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
      // Strip off quote char included for FieldNameBegin
      case token_t::FieldNameBegin: return token_index + quote_char_size;
      default: return token_index;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin: return token_t::StringEnd;
      case token_t::ValueBegin: return token_t::ValueEnd;
      case token_t::FieldNameBegin: return token_t::FieldNameEnd;
      default: return token_t::ErrorBegin;
    };
  };

#ifdef NJP_DEBUG_PRINT
  auto column_type_string = [](json_col_t column_type) {
    switch (column_type) {
      case json_col_t::Unknown: return "Unknown";
      case json_col_t::ListColumn: return "List";
      case json_col_t::StructColumn: return "Struct";
      case json_col_t::StringColumn: return "String";
      default: return "Unknown";
    }
  };

  auto token_to_string = [](PdaTokenT token_type) {
    switch (token_type) {
      case token_t::StructBegin: return "StructBegin";
      case token_t::StructEnd: return "StructEnd";
      case token_t::ListBegin: return "ListBegin";
      case token_t::ListEnd: return "ListEnd";
      case token_t::StructMemberBegin: return "StructMemberBegin";
      case token_t::StructMemberEnd: return "StructMemberEnd";
      case token_t::FieldNameBegin: return "FieldNameBegin";
      case token_t::FieldNameEnd: return "FieldNameEnd";
      case token_t::StringBegin: return "StringBegin";
      case token_t::StringEnd: return "StringEnd";
      case token_t::ValueBegin: return "ValueBegin";
      case token_t::ValueEnd: return "ValueEnd";
      case token_t::ErrorBegin: return "ErrorBegin";
      case token_t::LineEnd: return "LineEnd";
      default: return "Unknown";
    }
  };
#endif

  /**
   * @brief Updates the given row in the given column with a new string_end and child_count. In
   * particular, updating the child count is relevant for list columns.
   */
  auto update_row =
    [](json_column* column, uint32_t row_index, uint32_t string_end, uint32_t child_count) {
#ifdef NJP_DEBUG_PRINT
      std::cout << "  -> update_row()\n";
      std::cout << "  ---> col@" << column << "\n";
      std::cout << "  ---> row #" << row_index << "\n";
      std::cout << "  ---> string_lengths = " << (string_end - column->string_offsets[row_index])
                << "\n";
      std::cout << "  ---> child_offsets = " << (column->child_offsets[row_index + 1] + child_count)
                << "\n";
#endif
      column->string_lengths[row_index]    = column->child_offsets[row_index + 1] + child_count;
      column->child_offsets[row_index + 1] = column->child_offsets[row_index + 1] + child_count;
    };

  /**
   * @brief Gets the currently selected child column given a \p current_data_path.
   *
   * That is, if \p current_data_path top-of-stack is
   * (a) a struct, the selected child column corresponds to the child column of the last field name
   * node encountered.
   * (b) a list, the selected child column corresponds to single child column of
   * the list column. In this case, the child column may not exist yet.
   */
  auto get_selected_column = [](std::stack<tree_node>& current_data_path) {
    json_column* selected_col = current_data_path.top().current_selected_col;

    // If the node does not have a selected column yet
    if (selected_col == nullptr) {
      // We're looking at the child column of a list column
      if (current_data_path.top().column->type == json_col_t::ListColumn) {
        CUDF_EXPECTS(current_data_path.top().column->child_columns.size() <= 1,
                     "Encountered a list column with more than a single child column");
        // The child column has yet to be created
        if (current_data_path.top().column->child_columns.empty()) {
          current_data_path.top().column->child_columns.emplace(std::string{list_child_name},
                                                                json_column{json_col_t::Unknown});
          current_data_path.top().column->column_order.emplace_back(list_child_name);
        }
        current_data_path.top().current_selected_col =
          &current_data_path.top().column->child_columns.begin()->second;
        selected_col = current_data_path.top().current_selected_col;
      } else {
        CUDF_FAIL("Trying to retrieve child column without encountering a field name.");
      }
    }
#ifdef NJP_DEBUG_PRINT
    std::cout << "  -> get_selected_column()\n";
    std::cout << "  ---> selected col@" << selected_col << "\n";
#endif
    return selected_col;
  };

  /**
   * @brief Returns a pointer to the child column with the given \p field_name within the current
   * struct column.
   */
  auto select_column = [](std::stack<tree_node>& current_data_path, std::string const& field_name) {
#ifdef NJP_DEBUG_PRINT
    std::cout << "  -> select_column(" << field_name << ")\n";
#endif
    // The field name's parent struct node
    auto& current_struct_node = current_data_path.top();

    // Verify that the field name node is actually a child of a struct
    CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::StructColumn,
                 "Invalid JSON token sequence");

    json_column* struct_col  = current_struct_node.column;
    auto const& child_col_it = struct_col->child_columns.find(field_name);

    // The field name's column exists already, select that as the struct node's currently selected
    // child column
    if (child_col_it != struct_col->child_columns.end()) { return &child_col_it->second; }

    // The field name's column does not exist yet, so we have to append the child column to the
    // struct column
    struct_col->column_order.push_back(field_name);
    return &struct_col->child_columns.emplace(field_name, json_column{}).first->second;
  };

  /**
   * @brief Gets the row offset at which to insert. I.e., for a child column of a list column, we
   * just have to append the row to the end. Otherwise we have to propagate the row offset from the
   * parent struct column.
   */
  auto get_target_row_index = [](std::stack<tree_node> const& current_data_path,
                                 json_column* target_column) {
#ifdef NJP_DEBUG_PRINT
    std::cout << " -> target row: "
              << ((current_data_path.top().column->type == json_col_t::ListColumn)
                    ? target_column->current_offset
                    : current_data_path.top().row_index)
              << "\n";
#endif
    return (current_data_path.top().column->type == json_col_t::ListColumn)
             ? target_column->current_offset
             : current_data_path.top().row_index;
  };

  // The offset of the token currently being processed
  std::size_t offset = 0;

  // Giving names to magic constants
  constexpr uint32_t zero_child_count = 0;

  CUDF_EXPECTS(tokens.size() == token_indices_gpu.size(),
               "Unexpected mismatch in number of token types and token indices");
  CUDF_EXPECTS(tokens.size() > 0, "Empty JSON input not supported");

  // The JSON root may only be a struct, list, string, or value node
  CUDF_EXPECTS(is_valid_root_token(tokens[offset]), "Invalid beginning of JSON document");

  while (offset < tokens.size()) {
    // Verify there's at least the JSON root node left on the stack to which we can append data
    CUDF_EXPECTS(current_data_path.size() > 0, "Invalid JSON structure");

    // Verify that the current node in the tree (which becomes this nodes parent) can have children
    CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::ListColumn or
                   current_data_path.top().column->type == json_col_t::StructColumn,
                 "Invalid JSON structure");

    // The token we're currently parsing
    auto const& token = tokens[offset];

#ifdef NJP_DEBUG_PRINT
    std::cout << "[" << token_to_string(token) << "]\n";
#endif

    // StructBegin token
    if (token == token_t::StructBegin) {
      // Get this node's column. That is, the parent node's selected column:
      // (a) if parent is a list, then this will (create and) return the list's only child column
      // (b) if parent is a struct, then this will return the column selected by the last field name
      // encountered.
      json_column* selected_col = get_selected_column(current_data_path);

      // Get the row offset at which to insert
      auto const target_row_index = get_target_row_index(current_data_path, selected_col);

      // Increment parent's child count and insert this struct node into the data path
      current_data_path.top().num_children++;
      current_data_path.push({selected_col, target_row_index, nullptr, zero_child_count});

      // Add this struct node to the current column
      selected_col->append_row(target_row_index,
                               token_to_column_type(tokens[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               zero_child_count);
    }

    // StructEnd token
    else if (token == token_t::StructEnd) {
      // Verify that this node in fact a struct node (i.e., it was part of a struct column)
      CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::StructColumn,
                   "Broken invariant while parsing JSON");
      CUDF_EXPECTS(current_data_path.top().column != nullptr,
                   "Broken invariant while parsing JSON");

      // Update row to account for string offset
      update_row(current_data_path.top().column,
                 current_data_path.top().row_index,
                 get_token_index(tokens[offset], token_indices_gpu[offset]),
                 current_data_path.top().num_children);

      // Pop struct from the path stack
      current_data_path.pop();
    }

    // ListBegin token
    else if (token == token_t::ListBegin) {
      // Get the selected column
      json_column* selected_col = get_selected_column(current_data_path);

      // Get the row offset at which to insert
      auto const target_row_index = get_target_row_index(current_data_path, selected_col);

      // Increment parent's child count and insert this struct node into the data path
      current_data_path.top().num_children++;
      current_data_path.push({selected_col, target_row_index, nullptr, zero_child_count});

      // Add this struct node to the current column
      selected_col->append_row(target_row_index,
                               token_to_column_type(tokens[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               zero_child_count);
    }

    // ListEnd token
    else if (token == token_t::ListEnd) {
      // Verify that this node in fact a list node (i.e., it was part of a list column)
      CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::ListColumn,
                   "Broken invariant while parsing JSON");
      CUDF_EXPECTS(current_data_path.top().column != nullptr,
                   "Broken invariant while parsing JSON");

      // Update row to account for string offset
      update_row(current_data_path.top().column,
                 current_data_path.top().row_index,
                 get_token_index(tokens[offset], token_indices_gpu[offset]),
                 current_data_path.top().num_children);

      // Pop list from the path stack
      current_data_path.pop();
    }

    // Error token
    else if (token == token_t::ErrorBegin) {
#ifdef NJP_DEBUG_PRINT
      std::cout << "[ErrorBegin]\n";
      std::cout << "@" << get_token_index(tokens[offset], token_indices_gpu[offset]);
#endif
      CUDF_FAIL("Parser encountered an invalid format.");
    }

    // FieldName, String, or Value (begin, end)-pair
    else if (token == token_t::FieldNameBegin or token == token_t::StringBegin or
             token == token_t::ValueBegin) {
      // Verify that this token has the right successor to build a correct (being, end) token pair
      CUDF_EXPECTS((offset + 1) < tokens.size(), "Invalid JSON token sequence");
      CUDF_EXPECTS(tokens[offset + 1] == end_of_partner(token), "Invalid JSON token sequence");

      // The offset to the first symbol from the JSON input associated with the current token
      auto const& token_begin_offset = get_token_index(tokens[offset], token_indices_gpu[offset]);

      // The offset to one past the last symbol associated with the current token
      auto const& token_end_offset =
        get_token_index(tokens[offset + 1], token_indices_gpu[offset + 1]);

      // FieldNameBegin
      // For the current struct node in the tree, select the child column corresponding to this
      // field name
      if (token == token_t::FieldNameBegin) {
        std::string field_name{input.data() + token_begin_offset,
                               (token_end_offset - token_begin_offset)};
        current_data_path.top().current_selected_col = select_column(current_data_path, field_name);
      }
      // StringBegin
      // ValueBegin
      // As we currently parse to string columns there's no further differentiation
      else if (token == token_t::StringBegin or token == token_t::ValueBegin) {
        // Get the selected column
        json_column* selected_col = get_selected_column(current_data_path);

        // Get the row offset at which to insert
        auto const target_row_index = get_target_row_index(current_data_path, selected_col);

        current_data_path.top().num_children++;

        selected_col->append_row(target_row_index,
                                 token_to_column_type(token),
                                 token_begin_offset,
                                 token_end_offset,
                                 zero_child_count);
      } else {
        CUDF_FAIL("Unknown JSON token");
      }

      // As we've also consumed the end-of-* token, we advance the processed token offset by one
      offset++;
    }

    offset++;
  }

  // Make sure all of a struct's child columns have the same length
  root_column.level_child_cols_recursively(root_column.current_offset);
}

/**
 * @brief Retrieves the parse_options to be used for type inference and type casting
 *
 * @param options The reader options to influence the relevant type inference and type casting
 * options
 * @param stream The CUDA stream to which kernels are dispatched
 */
cudf::io::parse_options parsing_options(cudf::io::json_reader_options const& options,
                                        rmm::cuda_stream_view stream)
{
  auto parse_opts = cudf::io::parse_options{',', '\n', '\"', '.'};

  parse_opts.dayfirst              = options.is_enabled_dayfirst();
  parse_opts.keepquotes            = options.is_enabled_keep_quotes();
  parse_opts.normalize_whitespace  = options.is_enabled_normalize_whitespace();
  parse_opts.mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  parse_opts.trie_true             = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false            = cudf::detail::create_serialized_trie({"false"}, stream);
  std::vector<std::string> na_values{"", "null"};
  na_values.insert(na_values.end(), options.get_na_values().begin(), options.get_na_values().end());
  parse_opts.trie_na = cudf::detail::create_serialized_trie(na_values, stream);
  return parse_opts;
}

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> json_column_to_cudf_column(
  json_column const& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::json_reader_options const& options,
  std::optional<schema_element> schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Range of orchestrating/encapsulating function
  CUDF_FUNC_RANGE();

  auto make_validity =
    [stream, mr](json_column const& json_col) -> std::pair<rmm::device_buffer, size_type> {
    auto const null_count = json_col.current_offset - json_col.valid_count;
    if (null_count == 0) { return {rmm::device_buffer{}, null_count}; }
    return {rmm::device_buffer{json_col.validity.data(),
                               bitmask_allocation_size_bytes(json_col.current_offset),
                               stream,
                               mr},
            null_count};
  };

  auto get_child_schema = [schema](auto child_name) -> std::optional<schema_element> {
    if (schema.has_value()) {
      auto const result = schema.value().child_types.find(child_name);
      if (result != std::end(schema.value().child_types)) { return result->second; }
    }
    return {};
  };

  switch (json_col.type) {
    case json_col_t::StringColumn: {
      auto const col_size = json_col.string_offsets.size();
      CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                   "string offset, string length mismatch");

      // Move string_offsets and string_lengths to GPU
      rmm::device_uvector<json_column::row_offset_t> d_string_offsets =
        cudf::detail::make_device_uvector_async(
          json_col.string_offsets, stream, cudf::get_current_device_resource_ref());
      rmm::device_uvector<json_column::row_offset_t> d_string_lengths =
        cudf::detail::make_device_uvector_async(
          json_col.string_lengths, stream, cudf::get_current_device_resource_ref());

      // Prepare iterator that returns (string_offset, string_length)-tuples
      auto offset_length_it =
        cuda::make_zip_iterator(d_string_offsets.begin(), d_string_lengths.begin());

      data_type target_type{};

      if (schema.has_value()) {
#ifdef NJP_DEBUG_PRINT
        std::cout << "-> explicit type: "
                  << (schema.has_value() ? std::to_string(static_cast<int>(schema->type.id()))
                                         : "n/a");
#endif
        target_type = schema.value().type;
      }
      // Infer column type, if we don't have an explicit type for it
      else {
        target_type =
          cudf::io::detail::infer_data_type(parsing_options(options, stream).json_view(),
                                            d_input,
                                            offset_length_it,
                                            col_size,
                                            stream);
      }

      auto [result_bitmask, null_count] = make_validity(json_col);

      // Convert strings to the inferred data type
      auto col = parse_data(d_input.data(),
                            offset_length_it,
                            col_size,
                            target_type,
                            std::move(result_bitmask),
                            null_count,
                            parsing_options(options, stream).view(),
                            stream,
                            mr);

      // Reset nullable if we do not have nulls
      // This is to match the existing JSON reader's behaviour:
      // - Non-string columns will always be returned as nullable
      // - String columns will be returned as nullable, iff there's at least one null entry
      if (col->null_count() == 0) { col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0); }

      // For string columns return ["offsets"] schema
      if (target_type.id() == type_id::STRING) {
        return {std::move(col), std::vector<column_name_info>{{"offsets"}}};
      }
      // Non-string leaf-columns (e.g., numeric) do not have child columns in the schema
      else {
        return {std::move(col), std::vector<column_name_info>{}};
      }
      break;
    }
    case json_col_t::StructColumn: {
      std::vector<std::unique_ptr<column>> child_columns;
      std::vector<column_name_info> column_names{};
      size_type num_rows{json_col.current_offset};
      // Create children columns
      for (auto const& col_name : json_col.column_order) {
        auto const& col = json_col.child_columns.find(col_name);
        column_names.emplace_back(col->first);
        auto const& child_col      = col->second;
        auto [child_column, names] = json_column_to_cudf_column(
          child_col, d_input, options, get_child_schema(col_name), stream, mr);
        CUDF_EXPECTS(num_rows == child_column->size(),
                     "All children columns must have the same size");
        child_columns.push_back(std::move(child_column));
        column_names.back().children = names;
      }
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {
        make_structs_column(
          num_rows, std::move(child_columns), null_count, std::move(result_bitmask), stream, mr),
        column_names};
      break;
    }
    case json_col_t::ListColumn: {
      size_type num_rows = json_col.child_offsets.size();
      std::vector<column_name_info> column_names{};
      column_names.emplace_back("offsets");
      column_names.emplace_back(
        json_col.child_columns.empty() ? list_child_name : json_col.child_columns.begin()->first);

      rmm::device_uvector<json_column::row_offset_t> d_offsets =
        cudf::detail::make_device_uvector_async(json_col.child_offsets, stream, mr);
      auto offsets_column = std::make_unique<column>(
        data_type{type_id::INT32}, num_rows, d_offsets.release(), rmm::device_buffer{}, 0);
      // Create children column
      auto [child_column, names] =
        json_col.child_columns.empty()
          ? std::pair<std::unique_ptr<column>,
                      std::vector<column_name_info>>{std::make_unique<column>(),
                                                     std::vector<column_name_info>{}}
          : json_column_to_cudf_column(json_col.child_columns.begin()->second,
                                       d_input,
                                       options,
                                       get_child_schema(json_col.child_columns.begin()->first),
                                       stream,
                                       mr);
      column_names.back().children      = names;
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {make_lists_column(num_rows - 1,
                                std::move(offsets_column),
                                std::move(child_column),
                                null_count,
                                std::move(result_bitmask)),
              std::move(column_names)};
      break;
    }
    default: CUDF_FAIL("Unsupported column type, yet to be implemented"); break;
  }

  return {};
}

}  // namespace detail
}  // namespace cudf::io::json

// Debug print flag
#undef NJP_DEBUG_PRINT
