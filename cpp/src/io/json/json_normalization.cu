/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/fst/lookup_tables.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_copy.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace cudf::io::json {

// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
using StateT  = char;

// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;

namespace normalize_quotes {

enum class dfa_states : StateT { TT_OOS = 0U, TT_DQS, TT_SQS, TT_DEC, TT_SEC, TT_NUM_STATES };
enum class dfa_symbol_group_id : uint32_t {
  DOUBLE_QUOTE_CHAR,  ///< Quote character SG: "
  SINGLE_QUOTE_CHAR,  ///< Quote character SG: '
  ESCAPE_CHAR,        ///< Escape character SG: '\'
  DELIM_CHAR,         ///< Delimiter character SG
  OTHER_SYMBOLS,      ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS   ///< Total number of symbol groups
};

// Aliases for readability of the transition table
constexpr auto TT_OOS            = dfa_states::TT_OOS;
constexpr auto TT_DQS            = dfa_states::TT_DQS;
constexpr auto TT_SQS            = dfa_states::TT_SQS;
constexpr auto TT_DEC            = dfa_states::TT_DEC;
constexpr auto TT_SEC            = dfa_states::TT_SEC;
constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

auto get_sgid_lut(SymbolT delim)
{
  // The i-th string representing all the characters of a symbol group
  std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> symbol_groups{
    {{'\"'}, {'\''}, {'\\'}, {delim}}};
  return symbol_groups;
}

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const qna_state_tt{{
  /* IN_STATE      "       '       \     <delim>    OTHER  */
  /* TT_OOS */ {{TT_DQS, TT_SQS, TT_OOS, TT_OOS, TT_OOS}},
  /* TT_DQS */ {{TT_OOS, TT_DQS, TT_DEC, TT_OOS, TT_DQS}},
  /* TT_SQS */ {{TT_SQS, TT_OOS, TT_SEC, TT_OOS, TT_SQS}},
  /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_OOS, TT_DQS}},
  /* TT_SEC */ {{TT_SQS, TT_SQS, TT_SQS, TT_OOS, TT_SQS}},
}};

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(TT_OOS);

struct TransduceToNormalizedQuotes {
  /**
   * @brief Returns the <relative_offset>-th output symbol on the transition (state_id, match_id).
   */
  template <typename StateT, typename SymbolGroupT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE SymbolT operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                RelativeOffsetT const relative_offset,
                                                SymbolT const read_symbol) const
  {
    // -------- TRANSLATION TABLE ------------
    //      Let the alphabet set be Sigma
    // ---------------------------------------
    // ---------- NON-SPECIAL CASES: ----------
    //      Output symbol same as input symbol <s>
    // state | read_symbol <s> -> output_symbol <s>
    // DQS   | Sigma           -> Sigma
    // DEC   | Sigma           -> Sigma
    // OOS   | Sigma\{'}       -> Sigma\{'}
    // SQS   | Sigma\{', "}    -> Sigma\{', "}
    // ---------- SPECIAL CASES: --------------
    //    Input symbol translates to output symbol
    // OOS   | {'}             -> {"}
    // SQS   | {'}             -> {"}
    // SQS   | {"}             -> {\"}
    // SQS   | {\}             -> <nop>
    // DQS   | {\}             -> <nop>
    // SEC   | {'}             -> {'}
    // SEC   | Sigma\{'}       -> {\*}
    // DEC   | {'}             -> {'}
    // DEC   | Sigma\{'}       -> {\*}

    // Whether this transition translates to the escape sequence: \"
    bool const outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DOUBLE_QUOTE_CHAR));
    // Case when a double quote needs to be replaced by the escape sequence: \"
    if (outputs_escape_sequence) { return (relative_offset == 0) ? '\\' : '"'; }
    // Case when a single quote needs to be replaced by a double quote
    if ((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR)) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_SQS)) ||
         (state_id == static_cast<StateT>(dfa_states::TT_OOS)))) {
      return '"';
    }
    // Case when the read symbol is an escape character - the actual translation for \<s> for some
    // symbol <s> is handled by transitions from SEC. The same logic applies for the transition from
    // DEC. For now, there is no output for this transition
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR) &&
        (state_id == static_cast<StateT>(dfa_states::TT_SQS) ||
         state_id == static_cast<StateT>(dfa_states::TT_DQS))) {
      return 0;
    }
    // Case when an escaped single quote in an input single-quoted or double-quoted string needs
    // to be replaced by an unescaped single quote
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR) &&
        (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
         state_id == static_cast<StateT>(dfa_states::TT_DEC))) {
      return '\'';
    }
    // Case when an escaped symbol <s> that is not a single-quote needs to be replaced with \<s>
    if (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
        state_id == static_cast<StateT>(dfa_states::TT_DEC)) {
      return (relative_offset == 0) ? '\\' : read_symbol;
    }
    // In all other cases we simply output the input symbol
    return read_symbol;
  }

  /**
   * @brief Returns the number of output characters for a given transition. During quote
   * normalization, we always emit one output character (i.e., either the input character or the
   * single quote-input replaced by a double quote), except when we need to escape a double quote
   * that was previously inside a single-quoted string.
   */
  template <typename StateT, typename SymbolGroupT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                SymbolT const read_symbol) const
  {
    // Whether this transition translates to the escape sequence: \"
    bool const sqs_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DOUBLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sqs_outputs_escape_sequence) { return 2; }

    // Whether this transition translates to the escape sequence \<s> or unescaped '
    bool const sec_dec_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
       state_id == static_cast<StateT>(dfa_states::TT_DEC)) &&
      (match_id != static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sec_dec_outputs_escape_sequence) { return 2; }

    // Whether this transition translates to no output <nop>
    bool const sqs_dqs_outputs_nop =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS) ||
       state_id == static_cast<StateT>(dfa_states::TT_DQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR));
    // Number of characters to output on this transition
    if (sqs_dqs_outputs_nop) { return 0; }

    return 1;
  }
};

}  // namespace normalize_quotes

namespace normalize_whitespace {

// We do not need a symbol group for the delimiter character since whitespace normalization
// now occurs after tokenization.
enum class dfa_symbol_group_id : uint32_t {
  DOUBLE_QUOTE_CHAR,   ///< Quote character SG: "
  ESCAPE_CHAR,         ///< Escape character SG: '\\'
  WHITESPACE_SYMBOLS,  ///< Whitespace characters SG: '\t' or ' '
  OTHER_SYMBOLS,       ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS    ///< Total number of symbol groups
};
// Alias for readability of symbol group ids
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> const wna_sgs{{{'"'}, {'\\'}, {' ', '\t'}}};

/**
 * -------- FST states ---------
 * -----------------------------
 * TT_OOS | Out-of-string state handling whitespace and non-whitespace chars outside double
 *        |   quotes as well as any other character not enclosed by a string.
 * TT_DQS | Double-quoted string state handling all characters within double quotes
 * TT_DEC | State handling escaped characters inside double-quoted string. Note that this
 *        |   state is necessary to process escaped double-quote characters. Without this
 *        |   state, whitespaces following escaped double quotes inside strings may be removed.
 *
 */
enum class dfa_states : StateT { TT_OOS = 0U, TT_DQS, TT_DEC, TT_NUM_STATES };
// Aliases for readability of the transition table
constexpr auto TT_OOS        = dfa_states::TT_OOS;
constexpr auto TT_DQS        = dfa_states::TT_DQS;
constexpr auto TT_DEC        = dfa_states::TT_DEC;
constexpr auto TT_NUM_STATES = static_cast<StateT>(dfa_states::TT_NUM_STATES);

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const wna_state_tt{
  {/* IN_STATE      "       \     <SPC>   OTHER  */
   /* TT_OOS */ {{TT_DQS, TT_OOS, TT_OOS, TT_OOS}},
   /* TT_DQS */ {{TT_OOS, TT_DEC, TT_DQS, TT_DQS}},
   /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_DQS}}}};

// The DFA's starting state
constexpr StateT start_state = static_cast<StateT>(TT_OOS);

struct TransduceToNormalizedWS {
  /**
   * @brief Returns the <relative_offset>-th output symbol on the transition (state_id, match_id).
   */
  template <typename StateT, typename SymbolGroupT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE SymbolT operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                RelativeOffsetT const relative_offset,
                                                SymbolT const read_symbol) const
  {
    // -------- TRANSLATION TABLE ------------
    //      Let the alphabet set be Sigma
    // ---------------------------------------
    // ---------- NON-SPECIAL CASES: ----------
    //    Input symbol translates to output symbol
    // state | read_symbol <s>  -> output_symbol <s>
    // DQS   | Sigma            -> <nop>
    // OOS   | Sigma\{<SPC>,\t} -> <nop>
    // DEC   | Sigma            -> <nop>
    // ---------- SPECIAL CASES: --------------
    //      Output symbol same as input symbol <s>
    // OOS   | {<SPC>}          -> {<SPC>}
    // OOS   | {\t}             -> {\t}

    // Case when read symbol is not an unquoted space or tab
    // This will be the same condition as in `operator()(state_id, match_id, read_symbol)` function
    // However, since there is no output in this case i.e. the count returned by
    // operator()(state_id, match_id, read_symbol) is zero, this function is never called.
    // So skipping the check for this case.

    // In all other cases, we have an output symbol for the input symbol.
    // We simply output the input symbol
    return read_symbol;
  }

  /**
   * @brief Returns the number of output characters for a given transition.
   * During whitespace normalization, we always emit one output character i.e., the input
   * character, except when we need to remove the space/tab character
   */
  template <typename StateT, typename SymbolGroupT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE uint32_t operator()(StateT const state_id,
                                                 SymbolGroupT const match_id,
                                                 SymbolT const read_symbol) const
  {
    // Case when read symbol is a space or tab but is unquoted
    if (!(match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::WHITESPACE_SYMBOLS) &&
          state_id == static_cast<StateT>(dfa_states::TT_OOS))) {
      return 0;
    }
    return 1;
  }
};

}  // namespace normalize_whitespace

namespace detail {

void normalize_single_quotes(datasource::owning_buffer<rmm::device_buffer>& indata,
                             char delimiter,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  static constexpr std::int32_t min_out = 0;
  static constexpr std::int32_t max_out = 2;
  auto parser                           = fst::detail::make_fst(
    fst::detail::make_symbol_group_lut(normalize_quotes::get_sgid_lut(delimiter)),
    fst::detail::make_transition_table(normalize_quotes::qna_state_tt),
    fst::detail::make_translation_functor<SymbolT, min_out, max_out>(
      normalize_quotes::TransduceToNormalizedQuotes{}),
    stream);

  rmm::device_buffer outbuf(indata.size() * 2, stream, mr);
  cudf::detail::device_scalar<SymbolOffsetT> outbuf_size(stream, mr);
  parser.Transduce(reinterpret_cast<SymbolT const*>(indata.data()),
                   static_cast<SymbolOffsetT>(indata.size()),
                   static_cast<SymbolT*>(outbuf.data()),
                   cuda::make_discard_iterator(),
                   outbuf_size.data(),
                   normalize_quotes::start_state,
                   stream);

  outbuf.resize(outbuf_size.value(stream), stream);
  datasource::owning_buffer<rmm::device_buffer> outdata(std::move(outbuf));
  std::swap(indata, outdata);
}

std::
  tuple<rmm::device_uvector<char>, rmm::device_uvector<size_type>, rmm::device_uvector<size_type>>
  normalize_whitespace(device_span<char const> d_input,
                       device_span<size_type const> col_offsets,
                       device_span<size_type const> col_lengths,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
{
  /*
   * Algorithm:
    1. Create a single buffer by concatenating the rows of the string column. Create segment offsets
   and lengths array for concatenated buffer
    2. Run a whitespace normalization FST that performs NOP for non-whitespace and quoted
   whitespace characters, and outputs indices of unquoted whitespace characters
    3. Update segment lengths based on the number of output indices between segment offsets
    4. Remove characters at output indices from concatenated buffer.
    5. Return updated buffer, segment lengths and updated segment offsets
   */
  auto inbuf_lengths = cudf::detail::make_device_uvector_async(
    col_lengths, stream, cudf::get_current_device_resource_ref());
  std::size_t inbuf_lengths_size = inbuf_lengths.size();
  size_type inbuf_size =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   inbuf_lengths.begin(),
                   inbuf_lengths.end());
  rmm::device_uvector<char> inbuf(inbuf_size, stream);
  rmm::device_uvector<size_type> inbuf_offsets(inbuf_lengths_size, stream);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         inbuf_lengths.begin(),
                         inbuf_lengths.end(),
                         inbuf_offsets.begin(),
                         0);

  auto input_it = cuda::transform_iterator(
    col_offsets.begin(),
    cuda::proclaim_return_type<char const*>(
      [d_input = d_input.begin()] __device__(auto offset) { return &d_input[offset]; }));
  auto output_it = cuda::transform_iterator(
    inbuf_offsets.cbegin(),
    cuda::proclaim_return_type<char*>(
      [inbuf = inbuf.begin()] __device__(auto offset) { return &inbuf[offset]; }));

  {
    // cub device batched copy
    std::size_t temp_storage_bytes = 0;
    cub::DeviceCopy::Batched(nullptr,
                             temp_storage_bytes,
                             input_it,
                             output_it,
                             inbuf_lengths.begin(),
                             inbuf_lengths_size,
                             stream.get());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream);
    cub::DeviceCopy::Batched(temp_storage.data(),
                             temp_storage_bytes,
                             input_it,
                             output_it,
                             inbuf_lengths.begin(),
                             inbuf_lengths_size,
                             stream.get());
  }

  // whitespace normalization : get the indices of the unquoted whitespace characters
  auto parser =
    fst::detail::make_fst(fst::detail::make_symbol_group_lut(normalize_whitespace::wna_sgs),
                          fst::detail::make_transition_table(normalize_whitespace::wna_state_tt),
                          fst::detail::make_translation_functor<SymbolT, 0, 2>(
                            normalize_whitespace::TransduceToNormalizedWS{}),
                          stream);

  rmm::device_uvector<size_type> outbuf_indices(inbuf.size(), stream, mr);
  cudf::detail::device_scalar<SymbolOffsetT> outbuf_indices_size(stream, mr);
  parser.Transduce(inbuf.data(),
                   static_cast<SymbolOffsetT>(inbuf.size()),
                   cuda::make_discard_iterator(),
                   outbuf_indices.data(),
                   outbuf_indices_size.data(),
                   normalize_whitespace::start_state,
                   stream);

  auto const num_deletions = outbuf_indices_size.value(stream);
  outbuf_indices.resize(num_deletions, stream);

  // now these indices need to be removed
  // TODO: is there a better way to do this?
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    outbuf_indices.begin(),
    outbuf_indices.end(),
    [inbuf_offsets_begin = inbuf_offsets.begin(),
     inbuf_offsets_end   = inbuf_offsets.end(),
     inbuf_lengths       = inbuf_lengths.begin()] __device__(size_type idx) {
      auto it  = thrust::upper_bound(thrust::seq, inbuf_offsets_begin, inbuf_offsets_end, idx);
      auto pos = cuda::std::distance(inbuf_offsets_begin, it) - 1;
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*(inbuf_lengths + pos)};
      ref.fetch_add(-1, cuda::std::memory_order_relaxed);
    });

  auto stencil = cudf::detail::make_zeroed_device_uvector_async<bool>(
    static_cast<std::size_t>(inbuf_size), stream, cudf::get_current_device_resource_ref());
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  cuda::make_constant_iterator(true),
                  cuda::make_constant_iterator(true) + num_deletions,
                  outbuf_indices.begin(),
                  stencil.begin());
  thrust::remove_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    inbuf.begin(),
                    inbuf.end(),
                    stencil.begin(),
                    cuda::std::identity{});
  inbuf.resize(inbuf_size - num_deletions, stream);

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         inbuf_lengths.begin(),
                         inbuf_lengths.end(),
                         inbuf_offsets.begin(),
                         0);

  stream.sync();
  return std::tuple{std::move(inbuf), std::move(inbuf_offsets), std::move(inbuf_lengths)};
}

}  // namespace detail
}  // namespace cudf::io::json
