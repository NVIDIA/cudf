/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/fst/lookup_tables.cuh"
#include "io/utilities/hostdevice_vector.hpp"

#include <tests/io/fst/common.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdlib>
#include <vector>

namespace {

//------------------------------------------------------------------------------
// CPU-BASED IMPLEMENTATIONS FOR VERIFICATION
//------------------------------------------------------------------------------
/**
 * @brief CPU-based implementation of a finite-state transducer (FST).
 *
 * @tparam InputItT Forward input iterator type to symbols fed into the FST
 * @tparam StateT Type representing states of the finite-state machine
 * @tparam SymbolGroupLutT Sequence container of symbol groups. Each symbol group is a sequence
 * container to symbols within that group.
 * @tparam TransitionTableT Two-dimensional container type
 * @tparam TransducerTableT Two-dimensional container type
 * @tparam OutputItT Forward output iterator type
 * @tparam IndexOutputItT Forward output iterator type
 * @param[in] begin Forward iterator to the beginning of the symbol sequence
 * @param[in] end Forward iterator to one past the last element of the symbol sequence
 * @param[in] init_state The starting state of the finite-state machine
 * @param[in] symbol_group_lut Sequence container of symbol groups. Each symbol group is a sequence
 * container to symbols within that group. The index of the symbol group containing a symbol being
 * read will be used as symbol_gid of the transition and translation tables.
 * @param[in] transition_table The two-dimensional transition table, i.e.,
 * transition_table[state][symbol_gid] -> new_state
 * @param[in] translation_table The two-dimensional transducer table, i.e.,
 * translation_table[state][symbol_gid] -> range_of_output_symbols
 * @param[out] out_tape A forward output iterator to which the transduced input will be written
 * @param[out] out_index_tape A forward output iterator to which indexes of the symbols that
 * actually caused some output are written to
 * @return A pair of iterators to one past the last element of (1) the transduced output symbol
 * sequence and (2) the indexes of
 */
template <typename InputItT,
          typename StateT,
          typename SymbolGroupLutT,
          typename TransitionTableT,
          typename TransducerTableT,
          typename OutputItT,
          typename IndexOutputItT>
static std::pair<OutputItT, IndexOutputItT> fst_baseline(InputItT begin,
                                                         InputItT end,
                                                         StateT const& init_state,
                                                         SymbolGroupLutT symbol_group_lut,
                                                         TransitionTableT transition_table,
                                                         TransducerTableT translation_table,
                                                         OutputItT out_tape,
                                                         IndexOutputItT out_index_tape)
{
  // Initialize "FSM" with starting state
  StateT state = init_state;

  // To track the symbol offset within the input that caused the FST to output
  std::size_t in_offset = 0;
  for (auto it = begin; it < end; it++) {
    // The symbol currently being read
    auto const& symbol = *it;

    // Iterate over symbol groups and search for the first symbol group containing the current
    // symbol, if no match is found we use cend(symbol_group_lut) as the "catch-all" symbol group
    auto symbol_group_it =
      std::find_if(std::cbegin(symbol_group_lut), std::cend(symbol_group_lut), [symbol](auto& sg) {
        return std::find(std::cbegin(sg), std::cend(sg), symbol) != std::cend(sg);
      });
    auto symbol_group = std::distance(std::cbegin(symbol_group_lut), symbol_group_it);

    // Output the translated symbols to the output tape
    out_tape = std::copy(std::cbegin(translation_table[state][symbol_group]),
                         std::cend(translation_table[state][symbol_group]),
                         out_tape);

    auto out_size = std::distance(std::cbegin(translation_table[state][symbol_group]),
                                  std::cend(translation_table[state][symbol_group]));

    out_index_tape = std::fill_n(out_index_tape, out_size, in_offset);

    // Transition the state of the finite-state machine
    state = static_cast<char>(transition_table[state][symbol_group]);

    // Continue with next symbol from input tape
    in_offset++;
  }
  return {out_tape, out_index_tape};
}
}  // namespace

// Base test fixture for tests
struct FstTest : public cudf::test::BaseFixture {};

TEST_F(FstTest, GroundTruth)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT = char;

  // Type sufficiently large to index symbols within the input and output (may be unsigned)
  using SymbolOffsetT = uint32_t;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"(  {)"
                      R"("category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(~  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  {} [] [ ])";

  size_t string_size                 = input.size() * (1 << 10);
  auto d_input_scalar                = cudf::make_string_scalar(input);
  auto& d_string_scalar              = static_cast<cudf::string_scalar&>(*d_input_scalar);
  cudf::size_type const repeat_times = string_size / input.size();
  auto d_input_string                = cudf::strings::repeat_string(d_string_scalar, repeat_times);
  auto& d_input = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_string);
  input         = d_input.to_string(stream);

  // Prepare input & output buffers
  constexpr std::size_t single_item = 1;
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(input.size(), stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> out_indexes_gpu(input.size(), stream_view);

  // Run algorithm
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(pda_state_tt),
    cudf::io::fst::detail::make_translation_table<TT_NUM_STATES * NUM_SYMBOL_GROUPS,
                                                  min_translated_out,
                                                  max_translated_out>(pda_out_tt),
    stream);

  // Allocate device-side temporary storage & run algorithm
  parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   output_gpu.device_ptr(),
                   out_indexes_gpu.device_ptr(),
                   output_gpu_size.device_ptr(),
                   start_state,
                   stream.value());

  // Async copy results from device to host
  output_gpu.device_to_host_async(stream.view());
  out_indexes_gpu.device_to_host_async(stream.view());
  output_gpu_size.device_to_host_async(stream.view());

  // Prepare CPU-side results for verification
  std::string output_cpu{};
  std::vector<SymbolOffsetT> out_index_cpu{};
  output_cpu.reserve(input.size());
  out_index_cpu.reserve(input.size());

  // Run CPU-side algorithm
  fst_baseline(std::begin(input),
               std::end(input),
               start_state,
               pda_sgs,
               pda_state_tt,
               pda_out_tt,
               std::back_inserter(output_cpu),
               std::back_inserter(out_index_cpu));

  // Make sure results have been copied back to host
  stream.synchronize();

  // Verify results
  ASSERT_EQ(output_gpu_size[0], output_cpu.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output_cpu, output_cpu.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(out_indexes_gpu, out_index_cpu, output_cpu.size());
}

CUDF_TEST_PROGRAM_MAIN()
