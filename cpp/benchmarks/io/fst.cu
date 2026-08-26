/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/fst/lookup_tables.cuh"
#include "io/utilities/hostdevice_vector.hpp"  //TODO find better replacement

#include <benchmarks/common/generate_input.hpp>
#include <tests/io/fst/common.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>

#include <nvbench/nvbench.cuh>

#include <cstdlib>

namespace {
auto make_test_json_data(nvbench::state& state)
{
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};

  // Test input
  std::string input = R"(  {)"
                      R"("category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  {} [] [ ])";

  auto d_input_scalar                = cudf::make_string_scalar(input);
  auto& d_string_scalar              = static_cast<cudf::string_scalar&>(*d_input_scalar);
  cudf::size_type const repeat_times = string_size / input.size();
  return cudf::strings::repeat_string(d_string_scalar, repeat_times);
}

// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT                       = uint32_t;
constexpr std::size_t single_item         = 1;
constexpr auto max_translation_table_size = TT_NUM_STATES * NUM_SYMBOL_GROUPS;

}  // namespace

void BM_FST_JSON(nvbench::state& state)
{
  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<cudf::size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  cuda::stream_ref stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(d_input.size(), stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(pda_state_tt),
    cudf::io::fst::detail::make_translation_table<max_translation_table_size,
                                                  min_translated_out,
                                                  max_translated_out>(pda_out_tt),
    stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     output_gpu.device_ptr(),
                     out_indexes_gpu.device_ptr(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_outidx(nvbench::state& state)
{
  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<cudf::size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  cuda::stream_ref stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(d_input.size(), stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(pda_state_tt),
    cudf::io::fst::detail::make_translation_table<max_translation_table_size,
                                                  min_translated_out,
                                                  max_translated_out>(pda_out_tt),
    stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     output_gpu.device_ptr(),
                     cuda::make_discard_iterator(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_out(nvbench::state& state)
{
  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<cudf::size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  cuda::stream_ref stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);

  // Run algorithm
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(pda_state_tt),
    cudf::io::fst::detail::make_translation_table<max_translation_table_size,
                                                  min_translated_out,
                                                  max_translated_out>(pda_out_tt),
    stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     cuda::make_discard_iterator(),
                     cuda::make_discard_iterator(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_str(nvbench::state& state)
{
  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<cudf::size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  cuda::stream_ref stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(pda_state_tt),
    cudf::io::fst::detail::make_translation_table<max_translation_table_size,
                                                  min_translated_out,
                                                  max_translated_out>(pda_out_tt),
    stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     cuda::make_discard_iterator(),
                     out_indexes_gpu.device_ptr(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

NVBENCH_BENCH(BM_FST_JSON)
  .set_name("FST_JSON")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_outidx)
  .set_name("FST_JSON_no_outidx")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_out)
  .set_name("FST_JSON_no_out")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_str)
  .set_name("FST_JSON_no_str")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));
