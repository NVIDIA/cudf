/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/tokenize.hpp>

#include <nvbench/nvbench.cuh>

static void bench_vocab_tokenize(nvbench::state& state)
{
  auto const stream    = cudf::get_default_stream();
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  auto const column = [num_rows, min_width, max_width] {
    data_profile const profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
    auto const col = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
    return cudf::strings::filter_characters_of_type(
      cudf::strings_column_view(col->view()),
      cudf::strings::string_character_types::ALL_TYPES,
      cudf::string_scalar(" "),
      cudf::strings::string_character_types::ALPHANUM);
  }();
  cudf::strings_column_view input(column->view());

  auto const vocab_col = [] {
    data_profile const profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, 0, 15);
    auto const col = create_random_column(cudf::type_id::STRING, row_count{100}, profile);
    return cudf::strings::filter_characters_of_type(
      cudf::strings_column_view(col->view()),
      cudf::strings::string_character_types::ALL_TYPES,
      cudf::string_scalar(""),
      cudf::strings::string_character_types::ALPHANUM);
  }();
  auto const vocab = nvtext::load_vocabulary(cudf::strings_column_view(vocab_col->view()));

  auto token_count = [input, stream] {
    auto const counts = nvtext::count_tokens(input);
    auto const agg    = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto const count  = cudf::reduce(counts->view(), *agg, counts->type());
    return static_cast<cudf::scalar_type_t<cudf::size_type>*>(count.get())->value(stream);
  }();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto chars_size =
    input.chars_size(stream) + cudf::strings_column_view(vocab_col->view()).chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(token_count);

  auto const delimiter        = cudf::string_scalar("");
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::tokenize_with_vocabulary(input, *vocab, delimiter);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_vocab_tokenize)
  .set_name("vocab_tokenize")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
