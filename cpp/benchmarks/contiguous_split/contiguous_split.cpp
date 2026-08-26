/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

void contiguous_split(cudf::table_view const& src_table, std::vector<cudf::size_type> const& splits)
{
  auto result = cudf::contiguous_split(src_table, splits);
}

void chunked_pack(cudf::table_view const& src_table, std::vector<cudf::size_type> const&)
{
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const stream = cudf::get_default_stream();
  auto user_buffer  = rmm::device_uvector<std::uint8_t>(100L * 1024 * 1024, stream, mr);
  auto chunked_pack = cudf::chunked_pack::create(src_table, user_buffer.size());
  while (chunked_pack->has_next()) {
    (void)chunked_pack->next(user_buffer);
  }
  stream.synchronize();
}

template <typename ContigSplitImpl>
void contiguous_split_common(nvbench::state& state,
                             std::vector<std::unique_ptr<cudf::column>>& src_cols,
                             int64_t num_rows,
                             int64_t num_splits,
                             int64_t bytes_total,
                             ContigSplitImpl& impl)
{
  // generate splits
  std::vector<cudf::size_type> splits;
  if (num_splits > 0) {
    cudf::size_type const split_stride = num_rows / num_splits;
    // start after the first element.
    auto iter = cuda::counting_iterator<cudf::size_type>{1};
    splits.reserve(num_splits);
    std::transform(iter,
                   iter + num_splits,
                   std::back_inserter(splits),
                   [split_stride, num_rows](cudf::size_type i) {
                     return std::min(i * split_stride, static_cast<cudf::size_type>(num_rows));
                   });
  }

  auto const src_table = cudf::table(std::move(src_cols));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(src_table.alloc_size());
  state.add_global_memory_writes<int8_t>(src_table.alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { impl(src_table, splits); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_contiguous_split_strings(nvbench::state& state);

void bench_contiguous_split(nvbench::state& state)
{
  if (state.get_int64("string")) {
    bench_contiguous_split_strings(state);
    return;
  }

  auto const total_desired_bytes = 1L * 1024L * 1024L * 1024L;
  auto const num_cols            = state.get_int64("num_cols");
  auto const num_splits          = state.get_int64("num_splits");
  auto const include_validity    = static_cast<bool>(state.get_int64("oob_policy"));

  int64_t const num_rows = total_desired_bytes / (num_cols * sizeof(int32_t));

  // generate input table
  auto builder = data_profile_builder().cardinality(0).distribution<int32_t>(
    cudf::type_id::INT32, distribution_id::UNIFORM);
  if (not include_validity) builder.no_validity();

  auto src_cols = create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                                      row_count{static_cast<cudf::size_type>(num_rows)},
                                      data_profile{builder})
                    ->release();

  int64_t const total_bytes =
    total_desired_bytes +
    (include_validity ? (std::max(1L, (num_rows / 32)) * sizeof(cudf::bitmask_type) * num_cols)
                      : 0);

  if (state.get_int64("chunked")) {
    contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, chunked_pack);
  } else {
    contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, contiguous_split);
  }
}

void bench_contiguous_split_strings(nvbench::state& state)
{
  auto const total_desired_bytes = 1L * 1024L * 1024L * 1024L;
  auto const num_cols            = state.get_int64("num_cols");
  auto const num_splits          = state.get_int64("num_splits");
  auto const include_validity    = static_cast<bool>(state.get_int64("oob_policy"));

  auto const one_col = cudf::test::strings_column_wrapper({"aaaaaaaa",
                                                           "bbbbbbbb",
                                                           "cccccccc",
                                                           "dddddddd",
                                                           "eeeeeeee",
                                                           "ffffffff",
                                                           "gggggggg",
                                                           "hhhhhhhh"});
  auto const one_sv  = cudf::strings_column_view(one_col);

  constexpr auto string_len = 8;
  auto const col_len_bytes  = total_desired_bytes / num_cols;
  auto const num_rows       = col_len_bytes / string_len;
  auto const index_type     = cudf::type_to_id<cudf::size_type>();

  // generate input table
  data_profile profile = data_profile_builder().no_validity().cardinality(0).distribution(
    index_type,
    distribution_id::UNIFORM,
    0,
    (include_validity ? one_sv.size() * 2 : one_sv.size() - 1));  // out of bounds nullified
  std::vector<std::unique_ptr<cudf::column>> src_cols(num_cols);
  for (auto idx = 0L; idx < num_cols; ++idx) {
    auto random_indices = create_random_column(
      cudf::type_id::INT32, row_count{static_cast<cudf::size_type>(num_rows)}, profile);
    auto str_table = cudf::gather(cudf::table_view{{one_col}},
                                  *random_indices,
                                  (include_validity ? cudf::out_of_bounds_policy::NULLIFY
                                                    : cudf::out_of_bounds_policy::DONT_CHECK));
    src_cols[idx]  = std::move(str_table->release()[0]);
  }

  int64_t const total_bytes =
    total_desired_bytes + ((num_rows + 1) * sizeof(cudf::size_type)) +
    (include_validity
       ? (std::max(int64_t{1}, (num_rows / 32)) * sizeof(cudf::bitmask_type) * num_cols)
       : 0);

  if (state.get_int64("chunked")) {
    contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, chunked_pack);
  } else {
    contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, contiguous_split);
  }
}

NVBENCH_BENCH(bench_contiguous_split)
  .set_name("contiguous_split")
  .add_int64_axis("num_cols", {1, 4, 10, 512})
  .add_int64_axis("num_splits", {0, 256})
  .add_int64_axis("oob_policy", {false, true})
  .add_int64_axis("chunked", {false, true})
  .add_int64_axis("string", {false, true});
