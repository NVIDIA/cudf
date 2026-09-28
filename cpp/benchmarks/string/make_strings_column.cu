/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

#include <vector>

namespace {

constexpr int min_row_width = 0;
constexpr int max_row_width = 50;

using string_index_pair = cuda::std::pair<char const*, cudf::size_type>;

template <bool batch_construction>
std::vector<std::unique_ptr<cudf::column>> make_strings_columns(
  std::vector<cudf::device_span<string_index_pair const>> const& input, cuda::stream_ref stream)
{
  if constexpr (batch_construction) {
    return cudf::make_strings_column_batch(input, stream);
  } else {
    std::vector<std::unique_ptr<cudf::column>> output;
    output.reserve(input.size());
    for (auto const& column_input : input) {
      output.emplace_back(cudf::make_strings_column(column_input, stream));
    }
    return output;
  }
}

}  // namespace

static void BM_make_strings_column_batch(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const has_nulls  = true;

  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_row_width, max_row_width)
      .null_probability(has_nulls ? std::optional<double>{0.1} : std::nullopt);
  auto const data_table = create_random_table(
    cycle_dtypes({cudf::type_id::STRING}, batch_size), row_count{num_rows}, table_profile);

  auto const stream = cudf::get_default_stream();
  auto input_data   = std::vector<rmm::device_uvector<string_index_pair>>{};
  auto input        = std::vector<cudf::device_span<string_index_pair const>>{};
  input_data.reserve(batch_size);
  input.reserve(batch_size);
  for (auto const& cv : data_table->view()) {
    auto const d_data_ptr = cudf::column_device_view::create(cv, stream);
    auto batch_input      = rmm::device_uvector<string_index_pair>(cv.size(), stream);
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     batch_input.begin(),
                     batch_input.end(),
                     [data_col = *d_data_ptr] __device__(auto const idx) {
                       if (data_col.is_null(idx)) { return string_index_pair{nullptr, 0}; }
                       auto const row = data_col.element<cudf::string_view>(idx);
                       return string_index_pair{row.data(), row.size_bytes()};
                     });
    input_data.emplace_back(std::move(batch_input));
    input.emplace_back(input_data.back());
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto const output = make_strings_columns<true>(input, stream);
  });
}

NVBENCH_BENCH(BM_make_strings_column_batch)
  .set_name("make_strings_column_batch")
  .add_int64_axis("num_rows", {100'000, 500'000, 1'000'000, 2'000'000})
  .add_int64_axis("batch_size", {10, 20, 50, 100});
