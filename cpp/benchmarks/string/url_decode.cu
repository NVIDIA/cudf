/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <nvbench/nvbench.cuh>

struct url_string_generator {
  cudf::column_device_view d_strings;
  double esc_seq_chance;
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> esc_seq_dist{0, 1};

  __device__ void operator()(cudf::size_type idx)
  {
    engine.discard(idx);
    auto d_str = d_strings.element<cudf::string_view>(idx);
    auto chars = const_cast<char*>(d_str.data());
    for (auto i = 0; i < d_str.size_bytes() - 3; ++i) {
      if (esc_seq_dist(engine) < esc_seq_chance) {
        chars[i]     = '%';
        chars[i + 1] = '2';
        chars[i + 2] = '0';
        i += 2;
      }
    }
  }
};

auto generate_column(cudf::size_type num_rows, cudf::size_type chars_per_row, double esc_seq_chance)
{
  auto str_row    = std::string(chars_per_row, 'a');
  auto result_col = cudf::make_column_from_scalar(cudf::string_scalar(str_row), num_rows);
  auto d_strings  = cudf::column_device_view::create(result_col->view());

  auto engine = thrust::default_random_engine{};
  thrust::for_each_n(thrust::device,
                     cuda::counting_iterator<cudf::size_type>{0},
                     num_rows,
                     url_string_generator{*d_strings, esc_seq_chance, engine});
  return result_col;
}

static void bench_url_decode(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width   = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const esc_seq_pct = static_cast<cudf::size_type>(state.get_int64("esc_seq_pct"));

  auto column = generate_column(num_rows, row_width, esc_seq_pct / 100.0);
  auto input  = cudf::strings_column_view(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);

  {
    auto result = cudf::strings::url_decode(input);
    auto sv     = cudf::strings_column_view(result->view());
    state.add_global_memory_writes<nvbench::int8_t>(sv.chars_size(stream));
  }

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::url_decode(input); });
}

NVBENCH_BENCH(bench_url_decode)
  .set_name("url_decode")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("esc_seq_pct", {10, 50});
