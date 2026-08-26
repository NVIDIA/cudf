/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/filling.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/functional>
#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

void bm_tdigest_merge(nvbench::state& state)
{
  auto const num_tdigests = static_cast<cudf::size_type>(state.get_int64("num_tdigests"));
  auto const tdigest_size = static_cast<cudf::size_type>(state.get_int64("tdigest_size"));
  auto const tdigests_per_group =
    static_cast<cudf::size_type>(state.get_int64("tdigests_per_group"));
  auto const max_centroids   = static_cast<cudf::size_type>(state.get_int64("max_centroids"));
  auto const num_groups      = num_tdigests / tdigests_per_group;
  auto const total_centroids = num_tdigests * tdigest_size;

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  constexpr int base_value = 5;

  // construct inner means/weights
  auto val_iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<double>([tdigest_size](cudf::size_type i) {
      return static_cast<double>(base_value + (i % tdigest_size));
    }));
  auto one_iter = cuda::make_constant_iterator(1);
  cudf::test::fixed_width_column_wrapper<double> means(val_iter, val_iter + total_centroids);
  cudf::test::fixed_width_column_wrapper<double> weights(one_iter, one_iter + total_centroids);
  std::vector<std::unique_ptr<cudf::column>> inner_struct_children;
  inner_struct_children.push_back(means.release());
  inner_struct_children.push_back(weights.release());
  cudf::test::structs_column_wrapper inner_struct(std::move(inner_struct_children));

  // construct the tdigest lists themselves
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cudf::size_type>([tdigest_size](cudf::size_type i) {
      return i * tdigest_size;
    }));
  cudf::test::fixed_width_column_wrapper<int> offsets(offset_iter, offset_iter + num_tdigests + 1);
  auto list_col =
    cudf::make_lists_column(num_tdigests, offsets.release(), inner_struct.release(), 0, {});

  // min and max columns
  auto min_iter = cuda::make_constant_iterator(base_value);
  auto max_iter = cuda::make_constant_iterator(base_value + (tdigest_size - 1));
  cudf::test::fixed_width_column_wrapper<double> mins(min_iter, min_iter + num_tdigests);
  cudf::test::fixed_width_column_wrapper<double> maxes(max_iter, max_iter + num_tdigests);

  // assemble the whole thing
  std::vector<std::unique_ptr<cudf::column>> tdigest_children;
  tdigest_children.push_back(std::move(list_col));
  tdigest_children.push_back(mins.release());
  tdigest_children.push_back(maxes.release());
  cudf::test::structs_column_wrapper tdigest(std::move(tdigest_children));

  // group offsets, labels
  auto zero       = cudf::numeric_scalar<cudf::size_type>(0);
  auto indices    = cudf::sequence(num_tdigests, zero);
  auto tpg_scalar = cudf::numeric_scalar<cudf::size_type>(tdigests_per_group);

  auto group_offsets = cudf::sequence(num_groups + 1, zero, tpg_scalar, stream, mr);
  // expand 0, 1, 2, 3, 4, into 0, 0, 0, 1, 1, 1, 2, 2, 2, etc
  auto group_labels = std::move(
    cudf::repeat(cudf::table_view({cudf::slice(indices->view(), {0, num_groups}).front()}),
                 tdigests_per_group,
                 stream,
                 mr)
      ->release()
      .front());

  stream.synchronize();

  state.add_element_count(total_centroids);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               // re-fetch mr so allocations are routed through the statistics adaptor
               auto mr = cudf::get_current_device_resource_ref();
               timer.start();
               auto result = cudf::tdigest::detail::group_merge_tdigest(tdigest,
                                                                        group_offsets->view(),
                                                                        group_labels->view(),
                                                                        num_groups,
                                                                        max_centroids,
                                                                        stream,
                                                                        mr);
               timer.stop();
             });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bm_tdigest_reduce(nvbench::state& state)
{
  auto const rows_per_group = static_cast<cudf::size_type>(state.get_int64("rows_per_group"));
  auto const num_groups     = static_cast<cudf::size_type>(state.get_int64("num_groups"));
  auto const num_rows       = rows_per_group * num_groups;
  auto const max_centroids  = static_cast<cudf::size_type>(state.get_int64("max_centroids"));

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // construct input values
  auto zero  = cudf::numeric_scalar<cudf::size_type>(0);
  auto input = cudf::sequence(num_rows, zero);

  // group offsets, labels, valid counts
  auto rpg_scalar = cudf::numeric_scalar<cudf::size_type>(rows_per_group);

  auto group_offsets = cudf::sequence(num_groups + 1, zero, rpg_scalar, stream, mr);
  // expand 0, 1, 2, 3, 4, into 0, 0, 0, 1, 1, 1, 2, 2, 2, etc
  auto group_labels =
    std::move(cudf::repeat(cudf::table_view({cudf::slice(input->view(), {0, num_groups}).front()}),
                           rows_per_group,
                           stream,
                           mr)
                ->release()
                .front());
  auto group_valid_counts = cudf::sequence(num_groups, rpg_scalar, zero);

  stream.synchronize();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               // re-fetch mr so allocations are routed through the statistics adaptor
               auto mr = cudf::get_current_device_resource_ref();
               timer.start();
               auto result = cudf::tdigest::detail::group_tdigest(*input,
                                                                  group_offsets->view(),
                                                                  group_labels->view(),
                                                                  group_valid_counts->view(),
                                                                  num_groups,
                                                                  max_centroids,
                                                                  stream,
                                                                  mr);
               timer.stop();
             });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bm_tdigest_merge)
  .set_name("merge-many-tiny")
  .add_int64_axis("num_tdigests", {500'000})
  .add_int64_axis("tdigest_size", {1, 1000})
  .add_int64_axis("tdigests_per_group", {1})
  .add_int64_axis("max_centroids", {10000, 1000});

NVBENCH_BENCH(bm_tdigest_merge)
  .set_name("merge-many-small")
  .add_int64_axis("num_tdigests", {500'000})
  .add_int64_axis("tdigest_size", {1, 1000})
  .add_int64_axis("tdigests_per_group", {3})
  .add_int64_axis("max_centroids", {10000, 1000});

NVBENCH_BENCH(bm_tdigest_reduce)
  .set_name("reduce-many-small")
  .add_int64_axis("num_groups", {2000})
  .add_int64_axis("rows_per_group", {1, 32, 100})
  .add_int64_axis("max_centroids", {10000, 1000});

NVBENCH_BENCH(bm_tdigest_reduce)
  .set_name("reduce-few-large")
  .add_int64_axis("num_groups", {1, 16, 64})
  .add_int64_axis("rows_per_group", {5'000'000, 1'000'000})
  .add_int64_axis("max_centroids", {10000, 1000});
