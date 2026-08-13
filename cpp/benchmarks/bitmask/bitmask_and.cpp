/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

namespace {

constexpr float null_probability = 0.5;

auto setup_masks(nvbench::state& state)
{
  unsigned seed = 12345;

  // Get benchmark parameters
  auto const num_segments = static_cast<size_t>(state.get_int64("num_segments"));
  auto const expected_masks_per_segment =
    static_cast<size_t>(state.get_int64("expected_masks_per_segment"));
  auto const mask_size_bits = static_cast<size_t>(state.get_int64("mask_size_bits"));

  // Create segments
  std::mt19937 generator(seed);
  std::normal_distribution normal_dist(static_cast<double>(expected_masks_per_segment), 1.0);
  std::vector<cudf::size_type> segments(num_segments + 1);
  auto num_masks = 0;
  std::generate_n(segments.begin(), num_segments, [&normal_dist, &generator, &num_masks]() {
    cudf::size_type segment_size = normal_dist(generator);
    num_masks += segment_size;
    return segment_size;
  });
  std::exclusive_scan(segments.begin(), segments.end(), segments.begin(), 0);

  // Create masks
  std::vector<rmm::device_buffer> masks;
  std::vector<cudf::bitmask_type*> mask_pointers;
  masks.reserve(num_masks);
  std::generate_n(std::back_inserter(masks), num_masks, [mask_size_bits, seed, &mask_pointers]() {
    auto mask_pair = create_random_null_mask(mask_size_bits, null_probability, seed);
    mask_pointers.push_back(static_cast<cudf::bitmask_type*>(mask_pair.first.data()));
    return std::move(mask_pair.first);
  });

  // Create mask offsets
  std::vector<cudf::size_type> mask_begin_bits(num_masks, 0);

  // Total bytes processed
  auto const data_bytes = num_masks * std::ceil(static_cast<double>(mask_size_bits) / 8) +
                          (sizeof(cudf::size_type) * (num_masks + num_segments));

  return std::tuple{std::move(segments),
                    std::move(masks),
                    std::move(mask_pointers),
                    std::move(mask_begin_bits),
                    data_bytes};
}

void BM_segmented_bitmask_and(nvbench::state& state)
{
  auto const mask_size_bits = static_cast<size_t>(state.get_int64("mask_size_bits"));

  auto [segments, masks, mask_pointers, mask_begin_bits, data_bytes] = setup_masks(state);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(data_bytes, "input size");
  state.template add_global_memory_reads<nvbench::int8_t>(data_bytes);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::segmented_bitmask_and(mask_pointers, segments, mask_size_bits);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

void BM_multi_segment_bitmask_and(nvbench::state& state)
{
  auto const mask_size_bits = static_cast<size_t>(state.get_int64("mask_size_bits"));
  auto const num_segments   = static_cast<size_t>(state.get_int64("num_segments"));

  auto [segments, masks, mask_pointers, mask_begin_bits, data_bytes] = setup_masks(state);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(data_bytes, "input size");
  state.template add_global_memory_reads<nvbench::int8_t>(data_bytes);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    for (size_t i = 0; i < num_segments; i++) {
      auto segment_size = segments[i + 1] - segments[i];
      if (segment_size) {
        auto result = cudf::bitmask_and(
          cudf::host_span<cudf::bitmask_type*>(mask_pointers.data() + segments[i], segment_size),
          cudf::host_span<cudf::size_type>(mask_begin_bits.data() + segments[i], segment_size),
          mask_size_bits);
      }
    }
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

}  // anonymous namespace

/*
 * The benchmarks included in this file measure the performance of segmented reduction operation
 * , with the operator being bitwise AND, and the element being a bitmask vector.
 *
 * This operation is an important kernel in the construction of struct columns, and can be
 * particularly expensive for wide tables with deeply nested struct columns. The benchmark axes
 * chosen capture table properties such as child column count and table width (`num_segments`),
 * expected depth of nesting (`expected_masks_per_segment`), and table row count (`mask_size_bits`).
 *
 * `BM_segmented_bitmask_and` performs the segmented reduction in a single kernel, while
 * `BM_multi_segment_bitmask_and` performs the segmented reduction by launching a reduction kernel
 * for each segment iteratively.
 */
NVBENCH_BENCH(BM_segmented_bitmask_and)
  .set_name("segmented_bitmask_and")
  .add_int64_axis("num_segments", {100, 1000, 10000})
  .add_int64_axis("expected_masks_per_segment", {4, 8, 16})
  .add_int64_axis("mask_size_bits", {32, 64, 128});

NVBENCH_BENCH(BM_multi_segment_bitmask_and)
  .set_name("multi_segment_bitmask_and")
  .add_int64_axis("num_segments", {100, 1000, 10000})
  .add_int64_axis("expected_masks_per_segment", {4, 8, 16})
  .add_int64_axis("mask_size_bits", {32, 64, 128});
