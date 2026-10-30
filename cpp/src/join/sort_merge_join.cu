/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_merge_join.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_copy.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <memory>
#include <utility>

namespace cudf {

namespace {

auto make_cub_env(cuda::stream_ref stream)
{
  auto mr_prop = cuda::std::execution::prop{cuda::mr::get_memory_resource,
                                            cudf::get_current_device_resource_ref()};
  auto env     = cuda::std::execution::env{cuda::stream_ref{stream.get()}, mr_prop};
  return env;
}

/**
 * @brief Functor to map indices through a provided mapping container.
 *
 * Non-negative indices are mapped through the container, while negative indices
 * (e.g., JoinNoMatch) are passed through unchanged.
 *
 * @tparam T Type of the mapping container
 */
template <typename T>
struct index_mapping {
  T mapping;  ///< Mapping container that translates indices

  __device__ size_type operator()(size_type idx) const noexcept
  {
    return idx >= 0 ? mapping[idx] : idx;
  }
};

/**
 * @brief Functor to filter and update validity masks for list columns.
 *
 * Propagates null information from a reduced validity mask to specific child positions
 * in the output validity mask for nested list columns.
 */
struct list_nonnull_filter {
  bitmask_type* validity_mask;                   ///< Output validity mask to update
  bitmask_type const* reduced_validity_mask;     ///< Input reduced validity mask
  device_span<size_type const> child_positions;  ///< Positions in the child column
  size_type subset_offset;                       ///< Offset into child_positions

  __device__ void operator()(size_type idx) const noexcept
  {
    if (!bit_is_set(reduced_validity_mask, idx)) {
      clear_bit(validity_mask, child_positions[idx + subset_offset]);
    }
  };
};

/**
 * @brief Functor to check if a row is valid in an unprocessed table.
 *
 * Maps table row indices to boolean values based on the validity mask.
 */
struct is_row_valid {
  bitmask_type const* _validity_mask;  ///< Validity mask for the table

  __device__ bool operator()(size_type idx) const noexcept
  {
    return bit_is_set(_validity_mask, idx);
  }
};

/**
 * @brief Functor to identify null rows for left join with unequal null semantics.
 *
 * Returns true for rows that have nulls and were filtered out during preprocessing.
 */
struct is_row_null {
  bitmask_type const* const _validity_mask;  ///< Validity mask for the table

  __device__ bool operator()(size_type idx) const noexcept
  {
    return !cudf::bit_is_set(_validity_mask, idx);
  }
};

/**
 * @brief Compact index of the distinct key runs in a sorted table.
 *
 * `offsets` contains one start per run followed by a trailing row-count sentinel.
 */
struct right_run_index {
  std::unique_ptr<cuda::device_buffer<size_type>> rows;     ///< Representative row for each run
  std::unique_ptr<cuda::device_buffer<size_type>> offsets;  ///< Run starts and trailing sentinel
  size_type num_runs;                                       ///< Number of distinct key runs
};

/**
 * @brief Builds run representatives and offsets from a sorted row order.
 *
 * @tparam SortedOrderIterator Random-access iterator over sorted row indices
 * @tparam Less Row comparator type
 * @param sorted_order Iterator over sorted row indices
 * @param num_rows Number of rows in the sorted order
 * @param less Row comparator
 * @param stream CUDA stream used for device operations
 * @return The compact run index
 */
template <typename SortedOrderIterator, typename Less>
right_run_index build_right_run_index(SortedOrderIterator sorted_order,
                                      size_type num_rows,
                                      Less less,
                                      cuda::stream_ref stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  auto env     = make_cub_env(stream);
  auto rows =
    std::make_unique<cuda::device_buffer<size_type>>(stream, temp_mr, num_rows, cuda::no_init);
  auto offsets = std::make_unique<cuda::device_buffer<size_type>>(
    stream, temp_mr, static_cast<std::size_t>(num_rows) + 1, cuda::no_init);
  cudf::detail::device_scalar<size_type> num_runs{0, stream, temp_mr};

  // Keep the expensive row comparator confined to this transform. The subsequent CUB selection
  // kernel consumes only byte flags and integer positions, avoiding another row-operator
  // instantiation and its register pressure.
  cuda::device_buffer<uint8_t> run_starts(stream, temp_mr, num_rows, cuda::no_init);
  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
    cuda::counting_iterator<size_type>{0},
    run_starts.begin(),
    num_rows,
    [sorted_order, less] __device__(size_type idx) -> uint8_t {
      return idx == 0 || less(sorted_order[idx - 1], sorted_order[idx]);
    },
    env));

  auto const input  = cuda::make_zip_iterator(sorted_order, cuda::counting_iterator<size_type>{0});
  auto const output = cuda::make_zip_iterator(rows->begin(), offsets->begin());

  CUDF_CUDA_TRY(
    cub::DeviceSelect::Flagged(input, run_starts.begin(), output, num_runs.data(), num_rows, env));

  auto const host_num_runs = num_runs.value(stream);
  CUDF_CUDA_TRY(cub::DeviceTransform::Fill(offsets->begin() + host_num_runs, 1, num_rows, env));
  return {std::move(rows), std::move(offsets), host_num_runs};
}

/**
 * @brief Builds a run index using the table's lexicographic row comparator.
 *
 * @tparam SortedOrderIterator Random-access iterator over sorted row indices
 * @param table Table whose key runs are indexed
 * @param sorted_order Iterator over sorted row indices
 * @param stream CUDA stream used for device operations
 * @return The compact run index
 */
template <typename SortedOrderIterator>
right_run_index build_right_run_index(table_view const& table,
                                      SortedOrderIterator sorted_order,
                                      cuda::stream_ref stream)
{
  auto const has_nulls = has_nested_nulls(table);
  std::vector<cudf::order> column_order(table.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(table.num_columns(), cudf::null_order::BEFORE);
  auto const row_less =
    detail::row::lexicographic::self_comparator{table, column_order, null_precedence, stream};
  if (cudf::has_nested_columns(table)) {
    return build_right_run_index(
      sorted_order, table.num_rows(), row_less.less<true>(nullate::DYNAMIC{has_nulls}), stream);
  }
  return build_right_run_index(
    sorted_order, table.num_rows(), row_less.less<false>(nullate::DYNAMIC{has_nulls}), stream);
}

/**
 * @brief Produces inner-join output iterator ranges for one probe row.
 */
template <typename SmallerIterator>
struct inner_input_range {
  size_type const* match_starts;  ///< Start of each probe row's matching build run
  SmallerIterator smaller_order;  ///< Iterator over build rows in sorted order

  using iterator_type = decltype(cuda::make_zip_iterator(cuda::constant_iterator<size_type>{},
                                                         std::declval<SmallerIterator>()));

  __device__ iterator_type operator()(size_type idx) const
  {
    return cuda::make_zip_iterator(cuda::constant_iterator<size_type>{idx},
                                   smaller_order + match_starts[idx]);
  }
};

/**
 * @brief Stores the matching build-run range for one probe row.
 */
template <typename Comparator>
struct match_range_output {
  size_type const* unique_smaller_rows;  ///< Representative build row for each run
  size_type const* smaller_run_offsets;  ///< Build-run starts and trailing sentinel
  size_type num_smaller_runs;            ///< Number of build key runs
  size_type* match_starts;               ///< Optional matching-run start output; nullptr skips it
  size_type* match_counts;               ///< Output matching-row count per probe row
  Comparator comparator;                 ///< Probe-to-build row comparator

  __device__ void operator()(size_type idx, size_type run_idx) const
  {
    auto const is_match = run_idx < num_smaller_runs &&
                          !comparator(detail::row::rhs_index_type{idx},
                                      detail::row::lhs_index_type{unique_smaller_rows[run_idx]});
    auto const start = is_match ? smaller_run_offsets[run_idx] : 0;
    if (match_starts != nullptr) { match_starts[idx] = start; }
    match_counts[idx] = is_match ? smaller_run_offsets[run_idx + 1] - start : 0;
  }
};

/**
 * @brief Produces output iterator ranges from per-probe-row offsets.
 */
template <typename OutputIterator>
struct output_range {
  int64_t const* offsets;          ///< Output start for each probe row
  OutputIterator larger_indices;   ///< Probe-side output indices
  OutputIterator smaller_indices;  ///< Build-side output indices

  using iterator_type = decltype(cuda::make_zip_iterator(std::declval<OutputIterator>(),
                                                         std::declval<OutputIterator>()));

  __device__ iterator_type operator()(size_type idx) const
  {
    auto const offset = offsets[idx];
    return cuda::make_zip_iterator(larger_indices + offset, smaller_indices + offset);
  }
};

/**
 * @brief Maps a left-join output offset to its build-side row index.
 */
template <typename SmallerIterator>
struct left_smaller_index {
  size_type idx;                  ///< Probe row index
  size_type const* match_starts;  ///< Matching-run start per probe row
  size_type const* match_counts;  ///< Matching-row count per probe row
  SmallerIterator smaller_order;  ///< Iterator over build rows in sorted order

  __device__ size_type operator()(size_type offset) const
  {
    return match_counts[idx] == 0 ? JoinNoMatch : smaller_order[match_starts[idx] + offset];
  }
};

/**
 * @brief Produces left-join output iterator ranges for one probe row.
 */
template <typename SmallerIterator>
struct left_input_range {
  size_type const* match_starts;  ///< Matching-run start per probe row
  size_type const* match_counts;  ///< Matching-row count per probe row
  SmallerIterator smaller_order;  ///< Iterator over build rows in sorted order

  using smaller_iterator = cuda::transform_iterator<left_smaller_index<SmallerIterator>,
                                                    cuda::counting_iterator<size_type>>;
  using iterator_type    = decltype(cuda::make_zip_iterator(cuda::constant_iterator<size_type>{},
                                                         std::declval<smaller_iterator>()));

  __device__ iterator_type operator()(size_type idx) const
  {
    auto const right_indices = cuda::transform_iterator(
      cuda::counting_iterator<size_type>{0},
      left_smaller_index<SmallerIterator>{idx, match_starts, match_counts, smaller_order});
    return cuda::make_zip_iterator(cuda::constant_iterator<size_type>{idx}, right_indices);
  }
};

struct is_matched {
  __device__ bool operator()(size_type count) const noexcept { return count != 0; }
};

/**
 * @brief Maps a probe row to its build row when every build key is unique.
 *
 * Without match counts, the probe row is known to match.
 */
template <typename SmallerIterator>
struct unique_match_index {
  size_type const* match_starts;  ///< Matching-run start per probe row
  size_type const* match_counts;  ///< Optional matching-row count per probe row
  SmallerIterator smaller_order;  ///< Iterator over build rows in sorted order

  __device__ size_type operator()(size_type idx) const
  {
    if (match_counts != nullptr && match_counts[idx] == 0) { return JoinNoMatch; }
    return smaller_order[match_starts[idx]];
  }
};

template <typename InputIts, typename OutputIts, typename SizeIt>
void batched_copy(InputIts input_iterators,
                  OutputIts output_iterators,
                  SizeIt sizes,
                  size_type num_ranges,
                  cuda::stream_ref stream)
{
  CUDF_CUDA_TRY(cub::DeviceCopy::Batched(
    input_iterators, output_iterators, sizes, num_ranges, make_cub_env(stream)));
}

/// Whether to materialize per-row match start offsets.
enum class compute_match_starts : bool { NO, YES };

template <typename T>
__device__ size_type typed_lower_bound(T const* keys, size_type lo, size_type hi, T key)
{
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T>
__device__ void store_typed_match(T const* run_keys,
                                  size_type num_runs,
                                  size_type const* run_offsets,
                                  size_type pos,
                                  T key,
                                  int64_t row,
                                  size_type* match_starts,
                                  size_type* match_counts)
{
  auto const is_match = pos < num_runs && run_keys[pos] == key;
  auto const start    = is_match ? run_offsets[pos] : 0;
  if (match_starts != nullptr) { match_starts[row] = start; }
  match_counts[row] = is_match ? run_offsets[pos + 1] - start : 0;
}

constexpr size_type typed_search_sample_stride = 32;

/**
 * @brief Finds the matching build run of each probe row with one binary search per row.
 */
template <typename T>
CUDF_KERNEL void typed_search_kernel(T const* run_keys,
                                     size_type num_runs,
                                     T const* samples,
                                     size_type num_samples,
                                     size_type const* run_offsets,
                                     T const* probe_keys,
                                     size_type num_probe_rows,
                                     size_type* match_starts,
                                     size_type* match_counts)
{
  auto const row = cudf::detail::grid_1d::global_thread_id();
  if (row >= num_probe_rows) { return; }
  auto const key = probe_keys[row];
  auto const s   = typed_lower_bound(samples, 0, num_samples, key);
  auto const lo  = s > 0 ? (s - 1) * typed_search_sample_stride + 1 : 0;
  auto const hi  = s < num_samples ? s * typed_search_sample_stride : num_runs;
  auto const pos = typed_lower_bound(run_keys, lo, hi, key);
  store_typed_match(run_keys, num_runs, run_offsets, pos, key, row, match_starts, match_counts);
}

struct typed_probe_fn {
  template <typename T>
  bool operator()(column_view const& build,
                  column_view const& probe,
                  device_span<size_type const> run_rows,
                  device_span<size_type const> run_offsets,
                  size_type* match_starts,
                  size_type* match_counts,
                  cuda::stream_ref stream) const
    requires(cudf::is_integral<T>() || cudf::is_chrono<T>())
  {
    auto const num_runs = static_cast<size_type>(run_rows.size());
    auto run_keys       = cuda::device_buffer<T>(
      stream, cudf::get_current_device_resource_ref(), num_runs, cuda::no_init);
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      run_rows.begin(),
      run_rows.end(),
      run_keys.begin(),
      [keys = build.data<T>()] __device__(size_type row) -> T { return keys[row]; });
    auto const num_samples = cudf::util::div_rounding_up_safe(num_runs, typed_search_sample_stride);
    auto samples           = cuda::device_buffer<T>(
      stream, cudf::get_current_device_resource_ref(), num_samples, cuda::no_init);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{num_samples},
                      samples.begin(),
                      [keys = run_keys.data()] __device__(size_type i) -> T {
                        return keys[i * typed_search_sample_stride];
                      });
    if (probe.size() == 0) { return true; }
    auto const grid = cudf::detail::grid_1d{probe.size(), 256};
    typed_search_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      run_keys.data(),
      num_runs,
      samples.data(),
      num_samples,
      run_offsets.data(),
      probe.data<T>(),
      probe.size(),
      match_starts,
      match_counts);
    CUDF_CHECK_CUDA(stream.get());
    return true;
  }

  template <typename T, typename... Args>
  bool operator()(Args&&...) const
    requires(!(cudf::is_integral<T>() || cudf::is_chrono<T>()))
  {
    return false;
  }
};

struct is_matched_pair {
  template <typename Tuple>
  __device__ bool operator()(Tuple const& t) const noexcept
  {
    return cuda::std::get<1>(t) != JoinNoMatch;
  }
};

constexpr int typed_search_ilp = 2;

template <typename T>
CUDF_KERNEL void typed_unique_probe_kernel(T const* run_keys,
                                           size_type num_runs,
                                           T const* samples,
                                           size_type num_samples,
                                           size_type const* run_rows,
                                           T const* probe_keys,
                                           size_type num_probe_rows,
                                           size_type* smaller_out,
                                           size_type* larger_out,
                                           size_type* num_matched)
{
  int64_t const row0 =
    static_cast<int64_t>(blockIdx.x) * blockDim.x * typed_search_ilp + threadIdx.x;
  T key[typed_search_ilp];
  size_type base[typed_search_ilp];
  size_type len[typed_search_ilp];
#pragma unroll
  for (int k = 0; k < typed_search_ilp; ++k) {
    auto const row = row0 + static_cast<int64_t>(k) * blockDim.x;
    key[k]         = probe_keys[row < num_probe_rows ? row : 0];
    base[k]        = 0;
  }
  size_type n = num_samples;
  while (n > 1) {
    auto const half = n / 2;
#pragma unroll
    for (int k = 0; k < typed_search_ilp; ++k) {
      base[k] = samples[base[k] + half - 1] < key[k] ? base[k] + half : base[k];
    }
    n -= half;
  }
#pragma unroll
  for (int k = 0; k < typed_search_ilp; ++k) {
    auto const s = base[k] + ((n == 1 && samples[base[k]] < key[k]) ? 1 : 0);
    base[k]      = s > 0 ? (s - 1) * typed_search_sample_stride + 1 : 0;
    len[k]       = (s < num_samples ? s * typed_search_sample_stride : num_runs) - base[k];
  }
  constexpr int segment_steps = 5;
  static_assert((1 << segment_steps) == typed_search_sample_stride);
#pragma unroll
  for (int it = 0; it < segment_steps; ++it) {
#pragma unroll
    for (int k = 0; k < typed_search_ilp; ++k) {
      if (len[k] > 1) {
        auto const half = len[k] / 2;
        if (run_keys[base[k] + half - 1] < key[k]) { base[k] += half; }
        len[k] -= half;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < typed_search_ilp; ++k) {
    auto const row = row0 + static_cast<int64_t>(k) * blockDim.x;
    bool matched   = false;
    if (row < num_probe_rows) {
      auto const pos   = base[k] + ((len[k] == 1 && run_keys[base[k]] < key[k]) ? 1 : 0);
      matched          = pos < num_runs && run_keys[pos] == key[k];
      smaller_out[row] = matched ? run_rows[pos] : JoinNoMatch;
      if (larger_out != nullptr) { larger_out[row] = static_cast<size_type>(row); }
    }
    if (num_matched != nullptr) {
      auto const votes = __ballot_sync(0xffffffffu, matched);
      if ((threadIdx.x & 31) == 0 && votes != 0) { atomicAdd(num_matched, __popc(votes)); }
    }
  }
}

struct typed_unique_probe_fn {
  template <typename T>
  void operator()(column_view const& build,
                  column_view const& probe,
                  device_span<size_type const> run_rows,
                  size_type* smaller_out,
                  size_type* larger_out,
                  size_type* num_matched,
                  cuda::stream_ref stream) const
    requires(cudf::is_integral<T>() || cudf::is_chrono<T>())
  {
    auto const num_runs = static_cast<size_type>(run_rows.size());
    auto run_keys       = cuda::device_buffer<T>(
      stream, cudf::get_current_device_resource_ref(), num_runs, cuda::no_init);
    auto const policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
    thrust::transform(
      policy,
      run_rows.begin(),
      run_rows.end(),
      run_keys.begin(),
      [keys = build.data<T>()] __device__(size_type row) -> T { return keys[row]; });
    auto const num_samples = cudf::util::div_rounding_up_safe(num_runs, typed_search_sample_stride);
    auto samples           = cuda::device_buffer<T>(
      stream, cudf::get_current_device_resource_ref(), num_samples, cuda::no_init);
    thrust::transform(policy,
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{num_samples},
                      samples.begin(),
                      [keys = run_keys.data()] __device__(size_type i) -> T {
                        return keys[i * typed_search_sample_stride];
                      });
    if (probe.size() == 0) { return; }
    constexpr int block_size = 256;
    auto const num_blocks =
      cudf::util::div_rounding_up_safe(probe.size(), block_size * typed_search_ilp);
    typed_unique_probe_kernel<<<num_blocks, block_size, 0, stream.get()>>>(run_keys.data(),
                                                                           num_runs,
                                                                           samples.data(),
                                                                           num_samples,
                                                                           run_rows.data(),
                                                                           probe.data<T>(),
                                                                           probe.size(),
                                                                           smaller_out,
                                                                           larger_out,
                                                                           num_matched);
    CUDF_CHECK_CUDA(stream.get());
  }

  template <typename T, typename... Args>
  void operator()(Args&&...) const
    requires(!(cudf::is_integral<T>() || cudf::is_chrono<T>()))
  {
    CUDF_FAIL("Unsupported key type");
  }
};

[[nodiscard]] inline bool unique_typed_eligible(table_view const& smaller, table_view const& larger)
{
  if (smaller.num_columns() != 1 || has_nested_nulls(smaller) || has_nested_nulls(larger)) {
    return false;
  }
  auto const type = smaller.column(0).type();
  return type == larger.column(0).type() && (cudf::is_integral(type) || cudf::is_chrono(type));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
fused_unique_join(bool is_left,
                  table_view const& smaller,
                  table_view const& larger,
                  device_span<size_type const> run_rows,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  auto temp_mr        = cudf::get_current_device_resource_ref();
  auto const n        = larger.num_rows();
  auto const dispatch = [&](size_type* smaller_out, size_type* larger_out, size_type* matched) {
    type_dispatcher(smaller.column(0).type(),
                    typed_unique_probe_fn{},
                    smaller.column(0),
                    larger.column(0),
                    run_rows,
                    smaller_out,
                    larger_out,
                    matched,
                    stream);
  };
  if (is_left) {
    auto smaller_indices = std::make_unique<rmm::device_uvector<size_type>>(n, stream, mr);
    auto larger_indices  = std::make_unique<rmm::device_uvector<size_type>>(n, stream, mr);
    dispatch(smaller_indices->data(), larger_indices->data(), nullptr);
    return {std::move(smaller_indices), std::move(larger_indices)};
  }
  cudf::detail::device_scalar<size_type> num_matched{0, stream, temp_mr};
  rmm::device_uvector<size_type> smaller_tmp(n, stream, temp_mr);
  dispatch(smaller_tmp.data(), nullptr, num_matched.data());
  auto const total     = num_matched.value(stream);
  auto smaller_indices = std::make_unique<rmm::device_uvector<size_type>>(total, stream, mr);
  auto larger_indices  = std::make_unique<rmm::device_uvector<size_type>>(total, stream, mr);
  cudf::detail::device_scalar<size_type> num_selected{0, stream, temp_mr};
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    cuda::make_zip_iterator(cuda::counting_iterator<size_type>{0}, smaller_tmp.begin()),
    cuda::make_zip_iterator(larger_indices->begin(), smaller_indices->begin()),
    num_selected.data(),
    n,
    is_matched_pair{},
    make_cub_env(stream)));
  return {std::move(smaller_indices), std::move(larger_indices)};
}

template <typename SmallerIterator>
class merge {
 private:
  table_view smaller;
  table_view larger;
  SmallerIterator sorted_smaller_order_begin;
  device_span<size_type const> unique_smaller_rows;
  device_span<size_type const> smaller_run_offsets;
  std::unique_ptr<detail::row::lexicographic::two_table_comparator> tt_comparator;

 public:
  struct match_ranges {
    std::unique_ptr<cuda::device_buffer<size_type>> starts;
    std::unique_ptr<rmm::device_uvector<size_type>> counts;
  };

  /// Whether every build row is its own key run, so each probe row matches at most one row.
  [[nodiscard]] bool has_unique_smaller_keys() const
  {
    return static_cast<size_type>(unique_smaller_rows.size()) == smaller.num_rows();
  }

  merge(table_view const& smaller,
        SmallerIterator sorted_smaller_order_begin,
        device_span<size_type const> unique_smaller_rows,
        device_span<size_type const> smaller_run_offsets,
        table_view const& larger,
        cuda::stream_ref stream)
    : smaller{smaller},
      larger{larger},
      sorted_smaller_order_begin{sorted_smaller_order_begin},
      unique_smaller_rows{unique_smaller_rows},
      smaller_run_offsets{smaller_run_offsets}
  {
    std::vector<cudf::order> column_order(smaller.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_precedence(smaller.num_columns(), cudf::null_order::BEFORE);
    tt_comparator = std::make_unique<detail::row::lexicographic::two_table_comparator>(
      smaller, larger, column_order, null_precedence, stream);
  }

  std::unique_ptr<rmm::device_uvector<size_type>> matches_per_row(
    cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  match_ranges find_match_ranges(compute_match_starts compute_starts,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left(cuda::stream_ref stream, rmm::device_async_resource_ref mr);
};

template <typename SmallerIterator>
typename merge<SmallerIterator>::match_ranges merge<SmallerIterator>::find_match_ranges(
  compute_match_starts compute_starts, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const has_nulls        = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows   = larger.num_rows();
  auto const num_smaller_runs = static_cast<size_type>(unique_smaller_rows.size());
  auto match_starts =
    compute_starts == compute_match_starts::YES
      ? std::make_unique<cuda::device_buffer<size_type>>(stream, mr, larger_numrows, cuda::no_init)
      : nullptr;
  auto const match_starts_data = match_starts == nullptr ? nullptr : match_starts->data();
  auto match_counts            = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    static_cast<std::size_t>(larger_numrows) + 1, stream, mr);

  if (!has_nulls && smaller.num_columns() == 1 &&
      smaller.column(0).type() == larger.column(0).type() &&
      type_dispatcher(smaller.column(0).type(),
                      typed_probe_fn{},
                      smaller.column(0),
                      larger.column(0),
                      unique_smaller_rows,
                      smaller_run_offsets,
                      match_starts_data,
                      match_counts.data(),
                      stream)) {
    return {std::move(match_starts),
            std::make_unique<rmm::device_uvector<size_type>>(std::move(match_counts))};
  }

  auto const unique_smaller_it = cuda::transform_iterator(
    unique_smaller_rows.data(),
    cuda::proclaim_return_type<detail::row::lhs_index_type>(
      [] __device__(size_type idx) { return static_cast<detail::row::lhs_index_type>(idx); }));

  auto const find_ranges = [&](auto comparator) {
    auto const ranges_output =
      cuda::tabulate_output_iterator(match_range_output{unique_smaller_rows.data(),
                                                        smaller_run_offsets.data(),
                                                        num_smaller_runs,
                                                        match_starts_data,
                                                        match_counts.data(),
                                                        comparator});
    // These comparisons are data-dependent binary-search probes. Materializing them ahead of
    // time would require a pass per search level (or a quadratic comparison table), so keep them
    // in one bulk search and emit the scalar start/count ranges directly.
    thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        unique_smaller_it,
                        unique_smaller_it + num_smaller_runs,
                        cudf::detail::row::rhs_iterator(0),
                        cudf::detail::row::rhs_iterator(0) + larger_numrows,
                        ranges_output,
                        comparator);
  };

  if (cudf::has_nested_columns(smaller)) {
    find_ranges(tt_comparator->less<true>(nullate::DYNAMIC{has_nulls}));
  } else {
    find_ranges(tt_comparator->less<false>(nullate::DYNAMIC{has_nulls}));
  }

  return {std::move(match_starts),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(match_counts))};
}

template <typename SmallerIterator>
std::unique_ptr<rmm::device_uvector<size_type>> merge<SmallerIterator>::matches_per_row(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return find_match_ranges(compute_match_starts::NO, stream, mr).counts;
}

template <typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<SmallerIterator>::inner(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (has_unique_smaller_keys() && unique_typed_eligible(smaller, larger)) {
    return fused_unique_join(false, smaller, larger, unique_smaller_rows, stream, mr);
  }
  auto temp_mr              = cudf::get_current_device_resource_ref();
  auto const larger_numrows = larger.num_rows();

  auto [match_starts, match_counts] = find_match_ranges(compute_match_starts::YES, stream, temp_mr);

  if (has_unique_smaller_keys()) {
    auto const policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
    auto const counts = match_counts->begin();
    auto const total  = thrust::count_if(policy, counts, counts + larger_numrows, is_matched{});
    rmm::device_uvector<size_type> larger_indices(total, stream, mr);
    rmm::device_uvector<size_type> smaller_indices(total, stream, mr);
    cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                                cuda::counting_iterator<size_type>{larger_numrows},
                                counts,
                                larger_indices.begin(),
                                is_matched{},
                                stream);
    thrust::transform(policy,
                      larger_indices.begin(),
                      larger_indices.end(),
                      smaller_indices.begin(),
                      unique_match_index<SmallerIterator>{
                        match_starts->data(), nullptr, sorted_smaller_order_begin});
    return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
            std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
  }

  // Use 64-bit prefix sums to handle large output sizes (> INT32_MAX rows)
  // The prefix sums can exceed INT32_MAX even though individual match counts are small
  auto match_offsets =
    cudf::detail::make_zeroed_device_buffer_async<int64_t>(match_counts->size(), stream, temp_mr);
  // Use pinned memory as bounce buffer for efficient device-to-host transfer of the last element
  auto last_element =
    cudf::detail::device_scalar<int64_t>(0, stream, cudf::get_current_device_resource_ref());
  auto output_itr = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), last_element.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         match_counts->begin(),
                         match_counts->end(),
                         output_itr,
                         int64_t{0});
  auto const total_matches = static_cast<std::size_t>(last_element.value(stream));

  rmm::device_uvector<size_type> larger_indices(total_matches, stream, mr);
  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);

  auto const row_indices     = cuda::counting_iterator<size_type>{0};
  auto const input_iterators = cuda::transform_iterator(
    row_indices,
    inner_input_range<SmallerIterator>{match_starts->data(), sorted_smaller_order_begin});
  auto const output_iterators = cuda::transform_iterator(
    row_indices,
    output_range{match_offsets.data(), larger_indices.begin(), smaller_indices.begin()});
  batched_copy(input_iterators, output_iterators, match_counts->begin(), larger_numrows, stream);

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

/**
 * @brief Performs a left join between the larger and smaller tables.
 *
 * This method performs a sort-merge left join, ensuring all rows from the larger table
 * are included in the output. Rows with no matches in the smaller table are paired with
 * JoinNoMatch sentinel values. Output indices are grouped in probe-row order.
 *
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource for allocations
 * @return Pair of device vectors containing (smaller_indices, larger_indices)
 */
template <typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<SmallerIterator>::left(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (has_unique_smaller_keys() && unique_typed_eligible(smaller, larger)) {
    return fused_unique_join(true, smaller, larger, unique_smaller_rows, stream, mr);
  }
  auto temp_mr              = cudf::get_current_device_resource_ref();
  auto const larger_numrows = larger.num_rows();

  auto [match_starts, match_counts] = find_match_ranges(compute_match_starts::YES, stream, temp_mr);

  if (has_unique_smaller_keys()) {
    auto const policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
    rmm::device_uvector<size_type> larger_indices(larger_numrows, stream, mr);
    rmm::device_uvector<size_type> smaller_indices(larger_numrows, stream, mr);
    thrust::sequence(policy, larger_indices.begin(), larger_indices.end(), 0);
    thrust::transform(policy,
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{larger_numrows},
                      smaller_indices.begin(),
                      unique_match_index<SmallerIterator>{
                        match_starts->data(), match_counts->data(), sorted_smaller_order_begin});
    return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
            std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
  }

  cudf::detail::device_scalar<int64_t> total_matches(stream, temp_mr);
  auto match_offsets =
    cudf::detail::make_zeroed_device_buffer_async<int64_t>(match_counts->size(), stream, temp_mr);
  auto const output_sizes = cuda::transform_iterator(
    cuda::counting_iterator<size_type>{0},
    [match_counts = match_counts->begin(), larger_numrows] __device__(auto idx) -> size_type {
      if (idx == larger_numrows) { return 0; }
      return cuda::std::max(match_counts[idx], size_type{1});
    });
  auto output_itr = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), total_matches.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         output_sizes,
                         output_sizes + match_counts->size(),
                         output_itr,
                         int64_t{0});
  auto const total_output_size = total_matches.value(stream);
  rmm::device_uvector<size_type> larger_indices(total_output_size, stream, mr);
  rmm::device_uvector<size_type> smaller_indices(total_output_size, stream, mr);

  auto const row_indices     = cuda::counting_iterator<size_type>{0};
  auto const input_iterators = cuda::transform_iterator(
    row_indices,
    left_input_range<SmallerIterator>{
      match_starts->data(), match_counts->data(), sorted_smaller_order_begin});
  auto const output_iterators = cuda::transform_iterator(
    row_indices,
    output_range{match_offsets.data(), larger_indices.begin(), smaller_indices.begin()});
  batched_copy(input_iterators, output_iterators, output_sizes, larger_numrows, stream);

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

}  // anonymous namespace

namespace detail {

void sort_merge_join::preprocessed_table::populate_nonnull_filter(cuda::stream_ref stream)
{
  auto table   = this->_table_view;
  auto temp_mr = cudf::get_current_device_resource_ref();
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(table, stream, temp_mr);

  // If the table has no nullable top-level columns, then we need to create
  // an all-valid bitmask that is passed to subsequent operations. This bitmask
  // is updated if any of the nested struct/list children columns have nulls.
  if (validity_mask.empty())
    validity_mask =
      cudf::create_null_mask(table.num_rows(), mask_state::ALL_VALID, stream, temp_mr);

  // step 2: identify nulls at non-root levels
  for (size_type col_idx = 0; col_idx < table.num_columns(); col_idx++) {
    auto col = table.column(col_idx);
    if (col.type().id() == type_id::LIST) {
      auto lcv     = lists_column_view(col);
      auto offsets = lcv.offsets();
      auto child   = lcv.child();

      cuda::device_buffer<int32_t> offsets_subset(stream, temp_mr, offsets.size(), cuda::no_init);
      cuda::device_buffer<int32_t> child_positions(stream, temp_mr, offsets.size(), cuda::no_init);
      auto unique_end = thrust::unique_by_key_copy(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::std::reverse_iterator(lcv.offsets_end()),
        cuda::std::reverse_iterator(lcv.offsets_end()) + offsets.size(),
        cuda::std::reverse_iterator(cuda::counting_iterator{offsets.size()}),
        cuda::std::reverse_iterator(offsets_subset.end()),
        cuda::std::reverse_iterator(child_positions.end()));
      auto subset_size   = cuda::std::distance(cuda::std::reverse_iterator(offsets_subset.end()),
                                             cuda::std::get<0>(unique_end));
      auto subset_offset = offsets.size() - subset_size;

      auto [reduced_validity_mask, num_nulls] =
        detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                              offsets_subset.data() + subset_offset,
                                              offsets_subset.data() + offsets_subset.size() - 1,
                                              offsets_subset.data() + subset_offset + 1,
                                              null_policy::INCLUDE,
                                              std::nullopt,
                                              stream,
                                              temp_mr);

      thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<cudf::size_type>{0},
                       cuda::counting_iterator<cudf::size_type>{0} + subset_size,
                       list_nonnull_filter{validity_mask.data(),
                                           reduced_validity_mask.data(),
                                           child_positions,
                                           static_cast<size_type>(subset_offset)});
    } else if (col.type().id() == type_id::STRUCT) {
      // Recursive lambda to traverse struct hierarchy and accumulate null information.
      // This lambda ANDs the column's null mask with the accumulated mask in-place and recursively
      // processes all child columns to capture nested nulls.
      auto and_bitmasks = [&](auto&& self, bitmask_type* mask, column_view const& colview) -> void {
        auto const num_rows = colview.size();
        if (colview.type().id() == cudf::type_id::EMPTY) { return; }

        if (colview.nullable()) {
          // AND this column's null mask with the accumulated mask
          auto colmask = colview.null_mask();
          std::vector<bitmask_type const*> masks{colmask, mask};
          std::vector<size_type> begin_bits{0, 0};
          cudf::detail::inplace_bitmask_and(
            device_span<bitmask_type>(mask, num_bitmask_words(num_rows)),
            masks,
            begin_bits,
            num_rows,
            stream);
        }

        if (colview.type().id() == cudf::type_id::STRUCT ||
            colview.type().id() == cudf::type_id::LIST) {
          CUDF_EXPECTS(
            std::all_of(colview.child_begin(),
                        colview.child_end(),
                        [&](auto const& child_col) { return num_rows == child_col.size(); }),
            "Child columns must have the same number of rows as the Struct column.");

          // Recursively process child columns to capture nulls at deeper nesting levels.
          for (auto it = colview.child_begin(); it != colview.child_end(); it++) {
            auto& child = *it;
            self(self, mask, child);
          }
        }
      };
      // Process all children of the struct column
      for (auto it = col.child_begin(); it != col.child_end(); it++) {
        auto& child = *it;
        and_bitmasks(and_bitmasks, validity_mask.data(), child);
      }
    }
  }
  this->_num_nulls     = null_count(validity_mask.data(), 0, table.num_rows(), stream);
  this->_validity_mask = std::move(validity_mask);
}

void sort_merge_join::preprocessed_table::apply_nonnull_filter(cuda::stream_ref stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  // construct bool column to apply mask
  cudf::scalar_type_t<bool> true_scalar(true, true, stream, temp_mr);
  auto bool_mask =
    cudf::make_column_from_scalar(true_scalar, _table_view.num_rows(), stream, temp_mr);
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(),
               "Something went wrong while dropping nulls in the unprocessed tables");
  bool_mask->set_null_mask(_validity_mask.value(), _num_nulls.value(), stream);

  // Use the internal apply_mask directly to avoid the public API overhead (NVTX range).
  _null_processed_table =
    detail::apply_mask(_table_view, *bool_mask, detail::mask_type::RETENTION, stream, temp_mr);
  _null_processed_table_view = _null_processed_table.value()->view();
}

void sort_merge_join::preprocessed_table::preprocess_unprocessed_table(cuda::stream_ref stream)
{
  populate_nonnull_filter(stream);
  apply_nonnull_filter(stream);
}

void sort_merge_join::preprocessed_table::compute_sorted_order(cuda::stream_ref stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::order> column_order(_null_processed_table_view.num_columns(),
                                        cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(_null_processed_table_view.num_columns(),
                                                cudf::null_order::BEFORE);
  this->_null_processed_table_sorted_order =
    cudf::sorted_order(_null_processed_table_view, column_order, null_precedence, stream, temp_mr);
}

sort_merge_join::preprocessed_table sort_merge_join::preprocessed_table::create(
  table_view const& table, null_equality compare_nulls, sorted is_sorted, cuda::stream_ref stream)
{
  preprocessed_table result;
  result._table_view = table;

  if (compare_nulls == null_equality::EQUAL) {
    result._null_processed_table_view = table;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    if (has_nested_nulls(table)) {
      result.preprocess_unprocessed_table(stream);
    } else {
      result._null_processed_table_view = table;
    }
  }

  if (is_sorted == cudf::sorted::NO) { result.compute_sorted_order(stream); }

  return result;
}

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 cuda::stream_ref stream)
  : preprocessed_right{preprocessed_table::create(right, compare_nulls, is_right_sorted, stream)},
    compare_nulls{compare_nulls}
{
  cudf::scoped_range range{"sort_merge_join::sort_merge_join"};
  // Sanity checks
  CUDF_EXPECTS(right.num_columns() != 0,
               "Number of columns the keys table must be non-zero for a join",
               std::invalid_argument);

  auto const& right_view = preprocessed_right._null_processed_table_view;
  auto run_index         = [&] {
    if (preprocessed_right._null_processed_table_sorted_order.has_value()) {
      auto const order = preprocessed_right._null_processed_table_sorted_order.value()->view();
      return build_right_run_index(right_view, order.begin<size_type>(), stream);
    }
    return build_right_run_index(right_view, cuda::counting_iterator<size_type>{0}, stream);
  }();
  right_run_rows    = std::move(run_index.rows);
  right_run_offsets = std::move(run_index.offsets);
  num_right_runs    = run_index.num_runs;
}

cuda::device_buffer<size_type> sort_merge_join::preprocessed_table::map_table_to_unprocessed(
  cuda::stream_ref stream) const
{
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(), "Mapping is not possible");
  auto temp_mr                  = cudf::get_current_device_resource_ref();
  auto const table_mapping_size = _table_view.num_rows() - _num_nulls.value();
  cuda::device_buffer<size_type> table_mapping(stream, temp_mr, table_mapping_size, cuda::no_init);
  cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                              cuda::counting_iterator<size_type>{_table_view.num_rows()},
                              cuda::counting_iterator<size_type>{0},
                              table_mapping.begin(),
                              is_row_valid{_validity_mask.value().data()},
                              stream);
  return table_mapping;
}

void sort_merge_join::postprocess_indices(preprocessed_table const& preprocessed_left,
                                          device_span<size_type> smaller_indices,
                                          device_span<size_type> larger_indices,
                                          cuda::stream_ref stream) const
{
  if (compare_nulls == null_equality::UNEQUAL) {
    auto env = make_cub_env(stream);
    // if a table has no nullable column, then there's no postprocessing to be done
    if (has_nested_nulls(preprocessed_left._table_view)) {
      auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      CUDF_CUDA_TRY(
        cub::DeviceTransform::Transform(larger_indices.begin(),
                                        larger_indices.begin(),
                                        larger_indices.size(),
                                        index_mapping<device_span<size_type>>{left_mapping},
                                        env));
    }
    if (has_nested_nulls(preprocessed_right._table_view)) {
      auto right_mapping = preprocessed_right.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      CUDF_CUDA_TRY(
        cub::DeviceTransform::Transform(smaller_indices.begin(),
                                        smaller_indices.begin(),
                                        smaller_indices.size(),
                                        index_mapping<device_span<size_type>>{right_mapping},
                                        env));
    }
  }
}

template <typename MergeOperation>
auto sort_merge_join::invoke_merge(table_view right_view,
                                   table_view left_view,
                                   MergeOperation&& op,
                                   cuda::stream_ref stream) const
{
  auto const unique_right_rows =
    device_span<size_type const>{right_run_rows->data(), static_cast<std::size_t>(num_right_runs)};
  auto const right_offsets = device_span<size_type const>{
    right_run_offsets->data(), static_cast<std::size_t>(num_right_runs) + 1};
  auto has_right_sorting_order = preprocessed_right._null_processed_table_sorted_order.has_value();
  if (has_right_sorting_order) {
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    merge obj(
      right_view, r_view.begin<size_type>(), unique_right_rows, right_offsets, left_view, stream);
    return op(obj);
  }
  merge obj(right_view,
            cuda::counting_iterator<cudf::size_type>{0},
            unique_right_rows,
            right_offsets,
            left_view,
            stream);
  return op(obj);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Match discovery probes rows in their original order, so skip the unused probe-side sort.
  auto preprocessed_left =
    preprocessed_table::create(left, compare_nulls, cudf::sorted::YES, stream);

  return invoke_merge(
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.inner(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
      return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
    },
    stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::left_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Match discovery probes rows in their original order, so skip the unused probe-side sort.
  auto preprocessed_left =
    preprocessed_table::create(left, compare_nulls, cudf::sorted::YES, stream);

  return invoke_merge(
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.left(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);

      //  For left join with UNEQUAL nulls, we need to add back rows that were filtered out.
      //  Remaining configs can return directly
      if (compare_nulls == null_equality::EQUAL ||
          !has_nested_nulls(preprocessed_left._table_view)) {
        return std::pair{std::move(preprocessed_left_indices),
                         std::move(preprocessed_right_indices)};
      }

      // Special handling for UNEQUAL null semantics with nested nulls:
      // Rows containing nulls were filtered during preprocessing and must be reinserted.
      // These rows have no matches by definition (nulls are unequal), so they're added
      // to the output with JoinNoMatch sentinel values for the right side.

      auto const num_filtered_nulls = preprocessed_left._num_nulls.value();
      auto const total_output_size =
        preprocessed_left_indices->size() + static_cast<int64_t>(num_filtered_nulls);

      auto const validity_mask = preprocessed_left._validity_mask.value().data();
      cuda::device_buffer<size_type> null_left_indices(stream,
                                                       cudf::get_current_device_resource_ref(),
                                                       static_cast<std::size_t>(num_filtered_nulls),
                                                       cuda::no_init);
      cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                                  cuda::counting_iterator<size_type>{left.num_rows()},
                                  cuda::counting_iterator<size_type>{0},
                                  null_left_indices.begin(),
                                  is_row_null{validity_mask},
                                  stream);

      rmm::device_uvector<size_type> left_result_indices(total_output_size, stream, mr);
      rmm::device_uvector<size_type> right_result_indices(total_output_size, stream, mr);
      CUDF_CUDA_TRY(
        cub::DeviceMerge::MergePairs(preprocessed_left_indices->begin(),
                                     preprocessed_right_indices->begin(),
                                     static_cast<int64_t>(preprocessed_left_indices->size()),
                                     null_left_indices.begin(),
                                     cuda::constant_iterator<size_type>{JoinNoMatch},
                                     static_cast<int64_t>(null_left_indices.size()),
                                     left_result_indices.begin(),
                                     right_result_indices.begin(),
                                     cuda::std::less<>{},
                                     make_cub_env(stream)));

      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(std::move(left_result_indices)),
        std::make_unique<rmm::device_uvector<size_type>>(std::move(right_result_indices))};
    },
    stream);
}

std::unique_ptr<cudf::join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join_match_context"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Match counts are produced in original probe-row order, so skip the unused probe-side sort.
  auto preprocessed_left =
    preprocessed_table::create(left, compare_nulls, cudf::sorted::YES, stream);

  return invoke_merge(
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, left, &preprocessed_left, stream, mr](auto& obj) mutable {
      auto matches_per_row = obj.matches_per_row(stream, mr);
      matches_per_row->resize(matches_per_row->size() - 1, stream);
      if (compare_nulls == null_equality::UNEQUAL &&
          has_nested_nulls(preprocessed_left._table_view)) {
        // Now we need to post-process the matches i.e. insert zero counts for all the null
        // positions
        auto unprocessed_matches_per_row =
          cudf::detail::make_zeroed_device_uvector_async<size_type>(
            preprocessed_left._table_view.num_rows(), stream, mr);
        auto mapping = preprocessed_left.map_table_to_unprocessed(stream);
        thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        matches_per_row->begin(),
                        matches_per_row->end(),
                        mapping.begin(),
                        unprocessed_matches_per_row.begin());
        return std::make_unique<sort_merge_join_match_context>(
          left,
          std::make_unique<rmm::device_uvector<size_type>>(std::move(unprocessed_matches_per_row)),
          std::move(preprocessed_left));
      }
      return std::make_unique<sort_merge_join_match_context>(
        left, std::move(matches_per_row), std::move(preprocessed_left));
    },
    stream);
}

// left_partition_end exclusive
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::partitioned_inner_join"};

  // Extract preprocessed_left from the context
  auto const& preprocessed_left =
    static_cast<sort_merge_join_match_context const*>(context.left_table_context.get())
      ->preprocessed_left;

  auto const left_partition_start_idx = context.left_start_idx;
  auto const left_partition_end_idx   = context.left_end_idx;
  auto null_processed_table_start_idx = left_partition_start_idx;
  auto null_processed_table_end_idx   = left_partition_end_idx;
  if (compare_nulls == null_equality::UNEQUAL && has_nested_nulls(preprocessed_left._table_view)) {
    auto left_mapping              = preprocessed_left.map_table_to_unprocessed(stream);
    null_processed_table_start_idx = cuda::std::distance(
      left_mapping.begin(),
      thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          left_mapping.begin(),
                          left_mapping.end(),
                          left_partition_start_idx));
    null_processed_table_end_idx = cuda::std::distance(
      left_mapping.begin(),
      thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          left_mapping.begin(),
                          left_mapping.end(),
                          left_partition_end_idx - 1));
  }
  auto null_processed_left_partition =
    cudf::slice(preprocessed_left._null_processed_table_view,
                {null_processed_table_start_idx, null_processed_table_end_idx},
                stream)[0];

  auto [preprocessed_right_indices, preprocessed_left_indices] = invoke_merge(
    preprocessed_right._null_processed_table_view,
    null_processed_left_partition,
    [this, left_partition_start_idx, stream, mr](auto& obj) { return obj.inner(stream, mr); },
    stream);
  // Map from slice to total null processed table
  // Use cub API to handle large arrays (> INT32_MAX)
  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->size(),
    [null_processed_table_start_idx] __device__(auto idx) -> size_type {
      return null_processed_table_start_idx + idx;
    },
    make_cub_env(stream)));
  // Map from total null processed table to unprocessed table
  postprocess_indices(
    preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
  return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
}

}  // namespace detail

sort_merge_join::~sort_merge_join() = default;

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 cuda::stream_ref stream)
  : _impl{std::make_unique<impl_type>(right, is_right_sorted, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, stream, mr);
}

std::unique_ptr<join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr) const
{
  return _impl->partitioned_inner_join(context, stream, mr);
}

}  // namespace cudf
