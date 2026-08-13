/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <nvtext/jaccard.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::thread_index_type block_size       = 256;
constexpr cudf::thread_index_type bytes_per_thread = 4;

/**
 * @brief Retrieve the row data (span) for the given column/row-index
 *
 * @param values Flat vector of all values
 * @param offsets Offsets identifying rows within values
 * @param row_idx Row index to retrieve
 * @return A device-span of the row values
 */
__device__ auto get_row(uint32_t const* values, int64_t const* offsets, cudf::size_type row_idx)
{
  auto const offset = offsets[row_idx];
  auto const size   = offsets[row_idx + 1] - offset;
  auto const begin  = values + offset;
  return cudf::device_span<uint32_t const>(begin, size);
}

/**
 * @brief Kernel to count the unique values within each row of the input column
 *
 * This is called with a warp per row.
 *
 * @param d_values Sorted hash values to count uniqueness
 * @param d_offsets Offsets to each set of row elements in d_values
 * @param rows Number of rows in the output
 * @param d_results Number of unique values in each row
 */
CUDF_KERNEL void sorted_unique_fn(uint32_t const* d_values,
                                  int64_t const* d_offsets,
                                  cudf::size_type rows,
                                  cudf::size_type* d_results)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (static_cast<cudf::thread_index_type>(rows) * cudf::detail::warp_size)) { return; }

  using warp_reduce = cub::WarpReduce<cudf::size_type>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto const row_idx  = idx / cudf::detail::warp_size;
  auto const lane_idx = idx % cudf::detail::warp_size;
  auto const row      = get_row(d_values, d_offsets, row_idx);
  auto const begin    = row.begin();

  cudf::size_type count = 0;
  for (auto itr = begin + lane_idx; itr < row.end(); itr += cudf::detail::warp_size) {
    count += (itr == begin || *itr != *(itr - 1));
  }
  auto const result = warp_reduce(temp_storage).Sum(count);
  if (lane_idx == 0) { d_results[row_idx] = result; }
}

/**
 * @brief Count the unique values within each row of the input column
 *
 * @param values Sorted hash values to count uniqueness
 * @param offsets Offsets to each set of row elements in d_values
 * @param rows Number of rows in the output
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Number of unique values
 */
rmm::device_uvector<cudf::size_type> compute_unique_counts(uint32_t const* values,
                                                           int64_t const* offsets,
                                                           cudf::size_type rows,
                                                           rmm::cuda_stream_view stream)
{
  auto d_results        = rmm::device_uvector<cudf::size_type>(rows, stream);
  auto const num_blocks = cudf::util::div_rounding_up_safe(
    static_cast<cudf::thread_index_type>(rows) * cudf::detail::warp_size, block_size);
  sorted_unique_fn<<<num_blocks, block_size, 0, stream.value()>>>(
    values, offsets, rows, d_results.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return d_results;
}

/**
 * @brief Kernel to count the number of common values within each row of the 2 input columns
 *
 * This is called with a warp per row.
 *
 * @param d_values1 Sorted hash values to check against d_values2
 * @param d_offsets1 Offsets to each set of row elements in d_values1
 * @param d_values2 Sorted hash values to check against d_values1
 * @param d_offsets2 Offsets to each set of row elements in d_values2
 * @param rows Number of rows in the output
 * @param d_results Number of common values in each row
 */
CUDF_KERNEL void sorted_intersect_fn(uint32_t const* d_values1,
                                     int64_t const* d_offsets1,
                                     uint32_t const* d_values2,
                                     int64_t const* d_offsets2,
                                     cudf::size_type rows,
                                     cudf::size_type* d_results)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (static_cast<cudf::thread_index_type>(rows) * cudf::detail::warp_size)) { return; }

  using warp_reduce = cub::WarpReduce<cudf::size_type>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto const row_idx  = idx / cudf::detail::warp_size;
  auto const lane_idx = idx % cudf::detail::warp_size;

  auto const needles  = get_row(d_values1, d_offsets1, row_idx);
  auto const haystack = get_row(d_values2, d_offsets2, row_idx);

  auto begin     = haystack.begin();
  auto const end = haystack.end();

  cudf::size_type count = 0;
  for (auto itr = needles.begin() + lane_idx; itr < needles.end() && begin < end;
       itr += cudf::detail::warp_size) {
    if (itr != needles.begin() && *itr == *(itr - 1)) { continue; }  // skip duplicates
    // search haystack for this needle (*itr)
    auto const found = thrust::lower_bound(thrust::seq, begin, end, *itr);
    count += (found != end) && (*found == *itr);  // increment if found;
    begin = found;                                // shorten the next lower-bound range
  }
  // sum up the counts across this warp
  auto const result = warp_reduce(temp_storage).Sum(count);
  if (lane_idx == 0) { d_results[row_idx] = result; }
}

/**
 * @brief Count the number of common values within each row of the 2 input columns
 *
 * @param values1 Sorted hash values to check against values2
 * @param offsets1 Offsets to each set of row elements in values1
 * @param values2 Sorted hash values to check against values1
 * @param offsets2 Offsets to each set of row elements in values2
 * @param rows Number of rows in the output
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Number of common values
 */
rmm::device_uvector<cudf::size_type> compute_intersect_counts(uint32_t const* values1,
                                                              int64_t const* offsets1,
                                                              uint32_t const* values2,
                                                              int64_t const* offsets2,
                                                              cudf::size_type rows,
                                                              rmm::cuda_stream_view stream)
{
  auto d_results        = rmm::device_uvector<cudf::size_type>(rows, stream);
  auto const num_blocks = cudf::util::div_rounding_up_safe(
    static_cast<cudf::thread_index_type>(rows) * cudf::detail::warp_size, block_size);
  sorted_intersect_fn<<<num_blocks, block_size, 0, stream.value()>>>(
    values1, offsets1, values2, offsets2, rows, d_results.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return d_results;
}

/**
 * @brief Counts the number of substrings in each row of the given strings column
 *
 * Each warp processes a single string.
 * Formula is `count = max(1, str.length() - width + 1)`
 * If a string has less than width characters (but not empty), the count is 1
 * since the entire string is still hashed.
 *
 * @param d_strings Input column of strings
 * @param width Substring size in characters
 * @param d_counts Output number of substring per row of input
 */
CUDF_KERNEL void count_substrings_kernel(cudf::column_device_view const d_strings,
                                         cudf::size_type width,
                                         int64_t* d_counts)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (static_cast<cudf::thread_index_type>(d_strings.size()) * cudf::detail::warp_size)) {
    return;
  }

  auto const str_idx = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
  if (d_strings.is_null(str_idx)) {
    d_counts[str_idx] = 0;
    return;
  }

  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_counts[str_idx] = 0;
    return;
  }

  using warp_reduce = cub::WarpReduce<cudf::size_type>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto const end        = d_str.data() + d_str.size_bytes();
  auto const lane_idx   = idx % cudf::detail::warp_size;
  cudf::size_type count = 0;
  for (auto itr = d_str.data() + (lane_idx * bytes_per_thread); itr < end;
       itr += cudf::detail::warp_size * bytes_per_thread) {
    for (auto s = itr; (s < (itr + bytes_per_thread)) && (s < end); ++s) {
      count += static_cast<cudf::size_type>(cudf::strings::detail::is_begin_utf8_char(*s));
    }
  }
  auto const char_count = warp_reduce(temp_storage).Sum(count);
  if (lane_idx == 0) { d_counts[str_idx] = cuda::std::max(1, char_count - width + 1); }
}

/**
 * @brief Kernel to hash the substrings for each input row
 *
 * Each warp processes a single string.
 * Substrings of string "hello world" with width=4 produce:
 *   "hell", "ello", "llo ", "lo w", "o wo", " wor", "worl", "orld"
 * Each of these substrings is hashed and the hash stored in d_results
 *
 * @param d_strings Input column of strings
 * @param width Substring size in characters
 * @param d_output_offsets Offsets into d_results
 * @param d_results Hash values for each substring
 */
CUDF_KERNEL void substring_hash_kernel(cudf::column_device_view const d_strings,
                                       cudf::size_type width,
                                       int64_t const* d_output_offsets,
                                       uint32_t* d_results)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }

  __shared__ uint32_t hvs[block_size];  // temp store for hash values

  auto const hasher     = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{0};
  auto const end        = d_str.data() + d_str.size_bytes();
  auto const warp_count = (d_str.size_bytes() / cudf::detail::warp_size) + 1;

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto d_hashes = d_results + d_output_offsets[str_idx];
  auto itr      = d_str.data() + lane_idx;
  for (auto i = 0; i < warp_count; ++i) {
    uint32_t hash = 0;
    if (itr < end && cudf::strings::detail::is_begin_utf8_char(*itr)) {
      // resolve substring
      auto const sub_str =
        cudf::string_view(itr, static_cast<cudf::size_type>(cuda::std::distance(itr, end)));
      auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(sub_str, width);
      // hash only if we have the full width of characters or this is the beginning of the string
      if ((left == 0) || (itr == d_str.data())) { hash = hasher(cudf::string_view(itr, bytes)); }
    }
    hvs[threadIdx.x] = hash;  // store hash into shared memory
    warp.sync();
    if (lane_idx == 0) {
      // copy valid hash values for this warp into d_hashes
      auto const hashes     = &hvs[threadIdx.x];
      auto const hashes_end = hashes + cudf::detail::warp_size;
      d_hashes =
        thrust::copy_if(thrust::seq, hashes, hashes_end, d_hashes, [](auto h) { return h != 0; });
    }
    warp.sync();
    itr += cudf::detail::warp_size;
  }
}

void segmented_sort(uint32_t const* input,
                    uint32_t* output,
                    int64_t items,
                    cudf::size_type segments,
                    int64_t const* offsets,
                    rmm::cuda_stream_view stream)
{
  rmm::device_buffer temp;
  std::size_t temp_bytes = 0;
  cub::DeviceSegmentedSort::SortKeys(
    temp.data(), temp_bytes, input, output, items, segments, offsets, offsets + 1, stream.value());
  temp = rmm::device_buffer(temp_bytes, stream);
  cub::DeviceSegmentedSort::SortKeys(
    temp.data(), temp_bytes, input, output, items, segments, offsets, offsets + 1, stream.value());
}

/**
 * @brief Create hashes for each substring
 *
 * The hashes are sorted using a segmented-sort as setup to
 * perform the unique and intersect operations.
 *
 * @param input Input strings column to hash
 * @param width Substring width in characters
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The sorted hash values and offsets to each row
 */
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<int64_t>> hash_substrings(
  cudf::strings_column_view const& input, cudf::size_type width, rmm::cuda_stream_view stream)
{
  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  // count substrings
  auto offsets          = rmm::device_uvector<int64_t>(input.size() + 1, stream);
  auto const num_blocks = cudf::util::div_rounding_up_safe(
    static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size);
  count_substrings_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    *d_strings, width, offsets.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const total_hashes =
    cudf::detail::sizes_to_offsets(offsets.begin(), offsets.end(), offsets.begin(), 0, stream);

  // hash substrings
  rmm::device_uvector<uint32_t> hashes(total_hashes, stream);
  substring_hash_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    *d_strings, width, offsets.data(), hashes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // sort hashes
  rmm::device_uvector<uint32_t> sorted(total_hashes, stream);
  if (total_hashes < static_cast<int64_t>(std::numeric_limits<int>::max())) {
    segmented_sort(
      hashes.begin(), sorted.begin(), sorted.size(), input.size(), offsets.begin(), stream);
  } else {
    // The CUB segmented sort can only handle max<int> total values
    // so this code calls it in sections.
    auto const section_size   = std::numeric_limits<int>::max() / 2L;
    auto const sort_sections  = cudf::util::div_rounding_up_safe(total_hashes, section_size);
    auto const offset_indices = [&] {
      // build a set of indices that point to offsets subsections
      auto sub_offsets = rmm::device_uvector<int64_t>(sort_sections + 1, stream);
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       sub_offsets.begin(),
                       sub_offsets.end(),
                       0L,
                       section_size);
      auto indices = rmm::device_uvector<int64_t>(sub_offsets.size(), stream);
      thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          offsets.begin(),
                          offsets.end(),
                          sub_offsets.begin(),
                          sub_offsets.end(),
                          indices.begin());
      return cudf::detail::make_host_vector(indices, stream);
    }();

    // Call segmented sort with the sort sections
    for (auto i = 0L; i < sort_sections; ++i) {
      auto const index1 = offset_indices[i];
      auto const index2 = std::min(offset_indices[i + 1], static_cast<int64_t>(offsets.size() - 1));
      auto const offset1 = offsets.element(index1, stream);
      auto const offset2 = offsets.element(index2, stream);

      auto const num_items    = offset2 - offset1;
      auto const num_segments = index2 - index1;

      // There is a bug in the CUB segmented sort and the workaround is to
      // shift the offset values so the first offset is 0.
      // This transform can be removed once the bug is fixed.
      auto sort_offsets = rmm::device_uvector<int64_t>(num_segments + 1, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        offsets.begin() + index1,
                        offsets.begin() + index2 + 1,
                        sort_offsets.begin(),
                        [offset1] __device__(auto const o) { return o - offset1; });

      segmented_sort(hashes.begin() + offset1,
                     sorted.begin() + offset1,
                     num_items,
                     num_segments,
                     sort_offsets.begin(),
                     stream);
    }
  }
  return std::make_pair(std::move(sorted), std::move(offsets));
}

/**
 * @brief Compute the jaccard distance for each row
 *
 * Formula is J = |A ∩ B| / |A ∪ B|
 *              = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 * where |A ∩ B| is number of common values between A and B
 * and |x| is the number of unique values in x.
 */
struct jaccard_fn {
  cudf::size_type const* d_uniques1;
  cudf::size_type const* d_uniques2;
  cudf::size_type const* d_intersects;

  __device__ float operator()(cudf::size_type idx) const
  {
    auto const count1     = d_uniques1[idx];
    auto const count2     = d_uniques2[idx];
    auto const intersects = d_intersects[idx];
    // the intersect values are in both sets so a union count
    // would need to subtract the intersect count from one set
    // (see formula in comment above)
    auto const unions = count1 + count2 - intersects;
    return unions ? (static_cast<float>(intersects) / static_cast<float>(unions)) : 0.f;
  }
};

}  // namespace

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    input1.size() == input2.size(), "input columns must be the same size", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);

  constexpr auto output_type = cudf::data_type{cudf::type_id::FLOAT32};
  if (input1.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const [d_uniques1, d_uniques2, d_intersects] = [&] {
    // build hashes of the substrings
    auto const [hash1, offsets1] = hash_substrings(input1, width, stream);
    auto const [hash2, offsets2] = hash_substrings(input2, width, stream);

    // compute the unique counts in each set and the intersection counts
    auto d_uniques1   = compute_unique_counts(hash1.data(), offsets1.data(), input1.size(), stream);
    auto d_uniques2   = compute_unique_counts(hash2.data(), offsets2.data(), input2.size(), stream);
    auto d_intersects = compute_intersect_counts(
      hash1.data(), offsets1.data(), hash2.data(), offsets2.data(), input1.size(), stream);

    return std::tuple{std::move(d_uniques1), std::move(d_uniques2), std::move(d_intersects)};
  }();

  auto results = cudf::make_numeric_column(
    output_type, input1.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_results = results->mutable_view().data<float>();

  // compute the jaccard using the unique counts and the intersect counts
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{results->size()},
                    d_results,
                    jaccard_fn{d_uniques1.data(), d_uniques2.data(), d_intersects.data()});

  if (input1.null_count() || input2.null_count()) {
    auto [null_mask, null_count] =
      cudf::detail::bitmask_and(cudf::table_view({input1.parent(), input2.parent()}), stream, mr);
    results->set_null_mask(std::move(null_mask), null_count);
  }

  return results;
}

}  // namespace detail

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::jaccard_index(input1, input2, width, stream, mr);
}

}  // namespace nvtext
