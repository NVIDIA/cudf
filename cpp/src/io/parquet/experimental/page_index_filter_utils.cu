/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "page_index_filter_utils.hpp"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

namespace {

/**
 * @brief Functor to read the row mask, i.e. the zeroth Fenwick tree level
 *
 * Nulls are read as retained rows here so that pages aren't accidentally pruned due to
 * unavailable page-level statistics (represented as nulls)
 */
struct row_mask_accessor {
  bool const* data;               ///< Row mask data, adjusted for the column view offset
  bitmask_type const* null_mask;  ///< Null mask, or nullptr if there are no nulls
  cudf::size_type offset;         ///< Column view offset, needed to index into the null mask

  /**
   * @brief Constructs a row mask accessor from a nullable boolean row mask column
   *
   * @param row_mask Boolean row mask column
   */
  explicit row_mask_accessor(cudf::column_view const& row_mask)
    : data{row_mask.begin<bool>()},
      null_mask{row_mask.has_nulls() ? row_mask.null_mask() : nullptr},
      offset{row_mask.offset()}
  {
  }

  __device__ bool inline operator()(cudf::size_type row_idx) const noexcept
  {
    if (data[row_idx]) { return true; }
    return null_mask != nullptr and not bit_is_set(null_mask, offset + row_idx);
  }
};

/*
 * @brief Functor to build a Fenwick tree level from the previous level data
 *
 * @param tree_level_ptrs Pointers to the start of Fenwick tree level data
 * @param row_mask Accessor for the zeroth tree level (the row mask)
 * @param prev_level Previous tree level
 * @param prev_level_size Size of the previous tree level
 * @param current_level_size Size of the current tree level
 */
struct build_fenwick_tree_level_functor {
  bool** tree_level_ptrs;
  row_mask_accessor row_mask;
  cudf::size_type prev_level;
  cudf::size_type prev_level_size;
  cudf::size_type current_level_size;

  /**
   * @brief Reads an element from the previous tree level
   *
   * @param prev_level_idx Previous tree level element index
   * @return Value of the element at the previous tree level
   */
  __device__ bool inline read_prev_level(cudf::size_type prev_level_idx) const noexcept
  {
    // Use the row mask accessor for the zeroth tree level
    return prev_level == 0 ? row_mask(prev_level_idx) : tree_level_ptrs[prev_level][prev_level_idx];
  }

  /**
   * @brief Builds the next Fenwick tree level from the current level data
   * by ORing two elements at the current level.
   *
   * elem_current_level[idx] = elem_prev_level[idx * 2] OR elem_prev_level[idx * 2 + 1];
   *
   * @param current_level_idx Current tree level element index
   */
  __device__ void operator()(cudf::size_type current_level_idx) const noexcept
  {
    auto current_level_ptr = tree_level_ptrs[prev_level + 1];

    // Handle the odd-sized remaining element if prev_level_size is odd
    if (prev_level_size % 2 and current_level_idx == current_level_size - 1) {
      current_level_ptr[current_level_idx] = read_prev_level(prev_level_size - 1);
    } else {
      current_level_ptr[current_level_idx] =
        read_prev_level(current_level_idx * 2) or read_prev_level((current_level_idx * 2) + 1);
    }
  }
};

/**
 * @brief Functor to binary search a `true` value in the Fenwick tree in range [start, end)
 *
 * @param tree_level_ptrs Pointers to the start of Fenwick tree level data
 * @param row_mask Accessor for the zeroth tree level (the row mask)
 * @param page_offsets Pointer to page offsets describing each search range i as [page_offsets[i],
 * page_offsets[i+1))
 * @param num_ranges Number of search ranges
 */
struct search_fenwick_tree_functor {
  bool** tree_level_ptrs;
  row_mask_accessor row_mask;
  cudf::size_type const* page_offsets;
  cudf::size_type num_ranges;

  /**
   * @brief Enum class to represent which range boundary we are currently processing
   */
  enum class boundary : uint8_t {
    START = 0,
    END   = 1,
  };

  /**
   * @brief Checks if a value is a power of two
   *
   * @param value Value to check
   * @return Boolean indicating if the value is a power of two
   */
  __device__ bool inline constexpr is_power_of_two(cudf::size_type value) const noexcept
  {
    return (value & (value - 1)) == 0;
  }

  /**
   * @brief Finds the smallest power of two in the range [start, end). If no power of two is
   * found, returns a zero.
   *
   * @param start Range start
   * @param end Range end
   * @return Largest power of two in the range [start, end) or a zero if no power of two is found
   */
  __device__ cudf::size_type inline constexpr smallest_power_of_two_in_range(
    cudf::size_type start, cudf::size_type end) const noexcept
  {
    start--;
    start |= start >> 1;
    start |= start >> 2;
    start |= start >> 4;
    start |= start >> 8;
    start |= start >> 16;
    auto const result = start + 1;
    return result < end ? result : 0;
  }

  /**
   * @brief Finds the largest power of two in the range (start, end]. If no power of two is found,
   * returns a zero.
   *
   * @param start Range start
   * @param end Range end
   * @return Largest power of two in the range (start, end] or a zero if no power of two is found
   */
  __device__ size_type inline constexpr largest_power_of_two_in_range(size_type start,
                                                                      size_type end) const noexcept
  {
    auto constexpr nbits = cudf::detail::size_in_bits<size_type>() - 1;
    auto const result    = size_type{1} << (nbits - cuda::std::countl_zero<uint32_t>(end));
    return result > start ? result : 0;
  }

  /**
   * @brief Aligns a range boundary to the next power-of-two block
   *
   * @tparam Boundary Current boundary type (START or END)
   * @param start Range start
   * @param end Range end
   * @return A pair of the tree level and block size
   */
  template <boundary Boundary>
  __device__ auto inline constexpr align_range_boundary(cudf::size_type start,
                                                        cudf::size_type end) const noexcept
  {
    if constexpr (Boundary == boundary::START) {
      if (start == 0 or is_power_of_two(start)) {
        auto const block_size =
          cuda::std::max<size_type>(start & -start, largest_power_of_two_in_range(start, end));
        auto const tree_level = cuda::std::countr_zero<uint32_t>(block_size);
        return cuda::std::pair{tree_level, block_size};
      } else {
        auto const tree_level = cuda::std::countr_zero<uint32_t>(start);
        return cuda::std::pair{tree_level, size_type{1} << tree_level};
      }
    } else {
      auto block_size = end & -end;
      if (start > 0 and is_power_of_two(end)) {
        auto const next_alignment = cuda::std::max(smallest_power_of_two_in_range(start, end),
                                                   largest_power_of_two_in_range(0, end - start));
        block_size                = end - next_alignment;
      }
      return cuda::std::pair{cuda::std::countr_zero<uint32_t>(block_size), block_size};
    }
  }

  /**
   * @brief Queries the Fenwick tree for the given boundary position, tree level and block size
   *
   * @tparam Boundary Current boundary type (START or END)
   * @param boundary_pos Current boundary position
   * @param tree_level Corresponding tree level to query
   * @param block_size Alignment block size of the current boundary
   * @return Boolean indicating if a `true` value is found in the fenwick tree
   */
  template <boundary Boundary>
  __device__ bool inline constexpr query_fenwick_tree(cudf::size_type boundary_pos,
                                                      cudf::size_type tree_level,
                                                      cudf::size_type block_size) const noexcept
  {
    auto const position = (Boundary == boundary::START) ? boundary_pos : boundary_pos - block_size;
    auto const mask_index = position >> tree_level;
    return tree_level == 0 ? row_mask(mask_index) : tree_level_ptrs[tree_level][mask_index];
  }

  /**
   * @brief Searches the Fenwick tree to find a `true` value in range [start, end)
   *
   * Algorithm: While `start` < `end`, align `start` UP and `end` DOWN to the next power-of-two
   * searchable tree block. For the two aligned blocks, query the fenwick tree at corresponding
   * levels for a `true` value (larger block first). If found, return. Else, move the boundaries
   * to their alignments.
   *
   * @param range_idx Index of the range to search
   * @return Boolean indicating if a `true` value is found in the range
   */
  __device__ bool operator()(cudf::size_type range_idx) const noexcept
  {
    // Retrieve start and end for the current range [start, end)
    size_type start = page_offsets[range_idx];
    size_type end   = page_offsets[range_idx + 1];

    // Return early if the range is empty or invalid
    if (start >= end or range_idx >= num_ranges) { return false; }

    // Binary search decomposition loop
    while (start < end) {
      // Find the largest power-of-two block that aligns `start` up
      auto const [start_tree_level, start_block_size] =
        align_range_boundary<boundary::START>(start, end);

      // Find the largest power-of-two block that aligns `end` down
      auto const [end_tree_level, end_block_size] = align_range_boundary<boundary::END>(start, end);

      // Check the larger block first to minimize the number of queries
      if (start_block_size >= end_block_size) {
        // Check the `start` side alignment block first
        if (start + start_block_size <= end) {
          if (query_fenwick_tree<boundary::START>(start, start_tree_level, start_block_size)) {
            return true;
          }
          start += start_block_size;
        }
        // Check the `end` side alignment block if it's still in range
        if (end - end_block_size >= start) {
          if (query_fenwick_tree<boundary::END>(end, end_tree_level, end_block_size)) {
            return true;
          }
          end -= end_block_size;
        }
      } else {
        // Check the `end` side alignment block first
        if (end - end_block_size >= start) {
          if (query_fenwick_tree<boundary::END>(end, end_tree_level, end_block_size)) {
            return true;
          }
          end -= end_block_size;
        }
        // Check the `start` side alignment block if it's still in range
        if (start + start_block_size <= end) {
          if (query_fenwick_tree<boundary::START>(start, start_tree_level, start_block_size)) {
            return true;
          }
          start += start_block_size;
        }
      }
    }
    return false;
  }
};

/**
 * @brief Computes the offsets of the Fenwick tree levels (level 1 and higher) until the tree level
 * block size becomes larger than the maximum page (search range) size
 *
 * @param level0_size Size of the zeroth tree level (the row mask)
 * @param max_page_size Maximum page (search range) size
 * @return Fenwick tree level offsets
 */
std::vector<size_type> compute_fenwick_tree_level_offsets(cudf::size_type level0_size,
                                                          cudf::size_type max_page_size)
{
  std::vector<size_type> tree_level_offsets;
  tree_level_offsets.push_back(0);

  cudf::size_type current_level_size = cudf::util::div_rounding_up_safe(level0_size, 2);
  cudf::size_type current_level      = 1;

  while (current_level_size > 0) {
    auto const block_size = 1 << current_level;
    if (std::cmp_greater(block_size, max_page_size)) { break; }
    tree_level_offsets.push_back(tree_level_offsets.back() + current_level_size);
    current_level_size =
      current_level_size == 1 ? 0 : cudf::util::div_rounding_up_safe(current_level_size, 2);
    current_level++;
  }
  return tree_level_offsets;
}

}  // namespace

std::pair<cudf::detail::host_vector<size_type>, cudf::detail::host_vector<size_type>>
compute_page_row_offsets_and_colchunk_page_offsets(
  std::span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx,
  cuda::stream_ref stream)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  // Vector to store the cumulative number of rows in each page - - set initial capacity to two data
  // pages per row group
  auto page_row_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>((2 * total_row_groups) + 1, stream);
  // Vector to store the cumulative number of pages in each column chunk
  auto col_chunk_page_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>(total_row_groups + 1, stream);

  page_row_offsets.push_back(0);
  col_chunk_page_offsets.push_back(0);

  // For all data sources
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{row_group_indices.size()},
    [&](auto src_idx) {
      // For all column chunks in this data source
      auto const& rg_indices = row_group_indices[src_idx];
      std::optional<size_type> colchunk_iter_offset{};
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
        auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
        colchunk_iter_offset =
          parquet::detail::find_colchunk_iter_offset(row_group, schema_idx, colchunk_iter_offset);
        auto const& colchunk_iter = row_group.columns.begin() + colchunk_iter_offset.value();

        CUDF_EXPECTS(colchunk_iter->offset_index.has_value(),
                     "Offset index not found for column chunk",
                     std::invalid_argument);

        auto const& offset_index       = colchunk_iter->offset_index.value();
        auto const row_group_num_pages = offset_index.page_locations.size();

        col_chunk_page_offsets.push_back(col_chunk_page_offsets.back() + row_group_num_pages);

        // For all pages in this column chunk, update page row offsets.
        std::for_each(
          cuda::counting_iterator<std::size_t>{0},
          cuda::counting_iterator{row_group_num_pages},
          [&](auto const page_idx) {
            int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
            // For the last page, this is simply the total number of rows in the column chunk
            int64_t const last_row_idx =
              (page_idx < row_group_num_pages - 1)
                ? offset_index.page_locations[page_idx + 1].first_row_index
                : row_group.num_rows;

            // Update the page row offsets.
            page_row_offsets.push_back(page_row_offsets.back() + last_row_idx - first_row_idx);
          });
      });
    });

  return {std::move(page_row_offsets), std::move(col_chunk_page_offsets)};
}

std::pair<std::vector<size_type>, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::size_type schema_idx)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  std::vector<size_type> page_row_offsets;
  page_row_offsets.push_back(0);
  size_type max_page_size = 0;

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{row_group_indices.size()},
                [&](auto const src_idx) {
                  // For all row groups in this source
                  auto const& rg_indices = row_group_indices[src_idx];
                  std::optional<size_type> colchunk_iter_offset{};
                  std::for_each(rg_indices.begin(), rg_indices.end(), [&](auto const& rg_idx) {
                    auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
                    colchunk_iter_offset  = parquet::detail::find_colchunk_iter_offset(
                      row_group, schema_idx, colchunk_iter_offset);
                    auto const& colchunk_iter =
                      row_group.columns.begin() + colchunk_iter_offset.value();
                    CUDF_EXPECTS(colchunk_iter->offset_index.has_value(),
                                 "Offset index not found for column chunk",
                                 std::invalid_argument);
                    auto const& offset_index       = colchunk_iter->offset_index.value();
                    auto const row_group_num_pages = offset_index.page_locations.size();
                    std::for_each(cuda::counting_iterator<std::size_t>{0},
                                  cuda::counting_iterator{row_group_num_pages},
                                  [&](auto const page_idx) {
                                    int64_t const first_row_idx =
                                      offset_index.page_locations[page_idx].first_row_index;
                                    int64_t const last_row_idx =
                                      (page_idx < row_group_num_pages - 1)
                                        ? offset_index.page_locations[page_idx + 1].first_row_index
                                        : row_group.num_rows;
                                    auto const page_size = last_row_idx - first_row_idx;
                                    max_page_size = std::max<size_type>(max_page_size, page_size);
                                    page_row_offsets.push_back(page_row_offsets.back() + page_size);
                                  });
                  });
                });

  return {std::move(page_row_offsets), max_page_size};
}

rmm::device_uvector<size_type> compute_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto row_offsets = cudf::detail::make_device_uvector_async(
    page_row_offsets, stream, cudf::get_current_device_resource_ref());

  auto page_indices = rmm::device_uvector<cudf::size_type>(total_rows, stream, mr);
  cudf::detail::label_segments(
    row_offsets.begin(), row_offsets.end(), page_indices.begin(), page_indices.end(), stream);
  return page_indices;
}

bool are_all_rows_retained(cudf::column_view const& row_mask, cuda::stream_ref stream)
{
  return cudf::detail::all_of(cuda::counting_iterator<cudf::size_type>{0},
                              cuda::counting_iterator{row_mask.size()},
                              row_mask_accessor{row_mask},
                              stream);
}

thrust::host_vector<bool> compute_row_range_selection_mask(
  cudf::column_view const& row_mask,
  std::span<cudf::size_type const> page_row_offsets,
  cudf::size_type max_page_size,
  cuda::stream_ref stream)
{
  // Need at least two offsets (or one range) to search the Fenwick tree
  if (page_row_offsets.size() < 2) { return thrust::host_vector<bool>{}; }

  auto const total_rows         = row_mask.size();
  auto const mr                 = cudf::get_current_device_resource_ref();
  auto const tree_level_offsets = compute_fenwick_tree_level_offsets(total_rows, max_page_size);
  auto const num_levels         = static_cast<cudf::size_type>(tree_level_offsets.size());
  auto tree_levels_data         = rmm::device_uvector<bool>(tree_level_offsets.back(), stream, mr);
  auto host_tree_level_ptrs     = cudf::detail::make_pinned_vector_async<bool*>(num_levels, stream);
  // The zeroth level is the row mask itself, read through its accessor
  auto const d_row_mask   = row_mask_accessor{row_mask};
  host_tree_level_ptrs[0] = nullptr;
  std::for_each(cuda::counting_iterator<cudf::size_type>{1},
                cuda::counting_iterator<cudf::size_type>{num_levels},
                [&](auto const level_idx) {
                  host_tree_level_ptrs[level_idx] =
                    tree_levels_data.data() + tree_level_offsets[level_idx - 1];
                });
  auto tree_level_ptrs = cudf::detail::make_device_uvector_async(host_tree_level_ptrs, stream, mr);

  auto prev_level_size = total_rows;
  std::for_each(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{num_levels - 1},
    [&](auto const prev_level) {
      auto const current_level_size = cudf::util::div_rounding_up_safe(prev_level_size, 2);
      thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<cudf::size_type>{0},
                       cuda::counting_iterator{current_level_size},
                       build_fenwick_tree_level_functor{.tree_level_ptrs = tree_level_ptrs.data(),
                                                        .row_mask        = d_row_mask,
                                                        .prev_level      = prev_level,
                                                        .prev_level_size = prev_level_size,
                                                        .current_level_size = current_level_size});
      prev_level_size = current_level_size;
    });

  auto const num_ranges    = static_cast<cudf::size_type>(page_row_offsets.size() - 1);
  auto device_results      = rmm::device_uvector<bool>(num_ranges, stream, mr);
  auto pinned_page_offsets = cudf::detail::make_pinned_vector(page_row_offsets, stream);
  auto page_offsets = cudf::detail::make_device_uvector_async(pinned_page_offsets, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{num_ranges},
                    device_results.begin(),
                    search_fenwick_tree_functor{.tree_level_ptrs = tree_level_ptrs.data(),
                                                .row_mask        = d_row_mask,
                                                .page_offsets    = page_offsets.data(),
                                                .num_ranges      = num_ranges});

  auto results = cudf::detail::make_pinned_vector_async(device_results, stream);
  stream.sync();

  return thrust::host_vector<bool>(results.begin(), results.end());
}

}  // namespace cudf::io::parquet::experimental::detail
