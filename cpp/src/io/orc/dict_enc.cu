/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "orc_gpu.hpp"

#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/io/orc_types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::orc::detail {

/**
 * @brief Counts the number of characters in each rowgroup of each string column.
 */
CUDF_KERNEL void rowgroup_char_counts_kernel(device_2dspan<size_type> char_counts,
                                             device_span<orc_column_device_view const> orc_columns,
                                             device_2dspan<rowgroup_rows const> rowgroup_bounds,
                                             device_span<uint32_t const> str_col_indexes)
{
  // Index of the column in the `str_col_indexes` array
  auto const str_col_idx = blockIdx.x % str_col_indexes.size();
  // Index of the column in the `orc_columns` array
  auto const col_idx       = str_col_indexes[str_col_idx];
  auto const row_group_idx = (blockIdx.x / str_col_indexes.size()) * blockDim.x + threadIdx.x;
  if (row_group_idx >= rowgroup_bounds.size().first) { return; }

  auto const& str_col  = orc_columns[col_idx];
  auto const start_row = rowgroup_bounds[row_group_idx][col_idx].begin + str_col.offset();
  auto const num_rows  = rowgroup_bounds[row_group_idx][col_idx].size();

  size_type char_count = 0;
  if (num_rows > 0) {
    auto const& offsets    = str_col.child(strings_column_view::offsets_column_index);
    auto const offsets_itr = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    char_count = static_cast<size_type>(offsets_itr[start_row + num_rows] - offsets_itr[start_row]);
  }
  char_counts[str_col_idx][row_group_idx] = char_count;
}

void rowgroup_char_counts(device_2dspan<size_type> counts,
                          device_span<orc_column_device_view const> orc_columns,
                          device_2dspan<rowgroup_rows const> rowgroup_bounds,
                          device_span<uint32_t const> str_col_indexes,
                          rmm::cuda_stream_view stream)
{
  if (rowgroup_bounds.count() == 0) { return; }

  auto const num_rowgroups = rowgroup_bounds.size().first;
  if (str_col_indexes.empty()) { return; }

  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, rowgroup_char_counts_kernel));
  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe<unsigned int>(num_rowgroups, block_size) *
    str_col_indexes.size();

  rowgroup_char_counts_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    counts, orc_columns, rowgroup_bounds, str_col_indexes);
  CUDF_CUDA_TRY(cudaGetLastError());
}

struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // We don't call this for nulls so this is fine
    auto const equal = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<string_view>(lhs_idx), col.element<string_view>(rhs_idx));
  }
};

struct hash_functor {
  column_device_view const& col;
  __device__ auto operator()(size_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<string_view>{}(col.element<string_view>(idx));
  }
};

// Probing scheme to use for the hash map
using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_functor>;

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_dictionary_hash_maps_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                       device_span<orc_column_device_view const> columns)
{
  auto const col_idx    = blockIdx.x / dictionaries.size().second;
  auto const stripe_idx = blockIdx.x % dictionaries.size().second;
  auto const t          = threadIdx.x;
  auto& dict            = dictionaries[col_idx][stripe_idx];
  auto const& col       = columns[dict.column_idx];

  // Make a view of the hash map
  auto const hash_fn     = hash_functor{col};
  auto const equality_fn = equality_functor{col};

  storage_ref_type const storage_ref{dict.map_slots.size(), dict.map_slots.data()};
  // Make a view of the hash map.
  auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                           cuco::empty_value{VALUE_SENTINEL},
                                           equality_fn,
                                           probing_scheme_type{hash_fn},
                                           cuco::thread_scope_block,
                                           storage_ref};

  // Create a map ref with `cuco::insert` operator
  auto has_map_insert_ref = hash_map_ref.rebind_operators(cuco::insert);

  auto const start_row = dict.start_row;
  auto const end_row   = dict.start_row + dict.num_rows;

  size_type entry_count{0};
  size_type char_count{0};

  // all threads should loop the same number of times
  for (thread_index_type cur_row = start_row + t; cur_row - t < end_row; cur_row += block_size) {
    auto const is_valid = cur_row < end_row and col.is_valid(cur_row);

    if (is_valid) {
      // insert element at cur_row to hash map and count successful insertions
      auto const is_unique = has_map_insert_ref.insert(cuco::pair{cur_row, cur_row});

      if (is_unique) {
        ++entry_count;
        char_count += col.element<string_view>(cur_row).size_bytes();
      }
    }
    // ensure that threads access adjacent rows in each iteration
    __syncthreads();
  }

  using block_reduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  auto const block_entry_count = block_reduce(reduce_storage).Sum(entry_count);
  __syncthreads();
  auto const block_char_count = block_reduce(reduce_storage).Sum(char_count);

  if (t == 0) {
    dict.entry_count = block_entry_count;
    dict.char_count  = block_char_count;
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_2dspan<stripe_dictionary> dictionaries)
{
  auto const col_idx    = blockIdx.x / dictionaries.size().second;
  auto const stripe_idx = blockIdx.x % dictionaries.size().second;
  auto const& dict      = dictionaries[col_idx][stripe_idx];

  if (not dict.is_enabled) { return; }

  auto const t = threadIdx.x;
  __shared__ cuda::atomic<size_type, cuda::thread_scope_block> counter;

  using cuda::std::memory_order_relaxed;
  if (t == 0) { new (&counter) cuda::atomic<size_type, cuda::thread_scope_block>{0}; }
  __syncthreads();

  for (size_type i = 0; i < dict.map_slots.size(); i += block_size) {
    if (t + i < dict.map_slots.size()) {
      auto* slot     = dict.map_slots.data() + t + i;
      auto const key = slot->first;
      if (key != KEY_SENTINEL) {
        auto loc       = counter.fetch_add(1, memory_order_relaxed);
        dict.data[loc] = key;
        slot->second   = loc;
      }
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                device_span<orc_column_device_view const> columns)
{
  auto const col_idx    = blockIdx.x / dictionaries.size().second;
  auto const stripe_idx = blockIdx.x % dictionaries.size().second;
  auto const t          = threadIdx.x;
  auto const& dict      = dictionaries[col_idx][stripe_idx];
  auto const& col       = columns[dict.column_idx];

  if (not dict.is_enabled) { return; }

  // Make a view of the hash map
  auto const hash_fn     = hash_functor{col};
  auto const equality_fn = equality_functor{col};

  storage_ref_type const storage_ref{dict.map_slots.size(), dict.map_slots.data()};
  // Make a view of the hash map.
  auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                           cuco::empty_value{VALUE_SENTINEL},
                                           equality_fn,
                                           probing_scheme_type{hash_fn},
                                           cuco::thread_scope_block,
                                           storage_ref};

  // Create a map ref with `cuco::insert` operator
  auto has_map_find_ref = hash_map_ref.rebind_operators(cuco::find);

  auto const start_row = dict.start_row;
  auto const end_row   = dict.start_row + dict.num_rows;

  for (thread_index_type cur_row = start_row + t; cur_row < end_row; cur_row += block_size) {
    if (col.is_valid(cur_row)) {
      auto const found_slot = has_map_find_ref.find(cur_row);
      // Fail if we didn't find the previously inserted key.
      cudf_assert(found_slot != has_map_find_ref.end() &&
                  "Unable to find value in map in dictionary index construction");
      dict.index[cur_row] = found_slot->second;
    }
  }
}

void populate_dictionary_hash_maps(device_2dspan<stripe_dictionary> dictionaries,
                                   device_span<orc_column_device_view const> columns,
                                   rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 256;
  populate_dictionary_hash_maps_kernel<block_size>
    <<<dictionaries.count(), block_size, 0, stream.value()>>>(dictionaries, columns);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void collect_map_entries(device_2dspan<stripe_dictionary> dictionaries,
                         rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size>
    <<<dictionaries.count(), block_size, 0, stream.value()>>>(dictionaries);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void get_dictionary_indices(device_2dspan<stripe_dictionary> dictionaries,
                            device_span<orc_column_device_view const> columns,
                            rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  get_dictionary_indices_kernel<block_size>
    <<<dictionaries.count(), block_size, 0, stream.value()>>>(dictionaries, columns);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
