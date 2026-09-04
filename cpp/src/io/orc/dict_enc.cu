/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "orc_gpu.hpp"

#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/io/orc_types.hpp>

#include <cuda/stream>

#include <algorithm>

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
                          cuda::stream_ref stream)
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

  rowgroup_char_counts_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
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

/**
 * @brief Number of blocks to assign to each stripe dictionary.
 *
 * Splitting a dictionary across blocks stops a single block from having to work through a whole
 * stripe, which can be a million rows. It only pays off while there are idle SMs though: extra
 * blocks sharing one hash map cost either locality or atomic contention, so the budget collapses to
 * one block per dictionary once the dictionaries alone fill the device.
 *
 * @param kernel Kernel to be launched, used to query its occupancy
 * @param block_size Threads per block the kernel will be launched with
 * @param target_waves How many times over to fill the device before splitting any further
 * @param num_dictionaries Number of dictionaries the grid has to cover
 * @param max_dict_rows Largest row count of any dictionary
 */
template <typename Kernel>
int blocks_per_dictionary(Kernel kernel,
                          int block_size,
                          int target_waves,
                          std::size_t num_dictionaries,
                          size_type max_dict_rows)
{
  int blocks_per_sm = 0;
  CUDF_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_size, 0));
  auto const budget =
    target_waves * blocks_per_sm * cudf::detail::num_multiprocessors() / num_dictionaries;
  auto const blocks_to_cover = cudf::util::div_rounding_up_safe(max_dict_rows, block_size);
  return std::clamp<int>(budget, 1, std::max(blocks_to_cover, size_type{1}));
}

/**
 * @brief Builds the hash map of unique values for every stripe dictionary.
 *
 * `blockIdx.x` selects the dictionary and `blockIdx.y` splits its rows across blocks. Splitting
 * requires `Scope` to be `cuda::thread_scope_device` so that concurrent inserts from different
 * blocks stay atomic; with a y extent of one, `cuda::thread_scope_block` is both correct and
 * cheaper. The launcher picks accordingly.
 *
 * Which of several equal strings wins a slot depends on block order, but a slot is chosen by string
 * content, so the dictionary holds the same strings either way and the encoded output is
 * unaffected.
 */
template <int block_size, cuda::thread_scope Scope>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_dictionary_hash_maps_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                       device_span<orc_column_device_view const> columns)
{
  auto const num_stripes = dictionaries.size().second;
  auto const t           = threadIdx.x;
  auto& dict             = dictionaries[blockIdx.x / num_stripes][blockIdx.x % num_stripes];
  auto const& col        = columns[dict.column_idx];

  // Make a view of the hash map
  auto const hash_fn     = hash_functor{col};
  auto const equality_fn = equality_functor{col};

  storage_ref_type const storage_ref{dict.map_slots.size(), dict.map_slots.data()};
  // Make a view of the hash map.
  auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                           cuco::empty_value{VALUE_SENTINEL},
                                           equality_fn,
                                           probing_scheme_type{hash_fn},
                                           cuco::cuda_thread_scope<Scope>{},
                                           storage_ref};

  // Create a map ref with `cuco::insert` operator
  auto has_map_insert_ref = hash_map_ref.rebind_operators(cuco::insert);

  auto const end_row = dict.start_row + dict.num_rows;
  auto const first_row =
    dict.start_row + static_cast<thread_index_type>(blockIdx.y) * block_size + t;
  auto const row_stride = static_cast<thread_index_type>(block_size) * gridDim.y;

  size_type entry_count{0};
  size_type char_count{0};

  // all threads should loop the same number of times
  for (thread_index_type cur_row = first_row; cur_row - t < end_row; cur_row += row_stride) {
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
    // Accumulated rather than assigned, since several blocks may share a dictionary. The counts are
    // zeroed on the host before the launch.
    atomicAdd(&dict.entry_count, block_entry_count);
    atomicAdd(&dict.char_count, block_char_count);
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

/**
 * @brief Looks up the dictionary index of every row of every stripe dictionary.
 *
 * `blockIdx.x` selects the dictionary and `blockIdx.y` splits its rows across blocks, so one stripe
 * can be worked on by several blocks instead of just one. The y extent need not cover all of the
 * rows; blocks stride until they run out. Each row is handled independently, so no combining step
 * is needed.
 *
 * The hash maps are only read here; they are fully built by the preceding
 * `populate_dictionary_hash_maps` and `collect_map_entries` launches. `cuco::thread_scope_block`
 * therefore does not prevent spreading one dictionary over several blocks: the scope only governs
 * the atomicity of modifications, and there are none.
 */
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                device_span<orc_column_device_view const> columns)
{
  auto const num_stripes = dictionaries.size().second;
  auto const& dict       = dictionaries[blockIdx.x / num_stripes][blockIdx.x % num_stripes];
  if (not dict.is_enabled) { return; }

  auto const& col = columns[dict.column_idx];

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

  // Create a map ref with `cuco::find` operator
  auto has_map_find_ref = hash_map_ref.rebind_operators(cuco::find);

  auto const end_row = dict.start_row + dict.num_rows;
  auto const first_row =
    dict.start_row + static_cast<thread_index_type>(blockIdx.y) * block_size + threadIdx.x;
  auto const row_stride = static_cast<thread_index_type>(block_size) * gridDim.y;

  for (thread_index_type cur_row = first_row; cur_row < end_row; cur_row += row_stride) {
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
                                   size_type max_dict_rows,
                                   cuda::stream_ref stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size   = 256;
  constexpr int target_waves = 4;

  auto const blocks_per_dict = blocks_per_dictionary(
    populate_dictionary_hash_maps_kernel<block_size, cuda::thread_scope_device>,
    block_size,
    target_waves,
    dictionaries.count(),
    max_dict_rows);

  dim3 const grid{static_cast<unsigned int>(dictionaries.count()),
                  static_cast<unsigned int>(blocks_per_dict)};

  // Device-scoped inserts are only needed when a dictionary is shared by several blocks, and they
  // are more expensive, so keep the cheaper block scope whenever one block covers a dictionary.
  if (blocks_per_dict == 1) {
    populate_dictionary_hash_maps_kernel<block_size, cuda::thread_scope_block>
      <<<grid, block_size, 0, stream.get()>>>(dictionaries, columns);
  } else {
    populate_dictionary_hash_maps_kernel<block_size, cuda::thread_scope_device>
      <<<grid, block_size, 0, stream.get()>>>(dictionaries, columns);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

void collect_map_entries(device_2dspan<stripe_dictionary> dictionaries, cuda::stream_ref stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size>
    <<<dictionaries.count(), block_size, 0, stream.get()>>>(dictionaries);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void get_dictionary_indices(device_2dspan<stripe_dictionary> dictionaries,
                            device_span<orc_column_device_view const> columns,
                            size_type max_dict_rows,
                            cuda::stream_ref stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  // Chosen empirically to speed up narrow tables without slowing down wide ones.
  constexpr int target_waves = 4;

  auto const blocks_per_dict = blocks_per_dictionary(get_dictionary_indices_kernel<block_size>,
                                                     block_size,
                                                     target_waves,
                                                     dictionaries.count(),
                                                     max_dict_rows);

  dim3 const grid{static_cast<unsigned int>(dictionaries.count()),
                  static_cast<unsigned int>(blocks_per_dict)};
  get_dictionary_indices_kernel<block_size>
    <<<grid, block_size, 0, stream.get()>>>(dictionaries, columns);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
