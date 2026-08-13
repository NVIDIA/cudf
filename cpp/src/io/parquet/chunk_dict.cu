/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_gpu.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_scan.cuh>
#include <cuco/static_map_ref.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/bit>

namespace cudf::io::parquet::detail {

namespace {

/// Maximum number of chunk fragments for which spatially-local dictionary indices across fragment
/// keys are claimed using shared memory
constexpr size_type MAX_FRAGMENTS_PER_CHUNK = 1024;

/// Default block size for kernel launches
constexpr int DEFAULT_BLOCK_SIZE = 256;

/**
 * @brief Functor for checking equality of two keys.
 *
 * @tparam T Type of the keys
 * @param col Column view
 * @param lhs_idx Index of the left-hand side key
 * @param rhs_idx Index of the right-hand side key
 * @return Whether the keys are equal
 */
template <typename T>
struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(key_type lhs_idx, key_type rhs_idx) const
  {
    // We don't call this for nulls so this is fine.
    auto constexpr equal = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<T>(lhs_idx), col.element<T>(rhs_idx));
  }
};

/**
 * @brief Functor for hashing a key
 *
 * @tparam T Type of the key
 * @param col Column view
 * @param idx Key index of the key
 * @return Hash value of the key
 */
template <typename T>
struct hash_functor {
  column_device_view const& col;
  uint32_t const seed = 0;
  __device__ auto operator()(key_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(col.element<T>(idx));
  }
};

/**
 * @brief Functor for inserting a key into a hash map
 *
 * @tparam block_size Thread block size
 * @param storage_ref Hash map storage reference
 * @param chunk Column chunk pointer
 * @param frag Page fragment pointer
 * @param frag_idx Page fragment index
 */
template <int block_size>
struct map_insert_fn {
  storage_ref_type const& storage_ref;
  EncColumnChunk* const chunk;
  PageFragment* const frag;
  mapped_type const frag_idx;

  template <typename T>
  __device__ void operator()(size_type const start_value_idx, size_type const end_value_idx)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      using block_reduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename block_reduce::TempStorage reduce_storage;

      auto const t                       = threadIdx.x;
      auto const col                     = chunk->col_desc;
      column_device_view const& data_col = *col->leaf_column;
      __shared__ size_type total_num_dict_entries;
      __shared__ size_type num_dict_vals;
      if (t == 0) { num_dict_vals = 0; };
      __syncthreads();

      using equality_fn_type = equality_functor<T>;
      using hash_fn_type     = hash_functor<T>;
      // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
      // benchmarks.
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Make a view of the hash map.
      auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                               cuco::empty_value{VALUE_SENTINEL},
                                               equality_fn_type{data_col},
                                               probing_scheme_type{hash_fn_type{data_col}},
                                               cuco::thread_scope_block,
                                               storage_ref};

      // Create a map ref with `cuco::insert` operator
      auto map_insert_ref = hash_map_ref.rebind_operators(cuco::insert);

      // Create atomic refs to the current chunk's num_dict_entries and uniq_data_size
      cuda::atomic_ref<size_type, SCOPE> const chunk_num_dict_entries{chunk->num_dict_entries};
      cuda::atomic_ref<size_type, SCOPE> const chunk_uniq_data_size{chunk->uniq_data_size};

      // Note: Adjust the following loop to use `cg::tile<map_cg_size>` if needed in the future.
      for (size_type val_idx = start_value_idx + t; val_idx - t < end_value_idx;
           val_idx += block_size) {
        size_type is_unique      = 0;
        size_type uniq_elem_size = 0;

        // Check if this index is valid.
        auto const is_valid =
          val_idx < end_value_idx and val_idx < data_col.size() and data_col.is_valid(val_idx);

        // Insert fragment index to hash map using a single thread (for best performance for now)
        // and count successful insertions.
        if (is_valid) {
          // TODO(mh): Here we insert the fragment index of the CAS winner, which may not be the
          // smallest one (relies on monotonic block scheduling). Switch to static_map's
          // `insert_or_apply` with `cuco::op::min` for deterministic first-fragment semantics
          is_unique = map_insert_ref.insert(slot_type{static_cast<key_type>(val_idx), frag_idx});
          uniq_elem_size = [&]() -> size_type {
            if (not is_unique) { return 0; }
            switch (col->physical_type) {
              case Type::INT32: return sizeof(int32_t);
              case Type::INT64: return sizeof(int64_t);
              case Type::INT96: return sizeof(int32_t) + sizeof(int64_t);
              case Type::FLOAT: return sizeof(float);
              case Type::DOUBLE: return sizeof(double);
              case Type::BYTE_ARRAY: {
                auto const col_type = data_col.type().id();
                if (col_type == type_id::STRING) {
                  // Strings are stored as int32_t length + string bytes
                  return sizeof(int32_t) + data_col.element<string_view>(val_idx).size_bytes();
                } else if (col_type == type_id::LIST) {
                  // Binary is stored as int32_t length + bytes
                  return sizeof(int32_t) +
                         get_element<statistics::byte_array_view>(data_col, val_idx).size_bytes();
                }
                CUDF_UNREACHABLE(
                  "Byte array only supports string and list<byte> column types for dictionary "
                  "encoding!");
              }
              case Type::FIXED_LEN_BYTE_ARRAY:
                if (data_col.type().id() == type_id::DECIMAL128) { return sizeof(__int128_t); }
                CUDF_UNREACHABLE(
                  "Fixed length byte array only supports decimal 128 column types for dictionary "
                  "encoding!");
              default: CUDF_UNREACHABLE("Unsupported type for dictionary encoding");
            }
          }();
        }
        // Reduce num_unique and uniq_data_size from all tiles.
        auto num_unique = block_reduce(reduce_storage).Sum(is_unique);
        __syncthreads();
        auto uniq_data_size = block_reduce(reduce_storage).Sum(uniq_elem_size);
        // One thread atomically updates the number and data size of total unique values as well as
        // the number of unique values inserted by this fragment.
        if (t == 0) {
          total_num_dict_entries =
            chunk_num_dict_entries.fetch_add(num_unique, cuda::std::memory_order_relaxed);
          total_num_dict_entries += num_unique;
          num_dict_vals += num_unique;
          chunk_uniq_data_size.fetch_add(uniq_data_size, cuda::std::memory_order_relaxed);
        }
        __syncthreads();

        // Check if the num unique values in chunk has already exceeded max dict size and early exit
        if (total_num_dict_entries > MAX_DICT_SIZE) { break; }
      }  // for loop
      // Flush the number of unique values inserted by this fragment
      if (t == 0) { frag->num_dict_vals = num_dict_vals; };
    } else {
      CUDF_UNREACHABLE("Unsupported type to insert in map");
    }
  }
};

/**
 * @brief Functor for finding a key in a hash map
 *
 * @tparam block_size Thread block size
 * @param storage_ref Hash map storage reference
 * @param chunk Column chunk pointerk
 */
template <int block_size>
struct map_find_fn {
  storage_ref_type const& storage_ref;
  EncColumnChunk* const chunk;
  template <typename T>
  __device__ void operator()(size_type const start_value_idx,
                             size_type const end_value_idx,
                             size_type const ck_start_val_idx)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      auto const col       = chunk->col_desc;
      auto const& data_col = *col->leaf_column;

      using equality_fn_type = equality_functor<T>;
      using hash_fn_type     = hash_functor<T>;
      // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
      // benchmarks.
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Make a view of the hash map.
      auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                               cuco::empty_value{VALUE_SENTINEL},
                                               equality_fn_type{data_col},
                                               probing_scheme_type{hash_fn_type{data_col}},
                                               cuco::thread_scope_block,
                                               storage_ref};

      // Create a map ref with `cuco::find` operator
      auto const map_find_ref = hash_map_ref.rebind_operators(cuco::find);
      auto const t            = threadIdx.x;

      // Note: Adjust the following loop to use `cg::tiles<map_cg_size>` if needed in the future.
      for (key_type val_idx = start_value_idx + t; val_idx < end_value_idx; val_idx += block_size) {
        // Find the key using a single thread for best performance for now.
        if (data_col.is_valid(val_idx)) {
          auto const found_slot = map_find_ref.find(val_idx);
          // Fail if we didn't find the previously inserted key.
          cudf_assert(found_slot != map_find_ref.end() &&
                      "Unable to find value in map in dictionary index construction");
          // No need for atomic as this is not going to be modified by any other thread.
          chunk->dict_index[val_idx - ck_start_val_idx] = found_slot->second;
        }
      }
    } else {
      CUDF_UNREACHABLE("Unsupported type to find in map");
    }
  }
};

/**
 * @brief Populate the hash maps for all chunks (thread block per page fragment)
 *
 * @tparam block_size Thread block size
 * @param map_storage Hash map storage span
 * @param frags 2D span of page fragments
 */
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_chunk_hash_maps_kernel(device_span<slot_type> const map_storage,
                                  cudf::detail::device_2dspan<PageFragment> frags)
{
  auto const col_idx  = blockIdx.y;
  auto const frag_idx = blockIdx.x;
  auto& frag          = frags[col_idx][frag_idx];
  auto const chunk    = frag.chunk;
  auto col            = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  auto const start_row = frag.start_row;
  auto const end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be inserted into the map for current chunk.
  auto const start_value_idx = row_to_value_idx(start_row, *col);
  auto const end_value_idx   = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size,
                                     map_storage.data() + chunk->dict_map_offset};
  type_dispatcher(
    data_col.type(),
    map_insert_fn<block_size>{storage_ref, chunk, &frag, static_cast<mapped_type>(frag_idx)},
    start_value_idx,
    end_value_idx);
}

/**
 * @brief Collect the dictionary indices for all chunks (thread block per column chunk)
 *
 * @tparam block_size Thread block size
 * @param map_storage Hash map storage span
 * @param chunks Column chunks span
 * @param frags 2D span of page fragments
 */
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_span<slot_type> const map_storage,
                             device_span<EncColumnChunk> chunks,
                             cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto& chunk = chunks[blockIdx.x];
  if (not chunk.use_dictionary) { return; }

  auto t               = threadIdx.x;
  auto const num_frags = chunk.num_fragments;

  // If a chunk has less fragments than the maximum fragments per chunk, resolve its
  // column-relative fragment range.
  if (num_frags <= MAX_FRAGMENTS_PER_CHUNK) {
    auto const col_idx    = chunk.col_desc_id;
    auto const& col_frags = frags[col_idx];
    auto const frag_start = static_cast<size_type>(chunk.fragments - col_frags.data());

    // Initialize fragment_offsets with prefix sum (exclusive) of number of dictionary values
    // inserted by each fragment in this chunk.
    __shared__ size_type fragment_offsets[MAX_FRAGMENTS_PER_CHUNK];
    {
      using block_scan = cub::BlockScan<size_type, block_size>;
      __shared__ typename block_scan::TempStorage scan_storage;

      auto base_idx = 0;
      while (base_idx < num_frags) {
        auto const idx = base_idx + t;
        auto const per_thread_count =
          (idx < num_frags) ? col_frags[frag_start + idx].num_dict_vals : 0;
        auto per_thread_offset = 0;
        block_scan(scan_storage).ExclusiveSum(per_thread_count, per_thread_offset);
        if (idx < num_frags) { fragment_offsets[idx] = per_thread_offset; }
        base_idx += block_size;
        __syncthreads();
      }
    }

    // Iterate over slots and claim a dictionary index from the fragment offsets (spatial-locality
    // for keys in the same fragment).
    for (; t < chunk.dict_map_size; t += block_size) {
      auto* slot     = map_storage.data() + chunk.dict_map_offset + t;
      auto const key = slot->first;
      if (key != KEY_SENTINEL) {
        auto const frag_loc = static_cast<size_type>(slot->second) - frag_start;
        cudf_assert(frag_loc >= 0 && frag_loc < num_frags &&
                    "populate stamped a fragment hint outside this chunk's fragment range");
        auto const loc = atomicAdd(&fragment_offsets[frag_loc], 1);
        cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
        chunk.dict_data[loc] = key;
        slot->second         = loc;
      }
    }
  }
  // Otherwise, iterate over slots and claim monotonically increasing dictionary indices for each
  // key
  else {
    __shared__ cuda::atomic<size_type, SCOPE> counter;
    using cuda::std::memory_order_relaxed;
    if (t == 0) { new (&counter) cuda::atomic<size_type, SCOPE>{0}; }
    __syncthreads();

    for (; t < chunk.dict_map_size; t += block_size) {
      auto* slot     = map_storage.data() + chunk.dict_map_offset + t;
      auto const key = slot->first;
      if (key != KEY_SENTINEL) {
        auto const loc = counter.fetch_add(1, memory_order_relaxed);
        cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
        chunk.dict_data[loc] = key;
        slot->second         = loc;
      }
    }
  }
}

/**
 * @brief Get the dictionary indices for all chunks (thread block per page fragment)
 *
 * @tparam block_size Thread block size
 * @param map_storage Hash map storage span
 * @param frags 2D span of page fragments
 */
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_span<slot_type> const map_storage,
                                cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto const col_idx  = blockIdx.y;
  auto const frag_idx = blockIdx.x;
  auto const& frag    = frags[col_idx][frag_idx];
  auto const chunk    = frag.chunk;

  if (not chunk->use_dictionary) { return; }

  auto const start_row = frag.start_row;
  auto const end_row   = frag.start_row + frag.num_rows;

  auto const col = chunk->col_desc;
  // Find the bounds of values in leaf column to be searched in the map for current chunk
  auto const start_value_idx  = row_to_value_idx(start_row, *col);
  auto const end_value_idx    = row_to_value_idx(end_row, *col);
  auto const ck_start_val_idx = row_to_value_idx(chunk->start_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size,
                                     map_storage.data() + chunk->dict_map_offset};

  type_dispatcher(data_col.type(),
                  map_find_fn<block_size>{storage_ref, chunk},
                  start_value_idx,
                  end_value_idx,
                  ck_start_val_idx);
}

/**
 * @brief Compute the RLE bits for the dictionary indices for all pages (warp per page)
 *
 * @tparam block_size Thread block size
 * @param pages Pages span
 */
CUDF_KERNEL void __launch_bounds__(DEFAULT_BLOCK_SIZE)
  compute_page_dict_bits_kernel(device_span<EncPage> pages)
{
  namespace cg = cooperative_groups;
  using cudf::detail::warp_size;

  auto const page_idx = cudf::detail::grid_1d::global_thread_id() / warp_size;
  if (page_idx >= static_cast<size_type>(pages.size())) { return; }

  auto const warp  = cg::tiled_partition<warp_size>(cg::this_thread_block());
  auto& page       = pages[page_idx];
  auto const chunk = page.chunk;

  // Return if non-dict chunk: `dict_rle_bits` is unused by the encoder
  if (not chunk->use_dictionary) { return; }
  // Return if dictionary page itself does not encode dict_indices
  if (page.page_type == PageType::DICTIONARY_PAGE) { return; }

  auto const col = chunk->col_desc;

  // Return if BOOLEAN column: dict_bits=1 through a separate code path in
  // `gpuEncodeDictPages`, independent of `dict_rle_bits`
  if (col->physical_type == Type::BOOLEAN) { return; }

  auto const chunk_start_val      = row_to_value_idx(chunk->start_row, *col);
  auto const page_start_val       = row_to_value_idx(page.start_row, *col);
  auto const page_num_leaf_values = static_cast<size_type>(page.num_leaf_values);
  auto const begin                = page_start_val - chunk_start_val;
  auto const end                  = begin + page_num_leaf_values;

  auto const* dict_index = chunk->dict_index;
  auto const& leaf_col   = *col->leaf_column;
  auto const leaf_size   = leaf_col.size();

  // Accumulate per-lane max dict index for this page
  size_type lane_max = 0;
  for (size_type i = begin + warp.thread_rank(); i < end; i += warp_size) {
    auto const val_idx = chunk_start_val + i;
    // Null rows leave `dict_index` undefined; gate the read with the column's validity bitmap to
    // avoid pulling garbage bits into the max.
    if (val_idx < leaf_size && leaf_col.is_valid(val_idx)) {
      lane_max = cuda::std::max(lane_max, dict_index[i]);
    }
  }

  // Write this page's RLE bits
  auto const page_max = cg::reduce(warp, lane_max, cg::greater<cudf::size_type>{});
  cg::invoke_one(warp, [&] {
    // Floor at 1 to match the chunk-wide convention (all-null pages still emit a 1-bit RLE
    // preamble)
    auto const nbits   = cuda::std::max(cuda::std::bit_width(static_cast<uint32_t>(page_max)), 1);
    page.dict_rle_bits = static_cast<uint8_t>(nbits);
  });
}

}  // namespace

void populate_chunk_hash_maps(device_span<slot_type> const map_storage,
                              cudf::detail::device_2dspan<PageFragment> frags,
                              rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  populate_chunk_hash_maps_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void collect_map_entries(device_span<slot_type> const map_storage,
                         device_span<EncColumnChunk> chunks,
                         cudf::detail::device_2dspan<PageFragment const> frags,
                         rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  static_assert(block_size >= MAX_FRAGMENTS_PER_CHUNK,
                "block_size must be >= MAX_FRAGMENTS_PER_CHUNK so one BlockScan thread backs "
                "each histogram bucket.");
  collect_map_entries_kernel<block_size>
    <<<chunks.size(), block_size, 0, stream.value()>>>(map_storage, chunks, frags);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void get_dictionary_indices(device_span<slot_type> const map_storage,
                            cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  get_dictionary_indices_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void compute_per_page_dict_bits(device_span<EncPage> pages, rmm::cuda_stream_view stream)
{
  if (pages.empty()) { return; }
  auto constexpr warps_per_block = DEFAULT_BLOCK_SIZE / cudf::detail::warp_size;
  auto const num_blocks =
    cudf::util::div_rounding_up_safe(static_cast<size_type>(pages.size()), warps_per_block);
  compute_page_dict_bits_kernel<<<num_blocks, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(pages);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
