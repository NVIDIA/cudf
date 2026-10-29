/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudf {
namespace detail {
namespace {

// Block size of the fused kernel; matches the valid_if_n_kernel/concatenate precedent.
constexpr size_type block_size{256};

// Default consecutive descriptors per blockIdx.y tile: each thread loads a row's gather-map
// value once and reuses it across the tile's columns (per-column map re-reads dominated wide
// cells). Measured variable — the driver accepts 1/2/4/8 via the tile_cols overload.
constexpr size_type kDefaultTileCols{8};

// Map-locality routing. Misrouting a random map to the wide tile costs up to ~51% while the
// reverse only forfeits the ~17% coalesced win, so the classifier is strict: coalesced is
// reported only when EVERY sampled window is near-contiguous; anything else (and every regime
// where tile width measured K-insensitive) takes the safe K=1.
constexpr size_type kCoalescedTileCols{8};
constexpr size_type kRandomTileCols{1};
// 8-column kernels run 30-90us; the sampler's pinned-D2H sync (~10us flat, measured) is a
// 10-30% tax there, so they take the safe K=1 unsampled (win-or-tie at every 8-col cell).
constexpr size_type kMinSampledCols{9};
constexpr size_type kMinSampledRows{65536};
constexpr size_type kSampleWindows{16};
constexpr size_type kSampleWindowLen{8};

/**
 * @brief Type-erased per-column parameters consumed by `batch_gather_kernel`.
 *
 * One descriptor per fixed-width source column. All pointers are element bases, not byte bases:
 * `src_data` comes from `column_view::data<T>()`, which has the slice offset pre-applied in
 * element units, and `dst_data`/`dst_mask` belong to a freshly allocated destination whose
 * offset is zero. Nothing downstream may re-add an offset to these pointers.
 */
struct alignas(16) column_descriptor {
  void const* src_data;          // first source element; nullptr never (skip_data covers all-null)
  void* dst_data;                // first destination element; fresh allocation, offset 0
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr if source non-nullable or skip
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or fast-path
  int64_t mask_offset;           // bit offset into src_mask (== src.offset() for sliced sources)
  int32_t element_size;          // cudf::size_of(src.type()); one of {1, 2, 4, 8, 16}
  bool skip_data;                // all-null fast path: dst mask pre-set ALL_NULL, data undefined
};

// INVARIANT: element_size * row is computed in 64-bit below; it can exceed 2^31 for DECIMAL128.
// 16-byte copies stay aligned because device allocations are >=256B aligned and offsets are
// whole elements.
__device__ void copy_element(char* dst, char const* src, int32_t element_size)
{
  switch (element_size) {
    case 1: *dst = *src; break;
    case 2: *reinterpret_cast<int16_t*>(dst) = *reinterpret_cast<int16_t const*>(src); break;
    case 4: *reinterpret_cast<int32_t*>(dst) = *reinterpret_cast<int32_t const*>(src); break;
    case 8: *reinterpret_cast<int64_t*>(dst) = *reinterpret_cast<int64_t const*>(src); break;
    case 16: {
      auto* d8       = reinterpret_cast<int64_t*>(dst);
      auto const* s8 = reinterpret_cast<int64_t const*>(src);
      d8[0]          = s8[0];
      d8[1]          = s8[1];
      break;
    }
    default: CUDF_UNREACHABLE("Unexpected element size");
  }
}

// Returns the source column's element base pointer through its storage type (fixed-point
// dispatches to int32/int64/__int128), so the slice offset is applied exactly once, host-side.
// The dispatcher instantiates operator() for every type in the dispatch list, including the
// pass-through types (string_view/list_view/struct_view/dictionary32) that never carry data<T>;
// they are unreachable here because only fixed-width columns enter the descriptor path.
struct data_ptr_functor {
  template <typename T>
  void const* operator()(column_view const& col) const
  {
    if constexpr (cudf::is_rep_layout_compatible<T>()) {
      return static_cast<void const*>(col.data<T>());
    } else {
      CUDF_UNREACHABLE("Non-layout-compatible type on the descriptor path");
    }
  }
};

/**
 * @brief Gathers every column described by `descriptors` in one launch.
 *
 * The grid is 2D: x covers row blocks (grid-stride over rows); each y block owns a tile of
 * `kTileCols` consecutive descriptors and y-strides over tiles past the 65535 gridDim.y cap. A
 * row's gather-map value is loaded once and reused across the tile's columns; the tile's
 * descriptors are staged in shared memory so the row loop pays no per-column global reads.
 * `kTileCols == 1` degenerates to one column per y block (the pre-tiling geometry) with the
 * same single-launch contract.
 *
 * INVARIANT: every destination row r in [0, num_rows) is written by exactly one thread, per
 * column.
 * INVARIANT: each column's dst_mask word w is written exactly once, by lane 0 of the warp
 * covering rows [32w, 32w + 32): this requires BlockSize to be a multiple of warp_size (the
 * row stride is then a multiple of 32), which the driver guarantees. Trailing bits of the last
 * word are 0. Per-column bookkeeping (warp_valid[k], per-column ballots) keeps this true
 * independently of the tile a column belongs to.
 */
template <size_type BlockSize, size_type kTileCols>
CUDF_KERNEL __launch_bounds__(BlockSize) void batch_gather_kernel(
  input_indexalator gather_map,
  column_descriptor const* descriptors,
  size_type num_descriptors,
  size_type num_rows,
  size_type num_source_rows,
  bool check_bounds,
  bool negative_allowed,
  size_type* valid_counts)
{
  __shared__ column_descriptor tile_descs[kTileCols];
  auto const num_tiles = util::div_rounding_up_safe(num_descriptors, kTileCols);
  for (size_type tile = static_cast<size_type>(blockIdx.y); tile < num_tiles;
       tile += static_cast<size_type>(gridDim.y)) {
    auto const tile_begin = tile * kTileCols;
    // Stage the tile's descriptors in shared memory; the range is block-uniform and the reads
    // below are warp broadcasts.
    if (threadIdx.x < kTileCols &&
        tile_begin + static_cast<size_type>(threadIdx.x) < num_descriptors) {
      tile_descs[threadIdx.x] = descriptors[tile_begin + static_cast<size_type>(threadIdx.x)];
    }
    __syncthreads();
    size_type warp_valid[kTileCols]{};
    auto const row_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    auto row              = static_cast<int64_t>(blockIdx.x) * BlockSize + threadIdx.x;
    auto active           = __ballot_sync(0xFFFF'FFFFu, row < num_rows);
    while (row < num_rows) {
      // One map load (wrap + table-level bounds check) serves every column in the tile.
      auto idx = gather_map[static_cast<size_type>(row)];
      if (negative_allowed && idx < 0) { idx += num_source_rows; }
      bool const in_bounds = !check_bounds || (idx >= 0 && idx < num_source_rows);
#pragma unroll
      for (int32_t k = 0; k < kTileCols; ++k) {
        if (tile_begin + k < num_descriptors) {
          auto const d = tile_descs[k];
          bool valid   = in_bounds;
          if (valid && d.src_mask != nullptr) {
            valid = bit_is_set(d.src_mask, static_cast<size_type>(d.mask_offset + idx));
          }
          // Data at out-of-bounds/null rows may be left unwritten (fixed-width undefined-value
          // policy).
          if (valid && !d.skip_data) {
            copy_element(
              static_cast<char*>(d.dst_data) + static_cast<std::size_t>(row) * d.element_size,
              static_cast<char const*>(d.src_data) + static_cast<std::size_t>(idx) * d.element_size,
              d.element_size);
          }
          if (d.dst_mask != nullptr) {
            auto const word = __ballot_sync(active, valid);
            if (threadIdx.x % warp_size == 0) {
              d.dst_mask[word_index(static_cast<size_type>(row))] = word;
              warp_valid[k] += cuda::std::popcount(word);
            }
          }
        }
      }
      row += row_stride;
      active = __ballot_sync(active, row < num_rows);
    }
    if (valid_counts != nullptr) {
#pragma unroll
      for (int32_t k = 0; k < kTileCols; ++k) {
        auto const col = tile_begin + k;
        if (col < num_descriptors && tile_descs[k].dst_mask != nullptr) {
          auto const block_valid = single_lane_block_sum_reduce<BlockSize, 0>(warp_valid[k]);
          if (threadIdx.x == 0 && block_valid > 0) {
            cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{valid_counts[col]};
            ref.fetch_add(block_valid, cuda::std::memory_order_relaxed);
          }
        }
      }
    }
    // All threads are done reading tile_descs; the next tile iteration rewrites it.
    __syncthreads();
  }
}

}  // namespace

size_type select_gather_map_tile_cols(column_view const& gather_map,
                                      size_type num_fused_cols,
                                      cuda::stream_ref stream)
{
  auto const num_rows = gather_map.size();
  // Regimes where tile width measured K-insensitive: skip the sampler (its sync would dominate
  // a small kernel) and take the safe K=1. Also the guard for maps too short to sample.
  if (num_fused_cols < kMinSampledCols || num_rows < kMinSampledRows) { return kRandomTileCols; }
  switch (gather_map.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: break;
    default: return kRandomTileCols;  // Not an index type: unclassifiable, safe default.
  }

  // Sample kSampleWindows windows of kSampleWindowLen consecutive map elements, evenly spread
  // over the map, with one batched D2H into pinned staging (no kernel launch).
  auto const elem_size   = static_cast<size_t>(cudf::size_of(gather_map.type()));
  auto const window_size = static_cast<size_t>(kSampleWindowLen) * elem_size;
  auto staging           = cudf::detail::make_pinned_vector_async<std::byte>(
    static_cast<size_t>(kSampleWindows) * window_size, stream);
  std::array<void*, kSampleWindows> dsts{};
  std::array<void const*, kSampleWindows> srcs{};
  std::array<size_t, kSampleWindows> sizes{};
  auto const last_start = static_cast<int64_t>(num_rows) - static_cast<int64_t>(kSampleWindowLen);
  for (size_t w = 0; w < kSampleWindows; ++w) {
    auto const start =
      static_cast<size_type>(w == kSampleWindows - 1 ? last_start
                                                     : last_start * static_cast<int64_t>(w) /
                                                         static_cast<int64_t>(kSampleWindows - 1));
    srcs[w]  = gather_map.data<char>() + static_cast<std::size_t>(start) * elem_size;
    dsts[w]  = staging.data() + static_cast<size_t>(w) * window_size;
    sizes[w] = window_size;
  }
  CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
    dsts.data(), srcs.data(), sizes.data(), static_cast<size_t>(kSampleWindows), stream));
  // The decision is needed on host before the kernel launch; this waits only for the sample
  // copies and for work already enqueued on the stream (never for a kernel of ours).
  cudf::detail::sync_stream(stream);

  auto const read_element = [&](std::byte const* p) -> int64_t {
    switch (gather_map.type().id()) {
      case cudf::type_id::INT8: {
        int8_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT16: {
        int16_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT32: {
        int32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::INT64: {
        int64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
      }
      case cudf::type_id::UINT8: {
        uint8_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT16: {
        uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT32: {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      case cudf::type_id::UINT64: {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<int64_t>(v);
      }
      default: CUDF_UNREACHABLE("Map type excluded by the classifier guard");
    }
  };
  // A window is local when its values span at most one window length; classify coalesced only
  // if EVERY sampled window is local (reversed/ascending maps have spread kSampleWindowLen - 1).
  for (size_t w = 0; w < kSampleWindows; ++w) {
    auto const* window = staging.data() + w * window_size;
    auto const first   = read_element(window);
    auto min_value     = first;
    auto max_value     = first;
    for (size_type i = 1; i < kSampleWindowLen; ++i) {
      auto const v = read_element(window + static_cast<size_t>(i) * elem_size);
      min_value    = std::min(min_value, v);
      max_value    = std::max(max_value, v);
    }
    if (max_value - min_value > static_cast<int64_t>(kSampleWindowLen)) { return kRandomTileCols; }
  }
  return kCoalescedTileCols;
}

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr)
{
  // Routed entry: pick the tile width from sampled map locality. The fused-column count is
  // recomputed here (one host-side type pass) so K-insensitive regimes skip the sampler; a
  // null gather map skips it too (the delegate below rejects it).
  size_type num_fused_cols = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    num_fused_cols += cudf::is_fixed_width(source_table.column(i).type()) ? 1 : 0;
  }
  auto const tile_cols = (num_fused_cols == 0 or gather_map.has_nulls())
                           ? kDefaultTileCols
                           : select_gather_map_tile_cols(gather_map, num_fused_cols, stream);
  return batch_gather(source_table, gather_map, bounds_policy, neg_indices, stream, mr, tile_cols);
}

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols)
{
  cudf::scoped_range rng{"batch_gather"};

  CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

  // INVARIANT: gather_map.size() is the row count of every destination column.
  auto const num_rows     = gather_map.size();
  auto const num_src_rows = source_table.num_rows();
  auto const output_mr    = mr.get_output_mr();
  auto const temp_mr      = mr.get_temporary_mr();

  if (source_table.num_columns() == 0) {
    // Zero-column tables preserve their row count (stock gather parity).
    return std::make_unique<table>(std::vector<std::unique_ptr<column>>{}, num_rows);
  }

  std::vector<size_type> fused_indices;
  std::vector<size_type> fallback_indices;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    (cudf::is_fixed_width(source_table.column(i).type()) ? fused_indices : fallback_indices)
      .push_back(i);
  }

  std::vector<std::unique_ptr<column>> destination(source_table.num_columns());

  auto run_fallback = [&](
                        std::vector<size_type> const& indices, auto map_it_begin, auto map_it_end) {
    std::vector<std::unique_ptr<column>> fallback_columns;
    fallback_columns.reserve(indices.size());
    for (auto const idx : indices) {
      auto const& src = source_table.column(idx);
      fallback_columns.push_back(
        cudf::type_dispatcher<dispatch_storage_type>(src.type(),
                                                     column_gatherer{},
                                                     src,
                                                     map_it_begin,
                                                     map_it_end,
                                                     bounds_policy == out_of_bounds_policy::NULLIFY,
                                                     stream,
                                                     mr));
    }

    // Null-mask phase over the fallback subset, mirroring the stock gather's conditions.
    auto fallback_views = std::vector<cudf::column_view>{};
    fallback_views.reserve(indices.size());
    for (auto const idx : indices) {
      fallback_views.push_back(source_table.column(idx));
    }
    auto const fallback_table = cudf::table_view{fallback_views};
    if (bounds_policy == out_of_bounds_policy::NULLIFY or
        cudf::has_nested_nullable_columns(fallback_table)) {
      if (bounds_policy == out_of_bounds_policy::NULLIFY or
          cudf::has_nested_nulls(fallback_table)) {
        gather_bitmask(fallback_table,
                       map_it_begin,
                       fallback_columns,
                       bounds_policy == out_of_bounds_policy::NULLIFY
                         ? gather_bitmask_op::NULLIFY
                         : gather_bitmask_op::DONT_CHECK,
                       stream,
                       mr);
      } else {
        for (size_t i = 0; i < fallback_columns.size(); ++i) {
          cudf::detail::set_all_valid_null_masks(
            fallback_table.column(i), *fallback_columns[i], stream, output_mr);
        }
      }
    }

    for (size_t i = 0; i < indices.size(); ++i) {
      destination[indices[i]] = std::move(fallback_columns[i]);
    }
  };

  // Fused part: destination columns for the fixed-width subset, with per-column mask policy.
  std::vector<column_descriptor> descriptors;
  std::vector<size_type> kernel_masked_positions;  // descriptor positions with in-kernel masks
  if (not fused_indices.empty()) {
    descriptors.reserve(fused_indices.size());
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      auto dst        = cudf::detail::allocate_like(
        src, num_rows, cudf::mask_allocation_policy::NEVER, stream, output_mr);
      bitmask_type* dst_mask = nullptr;
      bool skip_data         = false;
      bool const all_null    = src.size() > 0 and src.null_count() == src.size();
      if (all_null) {
        // Every row is null regardless of the map: pre-set ALL_NULL and skip the data copy.
        dst->set_null_mask(
          cudf::detail::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, output_mr),
          num_rows);
        skip_data = true;
      } else if (bounds_policy == out_of_bounds_policy::NULLIFY or src.null_count() > 0) {
        auto mask = cudf::detail::create_null_mask(
          num_rows, cudf::mask_state::UNINITIALIZED, stream, output_mr);
        dst_mask = mask.data();
        dst->set_null_mask(std::move(mask), 0);
        kernel_masked_positions.push_back(pos);
      } else if (src.nullable()) {
        // Nullable source with zero nulls: ALL_VALID fast path, no kernel mask work.
        dst->set_null_mask(
          cudf::detail::create_null_mask(num_rows, cudf::mask_state::ALL_VALID, stream, output_mr),
          0);
      }
      auto const src_data =
        cudf::type_dispatcher<dispatch_storage_type>(src.type(), data_ptr_functor{}, src);
      descriptors.push_back(column_descriptor{src_data,
                                              dst->mutable_view().data<std::byte>(),
                                              skip_data ? nullptr : src.null_mask(),
                                              dst_mask,
                                              static_cast<int64_t>(src.offset()),
                                              static_cast<int32_t>(cudf::size_of(src.type())),
                                              skip_data});
      destination[fused_indices[pos]] = std::move(dst);
    }
  }

  if (not descriptors.empty() and num_rows > 0) {
    // Pinned staging: stream-ordered deallocation, so the H2D needs no host sync.
    auto staging =
      cudf::detail::make_pinned_vector_async<column_descriptor>(descriptors.size(), stream);
    std::copy(descriptors.begin(), descriptors.end(), staging.begin());
    auto d_descriptors =
      rmm::device_uvector<column_descriptor>(descriptors.size(), stream, temp_mr);
    cudf::detail::cuda_memcpy_async(device_span<column_descriptor>{d_descriptors},
                                    host_span<column_descriptor const>{staging},
                                    stream);

    auto const any_kernel_masks = not kernel_masked_positions.empty();
    rmm::device_uvector<size_type> valid_counts(0, stream, temp_mr);
    if (any_kernel_masks) {
      valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        descriptors.size(), stream, temp_mr);
    }

    // One launch per call; the tile width picks the kernel instantiation. y dimension: one
    // block per column tile (the kernel's y-stride loop covers tiles past the 65535 cap).
    auto const map_begin   = indexalator_factory::make_input_iterator(gather_map);
    auto const launch_tile = [&](auto tile_constant) {
      constexpr size_type kTileCols = decltype(tile_constant)::value;
      dim3 const grid{
        static_cast<unsigned>(cudf::util::div_rounding_up_safe(num_rows, block_size)),
        static_cast<unsigned>(std::min<size_type>(
          cudf::util::div_rounding_up_safe(static_cast<size_type>(descriptors.size()), kTileCols),
          65535)),
        1};
      batch_gather_kernel<block_size, kTileCols>
        <<<grid, block_size, 0, stream.get()>>>(map_begin,
                                                d_descriptors.data(),
                                                static_cast<size_type>(descriptors.size()),
                                                num_rows,
                                                num_src_rows,
                                                bounds_policy == out_of_bounds_policy::NULLIFY,
                                                neg_indices == negative_index_policy::ALLOWED,
                                                any_kernel_masks ? valid_counts.data() : nullptr);
      CUDF_CUDA_TRY(cudaGetLastError());
    };
    switch (tile_cols) {
      case 1: launch_tile(std::integral_constant<size_type, 1>{}); break;
      case 2: launch_tile(std::integral_constant<size_type, 2>{}); break;
      case 4: launch_tile(std::integral_constant<size_type, 4>{}); break;
      case 8: launch_tile(std::integral_constant<size_type, 8>{}); break;
      default: CUDF_FAIL("Unsupported batch_gather tile width", std::invalid_argument);
    }

    if (any_kernel_masks) {
      auto const h_valid_counts = cudf::detail::make_host_vector(valid_counts, stream);
      for (auto const pos : kernel_masked_positions) {
        destination[fused_indices[pos]]->set_null_count(num_rows - h_valid_counts[pos]);
      }
    }
  }

  if (not fallback_indices.empty()) {
    auto map_begin = indexalator_factory::make_input_iterator(gather_map);
    auto map_end   = map_begin + gather_map.size();
    if (neg_indices == negative_index_policy::ALLOWED) {
      auto const idx_converter = cuda::proclaim_return_type<size_type>(
        [num_src_rows] __device__(size_type in) { return in < 0 ? in + num_src_rows : in; });
      run_fallback(fallback_indices,
                   cuda::transform_iterator{map_begin, idx_converter},
                   cuda::transform_iterator{map_end, idx_converter});
    } else {
      run_fallback(fallback_indices, map_begin, map_end);
    }
  }

  return std::make_unique<table>(std::move(destination), num_rows);
}

}  // namespace detail
}  // namespace cudf
