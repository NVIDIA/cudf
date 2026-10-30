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
#include <cudf/detail/utilities/host_vector.hpp>
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
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudf {
namespace detail {

// Handle for an enqueued map-locality sample: the pinned staging plus its decode context. The
// staging frees stream-ordered, so it may outlive the resolve without a use-after-free.
// rmm_host_allocator has no default constructor, so empty handles come
// from empty(stream) — an unallocated pinned vector, not a plain std::vector.
struct sampled_map_locality {
  cudf::detail::host_vector<std::byte> staging;
  cudf::type_id map_type{cudf::type_id::EMPTY};
  size_t elem_size{};

  static sampled_map_locality empty(cuda::stream_ref stream)
  {
    return sampled_map_locality{
      cudf::detail::make_pinned_vector_async<std::byte>(0, stream), cudf::type_id::EMPTY, 0};
  }
};

namespace {

// Block size of the fused kernel; matches the valid_if_n_kernel/concatenate precedent.
constexpr size_type block_size{256};

// Default consecutive descriptors per blockIdx.y tile: each thread loads a row's gather-map
// value once and reuses it across the tile's columns (per-column map re-reads dominated wide
// cells). Measured variable — the driver accepts 1/2/4/8 via the tile_cols overload.
constexpr size_type kDefaultTileCols{8};

// Map-locality routing. Misrouting a random map to the wide tile costs up to ~51% while the
// reverse only forfeits the ~17% coalesced win, so the classifier is strict: coalesced is
// reported only when EVERY sampled window is near-contiguous; anything else takes the safe
// narrow tile.
constexpr size_type kCoalescedTileCols{8};
constexpr size_type kRandomTileCols{1};
constexpr size_type kUnsampledTileCols{2};
// Sampling floors (measured): the row gate keeps the sync off short/wide K-insensitive shapes,
// the work gate keeps it off narrow mid-height shapes; 1048576x8 crosses both and must sample.
constexpr size_type kMinSampledRows{262'144};
constexpr int64_t kMinSampledWork{4'000'000};
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
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr when no bit can be clear
                                 // (non-nullable, zero null count, or all-null skip)
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or fast-path
  int64_t mask_offset;           // bit offset into src_mask (== src.offset() for sliced sources)
  int32_t element_size;          // cudf::size_of(src.type()); one of {1, 2, 4, 8, 16}
  bool skip_data;                // all-null fast path: dst mask pre-set ALL_NULL, data undefined
};

// Evict-first cache policy (PTX ISA 74, SM_70+ via the generated cuda::ptx builtins), active
// only in the streaming regime: the sampled coalesced verdict marks the dst/src traffic
// don't-cache so L2 keeps the reuse-critical gather map. Random/unsampled verdicts, test-forced
// geometries, host compilation, and pre-sm_70 all take plain default-cached accesses.
template <bool kStreaming, typename T>
__device__ inline void streaming_copy(char* dst, char const* src)
{
  if constexpr (kStreaming) {
#if defined(__CUDA_ARCH__)
    if constexpr (__CUDA_ARCH__ >= 700) {
      auto const value =
        cuda::ptx::ld_L1_evict_first(cuda::ptx::space_global_t{}, reinterpret_cast<T const*>(src));
      cuda::ptx::st_L1_evict_first(cuda::ptx::space_global_t{}, reinterpret_cast<T*>(dst), value);
      return;
    }
#endif
  }
  *reinterpret_cast<T*>(dst) = *reinterpret_cast<T const*>(src);
}

// INVARIANT: element_size * row is computed in 64-bit below; it can exceed 2^31 for DECIMAL128.
// 16-byte copies stay aligned because device allocations are >=256B aligned and offsets are
// whole elements. The dst stores and src loads are evict-first only when kStreaming (the map
// load stays default-cached in every regime: it is the reuse-critical stream and the wide-tile
// win depends on its L2 residency).
template <bool kStreaming>
__device__ void copy_element(char* dst, char const* src, int32_t element_size)
{
  switch (element_size) {
    case 1: streaming_copy<kStreaming, int8_t>(dst, src); break;
    case 2: streaming_copy<kStreaming, int16_t>(dst, src); break;
    case 4: streaming_copy<kStreaming, int32_t>(dst, src); break;
    case 8: streaming_copy<kStreaming, int64_t>(dst, src); break;
    case 16: {
      streaming_copy<kStreaming, int64_t>(dst, src);
      streaming_copy<kStreaming, int64_t>(dst + 8, src + 8);
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
 *
 * `kMasked` selects the mask machinery: `true` reads `src_mask`, ballots validity into
 * `dst_mask` words, and accumulates `valid_counts`; `false` is launched only when no descriptor
 * carries either mask, so it drops the ballots, mask-word stores, source-mask reads, and
 * valid-bit tallies entirely.
 *
 * INVARIANT (kMasked == true): each column's dst_mask word w is written exactly once, by lane 0
 * of the warp covering rows [32w, 32w + 32): this requires BlockSize to be a multiple of
 * warp_size (the row stride is then a multiple of 32), which the driver guarantees. Trailing
 * bits of the last word are 0. Per-column bookkeeping (warp_valid[k], per-column ballots)
 * keeps this true independently of the tile a column belongs to.
 */
template <size_type BlockSize, size_type kTileCols, bool kStreaming, bool kMasked>
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
    // The masked instantiation tallies per-column valid bits and carries the row-loop predicate
    // in one active-lane ballot mask; the unmasked instantiation has neither.
    [[maybe_unused]] size_type warp_valid[kTileCols]{};
    auto const row_stride        = static_cast<int64_t>(blockDim.x) * gridDim.x;
    auto row                     = static_cast<int64_t>(blockIdx.x) * BlockSize + threadIdx.x;
    [[maybe_unused]] auto active = 0xFFFF'FFFFu;
    if constexpr (kMasked) { active = __ballot_sync(active, row < num_rows); }
    while (row < num_rows) {
      // One map load (wrap + table-level bounds check) serves every column in the tile.
      // Deliberately default-cached: the map is the reuse-critical stream and the wide-tile
      // win depends on its L2 residency (no eviction hint here).
      auto idx = gather_map[static_cast<size_type>(row)];
      if (negative_allowed && idx < 0) { idx += num_source_rows; }
      bool const in_bounds = !check_bounds || (idx >= 0 && idx < num_source_rows);
#pragma unroll
      for (int32_t k = 0; k < kTileCols; ++k) {
        if (tile_begin + k < num_descriptors) {
          auto const d = tile_descs[k];
          bool valid   = in_bounds;
          if constexpr (kMasked) {
            if (valid && d.src_mask != nullptr) {
              valid = bit_is_set(d.src_mask, static_cast<size_type>(d.mask_offset + idx));
            }
          }
          // Data at out-of-bounds/null rows may be left unwritten (fixed-width undefined-value
          // policy).
          if (valid && !d.skip_data) {
            copy_element<kStreaming>(
              static_cast<char*>(d.dst_data) + static_cast<std::size_t>(row) * d.element_size,
              static_cast<char const*>(d.src_data) + static_cast<std::size_t>(idx) * d.element_size,
              d.element_size);
          }
          if constexpr (kMasked) {
            if (d.dst_mask != nullptr) {
              auto const word = __ballot_sync(active, valid);
              if (threadIdx.x % warp_size == 0) {
                d.dst_mask[word_index(static_cast<size_type>(row))] = word;
                warp_valid[k] += cuda::std::popcount(word);
              }
            }
          }
        }
      }
      row += row_stride;
      if constexpr (kMasked) { active = __ballot_sync(active, row < num_rows); }
    }
    if constexpr (kMasked) {
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
    }
    // All threads are done reading tile_descs; the next tile iteration rewrites it.
    __syncthreads();
  }
}

// Enqueues the map-locality sample copies on the stream (one batched D2H into pinned staging —
// no kernel launch, no sync). Returns a handle with empty staging for unclassifiable maps.
sampled_map_locality enqueue_map_locality_sample(column_view const& gather_map,
                                                 cuda::stream_ref stream)
{
  switch (gather_map.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: break;
    default:
      return sampled_map_locality::empty(
        stream);  // Not an index type: unclassifiable, safe default.
  }
  auto const num_rows    = gather_map.size();
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
  return sampled_map_locality{std::move(staging), gather_map.type().id(), elem_size};
}

// Synchronizes the stream (the sample copies plus whatever was enqueued before them — never a
// kernel of ours) and classifies the staged windows.
size_type resolve_map_locality_tile_cols(sampled_map_locality const& sampled,
                                         cuda::stream_ref stream)
{
  if (sampled.staging.empty()) { return kRandomTileCols; }
  cudf::detail::sync_stream(stream);
  auto const elem_size    = sampled.elem_size;
  auto const read_element = [&](std::byte const* p) -> int64_t {
    switch (sampled.map_type) {
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
      default: CUDF_UNREACHABLE("Map type excluded at enqueue time");
    }
  };
  // Streaming needs genuine sequential locality, not merely local contiguity. Two checks per
  // sampled window, both on the staged data:
  // 1. The window's kSampleWindowLen values are a permutation of consecutive integers: a
  //    fanout map (map[i] = i/2, spread kSampleWindowLen/2) or any map with duplicates fails
  //    even though its spread stays small, and those rows are reuse-heavy.
  // 2. The window minima are strictly monotonic in one direction (ascending maps rise,
  //    descending maps fall): a periodic map such as map[i] = i % K repeats its per-window
  //    minimum and is reuse-heavy even when a window happens to be a permutation.
  // A map failing either keeps its routing decision and takes default-cached accesses; this
  // gates only the eviction hint, never the geometry.
  int minima_direction = 0;  // +1 rising, -1 falling; fixed by the first window pair.
  int64_t prev_min     = 0;
  for (size_t w = 0; w < kSampleWindows; ++w) {
    auto const* window = sampled.staging.data() + w * kSampleWindowLen * elem_size;
    auto const first   = read_element(window);
    auto min_value     = first;
    auto max_value     = first;
    for (size_type i = 1; i < kSampleWindowLen; ++i) {
      auto const v = read_element(window + static_cast<size_t>(i) * elem_size);
      min_value    = std::min(min_value, v);
      max_value    = std::max(max_value, v);
    }
    if (max_value - min_value != static_cast<int64_t>(kSampleWindowLen - 1)) {
      return kRandomTileCols;
    }
    uint32_t seen = 0;
    for (size_type i = 0; i < kSampleWindowLen; ++i) {
      seen |=
        uint32_t{1} << (read_element(window + static_cast<size_t>(i) * elem_size) - min_value);
    }
    if (seen != (uint32_t{1} << kSampleWindowLen) - 1) { return kRandomTileCols; }
    if (w > 0) {
      if (min_value == prev_min) { return kRandomTileCols; }  // flat run: periodic/broadcast
      auto const step = (min_value > prev_min) ? 1 : -1;
      if (minima_direction == 0) {
        minima_direction = step;
      } else if (step != minima_direction) {
        return kRandomTileCols;
      }
    }
    prev_min = min_value;
  }
  return kCoalescedTileCols;
}

// Tile width for regimes where sampling is skipped; nullopt means "sample the map". Zero/one
// fused columns keep K1; otherwise the unsampled choice is K2 while the two-column working set
// plus the map fit half of L2 (a failed L2 query reports 0 bytes and takes the safe K1).
std::optional<size_type> unsampled_tile_width(size_type num_map_rows,
                                              size_type num_src_rows,
                                              size_type num_fused_cols,
                                              size_type max_fused_element_size,
                                              size_type map_element_size)
{
  if (num_fused_cols < 2) { return kRandomTileCols; }
  auto const work = static_cast<int64_t>(num_map_rows) * static_cast<int64_t>(num_fused_cols);
  if (num_map_rows >= kMinSampledRows && work >= kMinSampledWork) { return std::nullopt; }
  static int const l2_bytes = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, device) != cudaSuccess) { return 0; }
    return value;
  }();
  auto const footprint =
    static_cast<int64_t>(num_src_rows) * static_cast<int64_t>(max_fused_element_size) * 2 +
    static_cast<int64_t>(num_map_rows) * static_cast<int64_t>(map_element_size);
  if (footprint * 2 > static_cast<int64_t>(l2_bytes)) { return kRandomTileCols; }
  return kUnsampledTileCols;
}

}  // namespace

size_type select_gather_map_tile_cols(table_view const& source_table,
                                      column_view const& gather_map,
                                      size_type num_fused_cols,
                                      size_type max_fused_element_size,
                                      cuda::stream_ref stream)
{
  auto const gate = unsampled_tile_width(gather_map.size(),
                                         source_table.num_rows(),
                                         num_fused_cols,
                                         max_fused_element_size,
                                         static_cast<size_type>(cudf::size_of(gather_map.type())));
  if (gate.has_value()) { return *gate; }
  auto const sampled = enqueue_map_locality_sample(gather_map, stream);
  return resolve_map_locality_tile_cols(sampled, stream);
}

// Shared body of the routed and explicit entry points; the routed entry enqueues the sampler
// before its allocation work and leaves the tile width unresolved until just before the launch.
std::unique_ptr<table> batch_gather_impl(table_view const& source_table,
                                         column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         negative_index_policy neg_indices,
                                         cuda::stream_ref stream,
                                         memory_resources mr,
                                         std::optional<size_type> explicit_tile_cols,
                                         sampled_map_locality const& sampled);

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr)
{
  // Routed entry: one host-side type pass gathers the fused-column count and the widest fused
  // element for the gate; when sampling is warranted it is ENQUEUED here and RESOLVED inside
  // the body just before the kernel launch, hiding its latency behind the allocation work.
  size_type num_fused_cols      = 0;
  size_type max_fused_elem_size = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    auto const& col = source_table.column(i);
    if (cudf::is_fixed_width(col.type())) {
      ++num_fused_cols;
      max_fused_elem_size =
        std::max(max_fused_elem_size, static_cast<size_type>(cudf::size_of(col.type())));
    }
  }
  if (num_fused_cols == 0 or gather_map.has_nulls()) {
    // Nothing to fuse (or the delegate rejects the map): no sampler, no measurable tile width.
    return batch_gather(
      source_table, gather_map, bounds_policy, neg_indices, stream, mr, kDefaultTileCols);
  }
  auto const tile_cols =
    unsampled_tile_width(gather_map.size(),
                         source_table.num_rows(),
                         num_fused_cols,
                         max_fused_elem_size,
                         static_cast<size_type>(cudf::size_of(gather_map.type())));
  if (tile_cols.has_value()) {
    return batch_gather(
      source_table, gather_map, bounds_policy, neg_indices, stream, mr, *tile_cols);
  }
  auto const sampled = enqueue_map_locality_sample(gather_map, stream);
  return batch_gather_impl(
    source_table, gather_map, bounds_policy, neg_indices, stream, mr, std::nullopt, sampled);
}

std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols)
{
  return batch_gather_impl(source_table,
                           gather_map,
                           bounds_policy,
                           neg_indices,
                           stream,
                           mr,
                           std::optional<size_type>{tile_cols},
                           sampled_map_locality::empty(stream));
}

std::unique_ptr<table> batch_gather_impl(table_view const& source_table,
                                         column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         negative_index_policy neg_indices,
                                         cuda::stream_ref stream,
                                         memory_resources mr,
                                         std::optional<size_type> explicit_tile_cols,
                                         sampled_map_locality const& sampled)
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

    // Null-mask phase over the fallback subset, mirroring the stock gather's WHOLE-TABLE
    // conditions: a null-bearing fused column must push the fallback
    // through gather_bitmask exactly as stock does. The APPLICATION stays subset-scoped — the
    // fused columns' masks are kernel-written per the mask-policy table, which matches stock's
    // per-column outcome (gather_bitmask is per-column self-guarding,
    // and set_all_valid_null_masks is a no-op for non-nullable inputs).
    auto fallback_views = std::vector<cudf::column_view>{};
    fallback_views.reserve(indices.size());
    for (auto const idx : indices) {
      fallback_views.push_back(source_table.column(idx));
    }
    auto const fallback_table = cudf::table_view{fallback_views};
    if (bounds_policy == out_of_bounds_policy::NULLIFY or
        cudf::has_nested_nullable_columns(source_table)) {
      if (bounds_policy == out_of_bounds_policy::NULLIFY or cudf::has_nested_nulls(source_table)) {
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
      // A cached zero null count means no source bit is clear, so the mask cannot change any
      // row's validity: drop it here to keep the call eligible for the mask-free instantiation.
      // The destination mask policy above keys off `dst_mask` only and is unaffected.
      bitmask_type const* src_mask = skip_data ? nullptr : src.null_mask();
      if (src.null_count() == 0) { src_mask = nullptr; }
      descriptors.push_back(column_descriptor{src_data,
                                              dst->mutable_view().data<std::byte>(),
                                              src_mask,
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

    // The mask-free instantiation is valid only when NO descriptor writes a destination mask or
    // reads a source mask. `kernel_masked_positions` covers the destination side only, so the
    // source side is checked here as well; both are host descriptor fields, not device state.
    auto const any_kernel_masks =
      not kernel_masked_positions.empty() or
      std::any_of(descriptors.begin(), descriptors.end(), [](column_descriptor const& d) {
        return d.src_mask != nullptr;
      });
    rmm::device_uvector<size_type> valid_counts(0, stream, temp_mr);
    if (any_kernel_masks) {
      valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        descriptors.size(), stream, temp_mr);
    }

    // One launch per call; the tile width picks the kernel instantiation. y dimension: one
    // block per column tile (the kernel's y-stride loop covers tiles past the 65535 cap).
    auto const map_begin   = indexalator_factory::make_input_iterator(gather_map);
    auto const launch_tile = [&](auto tile_constant, auto streaming_constant) {
      constexpr size_type kTileCols = decltype(tile_constant)::value;
      constexpr bool kStreaming     = decltype(streaming_constant)::value;
      dim3 const grid{
        static_cast<unsigned>(cudf::util::div_rounding_up_safe(num_rows, block_size)),
        static_cast<unsigned>(std::min<size_type>(
          cudf::util::div_rounding_up_safe(static_cast<size_type>(descriptors.size()), kTileCols),
          65535)),
        1};
      // The mask-free choice comes from host descriptor fields, so it selects the instantiation
      // directly; the unmasked one never receives a `valid_counts` buffer.
      auto const launch = [&](auto masked_constant) {
        constexpr bool kMasked = decltype(masked_constant)::value;
        batch_gather_kernel<block_size, kTileCols, kStreaming, kMasked>
          <<<grid, block_size, 0, stream.get()>>>(map_begin,
                                                  d_descriptors.data(),
                                                  static_cast<size_type>(descriptors.size()),
                                                  num_rows,
                                                  num_src_rows,
                                                  bounds_policy == out_of_bounds_policy::NULLIFY,
                                                  neg_indices == negative_index_policy::ALLOWED,
                                                  kMasked ? valid_counts.data() : nullptr);
        CUDF_CUDA_TRY(cudaGetLastError());
      };
      if (any_kernel_masks) {
        launch(std::true_type{});
      } else {
        launch(std::false_type{});
      }
    };
    // Resolve the routed tile width only now: the sync waits for the enqueued sample copies
    // AND for the allocation/descriptor work above, hiding the sampler latency behind it.
    auto const effective_tile_cols = explicit_tile_cols.has_value()
                                       ? *explicit_tile_cols
                                       : resolve_map_locality_tile_cols(sampled, stream);
    // Evict-first streaming hints ride the classifier verdict only: the sampled coalesced
    // route (which resolves to kCoalescedTileCols and nothing else). Unsampled regimes,
    // random verdicts, and test-forced geometries keep the default-cached accesses.
    auto const streaming =
      not explicit_tile_cols.has_value() and effective_tile_cols == kCoalescedTileCols;
    if (streaming) {
      launch_tile(std::integral_constant<size_type, kCoalescedTileCols>{}, std::true_type{});
    } else {
      switch (effective_tile_cols) {
        case 1: launch_tile(std::integral_constant<size_type, 1>{}, std::false_type{}); break;
        case 2: launch_tile(std::integral_constant<size_type, 2>{}, std::false_type{}); break;
        case 4: launch_tile(std::integral_constant<size_type, 4>{}, std::false_type{}); break;
        case 8: launch_tile(std::integral_constant<size_type, 8>{}, std::false_type{}); break;
        default: CUDF_FAIL("Unsupported batch_gather tile width", std::invalid_argument);
      }
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
