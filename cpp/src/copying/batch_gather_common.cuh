/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Internal header shared by the batch_gather translation units (batch_gather.cu and
// batch_gather_strings.cu). Nothing here is public API: the fused gather's device types,
// helpers, and the string fast path's deferred record live here so both TUs compile against
// one definition. The build is NOT relocatable-device-code (no CUDA_SEPARABLE_COMPILATION in
// cpp/CMakeLists.txt), so every __device__ helper and __global__ kernel must compile in the
// same TU as its launcher — this header carries them verbatim rather than splitting them
// across TUs.

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

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
 * One descriptor per fixed-width LEAF, mask-only STRUCT node, or dictionary index payload. All
 * pointers are element bases, not byte bases: a leaf's `src_data` comes from
 * `column_view::data<T>()`, which has that view's own offset pre-applied in element units, plus
 * the accumulated offsets of the struct ancestors the leaf sits under (applied exactly once
 * here); `dst_data`/`dst_mask` belong to a freshly allocated destination whose offset is zero.
 * Nothing downstream may re-add an offset to these pointers.
 *
 * `src_data`/`dst_data` are nullptr for mask-only STRUCT node descriptors (`skip_data` true,
 * nothing to copy). `src_mask` is a mask BASE pointer (nullptr if the source is non-nullable);
 * `mask_offset` is the bit offset of the first row of the viewed range within that
 * base. Descriptors are pushed only for kernel-visible work; an all-null destination (mask
 * pre-set ALL_NULL) gets none — see `plan_destination_mask`'s `skip_data`.
 */
struct alignas(16) column_descriptor {
  void const* src_data;          // first source element; nullptr for mask-only struct nodes
  void* dst_data;                // first destination element; nullptr for mask-only struct nodes
  bitmask_type const* src_mask;  // source mask BASE pointer; nullptr if source non-nullable
  bitmask_type* dst_mask;        // destination mask; nullptr if dst non-nullable or no mask
  int64_t mask_offset;           // bit offset into src_mask (sum of the viewed range's offsets)
  int32_t element_size;          // cudf::size_of(src.type()); one of {0, 1, 2, 4, 8, 16}
  bool skip_data;                // mask-only node: no data copy (all-null leaves get no descriptor)
};

// INVARIANT: element_size * row is computed in 64-bit below; it can exceed 2^31 for DECIMAL128.
// 16-byte copies stay aligned because device allocations are >=256B aligned and offsets are
// whole elements.
inline __device__ void copy_element(char* dst, char const* src, int32_t element_size)
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
// dispatches to int32/int64/__int128), so the view's own offset is applied exactly once,
// host-side. The dispatcher instantiates operator() for every type in the dispatch list,
// including the pass-through types (string_view/list_view/struct_view/dictionary32) that never
// carry data<T>; they are unreachable here because only fixed-width storage enters the
// descriptor path (plain fixed-width columns, struct leaves, dictionary indices).
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
 *
 * `valid_counts`, when non-null, holds one entry per kernel-masked descriptor (M entries): the
 * caller has stable-partitioned `descriptors` so masked ones occupy positions [0, M) in visit
 * order, so a masked descriptor's position IS its `valid_counts` slot and unmasked descriptors
 * (dst_mask null) leave no entry. With all descriptors masked (the common flat case) the
 * layout is exactly the pre-compaction one.
 */

// Row-length gate (bytes) between the warp-cooperative and the block-cooperative copy. A
// 32-byte row spans at most two 16-byte vectors, so one warp's uint4 span copies it without
// idle lanes; a longer row serialises across the warp's lanes and the whole block copies it.
// The 32-byte boundary matches the short/long strategy split of `strings::detail::gather`.
constexpr int64_t kBlockRowCopyThreshold{32};

// Row-count gate (rows) above which the CUB batched-memcpy engine runs instead of the payload
// kernel's chars phase. Mirrors `strings::detail::gather`'s engine split: stock measured
// `cub::DeviceMemcpy::Batched` as the winning short-string strategy at >= 524288 rows on H100,
// and the tall-cell profile shows the payload kernel's per-group barrier walk costing the same
// regime ~7x its CUB counterpart. Below the gate the payload kernel keeps its 65536-row wins.
constexpr int64_t kBatchMemcpyRowThreshold{524'288};

// Row-count gate (rows) between the N3 scan+payload string fast path and the stock per-column
// string path (run_fallback -> column_gatherer<string_view> -> strings::detail::gather). The
// payload's group-walk cost scales with the call's row count: measured A/B on tall cells shows
// every string shape at >= 262144 rows losing to the stock path (up to ~1.9x) while 65536 rows
// wins ~1.7x, and the loss persists even when every column's chars route to CUB — so the
// discriminator is the call's row count, not bytes-per-row. Above this threshold the
// chars-routing branch below becomes unreachable through this API; keep it (the stock path
// runs the same CUB engine there, and a lowered threshold or a1 coarsening revives the band).
constexpr size_type kStringFallbackRowThreshold{262'144};

// Preset-mask staging bands pay off only above this many preset destination masks in one call:
// a band memset costs ~24 µs (nsys-captured band-fill attribution) while a per-mask memset
// submission costs ~7 µs (submission-latency-bound — the device bytes are negligible at both
// the 8 KiB and 128 KiB mask extents), so the fill-method crossover sits at ~4 preset masks
// with one band populated to ~7 with both. 8 keeps a margin on the drip side while the
// measured win shapes (~60 preset masks) clear it by ~9x. Counted in preset MASKS, not
// destination columns: a wide table with few nullable columns would otherwise pay the fixed
// band cost for almost no replaced drip.
constexpr size_type kMaskBandPresetGate{8};

// Work item for one string column on the N3 path: source addressing plus the
// destination buffers the payload kernel writes. The src fields are complete
// when the item is registered (the scan reads only those); the dst fields are
// filled after the totals readback, before the payload launch. The
// user-provided host-only default ctor keeps the implicit one (which nvcc
// would otherwise attribute __host__ __device__ on this kernel-parameter
// struct) from calling the host-only constexpr offsetalator ctors in device
// context.
struct string_work_item {
  string_work_item() {}
  char const* src_chars;            // raw chars base; the offsets carry addressing
  bitmask_type const* src_mask;     // source mask BASE pointer; nullptr if non-nullable
  int64_t mask_bit_offset;          // bit offset of the viewed range within src_mask
  size_type src_rows;               // viewed source row count (bounds checks)
  input_offsetalator src_offsets;   // source offsets with the view offset applied
  int64_t const* dst_scratch;       // this column's range of the flat scratch
  char* dst_chars;                  // destination chars buffer base
  output_offsetalator dst_offsets;  // destination offsets child (narrowing stores)
  bitmask_type* dst_mask;           // destination mask; nullptr when pre-set
  size_type* valid_count;           // atomic valid-row counter; nullptr when pre-set
  // True when this column's chars ride cub::DeviceMemcpy::Batched (tall column whose
  // average gathered row is at or below the block-copy gate): the kernel's chars phase must
  // skip the column, masks/offsets stay kernel-owned.
  bool chars_via_batched_memcpy{false};
};

static_assert(std::is_trivially_copyable_v<string_work_item>,
              "string_work_item is staged through pinned memory to the device");

// Resolves destination row -> gathered source row, or -1 when the row carries
// no payload. Under DONT_CHECK the bounds test is skipped (stock parity: the
// source is then read at the raw index, undefined for out-of-bounds maps).
inline __device__ int64_t resolve_gathered_row(input_indexalator gather_map,
                                               size_type row,
                                               size_type num_source_rows,
                                               bool check_bounds,
                                               bool negative_allowed)
{
  auto idx = gather_map[row];
  if (negative_allowed && idx < 0) { idx += num_source_rows; }
  bool const in_bounds = !check_bounds || (idx >= 0 && idx < num_source_rows);
  return in_bounds ? static_cast<int64_t>(idx) : int64_t{-1};
}

// Payload validity of a resolved row: the source column's own mask only.
// Ancestor validity is never consulted (stock never ANDs ancestor masks).
inline __device__ bool string_row_valid(string_work_item const& item, int64_t src_row)
{
  return item.src_mask == nullptr ||
         bit_is_set(item.src_mask, static_cast<size_type>(item.mask_bit_offset + src_row));
}

// Layout of the flat string workspaces: the scan and the offsets phase index the int64
// scratch with `rows_per_col = num_rows + 1` slots per column, while the chars phase indexes
// flat ROWS with `num_rows` rows per column — two divisors. Every grid cell holds at most
// 41.9M flat rows, so every flat index fits uint32_t (the host enforces the bound); the
// precomputed 32-bit divisors replace the per-element 64-bit //% at each site.
struct string_flat_layout {
  cuda::fast_mod_div<uint32_t> rows_per_col;  // scan/offsets scratch space
  cuda::fast_mod_div<uint32_t> num_rows;      // chars-phase flat-row space
};

// Scan output: the exclusive prefix into the flat scratch; each column's tail
// slot doubles as the per-column total, copied out for the batched readback.
struct string_scan_out_fn {
  int64_t* scratch;
  int64_t* col_totals;
  string_flat_layout layout;

  __device__ void operator()(int64_t flat, int64_t value) const
  {
    scratch[flat] = value;
    auto const r  = static_cast<uint32_t>(flat) % layout.rows_per_col;
    if (r == layout.rows_per_col - 1u) {
      col_totals[static_cast<uint32_t>(flat) / layout.rows_per_col] = value;
    }
  }
};

// One row published by a warp for the block-cooperative copy; `size == 0` is the no-work marker.
struct block_row_copy {
  char const* src;
  char* dst;
  int64_t size;
};

// Char-parallel copy of one all-short block group's rows: every live byte of the group's span
// is addressed by exactly one thread. A thread owns the byte at `base + tid + k*blockDim.x`
// for k >= 0 until the span ends and finds its owning row by a binary search over the staged
// boundaries (thrust::upper_bound + cuda::std::prev, the same device idiom as stock's
// char-parallel kernel), then resolves that destination row through the gather map exactly
// like every other chars path. `group_offsets` is READ-ONLY here — the caller's common
// boundary stage staged it (and its first warp already published the offsets values from
// the same boundaries). Dead rows contribute no bytes (their scratch deltas are 0) but stage
// as zero-width boundaries. The caller fans only groups whose rows all belong to ONE column
// and none is published: a per-column scratch slice restarts at each column boundary, so a
// cross-column window would mix two byte spaces, and the block slot loop would race the fan
// on published rows' bytes.
inline __device__ void copy_group_chars_char_parallel(string_work_item const& item,
                                                      string_flat_layout layout,
                                                      int64_t group_start,
                                                      int64_t group_end,
                                                      int64_t const* group_offsets,
                                                      int64_t warps_per_block,
                                                      input_indexalator gather_map,
                                                      size_type num_source_rows,
                                                      bool check_bounds,
                                                      bool negative_allowed)
{
  auto const lane = static_cast<int64_t>(threadIdx.x);
  // The window's boundaries are ALREADY staged in `group_offsets` by the caller's common
  // boundary stage: slot w = the dst offset of flat row group_start + w (the +0 boundary of
  // that row), the tail slot closing the window's last row with its +1 scratch entry, slots
  // past the flat end INT64_MAX (an unreachable boundary). Both barriers that separated the
  // staging from the byte copy and from the next group's rewrite live in the caller.
  auto const base = group_offsets[0];
  auto const end  = group_offsets[warps_per_block];
  if (end <= base) { return; }  // no live bytes in the window
  for (int64_t b = base + lane; b < end; b += static_cast<int64_t>(blockDim.x)) {
    auto const it = cuda::std::prev(
      thrust::upper_bound(thrust::seq, group_offsets, group_offsets + warps_per_block, b));
    auto const dst_row = static_cast<size_type>(
      static_cast<uint32_t>(group_start + (it - group_offsets)) % layout.num_rows);
    auto const src =
      resolve_gathered_row(gather_map, dst_row, num_source_rows, check_bounds, negative_allowed);
    item.dst_chars[b] = item.src_chars[item.src_offsets[src] + (b - *it)];
  }
}

// Copies one gathered string row with `stride` threads starting at `tid`: the same alignment
// arithmetic, uint4 interior and scalar edges for both copy strategies. The warp path calls it
// with `(lane, warp_size)`, the block-cooperative path with `(threadIdx.x, BlockSize)`.
inline __device__ void copy_string_span(
  char* dst_ptr, char const* src_ptr, int64_t size, int64_t tid, int64_t stride)
{
  if (size <= 0) { return; }
  // The stock boundary arithmetic, on the per-row dst base (the base is
  // 16-aligned only at row 0 of the buffer; every row realigns).
  auto const alignment_offset =
    static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(dst_ptr) % 16);
  auto const out_start_aligned =
    (int64_t{4} + alignment_offset + int64_t{15}) / int64_t{16} * int64_t{16} - alignment_offset;
  auto const out_end_aligned =
    (size - int64_t{4} + alignment_offset) / int64_t{16} * int64_t{16} - alignment_offset;
  if (out_end_aligned > out_start_aligned) {
    for (int64_t p = out_start_aligned + tid * int64_t{16}; p < out_end_aligned;
         p += stride * int64_t{16}) {
      *reinterpret_cast<uint4*>(dst_ptr + p) = strings::detail::load_uint4(src_ptr + p);
    }
    if (tid < out_start_aligned) { dst_ptr[tid] = src_ptr[tid]; }
    auto const tail = out_end_aligned + tid;
    if (tail < size) { dst_ptr[tail] = src_ptr[tail]; }
  } else {
    for (int64_t p = tid; p < size; p += stride) {
      dst_ptr[p] = src_ptr[p];
    }
  }
}

// Batched payload kernel for the UNROUTED string columns (one launch per call; the routed
// cohort's metadata rides string_routed_metadata_kernel).
//
// Grid-stride phases over disjoint outputs, no ordering required between
// them:
//   chars   — every (column, row) pair with a live source takes one warp. The
//             warp re-resolves the row per lane (one map read each — the map
//             line is shared, and the resolve is 3 registers), giving every
//             lane the (src, dst, size) tuple without any shuffle; the copy
//             then follows the stock string-parallel strategy exactly (uint4
//             interior between the alignment-safe bounds, scalar edges). A row
//             longer than `kBlockRowCopyThreshold` is published to shared
//             memory and copied by the whole block instead, so it does not
//             serialise across one warp's lanes. A group with NO published row
//             (every live row at or under the gate) is instead copied
//             char-parallel by the whole block: one thread per output byte,
//             the row found by a binary search over the group's staged
//             boundaries — the stock char-parallel strategy applied to the
//             already-scanned offsets, which removes the rows-per-warp
//             starvation on tall short-row tables. EVERY group first runs a
//             common boundary stage that stages the group's row boundaries and
//             (from the same scratch reads) publishes the offsets values. The
//             launch grid covers ONLY the unrouted cohort's flat rows: a
//             column routed to `cub::DeviceMemcpy::Batched`
//             (`chars_via_batched_memcpy`) contributes no group visits at all
//             — CUB owns its bytes, and `string_routed_metadata_kernel` owns
//             its offsets/masks/counts.
//   masks   — one thread per (column, 32-row window): the thread re-resolves
//             each row of its window and packs the gathered validity into one
//             destination mask word, popping the count into the column's
//             atomic valid counter. Routed columns are excluded entirely
//             (their words/counts ride the routed kernel); pre-set masks
//             (ALL_NULL/ALL_VALID) skip the phase (`dst_mask` nullptr).
//   offsets — for unrouted columns, the group loop's boundary stage publishes every flat
//             row's offsets value (the group's first warp, adjacent lanes on adjacent flat
//             rows, from the staged shared-memory boundaries) plus each column's N+1
//             sentinel; the per-entry flat copy loop of the original design is deleted,
//             removing the second full scratch traversal and its per-entry 64-bit div/mod.
//             Routed columns' offsets children are written by the routed kernel instead.
template <size_type BlockSize>
CUDF_KERNEL void string_payload_kernel(string_work_item const* items,
                                       size_type const* unrouted,  // unrouted cohort -> items
                                       int64_t num_unrouted,
                                       string_flat_layout layout,  // uint32 rows_per_col, num_rows
                                       size_type num_source_rows,
                                       bool check_bounds,
                                       bool negative_allowed,
                                       input_indexalator gather_map)
{
  // The chars phase coordinates a block group: one slot per warp, a common boundary stage
  // that also publishes the offsets values, then one of three copy strategies per group.
  static_assert(BlockSize % warp_size == 0);
  auto const global_tid  = static_cast<int64_t>(blockIdx.x) * BlockSize + threadIdx.x;
  auto const grid_stride = static_cast<int64_t>(gridDim.x) * BlockSize;
  auto const lane        = static_cast<size_type>(threadIdx.x % warp_size);

  // ---- chars: one warp per row at or below the block-copy gate; the whole block copies a
  // longer row; an all-short group is copied char-parallel by the whole block ----
  //
  // Rows are handed out per block group: the block's warps take consecutive flat slots and the
  // group bound is block-uniform, so every thread reaches the barriers below.
  auto const warps_per_block = static_cast<int64_t>(BlockSize / warp_size);
  auto const warp_in_block   = static_cast<int64_t>(threadIdx.x) / static_cast<int64_t>(warp_size);
  auto const thread_in_block = static_cast<int64_t>(threadIdx.x);
  __shared__ block_row_copy block_rows[BlockSize / warp_size];
  // The fan window: the group's row boundaries plus the tail sentinel closing the last row.
  __shared__ int64_t group_offsets[BlockSize / warp_size + 1];
  __shared__ size_type pub_col;       // warp 0's publish: the window's column
  __shared__ size_type pub_base_row;  // ...and the first flat row's position inside it

  auto const num_rows   = layout.num_rows;
  auto const total_rows = num_unrouted * static_cast<int64_t>(num_rows);
  for (int64_t group = static_cast<int64_t>(blockIdx.x); group * warps_per_block < total_rows;
       group += static_cast<int64_t>(gridDim.x)) {
    auto const flat = group * warps_per_block + warp_in_block;

    string_work_item const* item_ptr = nullptr;
    size_type row                    = 0;
    char const* src_ptr              = nullptr;
    char* dst_ptr                    = nullptr;
    int64_t size                     = 0;
    if (flat < total_rows) {
      // The flat domain is the UNROUTED cohort's (column, row) space: the cohort table maps
      // the local column index back to the items slot. Every visited column is kernel-owned
      // by construction (the host built the cohort), so no per-item route flag is consulted.
      item_ptr = &items[unrouted[static_cast<uint32_t>(flat) / num_rows]];
      row      = static_cast<size_type>(static_cast<uint32_t>(flat) % num_rows);
      auto const src =
        resolve_gathered_row(gather_map, row, num_source_rows, check_bounds, negative_allowed);
      if (src >= 0 and string_row_valid(*item_ptr, src)) {
        auto const s = static_cast<size_type>(src);
        size         = item_ptr->src_offsets[s + 1] - item_ptr->src_offsets[s];
        if (size > 0) {
          src_ptr = item_ptr->src_chars + item_ptr->src_offsets[s];
          dst_ptr = item_ptr->dst_chars + item_ptr->dst_scratch[row];
        }
      }
    }

    // Publish this warp's row for the block-cooperative copy; an empty slot is the no-work
    // marker and also the all-short vote input (the vote reads the slots below).
    auto const publishable = size > kBlockRowCopyThreshold;
    if (lane == 0) {
      block_rows[warp_in_block] =
        publishable ? block_row_copy{src_ptr, dst_ptr, size} : block_row_copy{};
    }
    auto const group_start = group * warps_per_block;
    auto const group_end   = cuda::std::min(group_start + warps_per_block, total_rows);
    // Common boundary stage, shared by every chars strategy: each warp's lane 0 stages the
    // +0 boundary of its OWN flat row (a dead or empty row stages its zero-delta boundary
    // like any other); a warp past the flat end stages INT64_MAX, the unreachable boundary
    // that keeps the fan's binary search inside the live slots even when a partial final
    // group straddles a column boundary (stale values from the previous group are NOT
    // monotone across it). Slot 0 of the whole domain holds flat row 0's +0 scratch entry.
    if (flat < total_rows) {
      if (lane == 0) { group_offsets[warp_in_block] = item_ptr->dst_scratch[row]; }
    } else if (lane == 0) {
      group_offsets[warp_in_block] = cuda::std::numeric_limits<int64_t>::max();
    }
    // Stage the TAIL boundary: the +1 scratch entry of the window's last live row closes its
    // span for the binary search. (The +0 boundary of the NEXT flat row is staged by the next
    // group's own warp, so the tail must come from this group's rows.) Every executed group
    // has a nonempty window, so this always runs.
    if (thread_in_block == 0) {
      auto const last_row = static_cast<size_type>(static_cast<uint32_t>(group_end - 1) % num_rows);
      group_offsets[warps_per_block] =
        items[unrouted[static_cast<size_type>(static_cast<uint32_t>(group_end - 1) / num_rows)]]
          .dst_scratch[last_row + 1];
    }
    // Hoist the window's column geometry for warp 0's offsets publish: ONE div/mod pair per
    // group (thread 0, group-uniform) replaces the per-lane pair below. Warp 0 cannot
    // broadcast by shuffle — its lanes diverge on the lane_flat bound — so the values ride
    // the boundary-stage shared memory instead.
    if (thread_in_block == 0) {
      pub_col = unrouted[static_cast<size_type>(static_cast<uint32_t>(group_start) / num_rows)];
      pub_base_row = static_cast<size_type>(static_cast<uint32_t>(group_start) % num_rows);
    }
    __syncthreads();

    // Offsets publish FROM the staged boundaries: the group's first warp stores one offsets
    // value per active lane, adjacent lanes taking adjacent flat rows — for a single-column
    // group that is 8 consecutive INT32 entries of one child (the per-warp-lane-0 pattern
    // was rejected: one 4-byte transaction per warp). This replaces the deleted flat offsets
    // loop, whose per-entry 64-bit div/mod re-read every scratch entry a second time from
    // global memory; here the values are already resident in shared memory. One writer per
    // entry: each flat row belongs to exactly one group and one lane, covering dead/empty
    // rows and long/mixed groups alike (their windows stage real boundaries too). A row that
    // ENDS its column (+1 == num_rows) also publishes that column's N+1 tail sentinel —
    // its +1 boundary is the NEXT column's row-0 boundary, staged by the next group, so the
    // sentinel is read from the scratch (one global load per column per call). Row 0's value
    // IS the row-0 sentinel: scratch[0] is both the +0 boundary of flat row 0 and the child's
    // first entry, written by the ordinary row publish above.
    if (warp_in_block == 0) {
      auto const lane_flat = group_start + static_cast<int64_t>(lane);
      // warp_size (32) exceeds the 8-row window: bound by group_end, or lanes 8+ read
      // out-of-bounds shared slots and publish rows other blocks own.
      if (lane_flat < group_end) {
        if (pub_base_row + static_cast<int64_t>(warp_size) <= num_rows) {
          // Uniform window: the only div/mod pair per group is the hoisted one above; the
          // sentinel ride (column-ending lane reads scratch[row+1]) is unchanged.
          auto const& pub_item              = items[pub_col];
          auto const pub_row                = pub_base_row + static_cast<int64_t>(lane);
          *(pub_item.dst_offsets + pub_row) = group_offsets[lane];
          if (pub_row + 1 == num_rows) {
            *(pub_item.dst_offsets + num_rows) = pub_item.dst_scratch[pub_row + 1];
          }
        } else {
          // Cross-column window (num_rows < warp_size): rows span two unrouted columns, so
          // the general per-lane div/mod path applies (mapped through the cohort table).
          auto const& pub_item =
            items[unrouted[static_cast<size_type>(static_cast<uint32_t>(lane_flat) / num_rows)]];
          auto const pub_row = static_cast<size_type>(static_cast<uint32_t>(lane_flat) % num_rows);
          *(pub_item.dst_offsets + pub_row) = group_offsets[lane];
          if (pub_row + 1 == num_rows) {
            *(pub_item.dst_offsets + num_rows) = pub_item.dst_scratch[pub_row + 1];
          }
        }
      }
    }

    // A fan group must be single-column (a per-column scratch slice restarts at each column
    // boundary, so a cross-column window is not one byte space) — the check is block-uniform
    // because num_rows and warps_per_block are.
    auto const single_column = (static_cast<uint32_t>(group_start) / num_rows) ==
                               (static_cast<uint32_t>(group_end - 1) / num_rows);
    // The fan writes every live byte of the group's span by binary search over the staged
    // boundaries; every cohort column owns its chars on the kernel path, so the vote needs
    // no route predicate.
    auto const any_long = thread_in_block < warps_per_block
                            ? block_rows[thread_in_block].size > kBlockRowCopyThreshold
                            : false;
    // __syncthreads_or is barrier-safe here: every thread participates (the predicate is
    // computed unconditionally), only the VALUE differs.
    if (__syncthreads_or(any_long or not single_column)) {
      // Block-cooperative rows, one at a time; rows above the gate are rare, so the serial
      // walk over the block's slots costs nothing on short-row data.
      for (int64_t slot = 0; slot < warps_per_block; ++slot) {
        auto const& row_copy = block_rows[slot];
        if (row_copy.size == 0) { continue; }
        copy_string_span(row_copy.dst,
                         row_copy.src,
                         row_copy.size,
                         thread_in_block,
                         static_cast<int64_t>(BlockSize));
      }
      // The short rows of a mixed group keep the warp copy (the fan did not run).
      if (not publishable) {
        copy_string_span(
          dst_ptr, src_ptr, size, static_cast<int64_t>(lane), static_cast<int64_t>(warp_size));
      }
    } else {
      // The group is all-short, single-column, and kernel-owned: fan every live byte across
      // all BlockSize threads over the boundaries staged above. Offsets were already published.
      copy_group_chars_char_parallel(
        items[unrouted[static_cast<size_type>(static_cast<uint32_t>(group_start) / num_rows)]],
        layout,
        group_start,
        group_end,
        group_offsets,
        warps_per_block,
        gather_map,
        num_source_rows,
        check_bounds,
        negative_allowed);
    }
    __syncthreads();  // the next group rewrites the slots
  }

  // ---- masks + valid counts: one thread per (column, 32-row window) ----
  // Routed columns are outside the cohort domain: string_routed_metadata_kernel owns their
  // masks and counts entirely.
  auto const words_per_col =
    util::div_rounding_up_safe(static_cast<int64_t>(num_rows), static_cast<int64_t>(warp_size));
  auto const total_windows = num_unrouted * words_per_col;
  for (int64_t w = global_tid; w < total_windows; w += grid_stride) {
    auto const& item = items[unrouted[static_cast<size_type>(w / words_per_col)]];
    if (item.dst_mask == nullptr) { continue; }
    auto const word = static_cast<size_type>(w % words_per_col);
    auto const row0 = static_cast<int64_t>(word) * warp_size;
    auto const limit =
      cuda::std::min<int64_t>(int64_t{warp_size}, static_cast<int64_t>(num_rows) - row0);
    bitmask_type word_bits = 0;
    for (int64_t s = 0; s < limit; ++s) {
      auto const src = resolve_gathered_row(gather_map,
                                            static_cast<size_type>(row0 + s),
                                            num_source_rows,
                                            check_bounds,
                                            negative_allowed);
      if (src >= 0 and string_row_valid(item, src)) { word_bits |= bitmask_type{1} << s; }
    }
    item.dst_mask[word] = word_bits;
    if (item.valid_count != nullptr) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*item.valid_count};
      ref.fetch_add(cuda::std::popcount(word_bits), cuda::std::memory_order_relaxed);
    }
  }

  // ---- offsets children: unrouted columns fully covered by the group loop's boundary
  // stage; routed columns by string_routed_metadata_kernel ----
  //
  // Every unrouted flat row's offsets value was published by its group's first warp —
  // adjacent lanes taking adjacent flat rows — from the staged boundaries in shared memory
  // (covering dead/empty rows and long/mixed groups alike), and every column's N+1 sentinel
  // rode the lane whose row ends its column (its row-0 sentinel is the row-0 publish
  // itself). No separate pass remains: deleting the old per-entry flat loop removed the
  // second full scratch traversal and its per-entry 64-bit div/mod.
}

// Row-parallel metadata for the columns routed to cub::DeviceMemcpy::Batched (the group walk
// above does no metadata work for them). One block owns ONE column's BlockSize-row tile per
// iteration; the scan has already written the final exclusive prefix into `dst_scratch`, so
// the kernel only transports it:
//   offsets — every live lane narrows one scratch entry (row or the N+1 sentinel) into the
//             child's width directly from global memory. Tiles never straddle a column
//             boundary (the loop is indexed per column tile), so the tile holding the last
//             row has the unique N+1 writer.
//   masks   — each warp ballots its 32 gathered validities into one word and writes it; the
//             ballot runs unconditionally so every warp lane participates, and warps fully
//             past the column's words store nothing (their ballot is all-zero anyway). No
//             partial OR into uninitialized storage.
//   counts  — each warp pops its word into shared memory, the CTA sums the block total over
//             one barrier, and thread 0 adds it to the column's atomic counter once per
//             block-tile (replacing the per-word device atomics of the generic mask phase).
// A second barrier closes each iteration so the next tile's shared-count writes cannot race
// the just-finished reduction. Offsets publication needs no barrier: the stores are
// independent global writes, stream-ordered before any consumer of the child.
template <size_type BlockSize>
CUDF_KERNEL void string_routed_metadata_kernel(string_work_item const* routed_items,
                                               size_type num_routed,
                                               size_type num_rows,
                                               size_type num_source_rows,
                                               bool check_bounds,
                                               bool negative_allowed,
                                               input_indexalator gather_map)
{
  static_assert(BlockSize % warp_size == 0);
  auto const lane            = static_cast<size_type>(threadIdx.x % warp_size);
  auto const warp            = static_cast<size_type>(threadIdx.x / warp_size);
  auto const warps_per_block = static_cast<size_type>(BlockSize / warp_size);
  __shared__ size_type warp_counts[BlockSize / warp_size];

  auto const tiles_per_col = static_cast<int64_t>(
    util::div_rounding_up_safe(static_cast<int64_t>(num_rows), static_cast<int64_t>(BlockSize)));
  auto const total_tiles = static_cast<int64_t>(num_routed) * tiles_per_col;
  for (int64_t t = static_cast<int64_t>(blockIdx.x); t < total_tiles;
       t += static_cast<int64_t>(gridDim.x)) {
    auto const col       = static_cast<size_type>(t / tiles_per_col);
    auto const col_row0  = (t % tiles_per_col) * static_cast<int64_t>(BlockSize);
    auto const tile_rows = static_cast<int64_t>(
      cuda::std::min<int64_t>(BlockSize, static_cast<int64_t>(num_rows) - col_row0));
    auto const& item = routed_items[col];
    auto const row   = col_row0 + static_cast<int64_t>(threadIdx.x);
    auto const live  = static_cast<int64_t>(threadIdx.x) < tile_rows;

    // offsets: one narrowed store per lane; the lane holding the last row owns the sentinel.
    if (live) {
      *(item.dst_offsets + row) = item.dst_scratch[row];
      if (row + 1 == num_rows) { *(item.dst_offsets + num_rows) = item.dst_scratch[num_rows]; }
    }

    // masks + counts: warp ballots then one CTA reduction. A warp fully past the tile
    // contributes no mask bits and a zero count (its dead lanes all vote false).
    auto const word_row0 = col_row0 + static_cast<int64_t>(warp) * warp_size;
    auto const src       = live ? resolve_gathered_row(gather_map,
                                                 static_cast<size_type>(row),
                                                 num_source_rows,
                                                 check_bounds,
                                                 negative_allowed)
                                : int64_t{-1};
    // Dead rows (src < 0) are invalid — the guard must precede the mask read or the bit
    // index would underflow on NULLIFY calls.
    auto const valid =
      src >= 0 and (item.src_mask == nullptr or
                    bit_is_set(item.src_mask, static_cast<size_type>(item.mask_bit_offset + src)));
    auto const word_bits = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) {
      // Dead-lane ballot bits are already 0, so the packed word is exact for tail words too
      // (rows past num_rows stay clear).
      if (item.dst_mask != nullptr and word_row0 < static_cast<int64_t>(num_rows)) {
        item.dst_mask[static_cast<size_type>(word_row0 / warp_size)] = word_bits;
      }
      warp_counts[warp] = cuda::std::popcount(word_bits);
    }
    __syncthreads();
    size_type block_valid = 0;
    for (size_type w = 0; w < warps_per_block; ++w) {
      block_valid += warp_counts[w];
    }
    if (threadIdx.x == 0 and item.valid_count != nullptr) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*item.valid_count};
      ref.fetch_add(block_valid, cuda::std::memory_order_relaxed);
    }
    __syncthreads();  // the next tile rewrites warp_counts; the reads above must finish first
  }
}

// Enqueues the map-locality sample copies on the stream (one batched D2H into pinned staging —
// no kernel launch, no sync). Returns a handle with empty staging for unclassifiable maps.
inline sampled_map_locality enqueue_map_locality_sample(column_view const& gather_map,
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
inline size_type resolve_map_locality_tile_cols(sampled_map_locality const& sampled,
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
  // A window is local when its values span at most one window length; classify coalesced only
  // if EVERY sampled window is local (ascending/reversed maps have spread kSampleWindowLen - 1).
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
    if (max_value - min_value > static_cast<int64_t>(kSampleWindowLen)) { return kRandomTileCols; }
  }
  return kCoalescedTileCols;
}

// Top-level string columns on the N3 fast path: one record per column, held
// so the launch block can finish their destinations (allocate chars from the
// totals, build the offsets children) after the virtual scan's readback. A
// record owns every device allocation its destination needs; `dst_slot`
// points into `destination`, which is never resized after construction.
struct staged_string {
  column_view src;                  // full source view (offset applied by the offsets iterator)
  std::unique_ptr<column> offsets;  // destination offsets child (INT32 or INT64, allocated late)
  rmm::device_buffer chars;         // destination chars buffer (allocated late)
  std::optional<cuda::device_buffer<bitmask_type>> dst_mask;  // when allocated
  size_type null_count{};  // host-visible count; 0 placeholder when kernel-masked
  std::unique_ptr<column>* dst_slot{};
  // True when the payload kernel writes this column's mask: the kernel then
  // needs the mask pointer and the per-column atomic counter, and assembly
  // takes the count from string_counts (never kernel_masked_columns — the
  // descriptor readback's slots must stay descriptor-only).
  bool kernel_mask_written{false};
};

// These carve-layout helpers sit at `cudf::detail` scope, NOT the anonymous namespace:
// their concept-conformance members are odr-used only inside type-erased vtables built in
// librmm, and at internal linkage nvcc flags them "#177-D declared but never referenced"
// (same reason `column_descriptor` sits at this scope).

// Byte needs and slice positions of one destination strings column: the offsets child
// (whose width `create_offsets_child_column` selects — 4 or 8 bytes per element) plus the
// chars buffer. Every carve requests its bytes rounded up to the carve granularity and
// starts 256-aligned, so the running total tracks the bump allocator's position exactly and
// each slice start is 256-aligned. The offsets child and chars slice of one column must be
// carved in THAT order, and columns in `live` order, matching `carve_strings_buffers`.
struct strings_destination_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  struct slice_offsets {
    std::size_t offsets;
    std::size_t chars;
  };

  // Reserves one column's slices: offsets child (INT32 or INT64 per `chars_bytes`'s
  // threshold, mirroring `create_offsets_child_column`'s width selection) then chars;
  // `chars_bytes == 0` still round-ups the chars region so the reservation mirrors the bump
  // sequence (a zero-byte chars buffer is still constructed).
  slice_offsets carve(int64_t chars_bytes, size_type num_rows)
  {
    auto const threshold     = cudf::strings::detail::get_offset64_threshold();
    auto const offsets_elem  = chars_bytes < threshold ? sizeof(int32_t) : sizeof(int64_t);
    auto const offsets_bytes = static_cast<std::size_t>(num_rows + 1) * offsets_elem;
    slice_offsets offsets;
    offsets.offsets = cudf::util::round_up_safe(total_bytes, slice_alignment);
    total_bytes     = offsets.offsets + offsets_bytes;
    offsets.chars   = cudf::util::round_up_safe(total_bytes, slice_alignment);
    total_bytes = cudf::util::round_up_safe(offsets.chars + static_cast<std::size_t>(chars_bytes),
                                            slice_alignment);
    return offsets;
  }
};

// Stream-ordered resource handing out successive aligned slices of ONE pre-allocated device
// buffer, so a call's destination columns (string offsets/chars through
// `strings_destination_layout`, fixed-width masks/data through `fixed_width_destination_layout`)
// draw from a single upstream allocate/free pair instead of one pair per buffer. Allocation
// bumps a pointer; deallocation is a no-op — the arena block is freed stream-ordered when the
// resource itself dies. Every destination column keeps its slices alive independently: each
// buffer stores a value copy of the `cuda::mr::shared_resource` wrapper created for the call,
// and all copies share the one payload (bump pointer included), so the block outlives the last
// column. The upstream reference is captured as an owning `any_resource` deep copy, so a
// payload that outlives the call still deallocates through a live resource object, matching
// how stock columns capture their allocation resource. Because allocation never frees, every
// slice must be carved exactly once, in the reserving layout's order (strings: offsets child,
// then chars, per column, columns in live order; fixed-width: mask before data per column,
// struct node mask before its children, in `fused_indices` pre-order — see
// `fixed_width_destination_layout`).
class carve_memory_resource {
 public:
  static constexpr std::size_t slice_alignment{strings_destination_layout::slice_alignment};

  carve_memory_resource(rmm::device_async_resource_ref upstream,
                        cuda::stream_ref stream,
                        std::size_t total_bytes)
    : _upstream{upstream},  // deep copy: the payload may outlive the call's resource refs
      _stream{stream},
      _size{total_bytes},
      _arena{total_bytes > 0 ? _upstream.allocate(stream, total_bytes, slice_alignment) : nullptr}
  {
  }

  // A zero-sized carve never allocated (`_arena == nullptr`); deallocating it would still
  // trip instrumented adaptors' counters (e.g. statistics_resource_adaptor), so skip it.
  ~carve_memory_resource()
  {
    if (_size > 0) { _upstream.deallocate(_stream, _arena, _size, slice_alignment); }
  }

  carve_memory_resource(carve_memory_resource const&)            = delete;
  carve_memory_resource& operator=(carve_memory_resource const&) = delete;

  bool operator==(carve_memory_resource const& other) const noexcept
  {
    return this == std::addressof(other);
  }
  bool operator!=(carve_memory_resource const& other) const noexcept { return !(*this == other); }

  // Tag introspection only: CCCL queries this property in unevaluated concept checks, so the
  // function body never runs (precedent: `host_memory.cpp`'s pool resource).
  [[maybe_unused]] friend void get_property(carve_memory_resource const&,
                                            cuda::mr::device_accessible) noexcept
  {
  }

  // Buffer constructions request the full carve granularity (via the allocation-alignment
  // property or the explicit alignment argument), so the bump sequence matches the layout
  // reservation exactly and every slice keeps 256-byte alignment for its widest element.
  [[nodiscard]] void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    CUDF_EXPECTS(alignment <= slice_alignment, "Alignment exceeds the carve granularity");
    auto const offset = cudf::util::round_up_safe(_used, alignment);
    CUDF_EXPECTS(offset + bytes <= _size, "Destination arena exhausted");
    _used = offset + bytes;
    return static_cast<std::byte*>(_arena) + offset;
  }

  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}

  [[nodiscard]] void* allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) noexcept {}

  // The arena's base pointer, for a call-site-wide fill over the whole block (the preset-mask
  // bands' single memset — padding gaps inside the block are filled too, which no column
  // observes). Null on a zero-sized carve (nothing was allocated); callers guard on size.
  void* arena_base() const noexcept { return _arena; }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> _upstream;  // owns a deep copy
  cuda::stream_ref _stream;  // the arena block's stream-ordered free stream
  std::size_t _size;
  void* _arena;
  std::size_t _used{0};
};

// The buffers below type-erase through `cuda::mr::resource_with` — the concept surface is the
// contract, checked here so an interface drift fails at compile time, not at the erase point.
static_assert(cuda::mr::resource_with<carve_memory_resource, cuda::mr::device_accessible>,
              "carve_memory_resource must satisfy the stream-ordered device resource concepts");

// Byte needs of one destination fixed-width column: the Arrow-padded mask words (the sizing
// `create_null_mask` itself uses) plus the data payload. The reservation pass and the carve
// sites MUST visit columns in the same pre-order (top-level columns in `fused_indices` order;
// within a column: mask before data, struct node mask before its children recurse, dictionary
// mask before its indices) and consume the same round-up steps, so the summed total equals the
// bump pointer's final position exactly — a divergence trips the carve resource's loud
// `arena exhausted` expect rather than corrupting memory.
struct fixed_width_destination_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  // Reserves one column's mask region (`num_rows` words at the Arrow 64-word padding, zero
  // when the column carries no mask) and data region (`num_rows` fixed-width elements); each
  // region is rounded up to the carve granularity, mirroring the bump steps the constructions
  // will take.
  void carve(size_type num_rows, std::size_t data_bytes, bool with_mask)
  {
    auto const mask_words =
      with_mask ? cudf::bitmask_allocation_size_words(num_rows) : std::size_t{0};
    total_bytes =
      cudf::util::round_up_safe(total_bytes + mask_words * sizeof(bitmask_type), slice_alignment);
    total_bytes = cudf::util::round_up_safe(total_bytes + data_bytes, slice_alignment);
  }
};

// Preset-mask fill reservation for ONE of the two fill patterns (ALL_VALID 0xff / ALL_NULL
// 0x00). During the build walk each preset destination mask is instead carved from a pattern's
// staging arena; after the walk ONE memset per non-empty arena covers every preset fill in that
// pattern, replacing the per-mask cudaMemsetAsync submissions (the measured ~60-mask drip-feed
// class). The staging arenas are created alongside the fw carve arena — slices outside it, so
// the fw reservation mirror above is untouched. The slice sizing (`dst_mask_plan::mask_bytes`)
// MUST track `fixed_width_destination_layout::carve`'s mask sizing exactly
// (`bitmask_allocation_size_words`) so the fills stay byte-identical to per-mask memsets,
// padding words included.
struct preset_mask_fill_layout {
  static constexpr std::size_t slice_alignment{256};  // CUDA's allocation granularity

  std::size_t total_bytes{0};

  // Reserves one preset mask's padded word extent. `num_rows` is the DESTINATION row count
  // (sliced sources included), matching the carve sites' sizing.
  void carve(size_type num_rows)
  {
    total_bytes = cudf::util::round_up_safe(
      total_bytes + cudf::bitmask_allocation_size_words(num_rows) * sizeof(bitmask_type),
      slice_alignment);
  }
};

// Builds ONE destination strings column's carved buffers: the offsets child (a full column
// whose data buffer is a carve slice) and the chars buffer (a carve slice). Both store a value
// copy of the shared carve resource, so the column keeps its slices alive independently and
// the arena block is freed stream-ordered when the last shared copy dies. The offsets child is
// carved FIRST (matching `strings_destination_layout::carve`'s order), then chars. A zero-byte
// chars buffer constructs to a null data pointer exactly as stock does (RMM skips zero-size
// allocations) — pass `chars_bytes == 0` and `chars` is the empty-chars form. The destination
// mask is NOT carved: string masks keep the stock regime (allocated in `build_string`), since
// the strings carve arena only exists after the scan readback that sizes the chars.
struct carved_strings_buffers {
  std::unique_ptr<column> offsets;  // offsets child column, data carved (INT32 or INT64)
  rmm::device_buffer chars;         // chars slice; empty (null data) when chars_bytes == 0
};

inline carved_strings_buffers carve_strings_buffers(
  size_type num_rows,
  int64_t chars_bytes,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref output_mr)
{
  auto const threshold = cudf::strings::detail::get_offset64_threshold();
  if (!cudf::strings::detail::is_large_strings_enabled()) {
    // Stock `create_offsets_child_column` parity: the same overflow contract.
    CUDF_EXPECTS(
      chars_bytes < threshold, "Size of output exceeds the column size limit", std::overflow_error);
  }
  auto const offsets_type =
    chars_bytes < threshold ? data_type{type_id::INT32} : data_type{type_id::INT64};
  auto const offsets_bytes = (static_cast<std::size_t>(num_rows) + 1) *
                             (chars_bytes < threshold ? sizeof(int32_t) : sizeof(int64_t));

  // Construction order = reservation order: offsets child first, then chars. Each buffer
  // stores a value copy of the one shared carve resource and asks it for its slice (the
  // resource bumps once per buffer at the full carve granularity), so the carve sequence
  // matches the reservation exactly. `rmm::device_buffer`'s alignment-argument form carries
  // the carve granularity; `column`'s generic ctor forwards it into `_data`, so the child
  // keeps genuine `rmm::device_buffer` ownership of its slice.
  auto offsets_col = std::make_unique<column>(
    offsets_type,
    static_cast<size_type>(num_rows) + 1,
    rmm::device_buffer{offsets_bytes, carve_memory_resource::slice_alignment, stream, carve_mr},
    cudf::detail::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, output_mr),
    0,
    std::vector<std::unique_ptr<column>>{});
  auto chars = rmm::device_buffer{static_cast<std::size_t>(chars_bytes),
                                  carve_memory_resource::slice_alignment,
                                  stream,
                                  carve_mr};

  return carved_strings_buffers{std::move(offsets_col), std::move(chars)};
}

// One device staging upload drawn from the staging carve arena: a fresh `rmm::device_uvector`
// allocation against the arena's bump allocator (its deallocation is a no-op — the single
// upstream pair frees stream-ordered when the arena dies). Sizing the arena is a closed-form
// computation at worst case, so the caller's only obligation is that the reservation covers
// every carve it makes; any shortfall trips the carve resource's loud `arena exhausted` expect.
template <typename T>
inline rmm::device_uvector<std::remove_cv_t<T>> carve_device_vector(
  host_span<T> source_data,
  cuda::stream_ref stream,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr)
{
  rmm::device_uvector<std::remove_cv_t<T>> ret(source_data.size(), stream, carve_mr);
  if (source_data.size() > 0) {
    cudf::detail::cuda_memcpy_async<std::remove_cv_t<T>>(ret, source_data, stream);
  }
  return ret;
}

// Destination mask plan shared by every node on the descriptor path. A destination mask is
// needed iff the source is all-null (ALL_NULL pre-set), any gathered row can be null (NULLIFY
// policy or a null-bearing source — the fused kernel then writes it and reports the valid
// count), or the source is nullable with zero nulls (ALL_VALID pre-set). The plan carries the
// state and fill semantics only; the mask BUFFER is created by the caller through whichever
// allocation regime the column rides (the carve arenas or stock `create_null_mask`), so
// planning never allocates and the builder's carve reservations can be summed before the
// build walk starts. The policy inputs are host-side view accessors only — the unsliced null
// counts are cached values, never a device readback.
struct dst_mask_plan {
  mask_state state{mask_state::UNALLOCATED};
  // True when the mask must be memset here (ALL_VALID/ALL_NULL); kernel-written
  // (UNINITIALIZED) masks are left for the kernel, which writes every word.
  bool preset{};
  bitmask_type* kernel_mask{};  // non-null iff the fused kernel writes the mask
  size_type null_count{};       // host-visible count; 0 placeholder when kernel-masked
  bool skip_data{};             // all-null source: mask pre-set ALL_NULL, no descriptor pushed
  // Preset masks only: which pattern's staging arena the mask slices from (0 = ALL_VALID,
  // 1 = ALL_NULL) and its padded byte extent, so the carve consumes the pre-walk reservation
  // and the post-walk memset fills the matching pattern. `use_band` = the preset mask rides
  // the band; false (below `kMaskBandPresetGate`) takes the stock per-mask carve + memset.
  std::size_t band{};
  std::size_t mask_bytes{};
  bool use_band{};
};

// One preset-mask fill pattern's staging arena: a carve_memory_resource whose slices are
// filled by a SINGLE post-walk memset (pattern byte 0xff for ALL_VALID, 0x00 for ALL_NULL),
// plus the reservation that sized it. Index into the builder's two-element array with the
// pattern (see `dst_mask_plan::band`). The arena is created AFTER the builder's aggregate
// initialization (`shared_resource` has no default constructor), so — like the fw carve
// arena — it is owned by the enclosing scope: the member is a pointer null until the
// post-walk pass sets it; every use site runs after that point, and destination columns
// hold their own value copies so the wrapper's lifetime is not entangled with the builder's.
struct mask_band {
  preset_mask_fill_layout layout;
  std::unique_ptr<cuda::mr::shared_resource<carve_memory_resource>> mr;
};

// The exact condition under which `plan_destination_mask` hands the fused kernel a
// destination mask (and the builder then records a masked position). Host-side view
// accessors only — cached null counts, never a device readback — so the count estimator
// below evaluates it without planning (planning would allocate a device mask).
inline bool kernel_writes_dst_mask(column_view const& src, bool nullify)
{
  bool const all_null = src.size() > 0 and src.null_count() == src.size();
  return not all_null and (nullify or src.null_count() > 0);
}

// Exact descriptor/kernel-masked counts the build pass emits for one descriptor-path source,
// mirroring `build_struct`/`build_dictionary`/`emit_fixed_width`: a fixed-width leaf or a
// top-level dictionary emits one data descriptor, a struct node adds one MASK-ONLY descriptor
// iff the node's mask is kernel-written, and struct children recurse. Mirroring the recursion
// (rather than a closed formula) keeps the counts correct by construction when the builder's
// emission rules change; KEEP IN SYNC with the `plan.kernel_mask != nullptr` branch of
// `plan_destination_mask` — both must agree on when the kernel writes a mask.
struct descriptor_counts {
  size_type descriptors;
  size_type kernel_masked;
};

inline descriptor_counts estimate_descriptor_counts(column_view const& src, bool nullify)
{
  if (src.type().id() == type_id::STRING) { return {0, 0}; }  // strings ride the N3 fast path
  if (src.type().id() == type_id::STRUCT) {
    size_type descriptors = kernel_writes_dst_mask(src, nullify) ? 1 : 0;
    size_type masked      = descriptors;
    for (size_type i = 0; i < src.num_children(); ++i) {
      auto const child = estimate_descriptor_counts(src.child(i), nullify);
      descriptors += child.descriptors;
      masked += child.kernel_masked;
    }
    return {descriptors, masked};
  }
  // Fixed-width leaf or top-level dictionary: one data descriptor carrying the mask iff
  // kernel-written, none when the source is all-null (the builder takes the skip_data path).
  // (A dictionary never sits below a struct on this path: a DICTIONARY32 struct child fails
  // `is_fully_fixed_width_struct`.)
  if (src.size() > 0 and src.null_count() == src.size()) { return {0, 0}; }
  auto const masked = kernel_writes_dst_mask(src, nullify) ? 1 : 0;
  return {1, masked};
}

// The mask plan is evaluated against the DESTINATION row count (`num_rows`,
// i.e. `gather_map.size()`), not `src.size()`: a sliced source has
// `src.size() < num_rows`, so an all-null sliced source must still mark all
// `num_rows` destination rows null. Callers must therefore pass the call's
// `num_rows` even though only `null_count` consumes it.
inline dst_mask_plan plan_destination_mask(column_view const& src,
                                           bool nullify,
                                           size_type num_rows,
                                           bool bands_enabled)
{
  dst_mask_plan plan;
  bool const all_null = src.size() > 0 and src.null_count() == src.size();
  if (all_null) {
    // Every row is null regardless of the map: pre-set ALL_NULL.
    plan.state      = mask_state::ALL_NULL;
    plan.preset     = true;
    plan.null_count = num_rows;
    plan.skip_data  = true;
    plan.band       = 1;
  } else if (nullify or src.null_count() > 0) {
    // The kernel writes this mask; the caller assigns `kernel_mask` from whichever mask
    // buffer it constructs (carved or stock).
    plan.state = mask_state::UNINITIALIZED;
  } else if (src.nullable()) {
    // Nullable source with zero nulls: ALL_VALID fast path, no kernel mask work.
    plan.state  = mask_state::ALL_VALID;
    plan.preset = true;
  }
  if (plan.preset) {
    plan.mask_bytes = cudf::bitmask_allocation_size_words(num_rows) * sizeof(bitmask_type);
    plan.use_band   = bands_enabled;
  }
  return plan;
}

// Applies `plan`'s fill semantics to a freshly created destination mask. A preset mask fills
// its padded extent (`plan.mask_bytes`) with its pattern byte: banded — carved from its
// pattern's staging arena, filled by the post-walk band memset — when the caller took the band
// path (`bands_enabled`), else the per-mask memset (the stock form). A kernel-written
// UNINITIALIZED mask is left untouched; UNALLOCATED must never reach this helper. The caller
// assigns `plan.kernel_mask` from the buffer before pushing any descriptor.
inline void fill_destination_mask(dst_mask_plan& plan,
                                  cuda::device_buffer<bitmask_type>& mask,
                                  bool bands_enabled,
                                  cuda::stream_ref stream)
{
  if (plan.state == mask_state::UNALLOCATED) { return; }
  plan.kernel_mask = plan.state == mask_state::UNINITIALIZED ? mask.data() : nullptr;
  if (not plan.preset) { return; }
  if (bands_enabled) { return; }  // the post-walk band memset fills it
  CUDF_CUDA_TRY(cudaMemsetAsync(
    mask.data(), plan.state == mask_state::ALL_VALID ? 0xff : 0x00, plan.mask_bytes, stream.get()));
}

// Builds ONE destination fixed-width column whose mask and data buffers are slices of the
// call-wide carve arena. The mask is created FIRST (matching `fixed_width_destination_layout::
// carve`'s mask-then-data order), then the data slice is carved; a banded preset mask slices
// the matching pattern's staging arena (its fill rides the post-walk band memset), a
// stock-path preset mask carves the fw arena (filled here per-mask), and a kernel-written mask
// slices the fw arena untouched. `plan.kernel_mask` is assigned from the mask buffer so the
// caller can push its descriptor with it. Every slice stores a value copy of the shared
// resource it carves from, so the column keeps its slices alive independently and an arena
// block is freed stream-ordered when the last shared copy dies. A zero-size data slice
// (num_rows == 0) constructs to a null data pointer exactly as stock does (RMM skips zero-size
// allocations).
inline std::unique_ptr<column> make_carved_fixed_width(
  column_view const& src,
  size_type num_rows,
  dst_mask_plan& plan,
  cuda::mr::shared_resource<carve_memory_resource> const& carve_mr,
  mask_band* bands,
  bool bands_enabled,
  cuda::stream_ref stream)
{
  auto const data_bytes = static_cast<std::size_t>(num_rows) * cudf::size_of(src.type());

  // Construction order = reservation order: mask first, then data. The env property pins the
  // mask buffer's allocation alignment to the carve granularity (the default would be the
  // element's natural 4-byte alignment); `rmm::device_buffer`'s alignment-argument form carries
  // it for the data slice, and `column`'s generic ctor forwards both into their owned members.
  auto const mask_env =
    cuda::std::execution::prop{cuda::allocation_alignment, carve_memory_resource::slice_alignment};
  auto mask_slice =
    plan.use_band
      ? cuda::device_buffer<bitmask_type>{stream,
                                          *bands[plan.band].mr,
                                          plan.mask_bytes / sizeof(bitmask_type),
                                          cuda::no_init,
                                          mask_env}
      : cuda::device_buffer<bitmask_type>{
          stream,
          carve_mr,
          plan.state == mask_state::UNALLOCATED ? 0 : cudf::bitmask_allocation_size_words(num_rows),
          cuda::no_init,
          mask_env};
  fill_destination_mask(plan, mask_slice, bands_enabled, stream);
  auto data_slice =
    rmm::device_buffer{data_bytes, carve_memory_resource::slice_alignment, stream, carve_mr};

  auto dst = std::make_unique<column>(src.type(),
                                      num_rows,
                                      std::move(data_slice),
                                      std::move(mask_slice),
                                      plan.state == mask_state::ALL_NULL ? num_rows : 0,
                                      std::vector<std::unique_ptr<column>>{});
  return dst;
}

// Completes the deferred string columns (the N3 fast path): the virtual keyed exclusive scan
// over all string columns' gathered sizes, one batched totals readback, the batched payload
// kernel, then per-column assembly of the destination strings columns. Defined in
// batch_gather_strings.cu; the launch block calls it with the builder's staged state moved in
// and consumed (`staged_strings` is emptied before the call returns). The routed tile width
// is resolved under the totals synchronization when `explicit_tile_cols` arrives open.
void assemble_strings_impl(std::vector<staged_string>& staged_strings,
                           size_type num_rows,
                           column_view const& gather_map,
                           size_type num_source_rows,
                           bool check_bounds,
                           bool negative_allowed,
                           sampled_map_locality const& sampled,
                           std::optional<size_type>& explicit_tile_cols,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref output_mr,
                           rmm::device_async_resource_ref temp_mr);

}  // namespace detail
}  // namespace cudf
