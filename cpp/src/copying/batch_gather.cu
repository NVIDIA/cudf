/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "batch_gather_common.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
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
#include <cudf/null_mask.hpp>
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

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/bit>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudf {
namespace detail {

namespace {

// A struct column joins the descriptor path when every descendant is a fixed-width leaf; any
// list child at any depth keeps the WHOLE column in the fallback (partial flattening is a
// later stage). A STRING child is no longer disqualifying: top-level strings take the
// string fast path below, and a string inside a struct takes the same fast path via the
// string_descriptor machinery.
bool is_fully_fixed_width_struct(column_view const& col)
{
  for (size_type i = 0; i < col.num_children(); ++i) {
    auto const& child = col.child(i);
    if (child.type().id() == type_id::STRUCT) {
      if (not is_fully_fixed_width_struct(child)) { return false; }
    } else if (not cudf::is_fixed_width(child.type())) {
      return false;
    }
  }
  return true;
}

// Descriptor-path eligibility beyond plain fixed-width. Dictionaries qualify when their index
// child is a signed index type — the adopting factory's contract, which every public factory
// enforces; the child-count/signedness guard only deflects exotic hand-built columns to the
// stock path (which casts unsigned indices through the adopting factory).
bool is_descriptor_path_candidate(column_view const& col)
{
  if (cudf::is_fixed_width(col.type())) { return true; }
  switch (col.type().id()) {
    case type_id::STRUCT: return is_fully_fixed_width_struct(col);
    case type_id::STRING: return true;
    case type_id::DICTIONARY32: {
      // Unsigned index types defer to the stock gatherer, which recasts them signed. No public
      // factory can produce an unsigned-index dictionary — `dictionary::encode` and the
      // adopting factories either require signed indices or recast them signed — so this
      // deferral has no constructible fixture; do not re-add a test case for it.
      if (col.num_children() != 2) { return false; }
      auto const& indices = col.child(0);
      return cudf::is_index_type(indices.type()) and cudf::is_signed(indices.type());
    }
    default: return false;
  }
}

// Host-side count of what the descriptor path produces for one fused column: fixed-width
// columns contribute one data descriptor, fully-fixed-width structs contribute one descriptor
// per fixed-width leaf plus one mask-only descriptor per struct node that may need one, and
// dictionaries contribute one data descriptor over their indices. Mirrors the build walk in
// descriptor_path_builder; the kernel's num_descriptors is this same count.
size_type fused_column_descriptor_count(column_view const& col)
{
  if (col.type().id() == type_id::STRING) { return 0; }  // strings ride the N3 fast path
  if (col.type().id() == type_id::STRUCT) {
    size_type count = 1;  // mask-only descriptor for this node
    for (size_type i = 0; i < col.num_children(); ++i) {
      auto const& child = col.child(i);
      count += (child.type().id() == type_id::STRUCT) ? fused_column_descriptor_count(child)
                                                      : size_type{1};
    }
    return count;
  }
  return 1;  // fixed-width leaf or dictionary index payload
}

size_type fused_descriptor_count(table_view const& source_table)
{
  size_type count = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    if (is_descriptor_path_candidate(source_table.column(i))) {
      count += fused_column_descriptor_count(source_table.column(i));
    }
  }
  return count;
}

// Largest element size among the descriptor path's DATA descriptors; mask-only struct nodes
// carry no payload (element_size 0), so they never raise the working-set footprint.
size_type max_fused_data_element_size(column_view const& col)
{
  size_type max_size = 0;
  if (not is_descriptor_path_candidate(col) or col.type().id() == type_id::STRING) {
    return max_size;  // strings carry no fixed-width descriptor payload
  }
  if (col.type().id() == type_id::STRUCT) {
    for (size_type c = 0; c < col.num_children(); ++c) {
      max_size = std::max(max_size, max_fused_data_element_size(col.child(c)));
    }
  } else {
    // Fixed-width leaf or dictionary: data rides the leaf/indices element size.
    auto const& data_col = (col.type().id() == type_id::DICTIONARY32)
                             ? col.child(0)  // annotated indices; signed per the candidate check
                             : col;
    max_size             = static_cast<size_type>(cudf::size_of(data_col.type()));
  }
  return max_size;
}

size_type max_fused_data_element_size(table_view const& source_table)
{
  size_type max_size = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    max_size = std::max(max_size, max_fused_data_element_size(source_table.column(i)));
  }
  return max_size;
}

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

// ---------------------------------------------------------------------------
// String fast path (N3).
//
// Stock per-string-column cost (`strings::detail::gather`): a sizes scan that
// synchronizes on its last-element readback once PER COLUMN, then a chars
// kernel, then whole-table mask work — multiplied by the column count. This
// path batches every string column of the call:
//
//   1. ONE virtual keyed exclusive scan (`thrust::exclusive_scan_by_key`, the
//      `contiguous_split.cu` pattern) over the flat (column, row) domain.
//      Every gathered size is computed on the fly from the map, the source
//      mask, and the source offsets — the sizes are never materialized — and
//      the scan writes each destination offsets value into a flat int64
//      scratch. The scratch precedes the offsets children because a gathered
//      column's byte total is unbounded by its source (the map may repeat
//      rows), so the child widths are unknowable until the totals are read.
//      The scan's tabulate output also records each column's total.
//   2. ONE host totals readback: a pinned async D2H enqueued right after the
//      scan and synchronized just before the payload launch — the sampler's
//      enqueue-early/resolve-late pattern, and the only synchronization this
//      path adds. The totals size each column's chars buffer and select its
//      offsets child width (the same selection `create_offsets_child_column`
//      makes, including the column size limit), and every live column's
//      offsets child + chars buffer is carved from ONE call-wide arena
//      (`carve_memory_resource`): a single upstream allocate/free pair serves
//      all destination string allocations, while each column keeps owning its
//      slices through the shared resource copy.
//   3. ONE batched payload kernel for ALL string columns, phased over
//      disjoint outputs (no ordering between phases): characters with
//      warp-cooperative uint4 spans per row, except rows above the block-copy
//      gate which the whole block copies,
//      destination masks written in-kernel for kernel-masked columns
//      (constant masks are pre-set by `plan_destination_mask`), the offsets
//      children finalized from the scratch (narrowing to the child's width),
//      and per-column valid counts accumulated atomically for the
//      post-kernel readback. On tall calls (>= `kBatchMemcpyRowThreshold`)
//      each all-short column's chars runs through one
//      `cub::DeviceMemcpy::Batched` over the routed columns' rows (per-column
//      routing via `chars_via_batched_memcpy`); the routed columns also leave
//      the kernel entirely — their offsets/masks/counts are written by
//      `string_routed_metadata_kernel` — and the launch grid covers only the
//      unrouted cohort's rows.
//
// Trap compliance: chars are addressed from the unsliced base with the sliced
// offsets (the offsets iterator carries the view offset); null rows have zero
// bytes and distinct validity from empty strings; gathered sizes are scanned
// in 64 bits.
// ---------------------------------------------------------------------------

// The batched string payload kernel `string_payload_kernel` lives in `batch_gather_common.cuh`
// (only `batch_gather_strings.cu` launches it).

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
  // Routed entry: one host-side type pass gathers the flattened descriptor count (what the
  // kernel will actually receive — struct leaves and mask-only nodes, dictionary indices) and
  // the widest fused data element for the gate; when sampling is warranted it is ENQUEUED here
  // and RESOLVED inside the body just before the kernel launch, hiding its latency behind the
  // allocation work.
  size_type const num_fused_cols      = fused_descriptor_count(source_table);
  size_type const max_fused_elem_size = max_fused_data_element_size(source_table);
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

// Plans the descriptor-path columns of one gather call and builds their destination columns.
//
// Structs are flattened recursively: every fixed-width leaf under a row-preserving struct chain
// becomes a data descriptor carrying the leaf's OWN gathered mask, and every struct node
// becomes a MASK-ONLY descriptor (skip_data, null data pointers) carrying the node's OWN
// gathered mask — stock never ANDs ancestor validity into descendants, so each node keeps its
// own mask and the output hierarchy is reconstructed with `create_structs_hierarchy`, which
// (unlike `make_structs_column`) does not superimpose parent nulls into children. Dictionary
// columns contribute one data descriptor over their annotated fixed-width indices (source
// address = indices base + parent offset, source mask = the dictionary parent's own mask) and
// are assembled with the adopting `make_dictionary_column` over a deep-copied key catalog:
// full key order, keys unreferenced by the gathered indices retained, no dedup/remap/re-encode.
//
// An all-null source (leaf, struct node, or dictionary parent) still builds its destination
// column (ALL_NULL mask, undefined payload) but emits NO descriptor: the kernel's per-row map
// load and source-mask read would be dead work. A struct whose children are all skipped keeps
// its own MASK-ONLY descriptor as the only remaining work; when nothing is left for the launch
// the caller's `descriptors.empty()` guard skips it.
//
// Offset invariant: for a source path of views v1(top)...vk(leaf), each view's `offset()` is
// applied exactly once — a leaf adds the accumulated ancestor offsets to `data<T>()` (which
// already carries the leaf view's own offset, zero for raw children) and to its mask bit
// offset; a node's own mask descriptor uses the accumulated offset of every view down to and
// including the node. Raw struct children and dictionary index children always view their full
// underlying extent (offset 0), matching `get_sliced_child`/`get_indices_annotated` without
// paying their null-count readback.

namespace {

struct descriptor_path_builder {
  std::vector<column_descriptor>& descriptors;
  std::vector<size_type>& kernel_masked_positions;  // positions of kernel-masked descriptors
  std::vector<column*>& kernel_masked_columns;      // slot i's count target (columns[i])
  size_type num_rows;
  bool nullify;
  // The call-wide band-path decision (`kMaskBandPresetGate`): banded preset-mask fills vs the
  // stock per-mask memsets. Mirrored into every `dst_mask_plan` the builder plans.
  bool bands_enabled;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref output_mr;
  rmm::device_async_resource_ref temp_mr;
  fixed_width_destination_layout fw_layout;  // reservation, advanced during the build walk
  // One staging arena per preset-mask fill pattern (index 0 = ALL_VALID 0xff, 1 = ALL_NULL
  // 0x00). Slices are handed out during the build walk; each arena's layout was reserved by
  // the pre-walk pass and ONE memset per non-empty arena (issued post-walk, before the launch)
  // fills every preset mask in that pattern.
  mask_band mask_bands[2];
  // Set once the fw arena exists (after the build walk); the fw carve class checks it so a
  // misordered construction that would carve after teardown fails loudly.
  cuda::mr::shared_resource<carve_memory_resource>* fw_carve_mr{};

  // String-keyed dictionary columns defer their whole assembly to `assemble_staged_keys`:
  // one batched D2H copy of every catalog's offsets plus a single synchronization replaces
  // copy_slice's per-column blocking offset reads (two per string-keyed catalog). A staged
  // record holds everything the deferred factory call needs; the destination slot is a pointer
  // into `destination`, which batch_gather_impl never resizes after construction, so the
  // pointers stay valid until the assembly fills them (the builder outlives the assembly).
  struct staged_keys {
    column_view keys;                 // full keys view, including the parent's offset
    std::unique_ptr<column> offsets;  // deep copy of the full offsets child, unadjusted
    std::unique_ptr<column> indices;  // gathered indices; moves into the deferred factory call
    // The parent's destination mask, created and filled in `build_dictionary` (its pointer
    // already rides the pushed descriptor) and moved through here into the deferred factory
    // call. An empty buffer means the plan was UNALLOCATED.
    std::optional<cuda::device_buffer<bitmask_type>> dst_mask;  // when allocated
    size_type null_count{};  // host-visible parent null count (0 placeholder when kernel-masked)
    size_type keys_offset;   // keys view's own offset
    size_type keys_size;     // keys view's own size
    std::unique_ptr<column>* dst_slot{};  // destination[index] slot to fill
    // Position in kernel_masked_columns when the parent mask is kernel-written; the target
    // pointer is patched to the final column after the deferred assembly.
    std::optional<size_t> kernel_slot;
  };
  std::vector<staged_keys> pending_dictionaries;
  std::vector<staged_string> pending_strings;

  size_type push_descriptor(column_descriptor const& desc)
  {
    auto const position = static_cast<size_type>(descriptors.size());
    descriptors.push_back(desc);
    return position;
  }

  // Marks the descriptor at `position` as kernel-masked and returns the target-patch token.
  // Compact `valid_counts` slots are assigned after the build loop once the caller has
  // stable-partitioned the descriptors masked-first, so push order must stay natural. The
  // target column may arrive later (the owning factory owns it first); `set_target` patches
  // the slot after the column is assembled.
  size_t record_kernel_masked(size_type position)
  {
    kernel_masked_positions.push_back(position);
    kernel_masked_columns.push_back(nullptr);
    return kernel_masked_columns.size() - 1;
  }

  void set_target(size_t slot, column* target) { kernel_masked_columns[slot] = target; }
  // Batches the per-catalog offsets reads into ONE D2H staged copy plus ONE synchronization,
  // then assembles every deferred string-keyed dictionary column from the staged host offsets
  // and fills its destination slot. Called once per build walk, before the fused launch;
  // `output_mr` owns the produced columns' buffers.
  void assemble_staged_keys()
  {
    if (pending_dictionaries.empty()) { return; }

    // One pinned staging of every catalog's offsets child, copied in a single batched D2H
    // and read only after the single synchronization below. Catalog i's elements live at
    // `spans[i]` in the flat staging; per-catalog width is recorded because the offsets
    // width follows the chars size (INT64 only past the offset64 threshold), so a mixed
    // batch is legal. Spans are width-aligned so an 8-byte catalog is never read
    // misaligned when a narrower one lands at an odd 4-byte multiple.
    std::vector<size_t> spans;
    std::vector<bool> wide;
    spans.reserve(pending_dictionaries.size());
    wide.reserve(pending_dictionaries.size());
    size_t total_bytes = 0;
    for (auto const& staged : pending_dictionaries) {
      bool const is_wide = staged.offsets->type().id() == type_id::INT64;
      wide.push_back(is_wide);
      total_bytes = cudf::util::round_up_safe(total_bytes, sizeof(int64_t));
      spans.push_back(total_bytes);
      total_bytes +=
        (static_cast<size_t>(staged.keys_size) + 1) * (is_wide ? sizeof(int64_t) : sizeof(int32_t));
    }

    auto staging = cudf::detail::make_pinned_vector_async<std::byte>(total_bytes, stream);
    for (size_t i = 0; i < pending_dictionaries.size(); ++i) {
      auto const& staged = pending_dictionaries[i];
      auto const bytes   = wide[i] ? sizeof(int64_t) : sizeof(int32_t);
      auto const count   = static_cast<size_t>(staged.keys_size) + 1;
      auto const* src    = staged.offsets->view().head<std::byte>() +
                        static_cast<std::size_t>(staged.keys_offset) * bytes;
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(staging.data() + spans[i], src, count * bytes, stream));
    }
    cudf::detail::sync_stream(stream);
    auto const staged_bytes = staging.data();
    // Reads the staged copy at the span-local element index (index 0 is the catalog's
    // element at its own `keys_offset`); absolute child indices never index the staging.
    auto const read_offset = [&](size_t catalog, size_type index) -> int64_t {
      auto const bytes = wide[catalog] ? sizeof(int64_t) : sizeof(int32_t);
      auto const* p    = staged_bytes + spans[catalog] + static_cast<std::size_t>(index) * bytes;
      return wide[catalog] ? static_cast<int64_t>(*(reinterpret_cast<int64_t const*>(p)))
                           : static_cast<int64_t>(*(reinterpret_cast<int32_t const*>(p)));
    };

    for (size_t i = 0; i < pending_dictionaries.size(); ++i) {
      auto& staged            = pending_dictionaries[i];
      auto const chars_offset = staged.keys_offset == 0 ? 0L : read_offset(i, 0);
      auto const keys_bytes   = read_offset(i, staged.keys_size) - chars_offset;

      // The staged offsets copy holds the full underlying child; the copy needs an OWNED
      // column of exactly keys_size + 1 elements starting at keys_offset, rebased so element
      // 0 is zero — the same deep-copy-and-rebase copy_slice performs. `column(column_view)`
      // materializes the slice (honoring the view's offset), then the transform shifts it.
      auto out_offsets = std::make_unique<column>(
        cudf::detail::slice(staged.offsets->view(),
                            {staged.keys_offset, staged.keys_offset + staged.keys_size + 1},
                            stream)
          .front(),
        stream,
        output_mr);
      if (chars_offset > 0) {
        // Rebasing shifts every staged offset down by `chars_offset` so the copy's own
        // offsets start at zero; element 0 (the original left bound) becomes 0 exactly.
        auto const in =
          cudf::detail::offsetalator_factory::make_input_iterator(out_offsets->view(), 0);
        auto const out =
          cudf::detail::offsetalator_factory::make_output_iterator(out_offsets->mutable_view());
        thrust::transform(
          rmm::exec_policy_nosync(stream, temp_mr),
          in,
          in + staged.keys_size + 1,
          out,
          cuda::proclaim_return_type<int64_t>(
            [chars_offset] __device__(int64_t offset) { return offset - chars_offset; }));
      }
      // `keys.head<char>()` is the chars base (a strings column's data buffer IS the chars;
      // `chars_begin` does not apply the parent offset, the offsets carry the byte addressing).
      auto const* chars_src = staged.keys.head<char>() + chars_offset;
      auto chars_buffer =
        rmm::device_buffer{chars_src, static_cast<std::size_t>(keys_bytes), stream, output_mr};

      auto keys_mask = cudf::detail::copy_bitmask(staged.keys, stream, output_mr);
      auto keys_col =
        cudf::make_strings_column(staged.keys_size,
                                  std::move(out_offsets),
                                  std::move(chars_buffer),
                                  staged.keys.null_count(),
                                  cuda::device_buffer<cudf::bitmask_type>{std::move(keys_mask)});
      *staged.dst_slot =
        make_dictionary_column(std::move(keys_col),
                               std::move(staged.indices),
                               cuda::device_buffer<cudf::bitmask_type>{std::move(*staged.dst_mask)},
                               staged.null_count);
      if (staged.kernel_slot.has_value()) {
        kernel_masked_columns[*staged.kernel_slot] = staged.dst_slot->get();
      }
    }
    pending_dictionaries.clear();
  }

  // Fixed-width leaf (a top-level column or a struct descendant): payload plus the leaf's own
  // mask, both carved from the call-wide fw arena. `ancestor_offset` is the accumulated offset
  // of the struct views above the leaf — rows for the mask bit offset, elements for the data
  // base.
  std::unique_ptr<column> emit_fixed_width(column_view const& src, size_type ancestor_offset)
  {
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at leaf emission");
    auto plan               = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    auto const element_size = static_cast<int32_t>(cudf::size_of(src.type()));
    // The reservation for this leaf was already added by the pre-walk pass; the arena replay
    // below consumes it (mask first, then data — matching `fixed_width_destination_layout::
    // carve`'s per-column regions).
    auto dst =
      make_carved_fixed_width(src, num_rows, plan, *fw_carve_mr, mask_bands, bands_enabled, stream);
    if (plan.skip_data) {
      // Every destination row is null: pre-set ALL_NULL mask, undefined payload (stock parity),
      // and no descriptor — the kernel would only re-read the source mask per row.
      return dst;
    }
    auto const src_data =
      static_cast<char const*>(
        cudf::type_dispatcher<dispatch_storage_type>(src.type(), data_ptr_functor{}, src)) +
      static_cast<std::size_t>(ancestor_offset) * static_cast<std::size_t>(element_size);
    auto const position =
      push_descriptor(column_descriptor{src_data,
                                        dst->mutable_view().data<std::byte>(),
                                        src.null_mask(),
                                        plan.kernel_mask,
                                        static_cast<int64_t>(src.offset()) + ancestor_offset,
                                        element_size,
                                        false});
    if (plan.kernel_mask != nullptr) { set_target(record_kernel_masked(position), dst.get()); }
    return dst;
  }

  // Recursively flattens a fully-fixed-width struct node: leaves become data descriptors and
  // nested struct nodes recurse; the node's own mask rides a MASK-ONLY descriptor when the
  // kernel must write it. An all-null node emits no descriptor (its mask is pre-set ALL_NULL),
  // and its children are still gathered with their own masks — parent and child validity are
  // independent; when every child is skipped too, the launch keeps only the node's MASK-ONLY
  // descriptor (its mask is the remaining work) unless the node's mask is also pre-set. Returns
  // the assembled destination struct column.
  std::unique_ptr<column> build_struct(column_view const& node, size_type ancestor_offset)
  {
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at struct build");
    auto const base_offset = ancestor_offset + node.offset();
    auto plan              = plan_destination_mask(node, nullify, num_rows, bands_enabled);
    // The node's own destination mask is carved BEFORE the children recurse: the fw arena is
    // a single bump allocator replaying the reservation's pre-order (node mask first, then
    // each child's regions), so any other construction order could out-consume the reserved
    // advance and falsely trip `arena exhausted` on a later column. A banded preset mask
    // slices its pattern's staging arena (its band carve mirrors the band reservation, which
    // the walk also issued pre-order); the mask is filled by the post-walk band memset and
    // moves into the column at the end; the MASK-ONLY descriptor (when pushed) takes the
    // carved pointer, which the move preserves.
    auto node_mask_env = cuda::std::execution::prop{cuda::allocation_alignment,
                                                    carve_memory_resource::slice_alignment};
    auto node_mask =
      plan.use_band
        ? cuda::device_buffer<bitmask_type>{stream,
                                            *mask_bands[plan.band].mr,
                                            plan.mask_bytes / sizeof(bitmask_type),
                                            cuda::no_init,
                                            node_mask_env}
        : cuda::device_buffer<bitmask_type>{stream,
                                            *fw_carve_mr,
                                            plan.state == mask_state::UNALLOCATED
                                              ? 0
                                              : cudf::bitmask_allocation_size_words(num_rows),
                                            cuda::no_init,
                                            node_mask_env};
    fill_destination_mask(plan, node_mask, bands_enabled, stream);

    std::vector<std::unique_ptr<column>> children;
    children.reserve(node.num_children());
    for (size_type i = 0; i < node.num_children(); ++i) {
      auto const& child = node.child(i);
      if (child.type().id() == type_id::STRUCT) {
        children.push_back(build_struct(child, base_offset));
      } else {
        children.push_back(emit_fixed_width(child, base_offset));
      }
    }

    size_t mask_slot = 0;
    if (plan.kernel_mask != nullptr) {
      // MASK-ONLY descriptor: no payload; the kernel gathers the node's own mask.
      mask_slot =
        record_kernel_masked(push_descriptor(column_descriptor{nullptr,
                                                               nullptr,
                                                               node.null_mask(),
                                                               plan.kernel_mask,
                                                               static_cast<int64_t>(base_offset),
                                                               0,
                                                               true}));
    }
    // The mask buffer (when allocated) moves into the column; its data pointer captured in the
    // descriptor stays valid across the move. With no mask allocated, pass the same zero-size
    // mask stock's struct assembly uses.
    auto col = create_structs_hierarchy(
      num_rows, std::move(children), plan.null_count, std::move(node_mask), stream, output_mr);
    if (plan.kernel_mask != nullptr) { set_target(mask_slot, col.get()); }
    return col;
  }

  // Dictionary: the annotated indices are fixed-width, so their payload copies through a data
  // descriptor addressed as indices base + parent offset, with the PARENT's own mask as the
  // source mask; the gathered mask lands on the dictionary parent via the adopting factory.
  // The descriptor consumes the same `plan_destination_mask` policy as the fixed-width
  // columns: an all-null dictionary pre-sets its mask ALL_NULL and takes the skip_data
  // fast path (no per-row source-mask reads, indices left unwritten), a nullable
  // zero-null dictionary pre-sets ALL_VALID with no kernel mask work, and only a
  // null-bearing or NULLIFY dictionary carries a source mask / kernel-written mask.
  // Indices are gathered only at valid rows — the kernel gates on `valid`, like stock's
  // gather_if, so index values at null source rows stay uninitialized (every in-tree
  // consumer guards such reads through validity, never raw storage).
  //
  // The whole string-keyed dictionary column is DEFERRED: the stock column copy of a string
  // catalog reads two offset values back to the host with a stream synchronization each, and
  // one catalog per column would serialize the whole build walk. This method pushes the
  // descriptor and records a staged entry; `assemble_staged_keys` performs the batched
  // offsets read — one D2H copy plus one synchronization for every string-keyed catalog in
  // the call — and assembles each destination column. Full key order, keys unreferenced by
  // the gathered indices retained, no dedup/remap/re-encode: identical catalog bytes as the
  // stock path.
  std::unique_ptr<column> build_dictionary(column_view const& src,
                                           std::unique_ptr<column>* dst_slot)
  {
    auto const dict     = dictionary_column_view{src};
    auto const& indices = dict.indices();
    CUDF_EXPECTS(fw_carve_mr != nullptr, "fw carve arena missing at dictionary build");
    auto plan = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    // The indices payload and the parent mask are carved (their reservations were added by
    // the pre-walk pass). The mask must exist before the descriptor is pushed: the kernel
    // writes through the descriptor's pointer, and even the deferred string-keyed assembly
    // runs BEFORE the launch, so creating it here for every path is safe and keeps one
    // upstream pair for the whole class. The carve order mirrors the reservation's per-column
    // mask-then-data: mask first, then the indices data. A banded preset mask slices its
    // pattern's staging arena (filled by the post-walk band memset), a stock-path preset mask
    // or a kernel-written mask slices the fw arena.
    auto parent_mask_env = cuda::std::execution::prop{cuda::allocation_alignment,
                                                      carve_memory_resource::slice_alignment};
    auto parent_mask =
      plan.use_band
        ? cuda::device_buffer<bitmask_type>{stream,
                                            *mask_bands[plan.band].mr,
                                            plan.mask_bytes / sizeof(bitmask_type),
                                            cuda::no_init,
                                            parent_mask_env}
        : cuda::device_buffer<bitmask_type>{stream,
                                            *fw_carve_mr,
                                            plan.state == mask_state::UNALLOCATED
                                              ? 0
                                              : cudf::bitmask_allocation_size_words(num_rows),
                                            cuda::no_init,
                                            parent_mask_env};
    fill_destination_mask(plan, parent_mask, bands_enabled, stream);
    auto indices_dst = rmm::device_buffer{
      static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(cudf::size_of(indices.type())),
      carve_memory_resource::slice_alignment,
      stream,
      *fw_carve_mr};
    auto indices_col = std::make_unique<column>(
      indices.type(),
      num_rows,
      std::move(indices_dst),
      cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, output_mr),
      0,
      std::vector<std::unique_ptr<column>>{});
    if (plan.skip_data) {
      // Every destination row is null: pre-set ALL_NULL parent mask (stock parity), undefined
      // index payload, no descriptor, and keys still owned-copied.
      auto keys_copy = std::make_unique<column>(dict.keys(), stream, output_mr);
      return make_dictionary_column(
        std::move(keys_copy), std::move(indices_col), std::move(parent_mask), plan.null_count);
    }
    auto const element_size = static_cast<int32_t>(cudf::size_of(indices.type()));
    auto const src_data =
      static_cast<char const*>(
        cudf::type_dispatcher<dispatch_storage_type>(indices.type(), data_ptr_functor{}, indices)) +
      static_cast<std::size_t>(src.offset()) * static_cast<std::size_t>(element_size);
    auto const position =
      push_descriptor(column_descriptor{src_data,
                                        indices_col->mutable_view().data<std::byte>(),
                                        src.null_mask(),
                                        plan.kernel_mask,
                                        static_cast<int64_t>(src.offset()),
                                        element_size,
                                        false});
    size_t mask_slot = 0;
    if (plan.kernel_mask != nullptr) { mask_slot = record_kernel_masked(position); }

    if (dict.keys().type().id() == type_id::STRING) {
      // DEFERRED assembly: the stock column copy of a string catalog reads two offset values
      // back with a stream synchronization each, so this path records everything the deferred
      // factory call needs and lets `assemble_staged_keys` build the column after ONE batched
      // offsets read for all string-keyed catalogs in the call. The destination slot stays
      // null until then; the kernel-masked bookkeeping is patched there too. The carved
      // parent mask moves into the staged record (its pointer already rides the descriptor).
      staged_keys staged;
      staged.keys               = dict.keys();
      staged.keys_offset        = dict.keys().offset();
      staged.keys_size          = dict.keys().size();
      staged.null_count         = plan.null_count;
      auto const& offsets_child = dict.keys().child(strings_column_view::offsets_column_index);
      staged.offsets            = std::make_unique<column>(offsets_child, stream, output_mr);
      staged.indices            = std::move(indices_col);
      staged.dst_mask.emplace(std::move(parent_mask));
      staged.dst_slot = dst_slot;
      if (plan.kernel_mask != nullptr) { staged.kernel_slot = mask_slot; }
      pending_dictionaries.push_back(std::move(staged));
      return nullptr;
    }

    // Non-string keys: the stock device-side deep copy has no synchronization, so it runs
    // inline exactly as before. The keys copy is built HERE, not at function entry: the
    // string-keyed path defers assembly and would otherwise pay the copy's two offset
    // readbacks per catalog only to discard it.
    auto keys_copy = std::make_unique<column>(dict.keys(), stream, output_mr);
    auto col       = make_dictionary_column(
      std::move(keys_copy), std::move(indices_col), std::move(parent_mask), plan.null_count);
    if (plan.kernel_mask != nullptr) { set_target(mask_slot, col.get()); }
    return col;
  }

  // Top-level STRING column on the N3 fast path. The destination offsets
  // child and chars buffer are DEFERRED (their sizes are unknowable until the
  // virtual scan's totals are read back); this method only plans the mask,
  // records the deferred entry, and returns nullptr — the launch block fills
  // `*dst_slot` in `assemble_strings` after the payload launch, or pre-emptively
  // for the all-null / zero-row cases.
  //
  // Mask policy mirrors `plan_destination_mask`: all-null source → ALL_NULL
  // pre-set, no payload; nullable-with-zero-nulls → ALL_VALID; kernel-masked →
  // UNINITIALIZED with the kernel writing it and the valid count atomically
  // accumulated. The source view's own offset is NOT baked into src_chars:
  // `src_offsets` (an offsetalator over the FULL offsets child, offset
  // applied) carries the byte addressing, so chars are addressed from the
  // unsliced base through the sliced offsets exactly as stock does.
  std::unique_ptr<column> build_string(column_view const& src, std::unique_ptr<column>* dst_slot)
  {
    auto plan = plan_destination_mask(src, nullify, num_rows, bands_enabled);
    // Strings are NOT carved (chars bytes unknowable until the scan readback); their
    // destination masks keep the stock allocation regime.
    // A zero-row source is excluded from the live cohort by the assemble step, so its
    // UNINITIALIZED plan (NULLIFY on a source with no nulls) never reaches the kernel that
    // would write its mask: the empty-source completion would build a garbage mask with a
    // placeholder 0 null count where stock yields all-null. Every gathered row is
    // out-of-bounds for such a source, so it is exactly an all-null destination: build it
    // inline like the all-null case.
    if (plan.skip_data or src.size() == 0) {
      // All-null source: ALL_NULL mask, undefined payload, no scan entry. The
      // destination must still be a full strings column (full-size offsets
      // child, empty chars) — `allocate_like` is fixed-width-only, so the
      // column is assembled directly, matching the empty-source completion.
      auto offsets_col =
        cudf::strings::detail::create_offsets_child_column(0, num_rows + 1, stream, output_mr);
      // Empty rows: offsets are all zero. The child's data is uninitialized,
      // and the comparator flags a null row as dirty unless its span is zero.
      CUDF_CUDA_TRY(cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(),
                                    0,
                                    offsets_col->size() * sizeof(int32_t),
                                    stream.get()));
      return make_strings_column(
        num_rows,
        std::move(offsets_col),
        rmm::device_buffer{},
        src.size() == 0 ? num_rows : plan.null_count,
        cudf::detail::create_null_mask(num_rows, mask_state::ALL_NULL, stream, output_mr));
    }
    staged_string staged;
    staged.src        = src;
    staged.null_count = plan.null_count;
    staged.dst_slot   = dst_slot;
    if (plan.state != mask_state::UNALLOCATED) {
      staged.dst_mask.emplace(
        cudf::detail::create_null_mask(num_rows, plan.state, stream, output_mr));
    }
    if (plan.state == mask_state::UNINITIALIZED) {
      // A string's mask is kernel-written but never rides kernel_masked_columns
      // (the descriptor readback's slots): its null count comes from the
      // payload kernel's own string_counts readback at assembly time.
      staged.kernel_mask_written = true;
    }
    pending_strings.push_back(std::move(staged));
    return nullptr;
  }
};

}  // namespace

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
  // Same walk estimates the build pass's descriptor and kernel-masked counts so the three
  // descriptor-path vectors are reserved once, up front: a deeply nested struct otherwise
  // reallocates them (with element copies) on every push. The estimate mirrors the builder's
  // emission recursion exactly — same per-node conditions, same arithmetic — so the capacity
  // matches the build output with no slack; a pessimistic guess would silently reintroduce
  // the reallocations it exists to avoid.
  size_type estimated_descriptors   = 0;
  size_type estimated_kernel_masked = 0;
  for (size_type i = 0; i < source_table.num_columns(); ++i) {
    auto const& src = source_table.column(i);
    // String columns join the fused path at every row count: the batched-memcpy routing
    // engine engages above `kBatchMemcpyRowThreshold` (the host routing walk keys off
    // `chars_routing_active`).
    // Below the gate every string cohort stays fused: the engine serves short-row shapes (the
    // band cell is one of its measured win cells, engine-vs-stock) and an all-long-row table
    // falls to the payload kernel's group walk (the pre-engine fused behavior) — its loss on
    // that shape is the known admission risk the falsifier's regression checks watch. The
    // earlier band exclusion was a guard against a composed-tree tall regression that turned
    // out to be that tree losing the routed tall route, not a cost of admitting the band, and
    // it forfeited the measured band win while only the band's stock route differed; it is
    // reverted. Gathered row sizes are unknowable before the scan, so no stronger per-column
    // proxy is possible here; the exact post-totals routing still splits per column.
    if (is_descriptor_path_candidate(src)) {
      fused_indices.push_back(i);
      if (not(num_rows == 0 and not cudf::is_fixed_width(src.type()))) {
        // Mirror the build's skip: an empty map routes non-fixed-width columns to empty_like
        // without emitting descriptors, so those must not inflate the estimate.
        auto const counts =
          estimate_descriptor_counts(src, bounds_policy == out_of_bounds_policy::NULLIFY);
        estimated_descriptors += counts.descriptors;
        estimated_kernel_masked += counts.kernel_masked;
      }
    } else {
      fallback_indices.push_back(i);
    }
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
    // and set_all_valid_null_masks is a no-op for non-nullable inputs). Under DONT_CHECK a
    // null-FREE fallback cohort skips gather_bitmask entirely: its per-column outcome is
    // host-known (all-valid top-level masks, children untouched), which also drops the
    // valid_counts D2H readback and its stream sync.
    auto fallback_views = std::vector<cudf::column_view>{};
    fallback_views.reserve(indices.size());
    for (auto const idx : indices) {
      fallback_views.push_back(source_table.column(idx));
    }
    auto const fallback_table = cudf::table_view{fallback_views};
    if (bounds_policy == out_of_bounds_policy::NULLIFY or
        cudf::has_nested_nullable_columns(source_table)) {
      if (bounds_policy == out_of_bounds_policy::NULLIFY) {
        gather_bitmask(
          fallback_table, map_it_begin, fallback_columns, gather_bitmask_op::NULLIFY, stream, mr);
      } else if (cudf::has_nested_nulls(source_table)) {
        if (cudf::has_nested_nulls(fallback_table)) {
          // The fallback cohort itself carries nulls: per-row validity must be computed on the
          // device, exactly as stock does.
          gather_bitmask(fallback_table,
                         map_it_begin,
                         fallback_columns,
                         gather_bitmask_op::DONT_CHECK,
                         stream,
                         mr);
        } else {
          // Null-free fallback cohort under DONT_CHECK: every gathered row is valid, so stock's
          // per-column outcome is known host-side — no device-view build, no valid_if launch, no
          // counts readback, no blocking sync. Only nullable SOURCES need a destination mask
          // (stock's kernel skips null-mask-free sources); a fresh ALL_VALID top-level mask with
          // null_count 0 matches stock for any destination state, and children are deliberately
          // NOT rewritten: stock's gather_bitmask writes top-level masks only, and fabricating
          // child masks here diverges from stock (the child-nullability regression the
          // whole-table gate exists to prevent).
          for (size_t i = 0; i < fallback_columns.size(); ++i) {
            auto const& src = fallback_table.column(i);
            auto& dst       = *fallback_columns[i];
            if (not src.nullable()) { continue; }
            dst.set_null_mask(
              cudf::create_null_mask(dst.size(), mask_state::ALL_VALID, stream, output_mr), 0);
          }
        }
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

  // Fused part: destination columns for the descriptor-path subset, with per-node mask policy.
  // Plain fixed-width columns emit one data descriptor; fully-fixed-width structs flatten into
  // leaf descriptors plus mask-only node descriptors (reassembled after the launch); dictionary
  // columns emit one data descriptor over their annotated indices. All-null sources emit no
  // descriptor (their masks are pre-set ALL_NULL); if nothing remains, the launch is skipped.
  // The band-path decision, made once per call BEFORE any band arena exists: the band fill
  // trades a fixed ~24 µs-per-band memset for per-mask ~7 µs submissions, so with fewer than
  // `kMaskBandPresetGate` preset masks the per-mask drip wins and every preset mask must take
  // the stock carve + per-mask memset (a populated band memset ≈ 24 µs vs ~7 µs per mask
  // submission: 8 masks ≈ 56 µs > 2 × 24 µs). The
  // count is computed from the same `plan_destination_mask` classification the build walk
  // uses (struct children recursed), so the walk and this decision cannot diverge.
  bool const bands_enabled = [&] {
    if (fused_indices.empty()) { return false; }
    size_type preset_masks                                      = 0;
    std::function<void(column_view const&, bool)> count_presets = [&](column_view const& src,
                                                                      bool nullify_flag) {
      if (src.type().id() == type_id::STRING) { return; }
      auto const plan = plan_destination_mask(src, nullify_flag, num_rows, false);
      if (plan.preset) { ++preset_masks; }
      if (src.type().id() == type_id::STRUCT) {
        for (size_type i = 0; i < src.num_children(); ++i) {
          count_presets(src.child(i), nullify_flag);
        }
      }
    };
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) { continue; }
      count_presets(src, bounds_policy == out_of_bounds_policy::NULLIFY);
    }
    return preset_masks > kMaskBandPresetGate;
  }();

  std::vector<column_descriptor> descriptors;
  std::vector<size_type> kernel_masked_positions;  // positions of kernel-masked descriptors
  std::vector<column*> kernel_masked_columns;      // slot i's count target (columns[i])
  descriptor_path_builder builder{descriptors,
                                  kernel_masked_positions,
                                  kernel_masked_columns,
                                  num_rows,
                                  bounds_policy == out_of_bounds_policy::NULLIFY,
                                  bands_enabled,
                                  stream,
                                  output_mr,
                                  temp_mr};
  if (not fused_indices.empty()) {
    // Pre-walk reservation for the fixed-width carve: sum every fw destination's mask+data
    // bytes (dry planning — no allocation) so ONE arena covers the whole class with a single
    // upstream allocate/free pair. The reservation visits columns in the SAME pre-order the
    // build walk will carve them (see `fixed_width_destination_layout`), so the total equals
    // the bump pointer's final position exactly; a divergence trips the carve resource's loud
    // `arena exhausted` expect. The fw arena is created BEFORE the build walk because the fw
    // columns construct hierarchically during it (leaves, struct nodes, dictionaries);
    // strings stay outside it (chars bytes unknowable until the post-walk scan readback) and
    // keep their own carve in `assemble_strings`. The mirror must include every buffer the
    // build walk creates: an all-null destination still owns a full-size (uninitialized) data
    // buffer and a pre-set ALL_NULL mask.
    // std::function, not `auto`: the reservation mirrors the build walk's STRUCT recursion,
    // so the lambda must call itself from its own initializer (an `auto` initializer cannot
    // self-reference). One host-side allocation per call, outside the carve path.
    std::function<void(column_view const&, bool)> fw_reserve = [&](column_view const& src,
                                                                   bool nullify_flag) {
      if (src.type().id() == type_id::STRING) { return; }
      // A banded preset mask's reservation rides its pattern's band layout, not the fw arena
      // (the build walk carves banded preset masks from the band, kernel-written masks from
      // the arena — `plan_destination_mask` computes the same classification here as at build
      // time, so the two mirrors stay in lockstep). A stock-path preset mask (below the band
      // gate) carves the fw arena with the rest, matching `plan.use_band` at build time.
      auto const plan = plan_destination_mask(src, nullify_flag, num_rows, bands_enabled);
      if (src.type().id() == type_id::STRUCT) {
        if (plan.preset and plan.use_band) { builder.mask_bands[plan.band].layout.carve(num_rows); }
        if (plan.state != mask_state::UNALLOCATED and not plan.use_band) {
          builder.fw_layout.carve(num_rows, 0, true);
        }
        for (size_type i = 0; i < src.num_children(); ++i) {
          fw_reserve(src.child(i), nullify_flag);
        }
        return;
      }
      if (src.type().id() == type_id::DICTIONARY32) {
        auto const indices    = dictionary_column_view{src}.indices();
        auto const data_bytes = static_cast<std::size_t>(num_rows) *
                                static_cast<std::size_t>(cudf::size_of(indices.type()));
        if (plan.use_band) {
          // The parent mask rides the band; the gathered indices payload still carves the
          // fw arena at build time (`build_dictionary` carves it on every path), so its
          // region must be reserved here too or the replay out-consumes the layout.
          builder.mask_bands[plan.band].layout.carve(num_rows);
          builder.fw_layout.carve(num_rows, data_bytes, false);
        } else if (plan.state != mask_state::UNALLOCATED) {
          builder.fw_layout.carve(num_rows, data_bytes, true);
        } else {
          builder.fw_layout.carve(num_rows, data_bytes, false);
        }
        return;
      }
      auto const data_bytes =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(cudf::size_of(src.type()));
      if (plan.use_band) {
        builder.mask_bands[plan.band].layout.carve(num_rows);
        builder.fw_layout.carve(num_rows, data_bytes, false);
      } else {
        builder.fw_layout.carve(num_rows, data_bytes, plan.state != mask_state::UNALLOCATED);
      }
    };
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) { continue; }
      fw_reserve(src, bounds_policy == out_of_bounds_policy::NULLIFY);
    }
    auto fw_carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
      output_mr, stream, builder.fw_layout.total_bytes);
    builder.fw_carve_mr = &fw_carve_mr;
    for (auto& band : builder.mask_bands) {
      band.mr = std::make_unique<cuda::mr::shared_resource<carve_memory_resource>>(
        cuda::mr::make_shared_resource<carve_memory_resource>(
          output_mr, stream, band.layout.total_bytes));
    }

    descriptors.reserve(estimated_descriptors);
    kernel_masked_positions.reserve(estimated_kernel_masked);
    kernel_masked_columns.reserve(estimated_kernel_masked);
    for (size_type pos = 0; pos < static_cast<size_type>(fused_indices.size()); ++pos) {
      auto const& src = source_table.column(fused_indices[pos]);
      if (num_rows == 0 and not cudf::is_fixed_width(src.type())) {
        // Empty map: stock's per-type gatherers return exactly these shapes for these types.
        destination[fused_indices[pos]] =
          (src.type().id() == type_id::STRUCT)
            ? cudf::empty_like(src)
            : (src.type().id() == type_id::STRING ? cudf::make_empty_column(type_id::STRING)
                                                  : cudf::make_empty_column(type_id::DICTIONARY32));
        continue;
      }
      switch (src.type().id()) {
        case type_id::STRUCT: destination[fused_indices[pos]] = builder.build_struct(src, 0); break;
        case type_id::STRING:
          // The N3 fast path defers the whole column: the scan launch block
          // fills the slot in `assemble_strings`. Only an empty map (handled
          // above) or an all-null source (skip_data, built inline) returns a
          // column here.
          destination[fused_indices[pos]] =
            builder.build_string(src, &destination[fused_indices[pos]]);
          break;
        case type_id::DICTIONARY32:
          // String-keyed catalogs defer their assembly and fill the slot in
          // `assemble_staged_keys`; every other path builds the column here.
          destination[fused_indices[pos]] =
            builder.build_dictionary(src, &destination[fused_indices[pos]]);
          break;
        default: destination[fused_indices[pos]] = builder.emit_fixed_width(src, 0); break;
      }
    }
    // One memset per non-empty band replaces the per-mask fills: every banded preset mask
    // carved from that band gets the pattern byte in a single submission, stream-ordered
    // before any consumer (the kernel launch and the deferred assemblies both follow on the
    // same stream). Below the band gate the per-mask fills already ran during the walk.
    if (bands_enabled) {
      for (size_t band = 0; band < 2; ++band) {
        auto const& b = builder.mask_bands[band];
        if (b.layout.total_bytes == 0) { continue; }
        auto const pattern = band == 0 ? 0xff : 0x00;
        CUDF_CUDA_TRY(
          cudaMemsetAsync(b.mr->get().arena_base(), pattern, b.layout.total_bytes, stream.get()));
      }
    }
    // Zero rows never reach `build_dictionary` (the empty-map branch above returns early for
    // every non-fixed-width type), so `pending_dictionaries` is empty here; this call is a
    // defensive no-op that keeps the deferred contract satisfied on every path that skips
    // the launch block below.
    if (num_rows == 0) { builder.assemble_staged_keys(); }
  }

  auto const any_kernel_masks = not kernel_masked_positions.empty();
  if (any_kernel_masks) {
    // Masked descriptors must occupy the leading positions [0, M) so a masked descriptor's
    // position doubles as its `valid_counts` slot. Descriptors are position-independent (all
    // addressing is via their pointers), so the stable partition is semantically free; visit
    // order is preserved within each run. This must precede the staging copy below.
    std::stable_partition(descriptors.begin(), descriptors.end(), [](column_descriptor const& d) {
      return d.dst_mask != nullptr;
    });
    for (size_t i = 0; i < kernel_masked_positions.size(); ++i) {
      kernel_masked_positions[i] = static_cast<size_type>(i);
    }
  }

  // String fast path (N3): when string columns are present they ride the
  // batched scan+payload machinery, which also resolves the routed tile width
  // under its totals synchronization. The descriptor launch below reuses the
  // resolved width via `explicit_tile_cols` being set. When there are no
  // descriptors at all (a pure-strings call), the SAME assemble_strings call
  // performs everything — the launch block below is skipped.
  auto mutable_tile_cols = explicit_tile_cols;
  if (not builder.pending_strings.empty() and num_rows > 0) {
    assemble_strings_impl(builder.pending_strings,
                          num_rows,
                          gather_map,
                          num_src_rows,
                          bounds_policy == out_of_bounds_policy::NULLIFY,
                          neg_indices == negative_index_policy::ALLOWED,
                          sampled,
                          mutable_tile_cols,
                          stream,
                          output_mr,
                          temp_mr);
    explicit_tile_cols = mutable_tile_cols;
  } else if (not builder.pending_strings.empty()) {
    // Zero rows: every deferred string column becomes an empty strings column
    // (stock parity with gather_strings on an empty map).
    for (auto& staged : builder.pending_strings) {
      *staged.dst_slot = cudf::make_empty_column(type_id::STRING);
    }
    builder.pending_strings.clear();
  }

  if (not descriptors.empty() and num_rows > 0) {
    // Deferred key-catalog assembly (string-keyed dictionaries): one batched offsets read
    // plus one synchronization for every catalog in the call, in place of copy_slice's two
    // blocking reads per catalog. Must precede the launch so the sync does not wait behind
    // GPU work the gather could have overlapped with.
    builder.assemble_staged_keys();

    // Pinned staging: the pool's deallocation keeps the source readable for as long as
    // the copy may read it (stream-ordered pool free; the pool-exhaustion fallback frees
    // through cudaFreeHost, a synchronizing free — no host sync needed either way). The
    // device landing buffer is carved from a descriptor-staging arena (ONE upstream pair
    // instead of one malloc/free per call); its scope is this launch block, and
    // `batch_gather_kernel` — the buffer's only consumer — is enqueued inside it before
    // the staging vectors die.
    auto staging =
      cudf::detail::make_pinned_vector_async<column_descriptor>(descriptors.size(), stream);
    std::copy(descriptors.begin(), descriptors.end(), staging.begin());
    auto desc_staging_carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
      temp_mr, stream, descriptors.size() * sizeof(column_descriptor));
    auto d_descriptors = carve_device_vector(
      host_span<column_descriptor const>{staging}, stream, desc_staging_carve_mr);

    rmm::device_uvector<size_type> valid_counts(0, stream, temp_mr);
    if (any_kernel_masks) {
      // The buffer needs only M entries; the zero-fill covers exactly M.
      valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        kernel_masked_positions.size(), stream, temp_mr);
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
    // Resolve the routed tile width only now: the sync waits for the enqueued sample copies
    // AND for the allocation/descriptor work above, hiding the sampler latency behind it.
    // The string fast path may already have resolved it (under ITS totals sync); reuse.
    auto const effective_tile_cols = mutable_tile_cols.has_value()
                                       ? *mutable_tile_cols
                                       : resolve_map_locality_tile_cols(sampled, stream);
    switch (effective_tile_cols) {
      case 1: launch_tile(std::integral_constant<size_type, 1>{}); break;
      case 2: launch_tile(std::integral_constant<size_type, 2>{}); break;
      case 4: launch_tile(std::integral_constant<size_type, 4>{}); break;
      case 8: launch_tile(std::integral_constant<size_type, 8>{}); break;
      default: CUDF_FAIL("Unsupported batch_gather tile width", std::invalid_argument);
    }

    if (any_kernel_masks) {
      auto const h_valid_counts = cudf::detail::make_host_vector(valid_counts, stream);
      for (size_t i = 0; i < kernel_masked_positions.size(); ++i) {
        kernel_masked_columns[i]->set_null_count(num_rows - h_valid_counts[i]);
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
