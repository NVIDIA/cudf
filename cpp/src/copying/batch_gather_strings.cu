/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// String fast path (N3) of the fused gather, split out of batch_gather.cu so the two heavy
// template surfaces compile as independent TUs in parallel. The batched scan + payload kernel
// machinery lives here; the shared device types come from batch_gather_common.cuh, and the
// descriptor-path builder in batch_gather.cu hands its deferred string records to
// assemble_strings_impl below. No relocatable device code: this TU owns the launcher of
// string_payload_kernel, which the common header defines.

#include "batch_gather_common.cuh"

namespace cudf {
namespace detail {
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
//      offsets child width (`create_offsets_child_column`, which also
//      enforces the column size limit).
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
//      routing via `chars_via_batched_memcpy`); the kernel keeps masks/offsets
//      for every column and the chars of any UNROUTED column.
//
// Trap compliance: chars are addressed from the unsliced base with the sliced
// offsets (the offsets iterator carries the view offset); null rows have zero
// bytes and distinct validity from empty strings; gathered sizes are scanned
// in 64 bits.
// ---------------------------------------------------------------------------

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
                           rmm::device_async_resource_ref temp_mr)
{
  if (staged_strings.empty()) { return; }
  auto const num_strings = static_cast<size_type>(staged_strings.size());

  // The CUB batched-memcpy engine only runs above this row gate; every host-side addition
  // it requires (the routing walk, the flag init, the CUB launch) keys off this one
  // predicate, so below the gate the host path stays shaped like the pre-routing path.
  auto const chars_routing_active = num_rows >= kBatchMemcpyRowThreshold;

  // Live sources only: an empty SOURCE column (possibly no offsets child)
  // carries no payload work, but its destination still needs a full-size
  // offsets child (every row empty). Such columns are excluded from the
  // device items and completed host-side below.
  std::vector<size_type> live;
  live.reserve(num_strings);
  for (size_type i = 0; i < num_strings; ++i) {
    auto const& staged = staged_strings[i];
    if (staged.src.size() > 0 and staged.src.num_children() > 0) { live.push_back(i); }
  }

  std::vector<string_work_item> items(live.size());
  for (size_t k = 0; k < live.size(); ++k) {
    auto const& staged        = staged_strings[live[k]];
    auto const& offsets_child = staged.src.child(cudf::strings_column_view::offsets_column_index);
    items[k].src_chars        = staged.src.head<char>();
    items[k].src_mask         = staged.src.null_mask();
    items[k].mask_bit_offset  = staged.src.offset();
    items[k].src_rows         = staged.src.size();
    items[k].src_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(offsets_child, staged.src.offset());
  }

  // Staging-arena sizing, BEFORE the arena is created: a carve never frees, so the arena is
  // sized once, up front, at worst case — every upload cohort is bounded by live.size()
  // (the routing walk only SPLITS the live columns between the routed and unrouted lists),
  // so the arena covers the per-class maximum of all segments and the actual carve
  // sequence always consumes at most the reservation. The bound is the per-segment
  // aligned sum, NOT the raw byte sum: each carve request may shift the arena pointer by
  // up to (alignment − 1) bytes over an arbitrary start, so the reservation is the raw
  // byte sum plus that slack per segment slot. Reserving the aligned segment sizes covers
  // every round-up the sequence can make (the worst case is d_items + d_payload_items +
  // d_routed_items plus one size_type cohort — d_memcpy_live and d_unrouted are disjoint
  // splits of the live columns, and d_engine_cols fires only when the unrouted carve does
  // not); a shortfall trips the carve resource's loud `arena exhausted` expect rather than
  // corrupting memory. For the actual layouts every round-up is pure slack
  // (`string_work_item` is 144 B, a multiple of its 16-byte alignment).
  auto const staging_total =
    3 * (items.size() * sizeof(string_work_item) + (std::alignment_of_v<string_work_item> - 1)) +
    2 * (items.size() * sizeof(size_type) + (std::alignment_of_v<size_type> - 1));
  // The staging arena is created HERE — sized from the exact reservation, before the first
  // carve — and dies with this call scope. Its buffers must NOT outlive the call (the
  // design's own lifetime constraint): every consumer is a launch enqueued on `stream` while
  // the arena is alive, and the shared resource's stream-ordered free is enqueued when the
  // last buffer copy dies at this scope's exit, AFTER every consuming enqueue. The
  // destination carve classes keep the opposite lifetime (call-outliving).
  auto staging_carve_mr =
    cuda::mr::make_shared_resource<carve_memory_resource>(temp_mr, stream, staging_total);

  auto const rows_per_col = static_cast<int64_t>(num_rows) + 1;
  auto const flat_size    = static_cast<int64_t>(live.size()) * rows_per_col;
  // The 32-bit precomputed divisors are valid only while every flat index fits uint32_t;
  // this bound is the enforcement the string_flat_layout contract relies on. The fan-byte
  // dividend adds up to block_size row slots on top of a flat row index, so the bound is
  // tightened by that slack.
  CUDF_EXPECTS(flat_size <= cuda::std::numeric_limits<uint32_t>::max() - block_size,
               "Flat string index space exceeds uint32_t; reduce the number of string columns "
               "or rows");
  // The expects above pins flat_size within uint32_t, so the precomputed 32-bit divisors
  // replace the per-element 64-bit //% (string_flat_layout).
  string_flat_layout const scan_layout{
    cuda::fast_mod_div<uint32_t>{static_cast<uint32_t>(rows_per_col)},
    cuda::fast_mod_div<uint32_t>{static_cast<uint32_t>(num_rows)}};
  auto scratch = rmm::device_uvector<int64_t>(static_cast<std::size_t>(flat_size), stream, temp_mr);
  auto col_totals =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(live.size(), stream, temp_mr);

  // Every H2D staging source below is a pinned `host_vector`: with the CUDA 13+ batch
  // copy's stream-order source semantics a pageable source must outlive the stream's
  // execution of the copy, and the function returns without synchronizing after these
  // uploads. Pinned host memory is device-accessible, so the copy wrapper takes the
  // pinned path, and the pinned pool's deallocation keeps the source readable for as
  // long as the copy may still read it: the pool free is stream-ordered (an event
  // recorded on this stream fences the reuse), while the pool-exhaustion fallback
  // frees through cudaFreeHost, a synchronizing free that cannot complete while a
  // copy still reads the allocation.
  auto staging_items =
    cudf::detail::make_pinned_vector_async<string_work_item>(items.size(), stream);
  std::copy(items.begin(), items.end(), staging_items.begin());
  auto d_items =
    carve_device_vector(host_span<string_work_item const>{staging_items}, stream, staging_carve_mr);

  auto const map_begin = indexalator_factory::make_input_iterator(gather_map);
  {
    auto const keys = cuda::transform_iterator(
      cuda::counting_iterator<int64_t>{0},
      cuda::proclaim_return_type<int64_t>([scan_layout] __device__(int64_t flat) {
        return static_cast<int64_t>(static_cast<uint32_t>(flat) / scan_layout.rows_per_col);
      }));
    auto const values = cuda::transform_iterator(
      cuda::counting_iterator<int64_t>{0},
      cuda::proclaim_return_type<int64_t>([d_items = d_items.data(),
                                           map_begin,
                                           scan_layout,
                                           num_source_rows,
                                           check_bounds,
                                           negative_allowed] __device__(int64_t flat) -> int64_t {
        auto const col   = static_cast<uint32_t>(flat) / scan_layout.rows_per_col;
        auto const row   = static_cast<uint32_t>(flat) % scan_layout.rows_per_col;
        auto const& item = d_items[col];
        if (row >= scan_layout.rows_per_col - 1u) { return int64_t{0}; }
        auto const src = resolve_gathered_row(
          map_begin, static_cast<size_type>(row), num_source_rows, check_bounds, negative_allowed);
        if (src < 0 or not string_row_valid(item, src)) { return int64_t{0}; }
        return item.src_offsets[static_cast<size_type>(src) + 1] -
               item.src_offsets[static_cast<size_type>(src)];
      }));
    auto const out = cuda::make_tabulate_output_iterator(
      string_scan_out_fn{scratch.data(), col_totals.data(), scan_layout});
    thrust::exclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, temp_mr), keys, keys + flat_size, values, out, int64_t{0});
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // One batched totals D2H; its synchronization also resolves the routed
  // tile width, hiding the sampler latency behind the scan.
  auto h_col_totals = cudf::detail::make_pinned_vector_async<int64_t>(live.size(), stream);
  cudf::detail::cuda_memcpy_async<int64_t>(
    host_span<int64_t>{h_col_totals.data(), static_cast<std::size_t>(live.size())},
    device_span<int64_t const>{col_totals.data(), static_cast<std::size_t>(live.size())},
    stream);
  std::optional<size_type> tile_from_sampler;
  if (not explicit_tile_cols.has_value()) {
    tile_from_sampler = resolve_map_locality_tile_cols(sampled, stream);
  }
  explicit_tile_cols = explicit_tile_cols.has_value() ? explicit_tile_cols : tile_from_sampler;
  // resolve_map_locality_tile_cols already synchronized when it sampled; the
  // explicit sync is needed only on the paths that skipped it (explicit tile
  // width, or an unclassifiable map). Either way ONE synchronization covers
  // sampler decode, the scan, and this totals copy.
  if (tile_from_sampler.has_value() and sampled.staging.empty()) {
    cudf::detail::sync_stream(stream);
  } else if (not tile_from_sampler.has_value()) {
    cudf::detail::sync_stream(stream);
  }

  auto any_kernel_masked = false;
  for (size_t k = 0; k < live.size(); ++k) {
    if (staged_strings[live[k]].kernel_mask_written) { any_kernel_masked = true; }
  }
  // Landing buffer for the deferred counts readback (b3): allocated HERE,
  // right after the totals sync with an idle stream, because every
  // host_vector construction synchronizes on allocate. Allocating it after
  // the payload launch would block the host on the whole payload kernel.
  auto h_string_counts = cudf::detail::make_pinned_vector_async<size_type>(
    any_kernel_masked ? static_cast<std::size_t>(live.size()) : 0u, stream);

  // Destination allocation from the totals: ONE carve arena covers every live column's
  // offsets child and chars buffer — the allocation class b0 attributed ~783 µs/call to —
  // so the per-buffer `cudaMallocFromPoolAsync`/`cudaFreeAsync` storm collapses to a single
  // upstream allocate/free pair. Each column keeps owning its slices: the carve resource,
  // shared through every buffer's stored value copy, is the transferable part. Non-live
  // staged entries (empty SOURCE columns) and the all-null/empty-map completions below stay
  // on the stock `create_offsets_child_column` path: at most one small column per call, not
  // part of the measured allocation storm. String destination masks stay stock too (allocated
  // in `build_string`); the fixed-width destination class draws from its own arena.
  strings_destination_layout dst_layout;
  for (size_t k = 0; k < live.size(); ++k) {
    dst_layout.carve(h_col_totals[k], num_rows);
  }
  auto carve_mr = cuda::mr::make_shared_resource<carve_memory_resource>(
    output_mr, stream, dst_layout.total_bytes);

  std::vector<cudf::device_span<int64_t>> scratch_spans(live.size());
  std::vector<cudf::device_span<char>> chars_spans(live.size());
  for (size_t k = 0; k < live.size(); ++k) {
    auto& staged   = staged_strings[live[k]];
    auto carved    = carve_strings_buffers(num_rows, h_col_totals[k], carve_mr, stream, output_mr);
    staged.offsets = std::move(carved.offsets);
    staged.chars   = std::move(carved.chars);
    scratch_spans[k] = cudf::device_span<int64_t>{
      reinterpret_cast<int64_t*>(scratch.data()) + static_cast<std::ptrdiff_t>(k) * rows_per_col,
      static_cast<std::size_t>(rows_per_col)};
    auto const bytes = h_col_totals[k];
    if (bytes > 0) {
      chars_spans[k] = cudf::device_span<char>{static_cast<char*>(staged.chars.data()),
                                               static_cast<std::size_t>(bytes)};
    }
  }

  // ---- batched-memcpy engine: per-column routing BEFORE the items upload ----
  // After the totals sync the host knows each column's exact byte count, so the routing
  // predicate is exact: a column whose average gathered row is <= 32 B has its chars
  // served by one cub::DeviceMemcpy::Batched over the routed flat (column, row) domain —
  // stock parity, see `strings::detail::gather`'s engine split. The decision rides the
  // per-item `chars_via_batched_memcpy` flag, so the kernel skips ONLY the routed
  // columns' chars: a tall mixed-length call keeps every unrouted column's chars on the
  // kernel path, and every output region is written by exactly one engine. The items
  // upload below must stay AFTER this block: the flag is part of the staged struct.
  auto any_batched_memcpy = false;
  std::vector<size_type> memcpy_live;      // live indices routed to cub::DeviceMemcpy::Batched
  std::vector<size_type> memcpy_unrouted;  // live indices keeping the kernel chars+metadata
  if (chars_routing_active) {
    for (size_t k = 0; k < live.size(); ++k) {
      auto const bytes = h_col_totals[k];
      if (bytes > 0 and bytes / num_rows <= static_cast<int64_t>(kBlockRowCopyThreshold)) {
        memcpy_live.push_back(static_cast<size_type>(k));
        items[k].chars_via_batched_memcpy = true;
        any_batched_memcpy                = true;
      } else {
        // Unrouted cohort: this column keeps the kernel path AND its metadata rides the
        // payload kernel (the routed kernel covers only the routed columns).
        memcpy_unrouted.push_back(static_cast<size_type>(k));
      }
    }
  } else {
    // Below the routing gate every live column is unrouted by definition.
    for (size_t k = 0; k < live.size(); ++k) {
      memcpy_unrouted.push_back(static_cast<size_type>(k));
    }
  }

  // ---- fused short-string engine admission: BELOW the routing gate, the payload kernel's
  // group walk (3 CTA barriers per 8-row group) is replaced by the warp-tile engine when
  // every live column's average gathered row is short enough for it — the exact premise the
  // group walk was designed for. This includes the whole former seam band (short-row band
  // calls are measured engine win cells); at ≥524288 rows the routed-metadata path owns
  // string metadata, so the engine serves only below-gate short-row calls. Above the
  // gate the payload kernel serves only the UNROUTED cohort (long-row columns), which the
  // engine does NOT admit (subgroup copies serialize on long rows), so the group walk stays
  // authoritative there. The AVERAGE does not bound the maximum: long outlier rows are safe
  // on the engine (they ride the full-warp copy), so the short-average check alone admits.
  auto engine_active = false;
  std::vector<size_type> engine_cols;
  if (not chars_routing_active and not memcpy_unrouted.empty()) {
    auto all_short = true;
    for (size_t k = 0; k < memcpy_unrouted.size(); ++k) {
      auto const& idx  = memcpy_unrouted[k];
      auto const bytes = h_col_totals[idx];
      if (bytes / num_rows > static_cast<int64_t>(kBlockRowCopyThreshold)) {
        all_short = false;
        break;
      }
    }
    if (all_short) {
      engine_cols = std::move(memcpy_unrouted);
      memcpy_unrouted.clear();
      engine_active = true;
    }
  }

  // Wire the payload work items (num_rows > 0 guaranteed for live columns).
  // any_kernel_masked was computed above the landing-buffer allocation.
  rmm::device_uvector<size_type> string_counts(0, stream, temp_mr);
  if (any_kernel_masked) {
    string_counts =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(live.size(), stream, temp_mr);
  }
  for (size_t k = 0; k < live.size(); ++k) {
    auto const slot      = live[k];
    auto& staged         = staged_strings[slot];
    items[k].dst_scratch = scratch_spans[k].data();
    items[k].dst_chars   = chars_spans[k].empty() ? nullptr : chars_spans[k].data();
    items[k].dst_offsets =
      cudf::detail::offsetalator_factory::make_output_iterator(staged.offsets->mutable_view());
    items[k].dst_mask    = staged.kernel_mask_written ? staged.dst_mask->data() : nullptr;
    items[k].valid_count = staged.kernel_mask_written ? string_counts.data() + k : nullptr;
  }
  // The dst fields were patched after the first upload, so copy the final items into the
  // pinned staging buffer and upload from there (the d_items copy used the initial values).
  std::copy(items.begin(), items.end(), staging_items.begin());
  auto d_payload_items =
    carve_device_vector(host_span<string_work_item const>{staging_items}, stream, staging_carve_mr);

  if (any_batched_memcpy) {
    // The routed flat (column, row) domain, served by one cub::DeviceMemcpy::Batched
    // (`strings::detail::gather`'s engine split): destination and size read the scanned
    // prefixes in `dst_scratch` directly and the row's map/validity/source resolve runs
    // once per buffer, in the src iterator (`routed_memcpy_iterators` above) — no staging
    // arrays and no per-operator resolve chains. CUB classifies buffers from the sizes
    // iterator before dereferencing either pointer iterator, so a size-0 buffer's
    // nullptr src/dst is never dereferenced.
    // Pinned staging copy of the routed cohort (lifetime as pinned staging above).
    auto staging_memcpy_live =
      cudf::detail::make_pinned_vector_async<size_type>(memcpy_live.size(), stream);
    std::copy(memcpy_live.begin(), memcpy_live.end(), staging_memcpy_live.begin());
    auto const d_memcpy_live = carve_device_vector(
      host_span<size_type const>{staging_memcpy_live}, stream, staging_carve_mr);
    auto const num_buffers =
      static_cast<int64_t>(memcpy_live.size()) * static_cast<int64_t>(num_rows);
    auto const d_routed     = d_memcpy_live.data();
    auto const d_items_ptr  = d_payload_items.data();
    auto const lean_iters   = routed_memcpy_iterators{d_items_ptr,
                                                    d_routed,
                                                    map_begin,
                                                    num_source_rows,
                                                    check_bounds,
                                                    negative_allowed,
                                                    scan_layout};
    auto const src_addrs_it = cuda::make_transform_iterator(
      cuda::counting_iterator<int64_t>{0},
      cuda::proclaim_return_type<char const*>(
        [lean_iters] __device__(int64_t flat) -> char const* { return lean_iters.src_of(flat); }));
    auto const dst_addrs_it = cuda::make_transform_iterator(
      cuda::counting_iterator<int64_t>{0},
      cuda::proclaim_return_type<char*>(
        [lean_iters] __device__(int64_t flat) -> char* { return lean_iters.dst_of(flat); }));
    auto const copy_sizes_it = cuda::make_transform_iterator(
      cuda::counting_iterator<int64_t>{0},
      cuda::proclaim_return_type<int64_t>(
        [lean_iters] __device__(int64_t flat) -> int64_t { return lean_iters.size_of(flat); }));
    // The env-overload routes CUB's temporary storage to `temp_mr` (not the wrapper's
    // default-resource form), matching `strings::detail::gather`'s own batched call.
    // Explicit template args: the CTAD form trips nvcc 13.3 in TUs that lack an
    // earlier `<thrust/execution_policy.h>` include (include-order sensitive).
    auto const cub_env = cuda::std::execution::env<
      cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>,
      cuda::std::execution::prop<cuda::mr::get_memory_resource_t, rmm::device_async_resource_ref>>{
      cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>{cuda::get_stream_t{},
                                                                       stream},
      cuda::std::execution::prop<cuda::mr::get_memory_resource_t, rmm::device_async_resource_ref>{
        cuda::mr::get_memory_resource_t{}, temp_mr}};
    CUDF_CUDA_TRY(
      cub::DeviceMemcpy::Batched(src_addrs_it, dst_addrs_it, copy_sizes_it, num_buffers, cub_env));
  }

  // ---- the ONE batched payload launch (unrouted cohort only) ----
  // The grid covers only the UNROUTED cohort's flat rows: a routed column contributes no
  // group visits at all (its warps' flat slots are out of the launched domain), so the
  // group walk's barriers/visits disappear for it — the routed kernel owns its metadata.
  auto const unrouted_rows =
    static_cast<int64_t>(memcpy_unrouted.size()) * static_cast<int64_t>(num_rows);
  if (unrouted_rows > 0) {
    // Pinned staging copy of the unrouted cohort (lifetime as pinned staging above).
    auto staging_unrouted =
      cudf::detail::make_pinned_vector_async<size_type>(memcpy_unrouted.size(), stream);
    std::copy(memcpy_unrouted.begin(), memcpy_unrouted.end(), staging_unrouted.begin());
    auto const d_unrouted =
      carve_device_vector(host_span<size_type const>{staging_unrouted}, stream, staging_carve_mr);
    auto const grid = static_cast<unsigned>(
      std::min<int64_t>(util::div_rounding_up_safe(unrouted_rows, static_cast<int64_t>(block_size)),
                        static_cast<int64_t>(65535)));
    string_payload_kernel<block_size>
      <<<grid, block_size, 0, stream.get()>>>(d_payload_items.data(),
                                              d_unrouted.data(),
                                              static_cast<int64_t>(memcpy_unrouted.size()),
                                              scan_layout,
                                              num_source_rows,
                                              check_bounds,
                                              negative_allowed,
                                              map_begin);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else if (engine_active) {
    // Pinned staging copy of the engine cohort (lifetime as pinned staging above).
    auto staging_engine_cols =
      cudf::detail::make_pinned_vector_async<size_type>(engine_cols.size(), stream);
    std::copy(engine_cols.begin(), engine_cols.end(), staging_engine_cols.begin());
    auto const d_engine_cols = carve_device_vector(
      host_span<size_type const>{staging_engine_cols}, stream, staging_carve_mr);
    auto const engine_tiles =
      static_cast<int64_t>(engine_cols.size()) *
      util::div_rounding_up_safe(static_cast<int64_t>(num_rows), static_cast<int64_t>(warp_size));
    auto const engine_grid = static_cast<unsigned>(std::min<int64_t>(
      util::div_rounding_up_safe(engine_tiles, static_cast<int64_t>(block_size / warp_size)),
      static_cast<int64_t>(65535)));
    string_fused_short_engine_kernel<block_size>
      <<<engine_grid, block_size, 0, stream.get()>>>(d_payload_items.data(),
                                                     d_engine_cols.data(),
                                                     static_cast<int64_t>(engine_cols.size()),
                                                     static_cast<int64_t>(num_rows),
                                                     num_source_rows,
                                                     check_bounds,
                                                     negative_allowed,
                                                     map_begin);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // ---- routed metadata: one launch over the routed cohort's per-column tiles ----
  // The routed items reference the SAME dst buffers as the payload items (distinct output
  // buffers stay owned by the columns; the count slot indexing rides the same live slot
  // index), so the two kernels write disjoint metadata domains. Empty (unrouted) cohort is
  // skipped: no launch, no allocation.
  if (not memcpy_live.empty()) {
    std::vector<string_work_item> routed_items(memcpy_live.size());
    for (size_t r = 0; r < memcpy_live.size(); ++r) {
      routed_items[r] = items[memcpy_live[r]];
    }
    // Pinned staging copy of the routed cohort's items (lifetime as pinned staging above).
    auto staging_routed_items =
      cudf::detail::make_pinned_vector_async<string_work_item>(routed_items.size(), stream);
    std::copy(routed_items.begin(), routed_items.end(), staging_routed_items.begin());
    auto d_routed_items = carve_device_vector(
      host_span<string_work_item const>{staging_routed_items}, stream, staging_carve_mr);
    auto const routed_rows = static_cast<int64_t>(memcpy_live.size()) * num_rows;
    auto const meta_grid   = static_cast<unsigned>(
      std::min<int64_t>(util::div_rounding_up_safe(routed_rows, static_cast<int64_t>(block_size)),
                        static_cast<int64_t>(65535)));
    string_routed_metadata_kernel<block_size>
      <<<meta_grid, block_size, 0, stream.get()>>>(d_routed_items.data(),
                                                   static_cast<size_type>(memcpy_live.size()),
                                                   num_rows,
                                                   num_source_rows,
                                                   check_bounds,
                                                   negative_allowed,
                                                   map_begin);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Deferred counts readback (b3): the D2H is only enqueued here — the
  // blocking sync moved below the assembly loops, so the host builds the
  // destination columns while the payload kernel is still running. The
  // counts' sole consumer is the per-column null count, applied after the
  // sync via set_null_count (the same post-hoc patch the descriptor path
  // uses); every other assembly input comes from the totals readback.
  // The destination must be the pinned landing buffer allocated above: a
  // pageable destination would make this copy complete synchronously at
  // enqueue, re-blocking the host on the payload kernel.
  if (any_kernel_masked) {
    cudf::detail::cuda_memcpy_async<size_type>(
      host_span<size_type>{h_string_counts.data(), static_cast<std::size_t>(live.size())},
      device_span<size_type const>{string_counts.data(), static_cast<std::size_t>(live.size())},
      stream);
  }

  // Assemble the destination columns: chars, offsets, mask, null count.
  // Non-live staged entries (empty SOURCE columns) still get a full-size
  // offsets child and no chars: every gathered row is empty. Their masks ride
  // the same plan (all-null was handled in build_string; NULLIFY/all-valid
  // arrive here kernel-masked or pre-set).
  for (size_t k = 0; k < live.size(); ++k) {
    auto& staged = staged_strings[live[k]];
    // The kernel-written mask is already correct on device; the host-visible
    // null count is patched in after the deferred readback's sync below.
    size_type null_count = staged.kernel_mask_written ? 0 : staged.null_count;
    auto null_mask       = staged.dst_mask.has_value()
                             ? cuda::device_buffer<bitmask_type>{std::move(*staged.dst_mask)}
                             : cuda::device_buffer<bitmask_type>{stream, output_mr};
    auto col             = make_strings_column(num_rows,
                                   std::move(staged.offsets),
                                   std::move(staged.chars),
                                   null_count,
                                   std::move(null_mask));
    // The count is patched in below (deferred readback); nothing to patch
    // into kernel_masked_columns (descriptor slots stay descriptor-only).
    *staged.dst_slot = std::move(col);
  }
  // The deferred counts readback is only now synchronized: everything above
  // enqueued without waiting on the payload kernel. The null count is the
  // counts' sole host consumer, so the post-hoc set_null_count patch (the
  // pattern the descriptor path already uses for fixed-width columns)
  // applies it after the fact.
  if (any_kernel_masked) {
    cudf::detail::sync_stream(stream);
    for (size_t k = 0; k < live.size(); ++k) {
      if (not staged_strings[live[k]].kernel_mask_written) { continue; }
      staged_strings[live[k]].dst_slot->get()->set_null_count(num_rows - h_string_counts[k]);
    }
  }
  // Empty-source completions: offsets only, no kernel slot (no mask work is
  // possible on a zero-row source beyond ALL_VALID/ALL_NULL pre-sets, and a
  // kernel-written mask on an empty source still needs the kernel — but the
  // scan never saw this column, so it completes host-side with its planned
  // mask).
  for (auto& staged : staged_strings) {
    if (staged.dst_slot == nullptr or *staged.dst_slot != nullptr) { continue; }
    auto offsets_col =
      cudf::strings::detail::create_offsets_child_column(0, num_rows + 1, stream, output_mr);
    // Empty rows: offsets are all zero. The child's data is uninitialized,
    // and the comparator flags a null row as dirty unless its span is zero.
    CUDF_CUDA_TRY(cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(),
                                  0,
                                  offsets_col->size() * sizeof(int32_t),
                                  stream.get()));
    auto null_mask   = staged.dst_mask.has_value()
                         ? cuda::device_buffer<bitmask_type>{std::move(*staged.dst_mask)}
                         : cuda::device_buffer<bitmask_type>{stream, output_mr};
    auto col         = make_strings_column(num_rows,
                                   std::move(offsets_col),
                                   rmm::device_buffer{},
                                   staged.null_count,
                                   std::move(null_mask));
    *staged.dst_slot = std::move(col);
  }
  staged_strings.clear();
}

}  // namespace detail
}  // namespace cudf
