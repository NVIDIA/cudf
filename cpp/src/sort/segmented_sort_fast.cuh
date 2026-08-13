/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "segmented_sort_keys.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @brief Average and total element-count bounds below which CUB `DeviceSegmentedSort` is preferred
 *
 * Benchmark-chosen; the packed-radix path takes exactly the complement of this gate.
 */
constexpr size_type MAX_AVG_LIST_SIZE_FOR_FAST_SORT{100};
constexpr size_type MAX_LIST_SIZE_FOR_FAST_SORT{1 << 18};

/**
 * @brief Average list size at or below which a no-null `DECIMAL128` column prefers CUB
 * `DeviceSegmentedSort`
 *
 * Measured band top shared by both fast paths; above it CUB loses several-fold to the comparison
 * sort.
 */
constexpr size_type DECIMAL128_CUB_MAX_AVG_LIST_SIZE{16};

/**
 * @brief Whether a single fixed-width column is small enough to prefer CUB `DeviceSegmentedSort`
 *
 * @param num_rows Total element count; the average-size disjunct divides it by `num_offsets`
 * @param num_offsets Number of segments plus one, matching the historical average-size heuristic;
 *        the caller guarantees it is nonzero
 */
inline bool prefer_cub_segmented_sort(size_type num_rows, size_type num_offsets)
{
  return (num_rows / num_offsets) < MAX_AVG_LIST_SIZE_FOR_FAST_SORT or
         num_rows < MAX_LIST_SIZE_FOR_FAST_SORT;
}

/**
 * @brief Fixed-width fast path a single key column takes within the explicit-(order, null_order) /
 * unstable envelope
 */
enum class fixed_width_sort_path {
  comparison,     ///< No fast path applies
  tiered,         ///< Register / warp / block tiered kernel
  cub_segmented,  ///< CUB `DeviceSegmentedSort` over the packed rep
  packed_radix    ///< One global packed-radix sort
};

/**
 * @brief Routes a single fixed-width key column to the fast path measured best for its shape
 *
 * Four engines picked by type x null-presence x average list size, since their costs scale
 * differently:
 * - Non-tiered numerics (narrow/unsigned ints, `bool`, `DECIMAL32`/`DECIMAL64`) -> packed radix if
 *   null-bearing or long, else CUB/comparison: the tiered key can't encode them.
 * - Null-bearing tiered types -> tiered: validity folds into the key, no separate null pass.
 * - No-null past the fast cutoff -> packed radix: bandwidth-bound, amortized by long lists.
 * - No-null `DECIMAL128` -> tiered, CUB in a sparse-large mid band, tiered above it: CUB's
 *   16-byte-pair merge tiles cap at 32 elements, so dense large segments would explode into a
 *   radix.
 * - Other no-null tiered types -> tiered (register networks make tiny-segment cost
 *   width-independent; the warp tiers beat CUB), except a small-scale int64 tiny-average pocket
 *   that stays on CUB.
 * - Outside the envelope -> comparison sort.
 *
 * @param key The fixed-width key column being routed
 * @param num_rows Element count of `key`, used to compute the average list size
 * @param segment_offsets The segment offsets; the caller guarantees non-empty
 * @param stream Used only by the `DECIMAL128` mid-band shape gate
 */
fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   rmm::cuda_stream_view stream);

/**
 * @brief Faster segmented sorted-order for a single fixed-width key column via a tiered sort
 *
 * Four size tiers whose cost tracks segment size rather than key width -- the win over packed
 * radix when segments are tiny but values are wide: one thread with a fixed Batcher network
 * (<= `TIERED_NETWORK_CAP`), a warp with `cub::WarpMergeSort` (<= `TIERED_WARP_CAP`), a block with
 * `cub::BlockMergeSort` (<= `TIERED_BLOCK_TIER_CAP`), else a packed radix over just that segment's
 * elements. `polarity` folds nulls (three-valued class flag) and descending (value complement)
 * into the key; the result matches the comparison path's order.
 *
 * @throw cudf::logic_error If `polarity.nulls_first` is requested on a column with no null mask
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_tiered(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Faster segmented sorted-order for a single fixed-width key column via one radix sort
 *
 * One non-segmented radix over a packed [segment | null class | transformed value] key whose
 * unsigned order equals the requested order (`polarity` complements the value field for descending
 * and picks the null side). The value is fully encoded, so the radix permutation is final -- no
 * tie-break, and no null post-pass since null-vs-null order is immaterial under the unstable
 * contract.
 *
 * @throw cudf::logic_error If `polarity.nulls_first` is requested on a column with no null mask
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_numeric_packed(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Faster segmented sorted-order for a single STRING key column via iterative radix sort
 *
 * The first pass radixes one `uint64` per element laid out `[segment : S bits][class : 1 bit when
 * nullable][prefix : P bits]` (see `packed_key_layout`) -- eight byte-passes, not the twelve a
 * 96-bit key needs. Elements still tied are compacted out and re-sorted by successive eight-byte
 * windows: each pass radixes a `prefix_key96` whose most-significant field is a dense run rank
 * encoding the order resolved so far, so a pass only reorders within a tied run; elements that
 * become singletons freeze at their final positions and drop out, and the loop exits once nothing
 * stays tied. Runs still tied after the pass cap go to one comparison cleanup. Unlike rank
 * encoding, distinct strings are never sorted, so per-pass cost does not grow with key
 * cardinality.
 *
 * Ordering equals the lexicographic comparison sort under the requested `polarity`:
 * - The prefix and the windows pack leading bytes big-endian, so unsigned key compares reproduce
 *   unsigned-byte order. `P` need not be a byte multiple: a packed-key match proves only
 *   `floor(P/8)` whole bytes equal, so the windows and the cleanup both begin there and the run
 *   rank is seeded from the packed keys, leaving `P`-bit ties in one run.
 * - A string with no byte left in a window packs to zero (the minimum), reproducing the
 *   shorter-is-less length tie-break; the residual all-zero-tail case is settled by the cleanup's
 *   length compare.
 * - The class bit sits above every prefix bit, so nulls land on the requested side with no
 *   sentinel prefix (no all-0xFF collision) and are position-final after the first pass.
 * - A descending sort XOR-complements only the byte fields (prefix and windows), leaving segment
 *   and null placement intact; the complement sends an exhausted window's zero to the maximum, so
 *   the length tie-break inverts to shorter-is-greater consistently. A zero-null column relaxes to
 *   nulls-last, keeping its keys bit-identical to the shipped ascending configuration.
 *
 * @throw cudf::logic_error If `polarity.nulls_first` is requested on a column with no null mask
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_prefix(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Whether every segment fits the graduated path's largest warp tile
 * (`STRINGS_GRAD_WARP_CAP`)
 *
 * Synchronizing device probe; the caller runs the cheap scalar gates first and guarantees at
 * least two offsets, so `segment_offsets.size() - 1` is a valid segment count.
 */
bool strings_grad_all_segments_fit(column_view const& segment_offsets,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Segmented sorted-order for a STRING column via graduated in-warp sorts
 *
 * One virtual warp per segment with `cub::WarpMergeSort` under a string comparator; the warp width
 * follows the segment-size band (W8/W16/W32, two items per lane except the (0,8] slice at one) so
 * a tiny segment never occupies a full warp. Every band launches over the full segment list and
 * self-filters to its size slice; the bands partition [1, 64], so every output slot is written
 * exactly once. The caller guarantees every segment fits `STRINGS_GRAD_WARP_CAP`
 * (`strings_grad_all_segments_fit`) and offsets spanning all rows; `polarity` folds order and
 * null placement into the keys as in `fast_segmented_sorted_order_strings_prefix`.
 *
 * The cap precondition is re-verified in debug builds only, so a release-build violation is an
 * unchecked out-of-bounds hazard.
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_grad(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
