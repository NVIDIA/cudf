/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort_fast.cuh"
#include "segmented_sort_keys.cuh"
#include "segmented_sort_warp_kernel.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

/// Leading string bytes packed per big-endian window (the `uint64` byte capacity).
constexpr size_type PREFIX_BYTES = sizeof(cuda::std::uint64_t);

/**
 * @brief Bit layout of the single-`uint64` first-pass radix key
 *
 * Most- to least-significant: `[segment_id : S][is_null : 1, only when nullable][prefix : P]`; an
 * unsigned compare reproduces the `prefix_key96` ordering (segment, non-nulls before nulls, packed
 * leading bytes) in eight radix byte-passes instead of twelve. `is_null` sits just above the
 * prefix, a bit no non-null prefix can reach, so an all-0xFF prefix cannot collide with a null and
 * no sentinel is needed.
 *
 * A key match proves only `floor(P/8)` *whole* equal bytes -- the partial trailing bits of `P` say
 * nothing about the next byte -- so the windows and comparison cleanup start there, not at a fixed
 * eight.
 */
struct packed_key_layout {
  int segment_bits;  // S: bit_width of the maximum segment label.
  int null_bits;     // 1 when the column has nulls, else 0.
  int prefix_bits;   // P = 64 - S - null_bits.
};

/**
 * @brief Computes the `uint64` key layout for `num_segment_labels` distinct segment-label values
 *
 * The caller labels elements with dense segment ordinals, so the bound is `num_segments` rather
 * than the row count; the tighter `S` widens `P`, packing more leading bytes per key.
 */
inline packed_key_layout make_packed_key_layout(size_type num_segment_labels, bool has_nulls)
{
  auto const max_label    = static_cast<cuda::std::uint64_t>(num_segment_labels);
  auto const segment_bits = cuda::std::bit_width(max_label);
  auto const null_bits    = has_nulls ? 1 : 0;
  return packed_key_layout{segment_bits, null_bits, 64 - segment_bits - null_bits};
}

/**
 * @brief Big-endian packs a string's eight bytes starting at `window_start` into a `uint64`
 *
 * Trailing bytes are zero-filled, so an unsigned compare reproduces unsigned-byte order over the
 * window and an exhausted string packs to zero -- the minimum -- reproducing the shorter-is-less
 * tie-break. Descending complements the window at the call site, sending that zero to the maximum
 * so the shorter string orders last instead. Nulls are never inspected here.
 */
__device__ inline cuda::std::uint64_t pack_window(string_view const& d_str, size_type window_start)
{
  auto const bytes = d_str.size_bytes();
  if (window_start >= bytes) { return 0; }
  auto const* ptr = reinterpret_cast<unsigned char const*>(d_str.data());
  // Value copy: ODR-using the PREFIX_BYTES constexpr by reference is ill-formed in device code.
  auto const window_bytes    = cuda::std::min(bytes - window_start, size_type{PREFIX_BYTES});
  cuda::std::uint64_t window = 0;
  if (window_bytes == PREFIX_BYTES) {
    // One wide load and a byte-perm endian swap replace eight dependent single-byte loads; the
    // full-width branch guarantees eight in-range bytes, so the load never overreads.
    cuda::std::uint64_t raw;
    cuda::std::memcpy(&raw, ptr + window_start, sizeof(raw));
    window = cudf::hashing::detail::swap_endian(raw);
  } else {
    for (size_type i = 0; i < window_bytes; ++i) {
      window |= static_cast<cuda::std::uint64_t>(ptr[window_start + i]) << (56 - i * 8);
    }
  }
  return window;
}

/**
 * @brief Builds the per-element single-`uint64` first-pass key (see `packed_key_layout`)
 *
 * The class bit follows `polarity.element_class`, collecting nulls on the requested side of each
 * segment; a null zeroes the prefix so all nulls in a segment share one key (order immaterial,
 * bytes never read). Descending complements only the `P` prefix bits, never the class or segment
 * fields, so one key drives every (order, null_order) combination.
 */
struct packed_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_strings;
  bool const has_nulls;
  packed_key_layout const layout;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_id   = static_cast<cuda::std::uint64_t>(d_segment_ids[idx]);
    auto const segment_part = segment_id << (64 - layout.segment_bits);
    if (has_nulls && d_strings.is_null(idx)) {
      return segment_part |
             (static_cast<cuda::std::uint64_t>(polarity.element_class(true)) << layout.prefix_bits);
    }
    auto const full_prefix = pack_window(d_strings.element<string_view>(idx), 0);
    auto const prefix      = (full_prefix >> (64 - layout.prefix_bits)) ^
                        (polarity.value_mask64() >> (64 - layout.prefix_bits));
    // Null-free layouts reserve no class bit; the OR is then provably zero: callers resolve
    // `nulls_first == false` for null-free columns, and `element_class(false) == nulls_first`.
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false)) << layout.prefix_bits) |
           prefix;
  }
};

/**
 * @brief Flags the first position of each maximal run of equal `uint64` keys (1 = head, else 0)
 *
 * The `uint64` analogue of `key_head_flag`: inclusive-summing the flags yields the dense run rank
 * seeding the window path, capturing exactly the order the first sort resolved so packed-key ties
 * stay in one run for the windows rather than frozen apart.
 */
struct key_head_flag_packed {
  cuda::std::uint64_t const* d_keys;
  __device__ cuda::std::uint32_t operator()(size_type i) const
  {
    if (i == 0) { return 1u; }
    return d_keys[i - 1] != d_keys[i] ? 1u : 0u;
  }
};

/// Shared tie test over the `uint64` first-pass keys: a null-classed position never counts as
/// tied (nulls are position-final after the first pass); otherwise tied means equal to a
/// neighbor. One definition keeps `key_tied_flag_packed` and `keep_tied_first` from drifting.
__device__ inline bool packed_key_is_tied(cuda::std::uint64_t const* d_keys,
                                          size_type num_elements,
                                          bool has_nulls,
                                          int prefix_bits,
                                          cuda::std::uint32_t null_flag,
                                          size_type i)
{
  auto const cur = d_keys[i];
  if (has_nulls && static_cast<cuda::std::uint32_t>((cur >> prefix_bits) & 1u) == null_flag) {
    return false;
  }
  auto const eq_prev = i > 0 && d_keys[i - 1] == cur;
  auto const eq_next = i + 1 < num_elements && d_keys[i + 1] == cur;
  return eq_prev || eq_next;
}

/**
 * @brief Flags `uint64`-key positions equal to a neighbor (a run of two or more), never a null
 *
 * The `uint64` analogue of `key_tied_flag`, sizing and gating the iterative tie-break. A null is
 * position-final after the first pass and its order among nulls immaterial, so it is reported
 * untied rather than dragged through every window. `null_flag` is the class-bit value marking a
 * null (`polarity.element_class(true)`); the bit sits at `prefix_bits`.
 */
struct key_tied_flag_packed {
  cuda::std::uint64_t const* d_keys;
  size_type const num_elements;
  bool const has_nulls;
  int const prefix_bits;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    return packed_key_is_tied(d_keys, num_elements, has_nulls, prefix_bits, null_flag, i);
  }
};

/**
 * @brief Per-run string-length range: the minimum and maximum byte length over a run of tied keys
 *
 * A length-uniform run fully covered by the compared windows holds only copies of one string, so
 * its order is final; a mixed-length run is a zero-extension family (a shorter string colliding
 * with a longer one's real or zero-filled bytes) the cleanup must still order shorter-first.
 */
struct len_minmax {
  size_type min_len;
  size_type max_len;
};

/// Range union; associative and commutative, as `reduce_by_key` requires.
struct len_minmax_combine {
  __device__ len_minmax operator()(len_minmax const& a, len_minmax const& b) const
  {
    return len_minmax{cuda::std::min(a.min_len, b.min_len), cuda::std::max(a.max_len, b.max_len)};
  }
};

/// Seeds each element's byte length as a degenerate range for the per-run reduction. The loop call
/// site's active set is null-free (the tie gate excludes nulls), but the first-pass site runs over
/// every element, so a null is guarded into a zero-length range here -- that slot is never read by
/// `keep_tied_first`/`keep_active_window`, which bail out on the class bit first.
struct string_length_minmax {
  size_type const* d_children;
  column_device_view const d_strings;
  bool const has_nulls;
  __device__ len_minmax operator()(size_type i) const
  {
    auto const idx = d_children[i];
    if (has_nulls && d_strings.is_null(idx)) { return len_minmax{0, 0}; }
    auto const len = d_strings.element<string_view>(idx).size_bytes();
    return len_minmax{len, len};
  }
};

/**
 * @brief First-pass tie flag refined to drop byte-identical exhausted runs
 *
 * Extends `key_tied_flag_packed`: a run whose lengths are uniform and no greater than
 * `known_equal_bytes` holds one repeated string, so every copy is position-final at its stable
 * first-pass slot and skips the loop and its O(N) buffers. A mixed-length run stays: a shorter
 * string can share a longer one's key through the zero-fill, and only the cleanup can order it.
 *
 * The drop is polarity-independent: lengths read real bytes, not the possibly-complemented key,
 * and the complement is a bijection so runs are identical sets under either order -- uniform-length
 * covered runs are duplicates whose shared slot is final under any order.
 */
struct keep_tied_first {
  cuda::std::uint64_t const* d_keys;
  size_type const num_elements;
  bool const has_nulls;
  int const prefix_bits;
  cuda::std::uint32_t const* d_run_ids;
  len_minmax const* d_run_minmax;
  size_type const known_equal_bytes;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    if (!packed_key_is_tied(d_keys, num_elements, has_nulls, prefix_bits, null_flag, i)) {
      return false;
    }
    auto const mm        = d_run_minmax[d_run_ids[i] - 1];
    auto const identical = mm.min_len == mm.max_len && mm.max_len <= known_equal_bytes;
    return !identical;
  }
};

/**
 * @brief Window-loop tie flag refined to drop byte-identical exhausted runs
 *
 * As `keep_tied_first`, with `covered` the leading bytes a run agrees on after this pass: a
 * length-uniform run within `covered` holds only copies of one string, so its stable radix order
 * is final and it is frozen like a singleton; a mixed-length run is kept for the length-aware
 * cleanup. Polarity-independent for the same reason as `keep_tied_first`.
 */
struct keep_active_window {
  prefix_key96 const* d_keys;
  size_type const num_elements;
  cuda::std::uint32_t const* d_run_ids;
  len_minmax const* d_run_minmax;
  size_type const covered;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    auto const cur = d_keys[i];
    if ((cur.seg_null & 1u) == null_flag) { return false; }
    auto const eq_prev = i > 0 && keys_equal(d_keys[i - 1], cur);
    auto const eq_next = i + 1 < num_elements && keys_equal(d_keys[i + 1], cur);
    if (!(eq_prev || eq_next)) { return false; }
    auto const mm        = d_run_minmax[d_run_ids[i] - 1];
    auto const identical = mm.min_len == mm.max_len && mm.max_len <= covered;
    return !identical;
  }
};

/**
 * @brief Builds the next-pass radix key for the element currently at a sorted position
 *
 * `seg_null` packs the run rank -- the dominant field, so the radix preserves all order resolved
 * by prior passes and reorders only within a still-tied run -- over the class bit; the window
 * words hold the next eight string bytes. Descending complements only the window, sending an
 * exhausted zero to the maximum so a shorter string sorts after longer ones sharing its prefix. A
 * null (excluded from the tie set, handled defensively) gets the class bit and a zero window.
 */
struct window_key_builder {
  cuda::std::uint32_t const* d_run_ids;
  size_type const* d_sorted_indices;
  column_device_view const d_strings;
  bool const has_nulls;
  size_type const window_start;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type i) const
  {
    auto const run_id = d_run_ids[i];
    auto const idx    = d_sorted_indices[i];
    if (has_nulls && d_strings.is_null(idx)) {
      return prefix_key96{pack_seg_null(run_id, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(run_id, polarity.element_class(false)), 0u, 0u};
    split_prefix(
      pack_window(d_strings.element<string_view>(idx), window_start) ^ polarity.value_mask64(),
      key.prefix_hi,
      key.prefix_lo);
    return key;
  }
};

/**
 * @brief Seeds the iterative window path's first-pass key from the single-`uint64` first sort
 *
 * The seed's run boundaries must match exactly what the first sort resolved and nothing finer, so
 * `seg_null` packs the dense run rank and the window words stay zero: embedding the first window
 * here would split packed-key ties into distinct ranks, freezing an arbitrary order the windows
 * must still be free to refine. The null flag is carried for consistency; the tie set excludes
 * nulls.
 */
struct tied_run_seed_builder {
  cuda::std::uint32_t const* d_run_ids;
  cuda::std::uint64_t const* d_packed_keys;
  int const prefix_bits;
  bool const has_nulls;
  __device__ prefix_key96 operator()(size_type i) const
  {
    auto const is_null =
      has_nulls ? static_cast<cuda::std::uint32_t>((d_packed_keys[i] >> prefix_bits) & 1u) : 0u;
    return prefix_key96{pack_seg_null(d_run_ids[i], is_null), 0u, 0u};
  }
};

/**
 * @brief Unsigned-byte comparison of two strings starting at a shared byte offset
 *
 * Reproduces `string_view::compare` exactly for any pair whose first `offset` bytes match -- that
 * compare would consume those equal bytes with no effect. On an all-equal common region the
 * full-length difference carries the sign of the length tie-break, including when a string is no
 * longer than `offset` (e.g. a string vs itself plus a trailing embedded null).
 */
__device__ inline int compare_suffix(string_view const& a, string_view const& b, size_type offset)
{
  auto const a_len = cuda::std::max(0, a.size_bytes() - offset);
  auto const b_len = cuda::std::max(0, b.size_bytes() - offset);
  auto const* pa =
    reinterpret_cast<unsigned char const*>(a.data()) + cuda::std::min(offset, a.size_bytes());
  auto const* pb =
    reinterpret_cast<unsigned char const*>(b.data()) + cuda::std::min(offset, b.size_bytes());
  auto const common = cuda::std::min(a_len, b_len);
  for (size_type i = 0; i < common; ++i) {
    if (pa[i] != pb[i]) { return static_cast<int>(pa[i]) - static_cast<int>(pb[i]); }
  }
  return a.size_bytes() - b.size_bytes();
}

/**
 * @brief Run length at or above which a prefix-tie run is sorted by heapsort instead of insertion
 *
 * Insertion sort is optimal for the typical tiny runs but O(run^2): a fully-shared-prefix segment
 * collapses to one run spanning it, turning the tie-break into a multi-second straggler. Heapsort
 * caps that worst case and leaves the common tiny-run path untouched.
 */
constexpr size_type TIE_HEAPSORT_THRESHOLD = 32;

/**
 * @brief Maximum number of iterative eight-byte radix windows after the initial prefix pass
 *
 * Bounds the worst case: an arbitrarily long shared prefix would otherwise demand one pass per
 * eight shared bytes. Whatever stays tied past the cap is finished by the comparison cleanup, so
 * correctness never depends on the cap, only the pass count does.
 */
constexpr size_type MAX_RADIX_PASSES = 8;

/**
 * @brief Reorders prefix-tied runs by suffix comparison to match the comparison sort exactly
 *
 * Each maximal run of equal keys is owned by its first position, which sorts the run's child
 * indices by unsigned bytes from `tie_offset` (the proven-shared prefix width) onward. Genuine
 * duplicates compare equal, so the path being unstable is immaterial. Tiny runs use insertion
 * sort; runs reaching `TIE_HEAPSORT_THRESHOLD` use heapsort so a shared-prefix run spanning a
 * whole segment cannot go quadratic.
 *
 * Descending inverts the comparison's sign, reversing both byte order and the length tie-break to
 * match the descending window keys' exhausted-zero-to-maximum complement. A null-classed run holds
 * only nulls, is never compared, and is left in place -- and none reach here, the tie gate
 * excludes them.
 */
struct prefix_tie_breaker {
  prefix_key96 const* d_keys;
  size_type* d_indices;
  column_device_view const d_strings;
  size_type const num_elements;
  size_type const tie_offset;
  bool const descending               = false;
  cuda::std::uint32_t const null_flag = 1u;

  // Strict "orders before" in the requested direction; descending inverts the suffix compare.
  __device__ bool index_less(size_type a, size_type b) const
  {
    auto const cmp = compare_suffix(
      d_strings.element<string_view>(a), d_strings.element<string_view>(b), tie_offset);
    return descending ? cmp > 0 : cmp < 0;
  }

  // Restores the max-heap property below `root` over the `n`-element run at `base`.
  __device__ void sift_down(size_type* base, int64_t root, int64_t n) const
  {
    while (true) {
      auto const left = 2 * root + 1;
      if (left >= n) { break; }
      auto largest     = left;
      auto const right = left + 1;
      if (right < n && index_less(base[left], base[right])) { largest = right; }
      if (!index_less(base[root], base[largest])) { break; }
      auto const tmp = base[root];
      base[root]     = base[largest];
      base[largest]  = tmp;
      root           = largest;
    }
  }

  // Insertion-sorts the run's indices [i, run_end) by suffix; quadratic but branch-cheap, used
  // only below TIE_HEAPSORT_THRESHOLD.
  __device__ void sort_run_insertion(size_type i, size_type run_end) const
  {
    for (auto j = i + 1; j < run_end; ++j) {
      auto const idx_j = d_indices[j];
      auto const str_j = d_strings.element<string_view>(idx_j);
      auto k           = j;
      while (k > i) {
        auto const cmp =
          compare_suffix(d_strings.element<string_view>(d_indices[k - 1]), str_j, tie_offset);
        // Shift a neighbor that must follow `str_j`: greater ascending, smaller descending.
        if (!(descending ? cmp < 0 : cmp > 0)) { break; }
        d_indices[k] = d_indices[k - 1];
        --k;
      }
      d_indices[k] = idx_j;
    }
  }

  // Heapsorts the run's indices [i, i+run_len) by suffix; guarantees O(n log n) so a
  // whole-segment shared-prefix run cannot go quadratic.
  __device__ void sort_run_heap(size_type i, size_type run_len) const
  {
    auto* const base = d_indices + i;
    for (auto start = run_len / 2; start > 0; --start) {
      sift_down(base, start - 1, run_len);
    }
    for (auto heap_size = run_len; heap_size > 1; --heap_size) {
      auto const tmp      = base[0];
      base[0]             = base[heap_size - 1];
      base[heap_size - 1] = tmp;
      sift_down(base, 0, heap_size - 1);
    }
  }

  __device__ void operator()(size_type i) const
  {
    // Only the first position of a run does the work; runs are disjoint so writes never overlap.
    auto const key = d_keys[i];
    if (i > 0 && keys_equal(d_keys[i - 1], key)) { return; }
    // A null-classed run's string data must not be read; skip it.
    if ((key.seg_null & 1u) == null_flag) { return; }

    auto run_end = i + 1;
    while (run_end < num_elements && keys_equal(d_keys[run_end], key)) {
      ++run_end;
    }
    auto const run_len = run_end - i;
    if (run_len < 2) { return; }

    if (run_len < TIE_HEAPSORT_THRESHOLD) {
      sort_run_insertion(i, run_end);
    } else {
      sort_run_heap(i, run_len);
    }
  }
};

// ==========================================================================================
// Graduated-warp per-segment sort for a single STRING key column.
//
// When every segment fits `STRINGS_GRAD_WARP_CAP`, sorting each segment in a virtual warp with
// `cub::WarpMergeSort` beats the global prefix-radix machine. The warp width graduates with the
// segment-size band (W8 <= 16, W16 17-32, W32 33-64) so a tiny segment never occupies a full warp.
// The 8-byte comparator key drives the bottom bands, where the 16-byte prekey's merge-exchange
// volume dominates over mostly-pad tiles; the prekey drives W16/W32, where its packed prefix
// window amortizes. The (0,8] slice sorts one item per lane, halving pad traffic. A column with
// any segment above the cap falls through to the prefix path.
// ==========================================================================================

/// Largest segment size the graduated-warp string path admits: the W32 x 2 register tile.
constexpr size_type STRINGS_GRAD_WARP_CAP = 64;

/**
 * @brief Ordering key for the comparator-key bands: the element's global index plus its class
 *
 * The comparator reads string bytes through `gidx`, keeping the key at eight bytes. `cls` collects
 * nulls on the requested side and pad strictly above both classes, as the fixed-width tiers
 * require. `gidx` is dereferenced only for the valid class, so a null or pad key never reads
 * string data.
 */
struct strings_grad_cmp_key {
  size_type gidx;
  cuda::std::uint32_t cls;
};
static_assert(sizeof(strings_grad_cmp_key) == 8, "strings_grad_cmp_key must stay eight bytes");

struct strings_grad_cmp_key_builder {
  column_device_view d_strings;
  bool has_nulls;
  sort_polarity polarity;
  __device__ strings_grad_cmp_key operator()(size_type idx) const
  {
    return strings_grad_cmp_key{idx, polarity.element_class(has_nulls && d_strings.is_null(idx))};
  }
};

/**
 * @brief Strict-weak less for the comparator key: class first, then the full byte comparison
 *
 * Same-class null or pad keys compare equivalent (unstable path, bytes never read). The valid
 * class ordinal follows the polarity -- `element_class(false)` is 1 under nulls-first, so a
 * hardcoded constant would misfire. Descending inverts the comparison's sign, reversing byte order
 * and the length tie-break.
 */
struct strings_grad_cmp_less {
  column_device_view d_strings;
  sort_polarity polarity;
  __device__ bool operator()(strings_grad_cmp_key const& a, strings_grad_cmp_key const& b) const
  {
    if (a.cls != b.cls) { return a.cls < b.cls; }
    if (a.cls != polarity.element_class(false)) { return false; }
    auto const cmp =
      d_strings.element<string_view>(a.gidx).compare(d_strings.element<string_view>(b.gidx));
    return polarity.descending ? cmp > 0 : cmp < 0;
  }
};

/**
 * @brief Ordering key for the prekey bands: a packed leading-byte prefix over the index
 *
 * `prefix` is `pack_window` at offset zero, complemented at build time when descending. A prefix
 * tie leaves two cases -- the strings genuinely share their first eight bytes, or a shorter
 * string's zero-fill collided with real zero bytes (`S` vs `S + "\0"`) -- and `compare_suffix`
 * from byte eight orders both: its byte walk the first, its full-length difference the second.
 */
struct strings_grad_prekey {
  cuda::std::uint64_t prefix;
  size_type gidx;
  cuda::std::uint32_t cls;
};
static_assert(sizeof(strings_grad_prekey) == 16 and alignof(strings_grad_prekey) == 8,
              "strings_grad_prekey must stay sixteen bytes with eight-byte alignment");

/// A null carries a zero prefix and is never inspected past the class compare.
struct strings_grad_prekey_builder {
  column_device_view d_strings;
  bool has_nulls;
  sort_polarity polarity;
  __device__ strings_grad_prekey operator()(size_type idx) const
  {
    if (has_nulls && d_strings.is_null(idx)) {
      return strings_grad_prekey{0, idx, polarity.element_class(true)};
    }
    return strings_grad_prekey{
      pack_window(d_strings.element<string_view>(idx), 0) ^ polarity.value_mask64(),
      idx,
      polarity.element_class(false)};
  }
};

/**
 * @brief Strict-weak less for the prekey: class, then packed prefix, bytes only on a tie
 *
 * The prefix was complemented at build time under descending, so its plain unsigned compare is
 * direction-correct. The byte comparison starts at `PREFIX_BYTES` -- the tie proves the window
 * equal -- and descending inverts its sign. The valid class follows the polarity.
 */
struct strings_grad_prekey_less {
  column_device_view d_strings;
  sort_polarity polarity;
  __device__ bool operator()(strings_grad_prekey const& a, strings_grad_prekey const& b) const
  {
    if (a.cls != b.cls) { return a.cls < b.cls; }
    if (a.cls != polarity.element_class(false)) { return false; }
    if (a.prefix != b.prefix) { return a.prefix < b.prefix; }
    auto const cmp = compare_suffix(d_strings.element<string_view>(a.gidx),
                                    d_strings.element<string_view>(b.gidx),
                                    size_type{PREFIX_BYTES});
    return polarity.descending ? cmp > 0 : cmp < 0;
  }
};

/// Launches one graduated band: `W`-lane virtual warps sort segments sized in `(band_lo, band_hi]`,
/// self-filtering over the full segment list as the fixed-width warp bands do.
template <int W,
          int IPT,
          typename KeyT,
          typename KeyBuilder,
          typename CompareOp,
          typename SegListIt>
void launch_strings_grad_band(KeyBuilder const& build_key,
                              CompareOp const& compare_op,
                              KeyT const pad_key,
                              size_type const* d_offsets,
                              SegListIt d_seg_list,
                              size_type num_segments,
                              size_type band_lo,
                              size_type band_hi,
                              size_type* d_out,
                              rmm::cuda_stream_view stream)
{
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_segments) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT, KeyBuilder, W, IPT, TIERED_BLOCK_THREADS, CompareOp>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_offsets, d_seg_list, num_segments, band_lo, band_hi, build_key, d_out, compare_op, pad_key);
  CUDF_CHECK_CUDA(stream.value());
}

/// In-place inclusive rank scan (two-call CUB idiom), shared by every tie-refinement site in
/// `fast_segmented_sorted_order_strings_prefix` -- the packed-key pass, the window-key seed, and
/// the per-pass loop body all scan a freshly built run-head-flag array the same way.
inline void inclusive_sum_scan(cuda::std::uint32_t* d_data,
                               size_type count,
                               rmm::cuda_stream_view stream)
{
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
    d_temp_storage.data(), temp_storage_bytes, d_data, d_data, count, stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceScan::InclusiveSum(
    d_temp_storage.data(), temp_storage_bytes, d_data, d_data, count, stream.value());
}

/**
 * @brief Owns the tie-resolution loop's per-pass working buffers and its initial (pointer, count)
 * state
 *
 * Returned by `setup_tie_resolution_loop` and consumed by the loop that follows it inline in
 * `fast_segmented_sorted_order_strings_prefix`: the loop itself stays a plain `for` over ordinary
 * local variables destructured from this struct, never a struct-taking function of its own.
 */
struct tie_loop_state {
  rmm::device_uvector<prefix_key96> tied_keys_a;
  rmm::device_uvector<prefix_key96> tied_keys_b;
  rmm::device_uvector<size_type> child_a;
  rmm::device_uvector<size_type> child_b;
  rmm::device_uvector<size_type> comp_pos;
  rmm::device_uvector<size_type> pos_b;
  rmm::device_uvector<cuda::std::uint32_t> run_ids;
  rmm::device_uvector<len_minmax> elem_minmax;
  rmm::device_uvector<len_minmax> run_minmax;
  cudf::detail::device_scalar<size_type> d_num_tied;
  size_type num_active;
  size_type consumed_bytes;
};

/**
 * @brief Compacts the tied subset, seeds its rank-only run keys, and allocates the tie-resolution
 * loop's per-pass buffers
 *
 * `tied_keys_packed` is local to this function -- unlike the original single-function version, no
 * explicit early release is needed, since it goes out of scope (and its device memory is freed,
 * stream-ordered) when this function returns, strictly before the loop it feeds ever runs.
 */
inline tie_loop_state setup_tie_resolution_loop(cuda::std::uint64_t const* keys_out,
                                                bool const* tied_flags,
                                                size_type const* d_indices_out,
                                                cuda::counting_iterator<size_type> counting,
                                                size_type num_elements,
                                                size_type num_tied,
                                                size_type known_equal_bytes,
                                                bool has_nulls,
                                                int prefix_bits,
                                                rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> comp_pos(num_tied, stream);
  rmm::device_uvector<size_type> child_a(num_tied, stream);
  rmm::device_uvector<cuda::std::uint64_t> tied_keys_packed(num_tied, stream);
  rmm::device_uvector<prefix_key96> tied_keys_a(num_tied, stream);
  cudf::detail::device_scalar<size_type> d_num_tied(stream);

  // The tie count is already known, so `d_num_tied` here only satisfies the API; it is re-read
  // per pass inside the loop.
  cub_select_flagged(
    counting, tied_flags, comp_pos.data(), d_num_tied.data(), num_elements, stream);
  cub_select_flagged(
    d_indices_out, tied_flags, child_a.data(), d_num_tied.data(), num_elements, stream);
  cub_select_flagged(
    keys_out, tied_flags, tied_keys_packed.data(), d_num_tied.data(), num_elements, stream);

  rmm::device_uvector<size_type> child_b(num_tied, stream);
  rmm::device_uvector<prefix_key96> tied_keys_b(num_tied, stream);
  rmm::device_uvector<size_type> pos_b(num_tied, stream);
  rmm::device_uvector<cuda::std::uint32_t> run_ids(num_tied, stream);
  rmm::device_uvector<len_minmax> elem_minmax(num_tied, stream);
  rmm::device_uvector<len_minmax> run_minmax(num_tied, stream);

  // Seed `tied_keys_a` with rank-only keys reproducing exactly the runs the uint64 sort left
  // tied; embedding a window here would over-split them (see `tied_run_seed_builder`).
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    counting,
                    counting + num_tied,
                    run_ids.begin(),
                    key_head_flag_packed{tied_keys_packed.data()});
  inclusive_sum_scan(run_ids.data(), num_tied, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    counting,
    counting + num_tied,
    tied_keys_a.begin(),
    tied_run_seed_builder{run_ids.data(), tied_keys_packed.data(), prefix_bits, has_nulls});

  return tie_loop_state{cuda::std::move(tied_keys_a),
                        cuda::std::move(tied_keys_b),
                        cuda::std::move(child_a),
                        cuda::std::move(child_b),
                        cuda::std::move(comp_pos),
                        cuda::std::move(pos_b),
                        cuda::std::move(run_ids),
                        cuda::std::move(elem_minmax),
                        cuda::std::move(run_minmax),
                        cuda::std::move(d_num_tied),
                        num_tied,
                        known_equal_bytes};
}

/**
 * @brief Resolves any pass-cap survivors by direct comparison, then scatters the tie-resolution
 * loop's final child order back to its output slots
 */
inline void finish_tie_resolution(prefix_key96 const* cur_keys,
                                  size_type* cur_child,
                                  size_type const* cur_pos,
                                  size_type num_active,
                                  size_type tie_offset,
                                  column_device_view const& d_input,
                                  sort_polarity polarity,
                                  size_type* d_indices_out,
                                  rmm::cuda_stream_view stream)
{
  auto const counting = cuda::counting_iterator<size_type>{0};
  // Comparison cleanup for runs still tied after the windows: those past the pass cap plus the
  // mixed-length zero-extension families the drop kept -- pure length ties no window can
  // separate but `compare_suffix`'s length difference does. Survivors agree on their first
  // `consumed_bytes` bytes, so the comparison skips them.
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   counting,
                   counting + num_active,
                   prefix_tie_breaker{cur_keys,
                                      cur_child,
                                      d_input,
                                      num_active,
                                      tie_offset,
                                      polarity.descending,
                                      polarity.element_class(true)});

  // Scatter the survivors back to their output slots; everything else is already final.
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  cur_child,
                  cur_child + num_active,
                  cur_pos,
                  d_indices_out);
}

}  // namespace

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_prefix(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;

  CUDF_EXPECTS(input.has_nulls() or not polarity.nulls_first,
               "nulls_first requires a nullable column for the packed key layout");
  CUDF_EXPECTS(num_segments >= 1, "the packed key layout requires at least one segment");

  // Label every element with its dense segment ordinal so one global radix sort orders within each
  // segment. Ordinals are bounded by the segment count -- unlike `get_segment_indices`'s
  // offset-derived labels (row-count bound) -- so the key's segment field needs only
  // `bit_width(num_segments)` bits, widening the prefix; they stay monotonic with the segments,
  // preserving cross-segment order. Offsets are normalized to span `[0, num_elements]`.
  rmm::device_uvector<size_type> segment_ids(num_elements, stream);
  label_segments(segment_offsets.begin<size_type>(),
                 segment_offsets.end<size_type>(),
                 segment_ids.begin(),
                 segment_ids.end(),
                 stream);

  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();

  auto const layout = make_packed_key_layout(num_segments, has_nulls);
  // Only floor(P/8) whole bytes are proven equal on a key match; windows and cleanup start here.
  auto const known_equal_bytes = layout.prefix_bits / 8;

  auto const counting = cuda::counting_iterator<size_type>{0};

  // First-pass outputs; the packed-key inputs are scoped to the sort block below and freed early.
  rmm::device_uvector<cuda::std::uint64_t> keys_out(num_elements, stream);
  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);
  auto const d_indices_out = sorted_indices->mutable_view().begin<size_type>();

  // The window path keys on the 96-bit `prefix_key96` {run rank, eight-byte window}. `seg_null`
  // packs the rank in bits 1..31 over the null flag; the rank is bounded by the element count (a
  // `size_type`), so `(rank << 1) | flag` fits 32 bits -- proven once here, not guarded per
  // element.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type run rank does not fit prefix_key96::seg_null after the null-flag shift");
  auto const decomposer   = prefix_decomposer{};
  auto constexpr key_bits = static_cast<int>(
    (sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t)) * 8);

  // First pass: one global radix sort of the packed key with child indices as the paired values.
  // The whole word is significant, so the bit range is the full [0, 64). Inputs and temporary
  // storage are scoped here, released before the tie loop.
  {
    rmm::device_uvector<cuda::std::uint64_t> keys_in(num_elements, stream);
    rmm::device_uvector<size_type> indices_in(num_elements, stream);
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      counting,
      counting + num_elements,
      keys_in.begin(),
      packed_key_builder{segment_ids.data(), *d_input, has_nulls, layout, polarity});
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices_in.begin(),
                     indices_in.end(),
                     0);
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    indices_in.data(),
                                    d_indices_out,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    keys_in.data(),
                                    keys_out.data(),
                                    indices_in.data(),
                                    d_indices_out,
                                    num_elements,
                                    0,
                                    static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
                                    stream.value());
  }

  // `segment_ids` fed only the first-pass key build; free it before the tie path grows.
  segment_ids = rmm::device_uvector<size_type>{0, stream};

  // Any-tied gate; reading the result is the tie-break's one host synchronization. On
  // high-cardinality data nothing is tied, so the common path is the first pass plus this probe --
  // no tie buffers, compaction, loop, or cleanup.
  auto const tied_pred = key_tied_flag_packed{
    keys_out.data(), num_elements, has_nulls, layout.prefix_bits, polarity.element_class(true)};
  auto const any_tied =
    thrust::any_of(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   counting,
                   counting + num_elements,
                   tied_pred);

  if (any_tied) {
    // Refine the raw packed-key ties by dropping byte-identical runs (see `keep_tied_first`). The
    // per-run length reduction runs only on this already-gated tie path, and its N-wide
    // temporaries are freed before the loop.
    rmm::device_uvector<bool> tied_flags(num_elements, stream);
    {
      rmm::device_uvector<cuda::std::uint32_t> first_run_ids(num_elements, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        first_run_ids.begin(),
                        key_head_flag_packed{keys_out.data()});
      inclusive_sum_scan(first_run_ids.data(), num_elements, stream);
      rmm::device_uvector<len_minmax> first_elem_minmax(num_elements, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        first_elem_minmax.begin(),
                        string_length_minmax{d_indices_out, *d_input, has_nulls});
      rmm::device_uvector<len_minmax> first_run_minmax(num_elements, stream);
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        first_run_ids.begin(),
        first_run_ids.end(),
        first_elem_minmax.begin(),
        cuda::make_discard_iterator(),
        first_run_minmax.begin(),
        cuda::std::equal_to<cuda::std::uint32_t>{},
        len_minmax_combine{});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_elements,
                        tied_flags.begin(),
                        keep_tied_first{keys_out.data(),
                                        num_elements,
                                        has_nulls,
                                        layout.prefix_bits,
                                        first_run_ids.data(),
                                        first_run_minmax.data(),
                                        known_equal_bytes,
                                        polarity.element_class(true)});
    }

    // Compact down to the still-tied positions -- everything untied is already final -- so the
    // windows and cleanup re-sort only this shrinking subset. `comp_pos` records each tied
    // element's output slot so the refined order scatters straight back.
    auto const num_tied = static_cast<size_type>(
      thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    tied_flags.begin(),
                    tied_flags.end(),
                    true));

    auto loop_state = setup_tie_resolution_loop(keys_out.data(),
                                                tied_flags.data(),
                                                d_indices_out,
                                                counting,
                                                num_elements,
                                                num_tied,
                                                known_equal_bytes,
                                                has_nulls,
                                                layout.prefix_bits,
                                                stream);
    // `keys_out` is fully captured; free it before the loop's working set peaks.
    keys_out = rmm::device_uvector<cuda::std::uint64_t>{0, stream};

    // Iterative deepening over the compacted tied set: each pass computes a dense run rank from
    // the current keys, builds the next key as {run rank, next 8-byte window}, and radixes just
    // the tied subset. The rank is the dominant field, so a pass fixes all prior order and
    // reorders only within still-tied runs; windows start at `known_equal_bytes` and advance eight
    // bytes per pass.
    //
    // A singleton owns a unique run rank -- no later window can move it -- so resolved elements
    // scatter to their final slots and drop from the working set; the loop stops once nothing
    // stays tied.
    //
    // `cur_pos` (the still-tied output slots) stays ascending: slots pair with the sorted children
    // by position and are only compacted, never permuted -- permuting them by the sort would
    // misassign slots, so the radix reorders only keys and children.
    auto* cur_keys      = loop_state.tied_keys_a.data();
    auto* nxt_keys      = loop_state.tied_keys_b.data();
    auto* cur_child     = loop_state.child_a.data();
    auto* nxt_child     = loop_state.child_b.data();
    auto* cur_pos       = loop_state.comp_pos.data();
    auto* nxt_pos       = loop_state.pos_b.data();
    auto& run_ids       = loop_state.run_ids;
    auto& elem_minmax   = loop_state.elem_minmax;
    auto& run_minmax    = loop_state.run_minmax;
    auto& d_num_tied    = loop_state.d_num_tied;
    auto consumed_bytes = loop_state.consumed_bytes;
    auto num_active     = loop_state.num_active;
    for (size_type pass = 0; pass < MAX_RADIX_PASSES && num_active > 0; ++pass) {
      // Dense one-based run rank over the active subset: head flags, then an inclusive sum.
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        run_ids.begin(),
                        key_head_flag{cur_keys});
      inclusive_sum_scan(run_ids.data(), num_active, stream);

      // Build the next key {run rank, next eight bytes} and radix the active subset; all 96 bits
      // are significant, so the bit range is the full [0, key_bits).
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        counting,
        counting + num_active,
        nxt_keys,
        window_key_builder{
          run_ids.data(), cur_child, *d_input, has_nulls, consumed_bytes, polarity});
      {
        rmm::device_buffer d_temp_storage;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        nxt_keys,
                                        cur_keys,
                                        cur_child,
                                        nxt_child,
                                        num_active,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
        d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        temp_storage_bytes,
                                        nxt_keys,
                                        cur_keys,
                                        cur_child,
                                        nxt_child,
                                        num_active,
                                        decomposer,
                                        0,
                                        key_bits,
                                        stream.value());
      }
      // The radix wrote sorted keys to `cur_keys` but sorted children to `nxt_child`; swap so
      // `cur_child` names the sorted indices.
      cuda::std::swap(cur_child, nxt_child);

      // Flag who stays tied under the refreshed order: singletons and byte-identical runs drop
      // (see `keep_active_window`); `covered` is `consumed_bytes` plus this pass's window.
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        elem_minmax.begin(),
                        string_length_minmax{cur_child, *d_input, has_nulls});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        run_ids.begin(),
                        key_head_flag{cur_keys});
      inclusive_sum_scan(run_ids.data(), num_active, stream);
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        run_ids.begin(),
        run_ids.begin() + num_active,
        elem_minmax.begin(),
        cuda::make_discard_iterator(),
        run_minmax.begin(),
        cuda::std::equal_to<cuda::std::uint32_t>{},
        len_minmax_combine{});
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        counting,
                        counting + num_active,
                        tied_flags.begin(),
                        keep_active_window{cur_keys,
                                           num_active,
                                           run_ids.data(),
                                           run_minmax.data(),
                                           consumed_bytes + PREFIX_BYTES,
                                           polarity.element_class(true)});

      // Freeze resolved elements: scatter each untied child index to its final output slot.
      thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cur_child,
                         cur_child + num_active,
                         cur_pos,
                         tied_flags.begin(),
                         d_indices_out,
                         cuda::std::logical_not<bool>{});

      // Compact keys, children, and slots to the still-tied remainder. Reading the surviving count
      // is the one host sync per pass; a device-side predicate could remove it, but the pass count
      // is tiny so it stays simple.
      cub_select_flagged(
        cur_keys, tied_flags.data(), nxt_keys, d_num_tied.data(), num_active, stream);
      cub_select_flagged(
        cur_child, tied_flags.data(), nxt_child, d_num_tied.data(), num_active, stream);
      cub_select_flagged(
        cur_pos, tied_flags.data(), nxt_pos, d_num_tied.data(), num_active, stream);
      cuda::std::swap(cur_keys, nxt_keys);
      cuda::std::swap(cur_child, nxt_child);
      cuda::std::swap(cur_pos, nxt_pos);
      num_active = d_num_tied.value(stream);
      consumed_bytes += PREFIX_BYTES;
    }

    finish_tie_resolution(cur_keys,
                          cur_child,
                          cur_pos,
                          num_active,
                          consumed_bytes,
                          *d_input,
                          polarity,
                          d_indices_out,
                          stream);
  }

  return sorted_indices;
}

bool strings_grad_all_segments_fit(column_view const& segment_offsets, rmm::cuda_stream_view stream)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();
  auto const oversized =
    thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{num_segments},
                     segment_exceeds_size{d_offsets, STRINGS_GRAD_WARP_CAP});
  return oversized == 0;
}

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_grad(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_input      = column_device_view::create(input, stream);
  auto const has_nulls    = input.has_nulls();
  auto const d_offsets    = segment_offsets.begin<size_type>();

#ifndef NDEBUG
  // The caller's admission gate already ran this synchronizing probe; re-check in debug builds
  // only, so a future direct caller cannot slip an oversized segment past the warp-tile cap into
  // an out-of-bounds gather.
  CUDF_EXPECTS(strings_grad_all_segments_fit(segment_offsets, stream),
               "every segment must fit STRINGS_GRAD_WARP_CAP for the graduated path");
#endif

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);
  auto* const d_out = sorted_indices->mutable_view().begin<size_type>();

  // The band kernel walks an explicit segment list; here it is the identity, expressed as a
  // counting iterator so no device list is allocated or filled.
  auto const seg_list = cuda::counting_iterator<size_type>{0};

  // Pad stays `tier_pad`, ranking strictly above either element class under any polarity.
  auto const cmp_build = strings_grad_cmp_key_builder{*d_input, has_nulls, polarity};
  auto const cmp_less  = strings_grad_cmp_less{*d_input, polarity};
  auto const cmp_pad =
    strings_grad_cmp_key{0, static_cast<cuda::std::uint32_t>(tiered_element_class::tier_pad)};
  auto const pre_build = strings_grad_prekey_builder{*d_input, has_nulls, polarity};
  auto const pre_less  = strings_grad_prekey_less{*d_input, polarity};
  auto const pre_pad =
    strings_grad_prekey{0, 0, static_cast<cuda::std::uint32_t>(tiered_element_class::tier_pad)};

  launch_strings_grad_band<8, 1>(
    cmp_build, cmp_less, cmp_pad, d_offsets, seg_list, num_segments, 0, 8, d_out, stream);
  launch_strings_grad_band<8, 2>(
    cmp_build, cmp_less, cmp_pad, d_offsets, seg_list, num_segments, 8, 16, d_out, stream);
  launch_strings_grad_band<16, 2>(
    pre_build, pre_less, pre_pad, d_offsets, seg_list, num_segments, 16, 32, d_out, stream);
  launch_strings_grad_band<32, 2>(pre_build,
                                  pre_less,
                                  pre_pad,
                                  d_offsets,
                                  seg_list,
                                  num_segments,
                                  32,
                                  STRINGS_GRAD_WARP_CAP,
                                  d_out,
                                  stream);
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
