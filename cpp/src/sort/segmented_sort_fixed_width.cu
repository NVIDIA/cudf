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
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <utility>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Maps a fixed-width value to the unsigned key whose ascending order equals value order
 *
 * Signed integrals flip the sign bit; chrono types encode their integer rep; floating point takes
 * the order-preserving IEEE-754 flip with every NaN mapped to all-ones (NaNs tie, sorting after
 * +Inf). The transform runs at the input width; the caller's zero-extension adds equal high bits
 * that cannot reorder. Non-integral branches resolve first: `make_unsigned_t` is ill-formed for
 * floating-point and chrono types.
 */
template <typename T>
__device__ inline cuda::std::uint32_t radix_encode_u32(T value)
{
  if constexpr (cuda::std::is_same_v<T, bool>) {
    return value ? cuda::std::uint32_t{1} : cuda::std::uint32_t{0};
  } else if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u32(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u32(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint32_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint32_t>(value);
    auto constexpr sign_mask = cuda::std::uint32_t{1} << (sizeof(cuda::std::uint32_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint32_t>(encoded);
  }
}

/**
 * @brief Eight-byte analogue of `radix_encode_u32`: the same transforms at 64 bits
 */
template <typename T>
__device__ inline cuda::std::uint64_t radix_encode_u64(T value)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u64(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u64(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint64_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint64_t>(value);
    auto constexpr sign_mask = cuda::std::uint64_t{1} << (sizeof(cuda::std::uint64_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint64_t>(encoded);
  }
}

/**
 * @brief 128-bit analogue for the `DECIMAL128` rep (`__int128_t`): always signed, so the sign bit
 * flips unconditionally
 */
template <typename T>
__device__ inline unsigned __int128 radix_encode_u128(T value)
{
  auto encoded = static_cast<unsigned __int128>(value);
  encoded ^= (static_cast<unsigned __int128>(1) << 127);
  return encoded;
}

/**
 * @brief Builds the single-`uint64` packed key for a fixed-width element of width four bytes or
 * less
 *
 * Layout, most- to least-significant: `[segment : S bits][class : 1 bit][value : 32 bits]` with
 * `S = bit_width(num_segments)`, so an unsigned compare orders by segment, then the polarity's
 * null placement, then the radix-encoded value (complemented within its field when descending).
 * The class bit sits above the value field and (since `S <= 31`) below the segment field, so a
 * null -- value zero, unread -- sorts on its configured side of every valid element; the gap bits
 * are zero for all elements.
 */
template <typename T>
struct numeric_packed_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_part = static_cast<cuda::std::uint64_t>(d_segment_ids[idx])
                              << (64 - segment_bits);
    if (has_nulls && d_input.is_null(idx)) {
      return segment_part | (static_cast<cuda::std::uint64_t>(polarity.element_class(true))
                             << (sizeof(cuda::std::uint32_t) * 8));
    }
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false))
            << (sizeof(cuda::std::uint32_t) * 8)) |
           static_cast<cuda::std::uint64_t>(radix_encode_u32<T>(d_input.element<T>(idx)) ^
                                            polarity.value_mask32());
  }
};

/**
 * @brief Builds the `prefix_key96` packed key for an eight-byte fixed-width element
 *
 * Reuses the strings path's key: `seg_null` packs the segment ordinal and the polarity's class bit
 * (via `pack_seg_null`), so the segment dominates and a null sorts on its configured side; the
 * window words carry the encoded 64-bit value split hi/lo -- complemented when descending, which
 * distributes over the split. A null leaves the value words zero.
 */
template <typename T>
struct numeric_packed_key_builder64 {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    split_prefix(radix_encode_u64<T>(d_input.element<T>(idx)) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

// A DECIMAL128 segment set sorts by the narrowest lossless radix key its value range admits,
// rather than an unconditional twenty-pass 160-bit key: a min-biased `uint64` (range < 2^32, eight
// passes), a `prefix_key96` (fits int64, twelve passes), or a two-phase hi64-then-lo64 sort.
// Shared by the full-column and compact-large-segment DECIMAL128 radix sites.

/**
 * @brief Inclusive value range [min, max] over the `__int128` elements a radix site is about to
 * sort
 *
 * Drives the key-width gate. Nulls contribute the identity: placed by key class alone, their
 * values are never compared.
 */
struct dec128_value_range {
  __int128_t min_val;
  __int128_t max_val;
};

// Identity: min = largest, max = smallest `__int128`; a null-only/empty set leaves min > max.
__host__ __device__ inline dec128_value_range dec128_value_range_identity()
{
  auto constexpr i128_max = static_cast<__int128_t>((static_cast<unsigned __int128>(1) << 127) - 1);
  return dec128_value_range{i128_max, static_cast<__int128_t>(-i128_max - 1)};
}

/// Min of minima, max of maxima; associative and commutative as `transform_reduce` requires.
struct dec128_value_range_combine {
  __device__ dec128_value_range operator()(dec128_value_range const& a,
                                           dec128_value_range const& b) const
  {
    return dec128_value_range{a.min_val < b.min_val ? a.min_val : b.min_val,
                              a.max_val > b.max_val ? a.max_val : b.max_val};
  }
};

/// Maps a global element index to its single-value range; a null contributes the identity
struct dec128_value_range_fn {
  column_device_view d_input;
  bool has_nulls;
  __device__ dec128_value_range operator()(size_type idx) const
  {
    if (has_nulls && d_input.is_null(idx)) { return dec128_value_range_identity(); }
    auto const value = d_input.element<__int128_t>(idx);
    return dec128_value_range{value, value};
  }
};

/**
 * @brief Builds the single-`uint64` gate key for a DECIMAL128 element when the column-wide value
 * range spans fewer than 2^32 (the min-biased path)
 *
 * `numeric_packed_key_builder`'s layout with the value `v - min_val`: monotonic in `v` and inside
 * 32 bits, so the same segment/class/value order at eight byte-passes instead of twenty.
 */
struct dec128_biased_u64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  __int128_t const min_val;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_part = static_cast<cuda::std::uint64_t>(d_segment_ids[idx])
                              << (64 - segment_bits);
    if (has_nulls && d_input.is_null(idx)) {
      return segment_part | (static_cast<cuda::std::uint64_t>(polarity.element_class(true))
                             << (sizeof(cuda::std::uint32_t) * 8));
    }
    auto const biased = static_cast<cuda::std::uint32_t>(
      static_cast<unsigned __int128>(d_input.element<__int128_t>(idx)) -
      static_cast<unsigned __int128>(min_val));
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false))
            << (sizeof(cuda::std::uint32_t) * 8)) |
           static_cast<cuda::std::uint64_t>(biased ^ polarity.value_mask32());
  }
};

/**
 * @brief Builds the `prefix_key96` gate key for a DECIMAL128 element whose value fits `int64`
 *
 * `numeric_packed_key_builder64`'s layout with the rep narrowed to `int64` -- lossless, the caller
 * guarantees representability -- at twelve byte-passes instead of twenty.
 */
struct dec128_int64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    split_prefix(radix_encode_u64<int64_t>(static_cast<int64_t>(d_input.element<__int128_t>(idx))) ^
                   polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Phase-one key of the two-phase DECIMAL128 sort: `prefix_key96` {seg_null, high 64 bits}
 *
 * The value words carry the high 64 bits of the sign-flipped encoding, complemented when
 * descending -- equal high words stay equal, so the phase-two tie set is polarity-independent. A
 * null sets its class bit and leaves the words zero.
 */
struct dec128_hi64_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    auto const encoded = radix_encode_u128<__int128_t>(d_input.element<__int128_t>(idx));
    split_prefix(static_cast<cuda::std::uint64_t>(encoded >> 64) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Phase-two key of the two-phase DECIMAL128 sort: `prefix_key96` {run rank, low 64 bits}
 *
 * Applied to the (all non-null) elements still tied after phase one. The dense one-based run rank
 * keeps distinct (segment, hi64) runs apart; the low word -- untouched by the bit-127 flip,
 * complemented when descending -- orders within a run. `d_child` maps each tied slot to its
 * element's global index.
 */
struct dec128_lo64_key_builder {
  cuda::std::uint32_t const* d_run_ids;
  size_type const* d_child;
  column_device_view const d_input;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type i) const
  {
    prefix_key96 key{pack_seg_null(d_run_ids[i], 0u), 0u, 0u};
    auto const value = static_cast<unsigned __int128>(d_input.element<__int128_t>(d_child[i]));
    split_prefix(static_cast<cuda::std::uint64_t>(value) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Run-head flag over the tied subset, reading the sorted phase-one keys through `d_pos`
 *
 * The indirection spares materializing the compacted keys -- a dominant peak-memory term when most
 * elements tie.
 */
struct gathered_key_head_flag {
  prefix_key96 const* d_keys;
  size_type const* d_pos;
  __device__ cuda::std::uint32_t operator()(size_type i) const
  {
    if (i == 0) { return 1u; }
    return keys_equal(d_keys[d_pos[i - 1]], d_keys[d_pos[i]]) ? 0u : 1u;
  }
};

/**
 * @brief Common per-column sort parameters shared by the packed-radix and DECIMAL128 engines
 *
 * Bundles the column, its null/segment/polarity metadata, and the stream so the four sort entry
 * points that operate on a whole segmented column share one parameter list instead of repeating
 * the same six arguments. `tiered_sort_fn` is intentionally excluded: it derives
 * `d_segment_ids`/`segment_bits` lazily, only inside its radix-spill branch, and forcing that work
 * eager for struct uniformity would change when it happens on the common path.
 */
struct segment_sort_context {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  sort_polarity const polarity;
  rmm::cuda_stream_view const stream;
};

/**
 * @brief Two-phase DECIMAL128 sort: order by the high 64 value bits, then resolve exact hi64 ties
 * by the low 64 bits
 *
 * `d_order[j]` receives the global index of the j-th smallest element. Correct because the
 * sign-flipped 128-bit encoding compares lexicographically hi64 then lo64: phase one orders every
 * element whose (segment, hi64) differs, and only exact ties -- nulls excluded, being
 * position-final -- are re-sorted by {dense run rank, lo64} and scattered back into their
 * (ascending) slots. Each phase trims the radix end bit to its leading field's significant width.
 *
 * Tie density can reach every element, so the working set is held to the live minimum: both radix
 * sorts run ping-pong in caller-owned `cub::DoubleBuffer`s (CUB then keeps no key/value copies in
 * its temp storage), phase-one scratch is released before phase two allocates, and the tied keys
 * are read through the compacted positions instead of re-materialized.
 */
inline void dec128_two_phase_sort(segment_sort_context const& ctx,
                                  size_type const* global_indices,
                                  size_type num_elements,
                                  size_type* d_order)
{
  // Locals preserve the body's original parameter names; only the call boundary changed.
  auto const& d_input      = ctx.d_input;
  auto const d_segment_ids = ctx.d_segment_ids;
  auto const has_nulls     = ctx.has_nulls;
  auto const polarity      = ctx.polarity;
  auto const segment_bits  = ctx.segment_bits;
  auto const stream        = ctx.stream;
  // seg_null packs (rank << 1) | flag, so the shift must fit a uint32.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type run rank does not fit prefix_key96::seg_null after the shift");
  auto const alloc      = cudf::get_current_device_resource_ref();
  auto const counting   = cuda::counting_iterator<size_type>{0};
  auto const decomposer = prefix_decomposer{};
  // seg_null needs only segment_bits + 1 bits; the constant-zero high bits skip radix passes.
  auto const phase1_end = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                         64 + cuda::std::min(32, segment_bits + 1));

  // Phase one: sort by (segment, null, hi64), ping-pong in `cub::DoubleBuffer`s. Either half of a
  // pair may hold the result, so the sorted halves are routed back to `k1` / `d_order` afterward.
  rmm::device_uvector<prefix_key96> k1(num_elements, stream);
  {
    rmm::device_uvector<prefix_key96> keys_in(num_elements, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                      global_indices,
                      global_indices + num_elements,
                      keys_in.begin(),
                      dec128_hi64_key_builder{d_segment_ids, d_input, has_nulls, polarity});
    // The sort scribbles on both value halves and the radix-tier caller reads `global_indices`
    // again as its scatter map, so the indices are staged into `d_order` + a scratch alternate.
    rmm::device_uvector<size_type> vals_alt(num_elements, stream);
    thrust::copy(rmm::exec_policy_nosync(stream, alloc),
                 global_indices,
                 global_indices + num_elements,
                 d_order);
    cub::DoubleBuffer<prefix_key96> d_keys(keys_in.data(), k1.data());
    cub::DoubleBuffer<size_type> d_values(d_order, vals_alt.data());
    {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      d_keys,
                                      d_values,
                                      num_elements,
                                      decomposer,
                                      0,
                                      phase1_end,
                                      stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      d_keys,
                                      d_values,
                                      num_elements,
                                      decomposer,
                                      0,
                                      phase1_end,
                                      stream.value());
    }
    // Swap so `k1` owns the current key buffer (the stale half dies with this scope); copy the
    // values back only if they landed in the scratch half.
    if (d_keys.Current() != k1.data()) { std::swap(k1, keys_in); }
    if (d_values.Current() != d_order) {
      thrust::copy(rmm::exec_policy_nosync(stream, alloc),
                   d_values.Current(),
                   d_values.Current() + num_elements,
                   d_order);
    }
  }

  // Zero-tie gate: on distinct-hi64 data nothing ties and phase one is the order.
  auto const any_tied =
    thrust::count_if(rmm::exec_policy_nosync(stream, alloc),
                     counting,
                     counting + num_elements,
                     key_tied_flag{k1.data(), num_elements, polarity.element_class(true)}) > 0;
  if (not any_tied) { return; }

  // Compact the tied slots' output positions and global indices; the keys stay uncompacted and
  // are read through `comp_pos` instead.
  rmm::device_uvector<bool> tied_flags(num_elements, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_elements,
                    tied_flags.begin(),
                    key_tied_flag{k1.data(), num_elements, polarity.element_class(true)});
  auto const num_tied = static_cast<size_type>(thrust::count(
    rmm::exec_policy_nosync(stream, alloc), tied_flags.begin(), tied_flags.end(), true));

  rmm::device_uvector<size_type> comp_pos(num_tied, stream);
  rmm::device_uvector<size_type> child(num_tied, stream);
  cudf::detail::device_scalar<size_type> d_num_tied(stream);
  cub_select_flagged(
    counting, tied_flags.data(), comp_pos.data(), d_num_tied.data(), num_elements, stream);
  cub_select_flagged(
    d_order, tied_flags.data(), child.data(), d_num_tied.data(), num_elements, stream);
  // Release phase-one-only scratch before the phase-two buffers stack on it: on tie-dense data
  // the live buffers, not any single sort, set the peak.
  tied_flags.resize(0, stream);
  tied_flags.shrink_to_fit(stream);

  // Dense one-based run rank over the tied subset: one rank per (segment, hi64) run.
  rmm::device_uvector<cuda::std::uint32_t> run_ids(num_tied, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_tied,
                    run_ids.begin(),
                    gathered_key_head_flag{k1.data(), comp_pos.data()});
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  run_ids.data(),
                                  run_ids.data(),
                                  num_tied,
                                  stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  run_ids.data(),
                                  run_ids.data(),
                                  num_tied,
                                  stream.value());
  }
  // The sorted keys are dead once the run heads exist; release before phase two allocates.
  k1.resize(0, stream);
  k1.shrink_to_fit(stream);

  // Phase two: sort the tied subset by {run rank, lo64}; the end bit trims to the rank field's
  // bit_width(num_tied) + 1 significant bits.
  auto const run_field_bits = cuda::std::min(
    32, static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_tied))) + 1);
  auto const phase2_end =
    cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8), 64 + run_field_bits);
  rmm::device_uvector<prefix_key96> p2_in(num_tied, stream);
  rmm::device_uvector<prefix_key96> p2_out(num_tied, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    counting,
                    counting + num_tied,
                    p2_in.begin(),
                    dec128_lo64_key_builder{run_ids.data(), child.data(), d_input, polarity});
  // The ranks are baked into the keys; release before the alternate value buffer allocates.
  run_ids.resize(0, stream);
  run_ids.shrink_to_fit(stream);
  // Ping-pong again: either value half may end up current, so the scatter reads Current().
  rmm::device_uvector<size_type> child_sorted(num_tied, stream);
  cub::DoubleBuffer<prefix_key96> p2_keys(p2_in.data(), p2_out.data());
  cub::DoubleBuffer<size_type> p2_values(child.data(), child_sorted.data());
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    p2_keys,
                                    p2_values,
                                    num_tied,
                                    decomposer,
                                    0,
                                    phase2_end,
                                    stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                    temp_storage_bytes,
                                    p2_keys,
                                    p2_values,
                                    num_tied,
                                    decomposer,
                                    0,
                                    phase2_end,
                                    stream.value());
  }
  auto const* d_refined = p2_values.Current();
  thrust::scatter(rmm::exec_policy_nosync(stream, alloc),
                  d_refined,
                  d_refined + num_tied,
                  comp_pos.begin(),
                  d_order);
}

/**
 * @brief Builds a radix key for each input index via `build_key`, then two-call
 * `cub::DeviceRadixSort::SortPairs` orders the paired index array by it
 *
 * Shared by every packed-radix / DECIMAL128 branch that builds a key column then radix-sorts an
 * index array by it: only the key type, key builder, optional decomposer, and end bit differ per
 * call site. This overload covers a directly radix-sortable unsigned integer key (no decomposer);
 * the sibling overload below covers a struct key (`prefix_key96`) CUB needs help decomposing.
 */
template <typename KeyT, typename IndexIteratorT, typename KeyBuilder>
void build_and_radix_sort_packed_keys(IndexIteratorT d_indices_in,
                                      size_type count,
                                      KeyBuilder const& build_key,
                                      int end_bit,
                                      size_type const* d_values_in,
                                      size_type* d_values_out,
                                      rmm::cuda_stream_view stream)
{
  auto const alloc = cudf::get_current_device_resource_ref();
  rmm::device_uvector<KeyT> keys_in(count, stream);
  rmm::device_uvector<KeyT> keys_out(count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    d_indices_in,
                    d_indices_in + count,
                    keys_in.begin(),
                    build_key);
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  0,
                                  end_bit,
                                  stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  0,
                                  end_bit,
                                  stream.value());
}

/// Decomposer overload of `build_and_radix_sort_packed_keys` for a struct key (e.g.
/// `prefix_key96`).
template <typename KeyT, typename IndexIteratorT, typename KeyBuilder, typename DecomposerT>
void build_and_radix_sort_packed_keys(IndexIteratorT d_indices_in,
                                      size_type count,
                                      KeyBuilder const& build_key,
                                      DecomposerT const& decomposer,
                                      int end_bit,
                                      size_type const* d_values_in,
                                      size_type* d_values_out,
                                      rmm::cuda_stream_view stream)
{
  auto const alloc = cudf::get_current_device_resource_ref();
  rmm::device_uvector<KeyT> keys_in(count, stream);
  rmm::device_uvector<KeyT> keys_out(count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    d_indices_in,
                    d_indices_in + count,
                    keys_in.begin(),
                    build_key);
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  decomposer,
                                  0,
                                  end_bit,
                                  stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  decomposer,
                                  0,
                                  end_bit,
                                  stream.value());
}

/**
 * @brief Sorts a DECIMAL128 element set within its segments, picking the narrowest lossless radix
 * key from the value range
 *
 * `global_indices` doubles as the paired sort value, so `d_order[j]` receives the global index of
 * the j-th smallest element. One range reduction (nulls skipped, one D->H sync) picks the key:
 * range < 2^32 -> min-biased `uint64` (eight passes), fits `int64` -> `prefix_key96` (twelve),
 * else the two-phase hi64-then-lo64 sort -- never the twenty-pass 160-bit key.
 */
inline void dec128_segmented_radix_sort(segment_sort_context const& ctx,
                                        size_type const* global_indices,
                                        size_type num_elements,
                                        size_type* d_order)
{
  // Locals preserve the body's original parameter names; only the call boundary changed.
  auto const& d_input      = ctx.d_input;
  auto const d_segment_ids = ctx.d_segment_ids;
  auto const has_nulls     = ctx.has_nulls;
  auto const polarity      = ctx.polarity;
  auto const segment_bits  = ctx.segment_bits;
  auto const stream        = ctx.stream;
  // seg_null packs (segment << 1) | flag, so the shift must fit a uint32.
  static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                  cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                "size_type segment label does not fit the packed key's seg_null after the shift");
  auto const alloc = cudf::get_current_device_resource_ref();

  auto const range = thrust::transform_reduce(rmm::exec_policy_nosync(stream, alloc),
                                              global_indices,
                                              global_indices + num_elements,
                                              dec128_value_range_fn{d_input, has_nulls},
                                              dec128_value_range_identity(),
                                              dec128_value_range_combine{});

  enum class key_width { u64_biased, key96, twophase };
  auto width         = key_width::twophase;
  __int128_t min_val = 0;
  if (range.max_val < range.min_val) {
    // Null-only set: the order is width-independent, so take the cheapest.
    width = key_width::u64_biased;
  } else {
    auto const span =
      static_cast<unsigned __int128>(range.max_val) - static_cast<unsigned __int128>(range.min_val);
    auto constexpr i64_min = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::min());
    auto constexpr i64_max = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::max());
    if ((span >> 32) == 0) {
      width   = key_width::u64_biased;
      min_val = range.min_val;
    } else if (range.min_val >= i64_min and range.max_val <= i64_max) {
      width = key_width::key96;
    } else {
      width = key_width::twophase;
    }
  }

  if (width == key_width::u64_biased) {
    // Min-biased uint64 key, eight byte-passes; the segment rides the high bits, so the end bit
    // cannot be tightened.
    build_and_radix_sort_packed_keys<cuda::std::uint64_t>(
      global_indices,
      num_elements,
      dec128_biased_u64_key_builder{
        d_segment_ids, d_input, has_nulls, segment_bits, min_val, polarity},
      static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
      global_indices,
      d_order,
      stream);
  } else if (width == key_width::key96) {
    // Value fits int64: prefix_key96, twelve byte-passes, end bit trimmed to seg_null's width.
    auto const key96_end = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                          64 + cuda::std::min(32, segment_bits + 1));
    build_and_radix_sort_packed_keys<prefix_key96>(
      global_indices,
      num_elements,
      dec128_int64_key_builder{d_segment_ids, d_input, has_nulls, polarity},
      prefix_decomposer{},
      key96_end,
      global_indices,
      d_order,
      stream);
  } else {
    // Genuine 128-bit range: two-phase hi64-then-lo64.
    dec128_two_phase_sort(ctx, global_indices, num_elements, d_order);
  }
}

/**
 * @brief Sorts a single fixed-width key column within its segments by one global radix sort
 *
 * The whole value is encoded into the key, so the radix order is final: elements of four bytes or
 * less use one `uint64` key, eight-byte elements the `prefix_key96`, and the `DECIMAL128` rep the
 * narrowest key its value range admits. The paired sort value is the element index, so the sorted
 * values are the segmented sorted order. The failing overload covers only the non-fixed-width
 * types, which the caller's gate never dispatches.
 */
struct numeric_packed_sort_fn {
  // Compile-time counterpart of `is_numeric_packed_radix_supported` after `dispatch_storage_type`
  // mapping: the two must accept exactly the same types, so widen them together.
  template <typename T>
  static constexpr bool is_supported()
  {
    return cudf::is_integral<T>() or cudf::is_floating_point<T>() or cudf::is_chrono<T>() or
           cuda::std::is_same_v<__int128, T>;
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(segment_sort_context const& ctx,
                  size_type num_elements,
                  size_type* d_indices_out) const
  {
    // Locals preserve the body's original parameter names; only the call boundary changed.
    auto const& d_input      = ctx.d_input;
    auto const d_segment_ids = ctx.d_segment_ids;
    auto const has_nulls     = ctx.has_nulls;
    auto const polarity      = ctx.polarity;
    auto const segment_bits  = ctx.segment_bits;
    auto const stream        = ctx.stream;
    auto const counting      = cuda::counting_iterator<size_type>{0};
    rmm::device_uvector<size_type> indices_in(num_elements, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices_in.begin(),
                     indices_in.end(),
                     0);

    if constexpr (sizeof(T) <= 4) {
      // S = bit_width(num_segments) <= 31, so S + 1 + 32 <= 64: the three fields always fit.
      static_assert(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(
                      cuda::std::numeric_limits<size_type>::max())) +
                        1 + 32 <=
                      64,
                    "packed numeric key fields exceed the 64-bit key for some segment count");
      // The segment rides the key's high bits, so the end bit cannot be tightened.
      build_and_radix_sort_packed_keys<cuda::std::uint64_t>(
        counting,
        num_elements,
        numeric_packed_key_builder<T>{d_segment_ids, d_input, has_nulls, segment_bits, polarity},
        static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
        indices_in.data(),
        d_indices_out,
        stream);
    } else if constexpr (sizeof(T) == 8) {
      // seg_null packs (segment << 1) | flag, so the shift must fit a uint32.
      static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                      cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                    "size_type segment label does not fit prefix_key96::seg_null after the shift");
      // seg_null needs only segment_bits + 1 bits; the constant-zero high bits skip radix passes.
      auto const key_bits = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                           64 + cuda::std::min(32, segment_bits + 1));
      build_and_radix_sort_packed_keys<prefix_key96>(
        counting,
        num_elements,
        numeric_packed_key_builder64<T>{d_segment_ids, d_input, has_nulls, polarity},
        prefix_decomposer{},
        key_bits,
        indices_in.data(),
        d_indices_out,
        stream);
    } else {
      // DECIMAL128: the narrowest lossless key from the value range; `indices_in` is the key-build
      // source and the paired sort value.
      dec128_segmented_radix_sort(ctx, indices_in.data(), num_elements, d_indices_out);
    }
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(segment_sort_context const&, size_type, size_type*) const
  {
    CUDF_FAIL("Column type cannot be used with the numeric packed-radix segmented sort");
  }
};

/**
 * @brief Run-time predicate for the numeric packed-radix fast path: every fixed-width type the
 * packed key encodes losslessly
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the packed-radix fast path
 */
inline bool is_numeric_packed_radix_supported(data_type type)
{
  return cudf::is_integral(type) or cudf::is_fixed_point(type) or cudf::is_floating_point(type) or
         cudf::is_chrono(type);
}

// ==========================================================================================
// Tiered per-segment sort for a single fixed-width key column.
//
// The packed-radix path's fixed per-element cost (a global radix over a key as wide as the value)
// dominates when segments are tiny. This path instead classifies segments by size into four
// tiers whose cost tracks the segment size: up to `TIERED_NETWORK_CAP` a single thread with a
// register sorting network, up to `TIERED_WARP_CAP` a full warp with `cub::WarpMergeSort`, up to
// `TIERED_BLOCK_TIER_CAP` one block with `cub::BlockMergeSort`, and rare larger outliers a packed
// radix over just their elements. Nulls sort via a class flag folded into the key; descending
// complements the key's value bits.
// ==========================================================================================

/**
 * @brief Ordering key for the DECIMAL128 tiered path: a class flag over the hi/lo-split 128-bit
 * value
 *
 * `flag` dominates (the polarity's classes below the pad); within the valid class the sign-flipped
 * value words -- complemented when descending -- give unsigned-compare == value order. 24 bytes,
 * the width the warp tier's register / shared budget is sized against; a null or pad leaves the
 * value words zero.
 */
struct tiered_key128 {
  cuda::std::uint64_t hi;
  cuda::std::uint64_t lo;
  cuda::std::uint32_t flag;
  __device__ bool operator<(tiered_key128 const& o) const
  {
    if (flag != o.flag) { return flag < o.flag; }
    if (hi != o.hi) { return hi < o.hi; }
    return lo < o.lo;
  }
};
static_assert(sizeof(tiered_key128) == 24 and alignof(tiered_key128) == 8,
              "tiered_key128 must stay 24 bytes with 8-byte alignment");

/**
 * @brief Maps a tiered storage type to its packed ordering key by value width
 *
 * The class flag rides the high bits, so a native compare orders valid < null < pad and then by
 * the `radix_encode_*` value: four bytes or fewer -> `uint64` (flag in bits 32-33), eight ->
 * `unsigned __int128` (bits 64-65), sixteen -> the 24-byte `tiered_key128`.
 */
template <typename T>
using tiered_key_t = cuda::std::conditional_t<
  sizeof(T) <= 4,
  cuda::std::uint64_t,
  cuda::std::conditional_t<sizeof(T) == 8, unsigned __int128, tiered_key128>>;

/**
 * @brief The pad key for type `T`: class flag `tier_pad`, value words zero
 *
 * Ordered strictly after every valid element and null, so the sorts keep pads beyond a segment's
 * valid-item boundary. Host-usable because the launch code builds it on the host.
 */
template <typename T>
__host__ __device__ inline tiered_key_t<T> tiered_pad_key()
{
  using KeyT = tiered_key_t<T>;
  if constexpr (cuda::std::is_same_v<KeyT, tiered_key128>) {
    return tiered_key128{0, 0, static_cast<cuda::std::uint32_t>(tiered_element_class::tier_pad)};
  } else if constexpr (cuda::std::is_same_v<KeyT, unsigned __int128>) {
    return static_cast<unsigned __int128>(tiered_element_class::tier_pad) << 64;
  } else {
    return static_cast<cuda::std::uint64_t>(tiered_element_class::tier_pad) << 32;
  }
}

/// Strict-weak less-than over any tiered key: native `<` or `tiered_key128::operator<`.
struct tiered_key_less {
  template <typename KeyT>
  __device__ bool operator()(KeyT const& a, KeyT const& b) const
  {
    return a < b;
  }
};

/**
 * @brief Builds the packed tiered key for one element: class flag then radix-encoded value
 *
 * A null (value zero, unread) sorts on the polarity's side regardless of values. A valid value is
 * complemented within the value bits when descending -- per-word for `tiered_key128`, where the
 * hi-then-lo compare equals the 128-bit unsigned compare -- never reaching the class flag.
 */
template <typename T>
struct tiered_key_builder {
  column_device_view d_input;
  bool has_nulls;
  sort_polarity polarity;
  __device__ tiered_key_t<T> operator()(size_type idx) const
  {
    using KeyT     = tiered_key_t<T>;
    bool const nul = has_nulls && d_input.is_null(idx);
    auto const cls = polarity.element_class(nul);
    if constexpr (cuda::std::is_same_v<KeyT, tiered_key128>) {
      if (nul) { return tiered_key128{0, 0, cls}; }
      auto const encoded = radix_encode_u128<T>(d_input.element<T>(idx));
      auto const mask    = polarity.value_mask64();
      return tiered_key128{static_cast<cuda::std::uint64_t>(encoded >> 64) ^ mask,
                           static_cast<cuda::std::uint64_t>(encoded) ^ mask,
                           cls};
    } else if constexpr (cuda::std::is_same_v<KeyT, unsigned __int128>) {
      if (nul) { return static_cast<unsigned __int128>(cls) << 64; }
      return (static_cast<unsigned __int128>(cls) << 64) |
             static_cast<unsigned __int128>(radix_encode_u64<T>(d_input.element<T>(idx)) ^
                                            polarity.value_mask64());
    } else {
      if (nul) { return static_cast<cuda::std::uint64_t>(cls) << 32; }
      return (static_cast<cuda::std::uint64_t>(cls) << 32) |
             static_cast<cuda::std::uint64_t>(radix_encode_u32<T>(d_input.element<T>(idx)) ^
                                              polarity.value_mask32());
    }
  }
};

// Network tier: one thread sorts a whole segment of <= `TIERED_NETWORK_CAP` elements in registers
// with a fixed Batcher network. The 19-comparator eight-key network sorts all 2^8 binary inputs,
// so by the zero-one principle it sorts any totally-ordered keys; even at the widest (24-byte)
// key, eight keys + indices fit in registers without spilling.
constexpr size_type TIERED_NETWORK_CAP = 8;
// The kernel hardcodes the eight-slot network: raising the cap would silently mis-sort, lowering
// it would over-index. Regenerate the network before changing the cap.
static_assert(TIERED_NETWORK_CAP == 8,
              "tiered_network_sort_kernel hardcodes the eight-key Batcher network");
// Warp tier: a full 32-lane warp per segment, `cub::WarpMergeSort` at two items per lane. The
// 64-slot tile stays within register / shared budget even at the widest key, so the cap is
// uniform across key widths.
constexpr int TIERED_WARP_LANES     = 32;
constexpr int TIERED_WARP_ITEMS     = 2;
constexpr size_type TIERED_WARP_CAP = TIERED_WARP_LANES * TIERED_WARP_ITEMS;
// Block tier: one 128-thread block per segment with `cub::BlockMergeSort`, in graduated
// items-per-thread bands so a segment just over the warp cap pays a 128-slot tile rather than a
// 1024-slot one. Without this rung mid-band list shapes spilled to the radix tier and regressed
// below CUB `DeviceSegmentedSort`. The largest band's shared tile is ~25KB at the widest key,
// inside the 48KB static budget.
constexpr size_type TIERED_BLOCK_TIER_CAP = 1'024;

/// Selects segments whose size is <= `cap` (the network tier) for `cub::DevicePartition::If`
struct segment_in_network_tier {
  size_type const* d_offsets;
  size_type cap;
  __device__ bool operator()(size_type seg) const
  {
    return (d_offsets[seg + 1] - d_offsets[seg]) <= cap;
  }
};

/**
 * @brief Selects segments whose size is in `(network_cap, warp_cap]` (the warp tier)
 *
 * The explicit lower bound keeps the predicate correct whether or not `DevicePartition::If`
 * evaluates it on items the first selector already selected.
 */
struct segment_in_warp_tier {
  size_type const* d_offsets;
  size_type network_cap;
  size_type warp_cap;
  __device__ bool operator()(size_type seg) const
  {
    auto const sz = d_offsets[seg + 1] - d_offsets[seg];
    return sz > network_cap && sz <= warp_cap;
  }
};

/**
 * @brief Selects segments whose size is <= `block_cap` (the block tier) from the large-segment
 * list
 *
 * Every entry already exceeds the warp cap, so only the upper bound needs testing; the rest stay
 * radix-spill outliers.
 */
struct segment_in_block_tier {
  size_type const* d_offsets;
  size_type block_cap;
  __device__ bool operator()(size_type seg) const
  {
    return (d_offsets[seg + 1] - d_offsets[seg]) <= block_cap;
  }
};

/**
 * @brief Selects elements whose segment is in the radix tier (size > `cap`)
 *
 * Scanning ascending element indices compacts them grouped by segment then position -- the
 * arrangement `radix_sort_large_segments` relies on for its scatter.
 */
struct element_in_radix_tier {
  size_type const* d_segment_ids;
  size_type const* d_offsets;
  size_type cap;
  __device__ bool operator()(size_type i) const
  {
    auto const seg = d_segment_ids[i];
    return (d_offsets[seg + 1] - d_offsets[seg]) > cap;
  }
};

/// One compare-exchange: orders `keys[a] <= keys[b]` under `cmp`, carrying the paired values.
template <typename KeyT, typename CompareOp>
__device__ inline void network_compare_exchange(
  KeyT* keys, size_type* vals, int a, int b, CompareOp cmp)
{
  if (cmp(keys[b], keys[a])) {
    KeyT const tk      = keys[a];
    keys[a]            = keys[b];
    keys[b]            = tk;
    size_type const tv = vals[a];
    vals[a]            = vals[b];
    vals[b]            = tv;
  }
}

/**
 * @brief Sorts one segment per thread with the fixed eight-key Batcher network
 *
 * Thread `t` loads segment `d_seg_list[t]`'s keys and global indices into registers, pads the
 * unused slots (pads sort last and are never written back), applies the network, and writes the
 * sorted indices back to the segment's slots. Register-only, so a segment's cost is independent
 * of its neighbours'.
 */
template <typename T, int BLOCK_THREADS, typename CompareOp>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_network_sort_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  bool has_nulls,
  sort_polarity polarity,
  size_type* d_out,
  CompareOp compare_op,
  tiered_key_t<T> pad_key)
{
  using KeyT     = tiered_key_t<T>;
  auto const tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  if (tid >= static_cast<thread_index_type>(num_class_segments)) { return; }

  auto const seg       = d_seg_list[tid];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;

  tiered_key_builder<T> const build_key{d_input, has_nulls, polarity};
  KeyT keys[TIERED_NETWORK_CAP];
  size_type vals[TIERED_NETWORK_CAP];
#pragma unroll
  for (int i = 0; i < TIERED_NETWORK_CAP; ++i) {
    if (i < seg_size) {
      auto const gidx = seg_start + i;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  // Eight-key Batcher network (19 compare-exchanges); pads settle above every real element.
  network_compare_exchange(keys, vals, 0, 1, compare_op);
  network_compare_exchange(keys, vals, 2, 3, compare_op);
  network_compare_exchange(keys, vals, 4, 5, compare_op);
  network_compare_exchange(keys, vals, 6, 7, compare_op);
  network_compare_exchange(keys, vals, 0, 2, compare_op);
  network_compare_exchange(keys, vals, 1, 3, compare_op);
  network_compare_exchange(keys, vals, 4, 6, compare_op);
  network_compare_exchange(keys, vals, 5, 7, compare_op);
  network_compare_exchange(keys, vals, 1, 2, compare_op);
  network_compare_exchange(keys, vals, 5, 6, compare_op);
  network_compare_exchange(keys, vals, 0, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 7, compare_op);
  network_compare_exchange(keys, vals, 1, 5, compare_op);
  network_compare_exchange(keys, vals, 2, 6, compare_op);
  network_compare_exchange(keys, vals, 1, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 6, compare_op);
  network_compare_exchange(keys, vals, 2, 4, compare_op);
  network_compare_exchange(keys, vals, 3, 5, compare_op);
  network_compare_exchange(keys, vals, 3, 4, compare_op);

#pragma unroll
  for (int i = 0; i < TIERED_NETWORK_CAP; ++i) {
    if (i < seg_size) { d_out[seg_start + i] = vals[i]; }
  }
}

/**
 * @brief Sorts one segment per virtual warp with `cub::WarpMergeSort` under a null-aware
 * comparator
 *
 * Lane `l` holds items `[l*IPT, l*IPT+IPT)` in blocked order, padded past the segment size; the
 * pad key plus `valid_items = seg_size` keep the real elements in `[0, seg_size)`, so only those
 * slots are written back (global indices, straight into the output gather map). `W*IPT` >= the
 * class's maximum segment size is guaranteed by the caps the caller partitions on.
 */
template <typename T, int W, int IPT, int BLOCK_THREADS, typename CompareOp>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_warp_sort_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  bool has_nulls,
  sort_polarity polarity,
  size_type* d_out,
  CompareOp compare_op,
  tiered_key_t<T> pad_key)
{
  using KeyT                     = tiered_key_t<T>;
  using WarpMergeSortT           = cub::WarpMergeSort<KeyT, IPT, W, size_type>;
  constexpr int VWARPS_PER_BLOCK = BLOCK_THREADS / W;
  __shared__ typename WarpMergeSortT::TempStorage temp_storage[VWARPS_PER_BLOCK];

  auto const global_tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  auto const vwarp_id   = static_cast<size_type>(global_tid / W);  // the class segment to sort
  auto const lane       = static_cast<int>(threadIdx.x % W);  // logical lane in the virtual warp
  auto const vwarp_slot = static_cast<int>(threadIdx.x / W);  // virtual warp's shared-mem slot
  // Uniform across the virtual warp's lanes, so the whole warp returns together and never
  // desynchronizes the `__syncwarp` inside `Sort`.
  if (vwarp_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[vwarp_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;

  tiered_key_builder<T> const build_key{d_input, has_nulls, polarity};
  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  WarpMergeSortT(temp_storage[vwarp_slot])
    .Sort(keys, vals, compare_op, static_cast<int>(seg_size), pad_key);

#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

/**
 * @brief Sorts one segment per thread block with `cub::BlockMergeSort` over a segment-size band
 *
 * A block whose segment size falls outside `(band_lo, band_hi]` returns before sorting, so
 * graduated `IPT` bands share one list without re-partitioning. `seg_size` is uniform across the
 * block, so every arriving thread reaches the `__syncthreads` inside `Sort`. Blocked layout with
 * pads past the segment size, as in the warp kernel; only `[0, seg_size)` is written back.
 */
template <typename T, int BLOCK_THREADS, int IPT, typename CompareOp>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_block_band_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  size_type band_lo,
  size_type band_hi,
  bool has_nulls,
  sort_polarity polarity,
  size_type* d_out,
  CompareOp compare_op,
  tiered_key_t<T> pad_key)
{
  using KeyT            = tiered_key_t<T>;
  using BlockMergeSortT = cub::BlockMergeSort<KeyT, BLOCK_THREADS, IPT, size_type>;
  __shared__ typename BlockMergeSortT::TempStorage temp_storage;

  auto const block_id = static_cast<size_type>(blockIdx.x);
  if (block_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[block_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;
  if (seg_size <= band_lo || seg_size > band_hi) { return; }
  // A band_hi above the BLOCK_THREADS*IPT tile would silently drop elements.
  cudf_assert(seg_size <= BLOCK_THREADS * IPT &&
              "band segment exceeds the block tile (band_hi > BLOCK_THREADS*IPT)");

  tiered_key_builder<T> const build_key{d_input, has_nulls, polarity};
  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = static_cast<int>(threadIdx.x) * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  BlockMergeSortT(temp_storage).Sort(keys, vals, compare_op, static_cast<int>(seg_size), pad_key);

#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = static_cast<int>(threadIdx.x) * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

// No-null INT32 / INT64 warp segments route to kernels measured faster than the packed
// `WarpMergeSort`: a raw key (no null field) at half the packed width cuts the sort's data
// traffic. Only when `has_nulls` is false -- a raw key cannot express null ordering.

/**
 * @brief Raw ordering key for the no-null warp tier: the encoded value alone, half the packed
 * key's width
 */
template <typename T>
using tiered_raw_key_t =
  cuda::std::conditional_t<sizeof(T) <= 4, cuda::std::uint32_t, cuda::std::uint64_t>;

/**
 * @brief Builds the raw no-null key for one element: its order-preserving encoded value,
 * complemented when descending
 */
template <typename T>
struct tiered_raw_key_builder {
  column_device_view d_input;
  sort_polarity polarity;
  __device__ tiered_raw_key_t<T> operator()(size_type idx) const
  {
    if constexpr (sizeof(T) <= 4) {
      return radix_encode_u32<T>(d_input.element<T>(idx)) ^ polarity.value_mask32();
    } else {
      return radix_encode_u64<T>(d_input.element<T>(idx)) ^ polarity.value_mask64();
    }
  }
};

/**
 * @brief The pad key for the raw no-null path: the maximum unsigned value
 *
 * A real element can encode to the same maximum (`INT*_MAX` ascending, the minimum descending), so
 * the warp band uses `StableSort`: reals precede pads in tile order, and stability keeps an
 * equal-valued real inside the valid range.
 */
template <typename T>
__host__ __device__ inline tiered_raw_key_t<T> tiered_raw_pad_key()
{
  return cuda::std::numeric_limits<tiered_raw_key_t<T>>::max();
}

/// Plain ascending less-than for the raw key -- the raw path is no-null, so no class flag.
struct tiered_raw_less {
  template <typename KeyT>
  __device__ bool operator()(KeyT const& a, KeyT const& b) const
  {
    return a < b;
  }
};

/**
 * @brief Ascending compare of a raw-key/index pair with a pad tie-break for the register bitonic
 *
 * Compares `(key, is_pad)` with `is_pad == (val < 0)` (a pad carries `val = -1`). Ordering a real
 * before an equal-keyed pad keeps every real inside `[0, seg_size)` even when a real key encodes
 * to the all-ones pad sentinel, so the write-back never drops one.
 */
template <typename KeyT>
__device__ inline bool bitonic_pad_less(KeyT ka, size_type va, KeyT kb, size_type vb)
{
  if (ka != kb) { return ka < kb; }
  return (va >= 0) && (vb < 0);
}

/**
 * @brief Register shuffle-bitonic sort of `W * IPT` elements across a logical warp
 *
 * Standard Batcher bitonic network, blocked layout `e = lane * IPT + item`: intra-lane exchanges
 * (`j < IPT`) run in registers, inter-lane ones swap with lane `lane ^ (j / IPT)` via
 * `__shfl_xor_sync` confined by `mask` and width `W`. Stage direction is `(e & k) == 0` and the
 * `j`-bit-clear index is the low element, keeping partners consistent. Validated exhaustively on
 * the host (zero-one principle) for the instantiated shapes.
 */
template <int W, int IPT, typename KeyT>
__device__ inline void bitonic_warp_sort(KeyT (&keys)[IPT],
                                         size_type (&vals)[IPT],
                                         int lane,
                                         unsigned mask)
{
  constexpr int n = W * IPT;
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      if (j >= IPT) {
        int const jl = j / IPT;
#pragma unroll
        for (int m = 0; m < IPT; ++m) {
          int const e          = lane * IPT + m;
          KeyT const pk        = __shfl_xor_sync(mask, keys[m], jl, W);
          size_type const pv   = __shfl_xor_sync(mask, vals[m], jl, W);
          bool const ascending = ((e & k) == 0);
          bool const keep_min  = (((e & j) == 0) == ascending);
          bool const take      = keep_min ? bitonic_pad_less(pk, pv, keys[m], vals[m])
                                          : bitonic_pad_less(keys[m], vals[m], pk, pv);
          if (take) {
            keys[m] = pk;
            vals[m] = pv;
          }
        }
      } else {
#pragma unroll
        for (int m = 0; m < IPT; ++m) {
          int const m2 = m ^ j;
          if (m2 > m) {
            int const e          = lane * IPT + m;
            bool const ascending = ((e & k) == 0);
            bool const inverted  = ascending
                                     ? bitonic_pad_less(keys[m2], vals[m2], keys[m], vals[m])
                                     : bitonic_pad_less(keys[m], vals[m], keys[m2], vals[m2]);
            if (inverted) {
              KeyT const tk      = keys[m];
              keys[m]            = keys[m2];
              keys[m2]           = tk;
              size_type const tv = vals[m];
              vals[m]            = vals[m2];
              vals[m2]           = tv;
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Register shuffle-bitonic warp kernel over a segment-size band (raw keys, no-null)
 *
 * One logical `W`-lane warp per segment of size in `(band_lo, band_hi]`; pads (max key, index -1)
 * stay beyond `[0, seg_size)` via the tie-break.
 */
template <typename T, int W, int IPT, int BLOCK_THREADS>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_bitonic_band_kernel(
  column_device_view const d_input,
  size_type const* d_offsets,
  size_type const* d_seg_list,
  size_type num_class_segments,
  size_type band_lo,
  size_type band_hi,
  sort_polarity polarity,
  size_type* d_out)
{
  using KeyT            = tiered_raw_key_t<T>;
  auto const global_tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  auto const vwarp_id   = static_cast<size_type>(global_tid / W);
  auto const lane       = static_cast<int>(threadIdx.x % W);
  if (vwarp_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[vwarp_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;
  if (seg_size <= band_lo || seg_size > band_hi) { return; }
  // The register tile holds W*IPT elements; a band_hi above that would silently drop elements.
  cudf_assert(seg_size <= W * IPT && "band segment exceeds the register tile (band_hi > W*IPT)");

  tiered_raw_key_builder<T> const build_key{d_input, polarity};
  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = tiered_raw_pad_key<T>();
      vals[i] = size_type{-1};
    }
  }
  // Member mask of this virtual warp's W lanes (W <= 16, so the shift is well-defined).
  unsigned const mask = ((1u << W) - 1u) << ((static_cast<unsigned>(threadIdx.x) % 32u / W) * W);
  bitonic_warp_sort<W, IPT>(keys, vals, lane, mask);
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

/**
 * @brief Common per-helper launch parameters shared by all four tiered band launchers
 *
 * Bundles the column, its offsets/polarity, the output map, and the stream: the fields that stay
 * fixed across every launcher call one tier helper makes, so it constructs one instance and reuses
 * it. `band_lo`/`band_hi` stay explicit despite appearing in all four launchers' own signatures
 * too -- they change on every call within a helper, so bundling them would force a fresh struct
 * per call instead of one shared instance. `has_nulls` and the segment-list pointer (with its
 * paired count) also stay explicit: only two of the four launchers use `has_nulls`, and the
 * segment-list differs by name and meaning (`d_warp_segs` vs `d_block_segs`).
 */
struct band_launch_ctx {
  column_device_view const d_input;
  size_type const* d_offsets;
  sort_polarity const polarity;
  size_type* d_out;
  rmm::cuda_stream_view const stream;
};

/// Launches one packed-key `WarpMergeSort` band over the warp-segment list (all tiered types)
template <typename T, int W, int IPT>
void launch_packed_warp_band(band_launch_ctx const& ctx,
                             size_type const* d_warp_segs,
                             size_type num_warp,
                             bool has_nulls,
                             size_type band_lo,
                             size_type band_hi)
{
  if (num_warp == 0) { return; }
  using KeyT = tiered_key_t<T>;
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT,
                          tiered_key_builder<T>,
                          W,
                          IPT,
                          TIERED_BLOCK_THREADS,
                          tiered_key_less>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, ctx.stream.value()>>>(
      ctx.d_offsets,
      d_warp_segs,
      num_warp,
      band_lo,
      band_hi,
      tiered_key_builder<T>{ctx.d_input, has_nulls, ctx.polarity},
      ctx.d_out,
      tiered_key_less{},
      tiered_pad_key<T>());
  CUDF_CHECK_CUDA(ctx.stream.value());
}

/// Launches one raw-key `WarpMergeSort` band over the warp-segment list (no-null only)
template <typename T, int W, int IPT>
void launch_raw_warp_band(band_launch_ctx const& ctx,
                          size_type const* d_warp_segs,
                          size_type num_warp,
                          size_type band_lo,
                          size_type band_hi)
{
  if (num_warp == 0) { return; }
  using KeyT = tiered_raw_key_t<T>;
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_warp_band_kernel<KeyT,
                          tiered_raw_key_builder<T>,
                          W,
                          IPT,
                          TIERED_BLOCK_THREADS,
                          tiered_raw_less>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, ctx.stream.value()>>>(
      ctx.d_offsets,
      d_warp_segs,
      num_warp,
      band_lo,
      band_hi,
      tiered_raw_key_builder<T>{ctx.d_input, ctx.polarity},
      ctx.d_out,
      tiered_raw_less{},
      tiered_raw_pad_key<T>());
  CUDF_CHECK_CUDA(ctx.stream.value());
}

/// Launches one register-bitonic band over the warp-segment list (raw keys, no-null only)
template <typename T, int W, int IPT>
void launch_bitonic_band(band_launch_ctx const& ctx,
                         size_type const* d_warp_segs,
                         size_type num_warp,
                         size_type band_lo,
                         size_type band_hi)
{
  if (num_warp == 0) { return; }
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_warp) * W, TIERED_BLOCK_THREADS);
  tiered_bitonic_band_kernel<T, W, IPT, TIERED_BLOCK_THREADS>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, ctx.stream.value()>>>(
      ctx.d_input, ctx.d_offsets, d_warp_segs, num_warp, band_lo, band_hi, ctx.polarity, ctx.d_out);
  CUDF_CHECK_CUDA(ctx.stream.value());
}

/// Launches one packed-key `BlockMergeSort` band over the block-segment list (all tiered types)
template <typename T, int IPT>
void launch_block_band(band_launch_ctx const& ctx,
                       size_type const* d_block_segs,
                       size_type num_block,
                       bool has_nulls,
                       size_type band_lo,
                       size_type band_hi)
{
  if (num_block == 0) { return; }
  auto const grid = cudf::detail::grid_1d(
    static_cast<thread_index_type>(num_block) * TIERED_BLOCK_THREADS, TIERED_BLOCK_THREADS);
  tiered_block_band_kernel<T, TIERED_BLOCK_THREADS, IPT, tiered_key_less>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, ctx.stream.value()>>>(ctx.d_input,
                                                                             ctx.d_offsets,
                                                                             d_block_segs,
                                                                             num_block,
                                                                             band_lo,
                                                                             band_hi,
                                                                             has_nulls,
                                                                             ctx.polarity,
                                                                             ctx.d_out,
                                                                             tiered_key_less{},
                                                                             tiered_pad_key<T>());
  CUDF_CHECK_CUDA(ctx.stream.value());
}

/**
 * @brief Run-time predicate for the tiered fast path on a key column's type
 *
 * Timestamps / durations flow through the same width-keyed path via their integer reps.
 * DECIMAL32/64, the narrower integrals, and non-fixed-width types fall through to the paths below.
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the tiered fast path
 */
inline bool is_tiered_sort_supported(data_type type)
{
  return type.id() == type_id::INT32 or type.id() == type_id::INT64 or
         type.id() == type_id::DECIMAL128 or cudf::is_floating_point(type) or cudf::is_chrono(type);
}

/**
 * @brief Sorts only the radix-tier segments' elements via the packed-radix key, scattering the
 * result into the output gather map
 *
 * Keys just the compacted radix-tier element indices rather than paying a full-column radix for a
 * rare tier. `d_large_gidx` is ascending, so its entries are exactly the radix-tier output slots
 * in order: `d_out[d_large_gidx[j]] = sorted_gidx[j]` writes each segment's k-th smallest element
 * to its k-th slot. `d_large_gidx` is read-only throughout, so it doubles as the scatter map.
 */
template <typename T>
void radix_sort_large_segments(segment_sort_context const& ctx,
                               size_type const* d_large_gidx,
                               size_type num_large_elems,
                               size_type* d_out)
{
  // Locals preserve the body's original parameter names; only the call boundary changed.
  auto const& d_input      = ctx.d_input;
  auto const d_segment_ids = ctx.d_segment_ids;
  auto const has_nulls     = ctx.has_nulls;
  auto const polarity      = ctx.polarity;
  auto const segment_bits  = ctx.segment_bits;
  auto const stream        = ctx.stream;
  auto const alloc         = cudf::get_current_device_resource_ref();
  rmm::device_uvector<size_type> sorted_gidx(num_large_elems, stream);

  // Each branch mirrors the packed-radix path's width-specific key, restricted to the subset.
  if constexpr (sizeof(T) <= 4) {
    build_and_radix_sort_packed_keys<cuda::std::uint64_t>(
      d_large_gidx,
      num_large_elems,
      numeric_packed_key_builder<T>{d_segment_ids, d_input, has_nulls, segment_bits, polarity},
      static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
      d_large_gidx,
      sorted_gidx.data(),
      stream);
  } else if constexpr (sizeof(T) == 8) {
    // Same seg_null end-bit trim as the full-column eight-byte branch.
    auto const key_bits = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                         64 + cuda::std::min(32, segment_bits + 1));
    build_and_radix_sort_packed_keys<prefix_key96>(
      d_large_gidx,
      num_large_elems,
      numeric_packed_key_builder64<T>{d_segment_ids, d_input, has_nulls, polarity},
      prefix_decomposer{},
      key_bits,
      d_large_gidx,
      sorted_gidx.data(),
      stream);
  } else {
    // DECIMAL128: the narrowest lossless key from the subset's value range; `sorted_gidx`
    // receives the subset's sorted order for the scatter below.
    dec128_segmented_radix_sort(ctx, d_large_gidx, num_large_elems, sorted_gidx.data());
  }

  thrust::scatter(rmm::exec_policy_nosync(stream, alloc),
                  sorted_gidx.begin(),
                  sorted_gidx.end(),
                  d_large_gidx,
                  d_out);
}

/**
 * @brief Per-call tier sizes and the shared out-of-segment pad key for one `tiered_sort_fn`
 * dispatch
 *
 * Bundles what segment classification hands to all three tier handlers below, even though each
 * handler reads only its own count: one shared type keeps the three call sites uniform instead of
 * three bespoke parameter lists. `pad_key` depends only on `T`, so computing it once here, right
 * after classification, is equivalent to each tier computing it independently at its old site.
 */
template <typename T>
struct tiered_tier_counts {
  size_type num_network;
  size_type num_warp;
  size_type num_large;
  tiered_key_t<T> pad_key;
};

/**
 * @brief Handles the large-segment tier: splits it into the block sub-tier and the rare
 * radix-spill outliers, sorting each
 *
 * `segment_offsets` (not just `d_offsets`) is needed to re-derive `segment_ids` for the
 * radix-spill branch's `label_segments` call.
 */
template <typename T>
void handle_large_tier(column_view const& segment_offsets,
                       column_device_view const& d_input,
                       size_type num_elements,
                       size_type num_segments,
                       bool has_nulls,
                       sort_polarity polarity,
                       size_type const* large_segs,
                       tiered_tier_counts<T> const& counts,
                       size_type* d_out,
                       rmm::cuda_stream_view stream)
{
  auto const num_large = counts.num_large;
  if (num_large == 0) { return; }
  auto const d_offsets = segment_offsets.begin<size_type>();
  auto const seg_iter  = cuda::counting_iterator<size_type>{0};

  // The partition writes the rejected (radix-spill) ids reversed to the rear of `block_segs`;
  // the spill is driven off element size below, so the rear is never read.
  rmm::device_uvector<size_type> block_segs(num_large, stream);
  rmm::device_uvector<size_type> d_num_block(1, stream);
  auto const select_block = segment_in_block_tier{d_offsets, TIERED_BLOCK_TIER_CAP};
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(d_temp_storage.data(),
                             temp_storage_bytes,
                             large_segs,
                             block_segs.data(),
                             d_num_block.data(),
                             num_large,
                             select_block,
                             stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DevicePartition::If(d_temp_storage.data(),
                             temp_storage_bytes,
                             large_segs,
                             block_segs.data(),
                             d_num_block.data(),
                             num_large,
                             select_block,
                             stream.value());
  }
  auto const h_num_block =
    cudf::detail::make_pinned_vector(device_span<size_type const>{d_num_block.data(), 1}, stream);
  auto const num_block = h_num_block[0];
  auto const num_radix = num_large - num_block;

  // Radix-spill outliers: sort only their elements and scatter into their disjoint slots.
  if (num_radix > 0) {
    rmm::device_uvector<size_type> segment_ids(num_elements, stream);
    label_segments(segment_offsets.begin<size_type>(),
                   segment_offsets.end<size_type>(),
                   segment_ids.begin(),
                   segment_ids.end(),
                   stream);
    auto const segment_bits =
      static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_segments)));

    // Compact the radix-tier elements' global indices (ascending), then read their count
    // back -- the radix tier's one extra sync.
    rmm::device_uvector<size_type> large_gidx(num_elements, stream);
    rmm::device_uvector<size_type> d_num_large_elems(1, stream);
    {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      auto const is_large =
        element_in_radix_tier{segment_ids.data(), d_offsets, TIERED_BLOCK_TIER_CAP};
      cub::DeviceSelect::If(d_temp_storage.data(),
                            temp_storage_bytes,
                            seg_iter,
                            large_gidx.data(),
                            d_num_large_elems.data(),
                            num_elements,
                            is_large,
                            stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DeviceSelect::If(d_temp_storage.data(),
                            temp_storage_bytes,
                            seg_iter,
                            large_gidx.data(),
                            d_num_large_elems.data(),
                            num_elements,
                            is_large,
                            stream.value());
    }
    auto const h_num_large_elems = cudf::detail::make_pinned_vector(
      device_span<size_type const>{d_num_large_elems.data(), 1}, stream);
    radix_sort_large_segments<T>(
      segment_sort_context{segment_ids.data(), d_input, has_nulls, segment_bits, polarity, stream},
      large_gidx.data(),
      h_num_large_elems[0],
      d_out);
  }

  // Block tier: graduated items-per-thread bands over the one block-segment list; each band
  // self-filters to its size slice, so the bands need no re-partition.
  if (num_block > 0) {
    auto const* bl = block_segs.data();
    band_launch_ctx const launch_ctx{d_input, d_offsets, polarity, d_out, stream};
    launch_block_band<T, 1>(launch_ctx, bl, num_block, has_nulls, TIERED_WARP_CAP, 128);
    launch_block_band<T, 2>(launch_ctx, bl, num_block, has_nulls, 128, 256);
    launch_block_band<T, 4>(launch_ctx, bl, num_block, has_nulls, 256, 512);
    launch_block_band<T, 8>(launch_ctx, bl, num_block, has_nulls, 512, TIERED_BLOCK_TIER_CAP);
  }
}

/// Handles the network tier: one warp-network sort kernel launch over the small-segment list.
template <typename T>
void handle_network_tier(column_device_view const& d_input,
                         size_type const* d_offsets,
                         bool has_nulls,
                         sort_polarity polarity,
                         size_type const* network_segs,
                         tiered_tier_counts<T> const& counts,
                         size_type* d_out,
                         rmm::cuda_stream_view stream)
{
  auto const num_network = counts.num_network;
  if (num_network == 0) { return; }
  auto const grid =
    cudf::detail::grid_1d(static_cast<thread_index_type>(num_network), TIERED_BLOCK_THREADS);
  tiered_network_sort_kernel<T, TIERED_BLOCK_THREADS>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(d_input,
                                                                         d_offsets,
                                                                         network_segs,
                                                                         num_network,
                                                                         has_nulls,
                                                                         polarity,
                                                                         d_out,
                                                                         tiered_key_less{},
                                                                         counts.pad_key);
  CUDF_CHECK_CUDA(stream.value());
}

/**
 * @brief Handles the warp tier: routes each size sub-band to the kernel measured best per (type,
 * null-presence)
 *
 * No-null INT32: register bitonic throughout; no-null INT64: bitonic to 32, then a raw-key
 * WarpMergeSort at half the key traffic; nullable INT32 / INT64: packed-key WarpMergeSort bands.
 * Every other tiered type keeps the whole-band packed WarpMergeSort -- the raw-key / bitonic
 * kernels are unmeasured there.
 */
template <typename T>
void handle_warp_tier(column_device_view const& d_input,
                      size_type const* d_offsets,
                      bool has_nulls,
                      sort_polarity polarity,
                      size_type const* warp_segs,
                      tiered_tier_counts<T> const& counts,
                      size_type* d_out,
                      rmm::cuda_stream_view stream)
{
  auto const num_warp = counts.num_warp;
  if (num_warp == 0) { return; }
  auto const* wl = warp_segs;
  if constexpr (cuda::std::is_same_v<T, int32_t> or cuda::std::is_same_v<T, int64_t>) {
    band_launch_ctx const launch_ctx{d_input, d_offsets, polarity, d_out, stream};
    if (has_nulls) {
      launch_packed_warp_band<T, TIERED_WARP_LANES, 1>(
        launch_ctx, wl, num_warp, has_nulls, TIERED_NETWORK_CAP, 32);
      launch_packed_warp_band<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS>(
        launch_ctx, wl, num_warp, has_nulls, 32, TIERED_WARP_CAP);
    } else {
      launch_bitonic_band<T, 4, 4>(launch_ctx, wl, num_warp, TIERED_NETWORK_CAP, 16);
      launch_bitonic_band<T, 8, 4>(launch_ctx, wl, num_warp, 16, 32);
      if constexpr (cuda::std::is_same_v<T, int32_t>) {
        launch_bitonic_band<T, 16, 4>(launch_ctx, wl, num_warp, 32, TIERED_WARP_CAP);
      } else {
        launch_raw_warp_band<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS>(
          launch_ctx, wl, num_warp, 32, TIERED_WARP_CAP);
      }
    }
  } else {
    auto const grid = cudf::detail::grid_1d(
      static_cast<thread_index_type>(num_warp) * TIERED_WARP_LANES, TIERED_BLOCK_THREADS);
    tiered_warp_sort_kernel<T, TIERED_WARP_LANES, TIERED_WARP_ITEMS, TIERED_BLOCK_THREADS>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(d_input,
                                                                           d_offsets,
                                                                           wl,
                                                                           num_warp,
                                                                           has_nulls,
                                                                           polarity,
                                                                           d_out,
                                                                           tiered_key_less{},
                                                                           counts.pad_key);
    CUDF_CHECK_CUDA(stream.value());
  }
}

/**
 * @brief Type-dispatched worker for `fast_segmented_sorted_order_tiered`
 *
 * One three-way `cub::DevicePartition::If` classifies segments (network / warp / large); the
 * large class is split again into the block tier and the radix-spill outliers. Each stage adds
 * its host sync only when its class exists.
 */
struct tiered_sort_fn {
  template <typename T>
  static constexpr bool is_supported()
  {
    return cuda::std::is_same_v<T, int32_t> or cuda::std::is_same_v<T, int64_t> or
           cuda::std::is_same_v<T, __int128_t> or cudf::is_floating_point<T>() or
           cudf::is_chrono<T>();
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(column_view const& segment_offsets,
                  column_device_view const& d_input,
                  size_type num_elements,
                  size_type num_segments,
                  bool has_nulls,
                  sort_polarity polarity,
                  size_type* d_out,
                  rmm::cuda_stream_view stream) const
  {
    auto const d_offsets = segment_offsets.begin<size_type>();

    // Three-way classify the segment indices by size; `large_segs` feeds the second-stage
    // partition below.
    rmm::device_uvector<size_type> network_segs(num_segments, stream);
    rmm::device_uvector<size_type> warp_segs(num_segments, stream);
    rmm::device_uvector<size_type> large_segs(num_segments, stream);
    rmm::device_uvector<size_type> d_counts(2, stream);
    auto const seg_iter       = cuda::counting_iterator<size_type>{0};
    auto const select_network = segment_in_network_tier{d_offsets, TIERED_NETWORK_CAP};
    auto const select_warp = segment_in_warp_tier{d_offsets, TIERED_NETWORK_CAP, TIERED_WARP_CAP};
    {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DevicePartition::If(d_temp_storage.data(),
                               temp_storage_bytes,
                               seg_iter,
                               network_segs.data(),
                               warp_segs.data(),
                               large_segs.data(),
                               d_counts.data(),
                               num_segments,
                               select_network,
                               select_warp,
                               stream.value());
      d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
      cub::DevicePartition::If(d_temp_storage.data(),
                               temp_storage_bytes,
                               seg_iter,
                               network_segs.data(),
                               warp_segs.data(),
                               large_segs.data(),
                               d_counts.data(),
                               num_segments,
                               select_network,
                               select_warp,
                               stream.value());
    }
    auto const h_counts =
      cudf::detail::make_pinned_vector(device_span<size_type const>{d_counts.data(), 2}, stream);
    tiered_tier_counts<T> const counts{
      h_counts[0], h_counts[1], num_segments - h_counts[0] - h_counts[1], tiered_pad_key<T>()};

    // Each handler is a no-op -- with its host sync -- when its tier is empty, so the
    // tiny-segment common case pays nothing for unused rungs; every output slot is written
    // exactly once across the tiers.
    handle_large_tier<T>(segment_offsets,
                         d_input,
                         num_elements,
                         num_segments,
                         has_nulls,
                         polarity,
                         large_segs.data(),
                         counts,
                         d_out,
                         stream);
    handle_network_tier<T>(
      d_input, d_offsets, has_nulls, polarity, network_segs.data(), counts, d_out, stream);
    handle_warp_tier<T>(
      d_input, d_offsets, has_nulls, polarity, warp_segs.data(), counts, d_out, stream);
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(column_view const&,
                  column_device_view const&,
                  size_type,
                  size_type,
                  bool,
                  sort_polarity,
                  size_type*,
                  rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type cannot be used with the tiered segmented sort");
  }
};

// Measured crossover below which the tiered kernel beats CUB and the packed radix for every type.
constexpr size_type TIERED_MAX_TINY_AVG_LIST_SIZE{4};

// Segment-count ceiling for the eight-byte tiny-average pocket: below it CUB `DeviceSegmentedSort`
// beats the tiered network tier, whose fixed per-launch setup has not yet amortized.
constexpr size_type EIGHT_BYTE_TINY_CUB_MAX_NUM_OFFSETS{1 << 17};

// Measured cap of CUB's 16-byte-pair merge tile; longer segments fall to a costly per-block radix.
constexpr size_type DECIMAL128_CUB_MAX_SEGMENT_SIZE{32};

/**
 * @brief Whether a `DECIMAL128` mid-band column's segment shape favors CUB `DeviceSegmentedSort`
 *
 * CUB degrades on long segments, so the mid band takes it only when they are sparse: the oversized
 * count scaled by the cap must stay within the segment count (a 64-bit product, against overflow).
 * Costs one device count and host sync, paid only on the branch that calls it.
 *
 * @param segment_offsets The segment offsets (segment count plus one)
 * @param stream CUDA stream used for the device count
 * @return true if the segment shape favors CUB `DeviceSegmentedSort`
 */
inline bool decimal128_cub_segment_shape_ok(column_view const& segment_offsets,
                                            rmm::cuda_stream_view stream)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();
  auto const oversized =
    thrust::count_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{num_segments},
                     segment_exceeds_size{d_offsets, DECIMAL128_CUB_MAX_SEGMENT_SIZE});
  return static_cast<int64_t>(oversized) * DECIMAL128_CUB_MAX_SEGMENT_SIZE <= num_segments;
}

}  // namespace

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_numeric_packed(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;
  CUDF_EXPECTS(num_segments >= 1, "the packed key layout requires at least one segment");

  // Dense segment ordinals: the key's segment field needs only bit_width(num_segments) bits; an
  // empty segment skips a label, preserving cross-segment order.
  rmm::device_uvector<size_type> segment_ids(num_elements, stream);
  label_segments(segment_offsets.begin<size_type>(),
                 segment_offsets.end<size_type>(),
                 segment_ids.begin(),
                 segment_ids.end(),
                 stream);

  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();

  // The packed key reserves a null class only when nullable; pin the caller-side coupling the
  // strings path also asserts, so a null-free column can never arrive with `nulls_first` set.
  CUDF_EXPECTS(has_nulls or not polarity.nulls_first,
               "nulls_first requires a nullable column for the packed key layout");

  auto const segment_bits =
    static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_segments)));

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);

  // Storage-type dispatch: DECIMAL32/64 sort by their int32/int64 rep.
  cudf::type_dispatcher<dispatch_storage_type>(
    input.type(),
    numeric_packed_sort_fn{},
    segment_sort_context{segment_ids.data(), *d_input, has_nulls, segment_bits, polarity, stream},
    num_elements,
    sorted_indices->mutable_view().begin<size_type>());
  return sorted_indices;
}

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_tiered(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;
  CUDF_EXPECTS(num_segments >= 1, "the tiered key layout requires at least one segment");
  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();

  // The tiered key reserves a null class only when nullable; pin the caller-side coupling the
  // strings path also asserts, so a null-free column can never arrive with `nulls_first` set.
  CUDF_EXPECTS(has_nulls or not polarity.nulls_first,
               "nulls_first requires a nullable column for the tiered key layout");

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);

  // Storage-type dispatch: DECIMAL128 sorts by its __int128 rep; DECIMAL32/64 and chrono types
  // by their integer reps.
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               tiered_sort_fn{},
                                               segment_offsets,
                                               *d_input,
                                               num_elements,
                                               num_segments,
                                               has_nulls,
                                               polarity,
                                               sorted_indices->mutable_view().begin<size_type>(),
                                               stream);
  return sorted_indices;
}

fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   rmm::cuda_stream_view stream)
{
  auto const type = key.type();
  // String / nested keys have no fixed-width fast path.
  if (not is_numeric_packed_radix_supported(type)) { return fixed_width_sort_path::comparison; }

  // avg_list_size uses num_rows / num_offsets, the heuristic prefer_cub_segmented_sort also uses.
  auto const num_offsets   = segment_offsets.size();
  auto const avg_list_size = num_rows / num_offsets;

  // Types the tiered key can't encode: keep main's packed-radix-or-CUB decision.
  if (not is_tiered_sort_supported(type)) {
    return (key.has_nulls() or not prefer_cub_segmented_sort(num_rows, num_offsets))
             ? fixed_width_sort_path::packed_radix
             : fixed_width_sort_path::comparison;
  }
  // Tiered sort folds validity into the key, so nulls need no separate pass, at any list size.
  if (key.has_nulls()) { return fixed_width_sort_path::tiered; }
  // Long lists amortize the global packed-key radix's bandwidth-bound pass.
  if (avg_list_size >= MAX_AVG_LIST_SIZE_FOR_FAST_SORT) {
    return fixed_width_sort_path::packed_radix;
  }

  // DECIMAL128 no-null: tiny -> tiered, sparse-large mid band -> lifted CUB, longer -> tiered.
  if (type.id() == type_id::DECIMAL128) {
    if (avg_list_size <= TIERED_MAX_TINY_AVG_LIST_SIZE) { return fixed_width_sort_path::tiered; }
    if (avg_list_size <= DECIMAL128_CUB_MAX_AVG_LIST_SIZE and
        decimal128_cub_segment_shape_ok(segment_offsets, stream)) {
      return fixed_width_sort_path::cub_segmented;
    }
    return fixed_width_sort_path::tiered;
  }
  // Floating point no-null short: tiered across the range.
  if (cudf::is_floating_point(type)) { return fixed_width_sort_path::tiered; }
  // Eight-byte no-null: in the tiny-average pocket at a small offset count, CUB (via comparison)
  // beats the tiered network tier until the offset count amortizes its launch setup
  // (`EIGHT_BYTE_TINY_CUB_MAX_NUM_OFFSETS`); `num_offsets >= 2` matches the coverage probe's
  // precondition. Above the pocket, tiered: the warp kernels beat CUB for INT64 mid-band lists.
  // Chrono must stay off `cub_segmented` -- the CUB engine sorts only integral reps and would
  // `CUDF_FAIL`.
  if (cudf::size_of(type) == 8) {
    // Chrono skips the pocket: the CUB engine sorts only integral reps, so the pocket's
    // comparison route would strand a timestamp/duration on the plain comparison sort.
    if (cudf::is_integral(type) and avg_list_size <= TIERED_NETWORK_CAP and num_offsets >= 2 and
        num_offsets < EIGHT_BYTE_TINY_CUB_MAX_NUM_OFFSETS) {
      return fixed_width_sort_path::comparison;
    }
    return fixed_width_sort_path::tiered;
  }
  // Four-byte no-null short: tiered; the block tier covers the mid band, so no CUB escape.
  return fixed_width_sort_path::tiered;
}

}  // namespace detail
}  // namespace cudf
