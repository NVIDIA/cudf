/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstdint>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Returns true if the column's top-k can be resolved by radix selection
 *
 * Radix selection needs a fixed-width key it can reduce to an order-preserving unsigned image.
 * Nulls do not prevent that: select_top_k_rows() partitions them out of the key stream rather
 * than encoding them into it. A floating-point column qualifies once its values are normalized
 * into that image; see key_at_row.
 */
bool is_fast_path(column_view const& column) { return cudf::is_fixed_width(column.type()); }

/**
 * @brief Returns the null precedence used by the overloads that take only a direction
 *
 * Nulls go to the far end of the requested direction, so a top-k contains a null only when k
 * exceeds the number of non-null rows.
 */
null_order default_null_precedence(order topk_order)
{
  return topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
}

/**
 * @brief Returns true if nulls belong at the front of the top-k result
 *
 * `null_precedence` is relative to the values and not to the output: `BEFORE` means a null
 * compares less than every value. A null is therefore selected first exactly when the direction
 * being selected agrees with where nulls sit -- descending selects the greatest first, so it takes
 * nulls first under `AFTER`.
 */
bool nulls_selected_first(order topk_order, null_order null_precedence)
{
  return (null_precedence == null_order::AFTER) == (topk_order == order::DESCENDING);
}

/**
 * @brief Reads `values[row]` and returns the key cub should compare
 *
 * For a floating-point column this is an order-preserving unsigned image of the value rather than
 * the value itself. Radix selection compares bits and not values, and the usual IEEE-754 mapping
 * -- complement a negative value, set the sign bit of a non-negative one -- is not that image under
 * cudf's ordering, because `sorting_physical_element_comparator` treats every NaN as equal to every
 * other NaN and greater than all other values, and treats -0.0 as equal to +0.0. Two
 * normalizations restore it, and they are the same ones
 * `hashing::detail::normalize_nans_and_zeros` applies for the same reason:
 *
 * - Every NaN becomes the canonical quiet NaN. Its image lands above the image of infinity, so
 *   NaNs order last ascending and first descending. Without this a negative NaN complements to
 *   below the image of -infinity and is selected as the smallest value.
 * - -0.0 becomes +0.0, so the two never order against each other.
 *
 * Every other fixed-width type is returned unchanged, since cub's own twiddle already puts its
 * bits in order.
 */
template <typename T>
using key_type_for =
  cuda::std::conditional_t<cudf::is_floating_point<T>(),
                           cuda::std::conditional_t<sizeof(T) == sizeof(::cuda::std::uint32_t),
                                                    ::cuda::std::uint32_t,
                                                    ::cuda::std::uint64_t>,
                           T>;

template <typename T>
struct key_at_row {
  T const* values;

  // Spelled out rather than deduced: nvcc cannot generate the kernel stub for a cub kernel whose
  // key iterator has a functor with a deduced return type.
  __device__ key_type_for<T> operator()(size_type row) const
  {
    if constexpr (cudf::is_floating_point<T>()) {
      using bits_type = key_type_for<T>;
      static_assert(sizeof(bits_type) == sizeof(T), "no unsigned type matches the key's width");

      auto const value = [v = values[row]] {
        if (cuda::std::isnan(v)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
        return v == T{0} ? T{0} : v;  // normalizes -0.0
      }();
      auto constexpr sign_bit = bits_type{1} << ((sizeof(bits_type) * 8) - 1);
      auto const bits         = cuda::std::bit_cast<bits_type>(value);
      return static_cast<bits_type>((bits & sign_bit) ? ~bits : (bits | sign_bit));
    } else {
      return values[row];
    }
  }
};

/// Selects a valid row, or a null one when `want_valid` is false. Only used where the input is
/// known to have nulls.
struct row_validity_is {
  bitmask_type const* null_mask;
  size_type offset;
  bool want_valid;

  __device__ bool operator()(size_type row) const
  {
    return cudf::bit_is_set(null_mask, offset + row) == want_valid;
  }
};

/**
 * @brief Selects `num_out` row indices with cub::DeviceTopK, writing them to `d_output`
 *
 * When `valid_rows` is null the selection runs over the whole column. Otherwise `valid_rows` holds
 * `num_items` indices of the non-null rows and the selection runs over those; cub carries those
 * indices through as its values, so the output is indices into `input` either way and never needs
 * remapping.
 */
struct dispatch_topk_fn {
  column_view input;
  size_type const* valid_rows;
  size_type num_items;
  size_type num_out;
  order topk_order;
  size_type* d_output;
  cuda::stream_ref stream;

  template <typename KeysIterator, typename ValuesIterator>
  void select(KeysIterator keys_in, ValuesIterator vals_in) const
  {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env          = cuda::std::execution::env{cuda::stream_ref{stream.get()}, requirements};
    auto tmp_size     = std::size_t{0};
    auto keys_out     = cuda::make_discard_iterator();

    if (topk_order == order::ASCENDING) {
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, d_output, num_items, num_out, env));
      auto tmp = rmm::device_buffer(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, d_output, num_items, num_out, env));
    } else {
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, d_output, num_items, num_out, env));
      auto tmp = rmm::device_buffer(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, d_output, num_items, num_out, env));
    }
  }

  template <typename T>
  void select_rows() const
  {
    auto const values = input.begin<T>();
    if (valid_rows == nullptr) {
      // Handing cub a raw pointer lets it vectorize its loads, so it is worth keeping for the
      // types whose values are already their own key.
      if constexpr (cudf::is_floating_point<T>()) {
        select(cudf::detail::make_counting_transform_iterator(size_type{0}, key_at_row<T>{values}),
               cuda::counting_iterator<size_type>{});
      } else {
        select(values, cuda::counting_iterator<size_type>{});
      }
    } else {
      select(thrust::make_transform_iterator(valid_rows, key_at_row<T>{values}), valid_rows);
    }
  }

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>())
  void operator()() const
  {
    select_rows<T>();
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  void operator()() const
  {
    select_rows<typename T::rep>();
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>())
  void operator()() const
  {
    CUDF_UNREACHABLE("unexpected type for top_k fast path");
  }
};

/**
 * @brief Returns the k selected row indices for a fixed-width column
 *
 * Nulls are partitioned out of the key stream instead of being encoded into it. Only a
 * floating-point type reliably has a spare bit pattern to spend on a null sentinel -- every
 * integral pattern is a valid key -- and widening the key to make room costs radix passes. The
 * split of k between null and non-null rows is host-side arithmetic on `null_count()`, which
 * `column_view` already carries, so partitioning costs one pass over the input and no
 * synchronization.
 */
rmm::device_uvector<size_type> select_top_k_rows(column_view const& col,
                                                 size_type k,
                                                 order topk_order,
                                                 null_order null_precedence,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const size       = col.size();
  auto const null_count = col.null_count();
  auto const num_valid  = size - null_count;

  auto output = rmm::device_uvector<size_type>(k, stream, mr);

  if (null_count == 0) {
    type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_topk_fn{col, nullptr, size, k, topk_order, output.data(), stream});
    return output;
  }

  // How much of the output each side of the partition supplies. Both counts are host-side
  // arithmetic on null_count(), so neither compaction below has to be sized on the device.
  auto const nulls_first = nulls_selected_first(topk_order, null_precedence);
  auto const num_nulls_out =
    nulls_first ? std::min(k, null_count) : std::max(size_type{0}, k - num_valid);
  auto const num_valid_out = k - num_nulls_out;
  auto* const nulls_out    = nulls_first ? output.data() : output.data() + num_valid_out;
  auto* const valid_out    = nulls_first ? output.data() + num_nulls_out : output.data();

  auto const rows   = cuda::counting_iterator<size_type>(0);
  auto const mask   = col.null_mask();
  auto const offset = col.offset();

  // cub::DeviceSelect::If rather than thrust::copy_if: copy_if, even under
  // rmm::exec_policy_nosync, unconditionally synchronizes the stream to read its
  // selected count back to the host, which would undo the host-side count split
  // above. cub::DeviceSelect::If needs a device pointer for that count, but both
  // counts are already known (num_nulls_out / num_valid_out), so it is written
  // to a throwaway and ignored -- and no synchronization occurs.
  auto d_num_selected = rmm::device_uvector<size_type>(1, stream, temp_mr);
  auto const compact  = [&](size_type* out, bool want_valid) {
    row_validity_is const pred{mask, offset, want_valid};
    size_t temp_bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceSelect::If(
      nullptr, temp_bytes, rows, out, d_num_selected.data(), size, pred, stream.get()));
    auto temp = rmm::device_buffer(temp_bytes, stream, temp_mr);
    CUDF_CUDA_TRY(cub::DeviceSelect::If(
      temp.data(), temp_bytes, rows, out, d_num_selected.data(), size, pred, stream.get()));
  };

  // Compacted only when some nulls are wanted: for a null placement that puts them at the far end,
  // k <= num_valid means none are. The compaction needs its own buffer because the select writes
  // every match, while the output holds only the num_nulls_out of them that are kept.
  auto null_rows =
    rmm::device_uvector<size_type>(num_nulls_out > 0 ? null_count : 0, stream, temp_mr);
  if (num_nulls_out > 0) {
    compact(null_rows.data(), false);
    CUDF_CUDA_TRY(cudaMemcpyAsync(nulls_out,
                                  null_rows.data(),
                                  num_nulls_out * sizeof(size_type),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
  }

  if (num_valid_out == 0) { return output; }

  // Compacting the non-null rows in input order gives cub a candidate list whose values are the
  // original row indices, so it carries them through and the output needs no remapping. Input order
  // fixes the values cub carries, not its tie choice: which of several equal-keyed candidates it
  // keeps at the k-th boundary is unspecified (determinism::not_guaranteed).
  auto valid_rows = rmm::device_uvector<size_type>(num_valid, stream, temp_mr);
  compact(valid_rows.data(), true);

  if (num_valid_out == num_valid) {
    // Every non-null row is in the answer, so there is no order statistic to find. This also
    // keeps the selection off cub's k == num_items edge case.
    CUDF_CUDA_TRY(cudaMemcpyAsync(valid_out,
                                  valid_rows.data(),
                                  num_valid_out * sizeof(size_type),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
    return output;
  }

  type_dispatcher<dispatch_storage_type>(
    col.type(),
    dispatch_topk_fn{
      col, valid_rows.data(), num_valid, num_valid_out, topk_order, valid_out, stream});

  return output;
}

}  // namespace

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              null_order null_precedence,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr); }

  auto const indices = [&]() -> std::unique_ptr<column> {
    auto const temp_mr = cudf::get_current_device_resource_ref();
    if (is_fast_path(col)) {
      return std::make_unique<column>(
        select_top_k_rows(col, k, topk_order, null_precedence, stream, temp_mr),
        rmm::device_buffer{},
        0);
    }
    return sorted_order<sort_method::STABLE>(col, topk_order, null_precedence, stream, temp_mr);
  }();

  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();

  auto const dont_check  = out_of_bounds_policy::DONT_CHECK;
  auto const not_allowed = negative_index_policy::NOT_ALLOWED;
  auto result =
    cudf::detail::gather(cudf::table_view({col}), k_indices, dont_check, not_allowed, stream, mr);
  return std::move(result->release().front());
}

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  return cudf::detail::top_k(col, k, topk_order, default_null_precedence(topk_order), stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    null_order null_precedence,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return make_empty_column(cudf::type_to_id<size_type>()); }
  if (k >= col.size()) {
    return cudf::detail::sequence(
      col.size(),
      numeric_scalar<size_type>(0, true, stream, cudf::get_current_device_resource_ref()),
      stream,
      mr);
  }

  if (is_fast_path(col)) {
    return std::make_unique<column>(
      select_top_k_rows(col, k, topk_order, null_precedence, stream, mr), rmm::device_buffer{}, 0);
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto indices =
    sorted_order<sort_method::STABLE>(col, topk_order, null_precedence, stream, temp_mr);

  return std::make_unique<column>(
    cudf::detail::split(indices->view(), {k}, stream).front(), stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  return cudf::detail::top_k_order(
    col, k, topk_order, default_null_precedence(topk_order), stream, mr);
}

}  // namespace detail

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, topk_order, stream, mr);
}

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              null_order null_precedence,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, topk_order, null_precedence, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    null_order null_precedence,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, null_precedence, stream, mr);
}

}  // namespace cudf
