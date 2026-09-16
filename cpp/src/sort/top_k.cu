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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Returns true if the column's top-k can be resolved by radix selection
 *
 * Radix selection needs a fixed-width key it can reduce to an order-preserving unsigned image.
 * A floating-point column qualifies once its values are normalized into that image; see
 * key_at_row.
 */
bool is_fast_path(column_view const& column)
{
  return !column.has_nulls() && cudf::is_fixed_width(column.type());
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

struct dispatch_topk_fn {
  column_view input;
  size_type k;
  order topk_order;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename KeysIterator>
  std::unique_ptr<column> select(KeysIterator keys_in)
  {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env          = cuda::std::execution::env{cuda::stream_ref{stream.get()}, requirements};
    auto tmp_size     = std::size_t{0};
    auto const size   = input.size();

    auto keys_out = cuda::make_discard_iterator();
    auto indices  = rmm::device_uvector<size_type>(k, stream);
    auto vals_in  = cuda::counting_iterator<size_type>();
    auto vals_out = indices.begin();

    if (topk_order == order::ASCENDING) {
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
      auto tmp = rmm::device_buffer(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
    } else {
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
      auto tmp = rmm::device_buffer(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
    }

    return std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);
  }

  template <typename T>
  std::unique_ptr<column> top_k()
  {
    auto const values = input.begin<T>();
    // Handing cub a raw pointer lets it vectorize its loads, so it is worth keeping for the types
    // whose values are already their own key.
    if constexpr (cudf::is_floating_point<T>()) {
      return select(
        cudf::detail::make_counting_transform_iterator(size_type{0}, key_at_row<T>{values}));
    } else {
      return select(values);
    }
  }

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    return top_k<T>();
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    using rep_type = typename T::rep;
    return top_k<rep_type>();
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>())
  std::unique_ptr<column> operator()()
  {
    CUDF_UNREACHABLE("unexpected type for top_k fast path");
  }
};

}  // namespace

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr); }

  auto const indices = [&] {
    auto const temp_mr = cudf::get_current_device_resource_ref();
    if (is_fast_path(col)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_topk_fn{col, k, topk_order, stream, temp_mr});
    }
    auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
    return sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, temp_mr);
  }();

  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();

  auto const dont_check  = out_of_bounds_policy::DONT_CHECK;
  auto const not_allowed = negative_index_policy::NOT_ALLOWED;
  auto result =
    cudf::detail::gather(cudf::table_view({col}), k_indices, dont_check, not_allowed, stream, mr);
  return std::move(result->release().front());
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
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

  auto const temp_mr = cudf::get_current_device_resource_ref();
  if (is_fast_path(col)) {
    return type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_topk_fn{col, k, topk_order, stream, temp_mr});
  }
  auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto indices     = sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, temp_mr);

  return std::make_unique<column>(
    cudf::detail::split(indices->view(), {k}, stream).front(), stream, mr);
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

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, stream, mr);
}

}  // namespace cudf
