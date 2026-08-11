/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rolling_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <memory>
#include <optional>

namespace cudf {
namespace detail {
namespace rolling {

/*
 * Spark requires that orderby columns with floating point type have a
 * total order on floats where all NaNs compare equal to one-another,
 * and greater than any non-nan value. These structs implement that logic.
 */
template <typename T>
struct less {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return false; }
      return cuda::std::isnan(y) || x < y;
    } else {
      return x < y;
    }
  }
};

template <typename T>
struct less_equal {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return cuda::std::isnan(y); }
      return cuda::std::isnan(y) || x <= y;
    } else {
      return x <= y;
    }
  }
};
template <typename T>
struct greater {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return !cuda::std::isnan(y); }
      return !cuda::std::isnan(y) && x > y;
    } else {
      return x > y;
    }
  }
};

template <typename T>
struct greater_equal {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return true; }
      return !cuda::std::isnan(y) && x >= y;
    } else {
      return x >= y;
    }
  }
};

/**
 * @brief Select the appropriate ordering comparator for the window type.
 *
 * @tparam T The type being compared.
 * @tparam WindowType The type of the window.
 */
template <typename T, typename WindowType>
struct comparator_impl {
  static_assert(cuda::std::is_same_v<WindowType, bounded_closed> ||
                  cuda::std::is_same_v<WindowType, current_row>,
                "Invalid window type");
  using op     = less<T>;
  using rev_op = greater<T>;
};

template <typename T>
struct comparator_impl<T, bounded_open> {
  using op     = less_equal<T>;
  using rev_op = greater_equal<T>;
};

/**
 * @brief Select the appropriate ordering comparator for the window type.
 *
 * @tparam T The type being compared.
 * @tparam WindowType The type of the window.
 * @param order The sort order of the column being searched in.
 */
template <typename T, typename WindowType>
struct comparator_t {
  order const order;

  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if (order == order::ASCENDING) {
      using op = typename comparator_impl<T, WindowType>::op;
      return op{}(x, y);
    } else {
      using op = typename comparator_impl<T, WindowType>::rev_op;
      return op{}(x, y);
    }
  }
};

/**
 * @brief Saturating addition or subtraction of values.
 *
 * @tparam Op The operation to perform, `cuda::std::plus` or `cuda::std::minus`.
 *
 * Performs `x + y` (respectively `x - y`) saturating at the numeric
 * limits for the type, returning the value and a flag indicating
 * whether overflow occurred.
 *
 * For arithmetic types, the usual arithmetic conversions are _not_
 * applied, and so it is required that `x` and `y` have the same type.
 *
 * If `x` is floating, then `y` must be neither `inf` nor `nan` (not
 * checked), otherwise behaviour is undefined. If `x` is `inf` or
 * `nan`, then the result is the same as `x` and overflow does not
 * occur.
 *
 * If `x` is a timestamp type, then `y` must be the matching
 * `duration` type.
 *
 * If `x` is a decimal type, then `y` must be the matching underlying
 * rep type, and _must_ have the same scale as `x` (not checked)
 * otherwise behaviour is undefined.
 *
 * All other types are unsupported.
 */
template <typename Op>
struct saturating {
  static_assert(cuda::std::is_same_v<Op, cuda::std::plus<>> ||
                  cuda::std::is_same_v<Op, cuda::std::minus<>>,
                "Only for addition or subtraction");
  template <typename T, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<T>)>
  [[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> operator()(
    T x, T y) const noexcept
  {
    // Mimicking spark requirements, inf/nan x propagates
    // Other consumers (pandas, polars) of this functionality do not
    // support orderby columns that have floating point type.
    if (cuda::std::isinf(x) || cuda::std::isnan(x)) { return {x, false}; }
    // Requirement, not checked, y is not inf or nan.
    T result = Op{}(x, y);
    // If the result is outside the range of finite values it can at
    // this point only be +- infinity (we can't generate a nan by
    // adding a non-nan/non-inf y to a non-nan/non-inf x).
    if (result < cuda::std::numeric_limits<T>::lowest()) {
      return {cuda::std::numeric_limits<T>::lowest(), true};
    } else if (result > cuda::std::numeric_limits<T>::max()) {
      return {cuda::std::numeric_limits<T>::max(), true};
    }
    return {result, false};
  }

  template <typename T, CUDF_ENABLE_IF(cuda::std::is_integral_v<T>&& cuda::std::is_signed_v<T>)>
  [[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> operator()(
    T x, T y) const noexcept
  {
    using U  = cuda::std::make_unsigned_t<T>;
    U ux     = static_cast<U>(x);
    U uy     = static_cast<U>(y);
    U result = Op{}(ux, uy);
    ux       = (ux >> cuda::std::numeric_limits<T>::digits) +
         static_cast<U>(cuda::std::numeric_limits<T>::max());
    // Note: the casts here are implementation defined (until C++20) but all
    // the platforms we care about do the twos-complement thing.
    if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>>) {
      auto const did_overflow = static_cast<T>((ux ^ uy) | ~(uy ^ result)) >= 0;
      return {did_overflow ? ux : result, did_overflow};
    } else {
      auto const did_overflow = static_cast<T>((ux ^ uy) & (ux ^ result)) < 0;
      return {did_overflow ? ux : result, did_overflow};
    }
  }

  template <typename T, CUDF_ENABLE_IF(cuda::std::is_integral_v<T>&& cuda::std::is_unsigned_v<T>)>
  [[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> operator()(
    T x, T y) const noexcept
  {
    T result = Op{}(x, y);
    if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>>) {
      // Only way we can overflow is in the positive direction
      // in which case result will be less than both of x and y.
      // To saturate, we bit-or with (T)-1 in this case
      auto const did_overflow = result < x;
      return {result | (-static_cast<T>(did_overflow)), did_overflow};
    } else {
      auto const did_overflow = result > x;
      return {result & (-static_cast<T>(!did_overflow)), did_overflow};
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
  [[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> operator()(
    T x, typename T::duration y) const noexcept
  {
    using Duration                   = typename T::duration;
    auto const [value, did_overflow] = saturating<Op>{}(x.time_since_epoch().count(), y.count());
    return {T{Duration{value}}, did_overflow};
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  [[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> operator()(
    T x, typename T::rep y) const noexcept
  {
    using Rep                        = typename T::rep;
    auto const [value, did_overflow] = saturating<Op>{}(x.value(), y);
    return {T{numeric::scaled_integer<Rep>{value, x.scale()}}, did_overflow};
  }
};

/**
 * @brief Functor to compute distance from current row for `unbounded` windows.
 *
 * An `unbounded` window runs to the boundary of the current row's group (inclusive).
 *
 * @tparam Grouping (inferred) type of object defining groups in the orderby column.
 * @param groups object defining groups in the orderby column.
 * @param direction direction of the window `PRECEDING` or `FOLLOWING`.
 */
template <typename Grouping>
struct unbounded_distance_functor {
  Grouping const groups;
  direction const direction;
  [[nodiscard]] __device__ size_type operator()(size_type i) const noexcept
  {
    auto const row_info = groups.row_info(i);
    if (direction == direction::PRECEDING) {
      return i - row_info.group_start() + 1;
    } else {
      return row_info.group_end() - i - 1;
    }
  }
};
// TODO: Remove this deduction guide when we require C++20
template <typename Grouping>
unbounded_distance_functor(Grouping, direction) -> unbounded_distance_functor<Grouping>;

/**
 * @brief Functor to compute distance from current row for `current_row` windows.
 *
 * A `current_row` window contains all rows whose value is the same as the current row.
 *
 * @tparam Grouping (inferred) type of object defining groups in the orderby column.
 * @tparam OrderbyT (inferred) type of the entries in the orderby column
 * @param groups object defining groups in the orderby column.
 * @param direction direction of the window `PRECEDING` or `FOLLOWING`.
 * @param order sort order of the orderby column.
 * @param begin iterator to the begin of the orderby column.
 */
template <typename Grouping, typename OrderbyT>
struct current_row_distance_functor {
  Grouping const groups;
  direction const direction;
  order const order;
  column_device_view::const_iterator<OrderbyT> const begin;

  [[nodiscard]] __device__ size_type operator()(size_type i) const noexcept
  {
    using Comp          = comparator_t<OrderbyT, current_row>;
    auto const row_info = groups.row_info(i);
    if (row_info.is_null(i)) {
      return direction == direction::PRECEDING ? i - row_info.null_start() + 1
                                               : row_info.null_end() - i - 1;
    }
    if (direction == direction::PRECEDING) {
      return 1 +
             cuda::std::distance(
               thrust::lower_bound(
                 thrust::seq, begin + row_info.non_null_start(), begin + i, begin[i], Comp{order}),
               begin + i);
    } else {
      return cuda::std::distance(
               begin + i,
               thrust::upper_bound(
                 thrust::seq, begin + i, begin + row_info.non_null_end(), begin[i], Comp{order})) -
             1;
    }
  }
};
// TODO: Remove this deduction guide when we require C++20
template <typename Grouping, typename OrderbyT>
current_row_distance_functor(Grouping,
                             direction,
                             order,
                             column_device_view::const_iterator<OrderbyT>)
  -> current_row_distance_functor<Grouping, OrderbyT>;

/**
 * @brief Functor to compute distance from current row for `bounded_open` and `bounded_closed`
 * windows.
 *
 * A `bounded_open` window contains all rows up to but not including the computed endpoint.
 * A `bounded_closed` window contains all rows up to and including the computed endpoint.
 *
 * @tparam Grouping type of object defining groups in the orderby column.
 * @tparam OrderbyT type of elements in the orderby columns.
 * @tparam DeltaT type of the elements in the scalar delta (returned
 * by `scalar.data()`).
 * @tparam WindowType type of window we're computing the distance for.
 * @param groups object defining groups in the orderby column.
 * @param direction direction of the window `PRECEDING` or `FOLLOWING`.
 * @param order sort order of the orderby column.
 * @param row_delta pointer to row delta on device.
 * @param begin iterator to the begin of orderby column on device.
 *
 * @note Let `x` be the value of the current row and `delta` the provided
 * row delta then for bounded windows the endpoints are computed as follows.
 *
 *           | ASCENDING | DESCENDING
 * ----------+-----------+-----------
 * PRECEDING | x - delta | x + delta
 * FOLLOWING | x + delta | x - delta
 *
 * See `saturating_op` for details of the implementation of saturating addition/subtraction.
 */
template <typename Grouping, typename OrderbyT, typename DeltaT, typename WindowType>
struct bounded_distance_functor {
  static_assert(cuda::std::is_same_v<WindowType, bounded_open> ||
                  cuda::std::is_same_v<WindowType, bounded_closed>,
                "Invalid WindowType, expecting bounded_open or bounded_closed.");
  Grouping const groups;
  direction const direction;
  order const order;
  column_device_view::const_iterator<OrderbyT> const begin;
  DeltaT const* row_delta;

  /**
   * @brief Compute the offset to the end of the window.
   *
   * @param i The current row index.
   * @return Offset to the current row's window endpoint.
   */
  [[nodiscard]] __device__ size_type operator()(size_type i) const
  {
    using Comp           = comparator_t<OrderbyT, WindowType>;
    using saturating_sub = saturating<cuda::std::minus<>>;
    using saturating_add = saturating<cuda::std::plus<>>;
    auto const row_info  = groups.row_info(i);
    if (row_info.is_null(i)) {
      // TODO: If the window is BOUNDED_OPEN, what does it mean for a row to fall in the null
      // group? Not that important because only spark allows nulls in the orderby column, and it
      // doesn't have BOUNDED_OPEN windows.
      return direction == direction::PRECEDING ? i - row_info.null_start() + 1
                                               : row_info.null_end() - i - 1;
    }
    auto const offset_value_did_overflow = [subtract = (order == order::ASCENDING) ==
                                                       (direction == direction::PRECEDING),
                                            delta     = *row_delta,
                                            row_value = begin[i]]() {
      return subtract ? saturating_sub{}(row_value, delta) : saturating_add{}(row_value, delta);
    }();
    OrderbyT const offset_value = cuda::std::get<0>(offset_value_did_overflow);
    bool const did_overflow     = cuda::std::get<1>(offset_value_did_overflow);
    auto const distance         = [preceding    = direction == direction::PRECEDING,
                           current      = begin + i,
                           start        = begin + row_info.non_null_start(),
                           end          = begin + row_info.non_null_end(),
                           offset_value = offset_value](auto&& cmp) {
      if (preceding) {
        // Search for first slot we can place the offset value
        return 1 + cuda::std::distance(
                     thrust::lower_bound(thrust::seq, start, end, offset_value, cmp), current);
      } else {
        // Search for last slot we can place the offset value
        return cuda::std::distance(
                 current, thrust::upper_bound(thrust::seq, start, end, offset_value, cmp)) -
               1;
      }
    };
    if (did_overflow) {
      // If computing the offset row value overflowed then we must
      // adapt the search comparator. Suppose that the window
      // direction is PRECEDING and orderby column is ASCENDING. The
      // offset value is computed as row_value - delta. There are two
      // cases:
      // 1. delta is positive: we overflowed towards -infinity;
      // 2. delta is negative, we overflowed towards +infinity.
      // For case 1, the saturating minimum value must be
      // included in the preceding window bound (because -infinity <
      // min). Conversely, for case 2, the saturating maximum value
      // must not be included in the preceding window bound (because
      // max < +infinity). This can be obtained by picking a
      // bounded_closed (respectively bounded_open) window.
      // For FOLLOWING windows (with ASCENDING orderby), positive
      // delta overflows towards +infinity and negative delta towards
      // -infinity, and the same logic applies: we should include for
      // positive overflow, but exclude for negative overflow.
      // Since, when the orderby columns is ASCENDING, delta is
      // treated with a sign flip, the above also applies in that case.
      if (*row_delta > DeltaT{0}) {
        return distance(comparator_t<OrderbyT, bounded_closed>{order});
      } else {
        return distance(comparator_t<OrderbyT, bounded_open>{order});
      }
    } else {
      return distance(Comp{order});
    }
  }
};

constexpr thread_index_type materialize_range_window_bounds_block_size{128};

/**
 * @brief Writes `transform(i)` to `result[i]` for each row `i`
 *
 * A custom kernel is used instead of CUB to reduce compile time and binary size.
 *
 * @tparam Transform Callable mapping a row index to a window bound
 * @param size Number of rows
 * @param result Device pointer to `size` output elements
 * @param transform Per-row window-bound computation
 */
template <typename Transform>
CUDF_KERNEL void __launch_bounds__(materialize_range_window_bounds_block_size)
  materialize_range_window_bounds_kernel(size_type size, size_type* result, Transform transform)
{
  auto const index = cudf::detail::grid_1d::global_thread_id();
  if (index < size) { result[index] = transform(static_cast<size_type>(index)); }
}

template <typename Transform>
void materialize_range_window_bounds(size_type size,
                                     size_type* result,
                                     Transform transform,
                                     rmm::cuda_stream_view stream)
{
  if (size == 0) { return; }

  cudf::detail::grid_1d config{size, materialize_range_window_bounds_block_size};
  materialize_range_window_bounds_kernel<<<config.num_blocks,
                                           config.num_threads_per_block,
                                           0,
                                           stream.value()>>>(size, result, transform);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Functor to dispatch computation of clamped range-based rolling window bounds.
 *
 * @tparam WindowType The tag indicating the type of window being computed
 */
template <typename WindowType>
struct range_window_clamper {
  static_assert(cuda::std::is_same_v<WindowType, unbounded> ||
                  cuda::std::is_same_v<WindowType, current_row> ||
                  cuda::std::is_same_v<WindowType, bounded_closed> ||
                  cuda::std::is_same_v<WindowType, bounded_open>,
                "Invalid WindowType descriptor");
  template <typename Grouping>
  void expand_unbounded(Grouping grouping,
                        direction direction,
                        size_type size,
                        mutable_column_view& result,
                        rmm::cuda_stream_view stream) const
  {
    materialize_range_window_bounds(
      size, result.data<size_type>(), unbounded_distance_functor{grouping, direction}, stream);
  }

  template <typename Grouping, typename OrderbyT>
  void expand_current_row(Grouping grouping,
                          direction direction,
                          order order,
                          column_device_view::const_iterator<OrderbyT> begin,
                          size_type size,
                          mutable_column_view& result,
                          rmm::cuda_stream_view stream) const
  {
    materialize_range_window_bounds(size,
                                    result.data<size_type>(),
                                    current_row_distance_functor{grouping, direction, order, begin},
                                    stream);
  }

  template <typename Grouping, typename OrderbyT, typename DeltaT>
  void expand_bounded(Grouping grouping,
                      direction direction,
                      order order,
                      column_device_view::const_iterator<OrderbyT> begin,
                      DeltaT const* row_delta,
                      size_type size,
                      mutable_column_view& result,
                      rmm::cuda_stream_view stream) const
  {
    materialize_range_window_bounds(
      size,
      result.data<size_type>(),
      bounded_distance_functor<Grouping, OrderbyT, DeltaT, WindowType>{
        grouping, direction, order, begin, row_delta},
      stream);
  }

  /**
   * @brief Compute the window bounds (possibly grouped) for an orderby column.
   *
   * @tparam OrderbyT element type of the orderby column (dispatched on)
   * @tparam ScalarT Concrete scalar type of the scalar row delta
   * @param orderby Column used to define windows.
   * @param direction The window direction
   * @param order The sort order of the orderby column
   * @param grouping optional pre-processed group information.
   * @param nulls_at_start If the orderby column contains nulls, are they are the start or the end?
   * @param row_delta the delta applied to each row, will be null if the window is of type
   * `UNBOUNDED` or `CURRENT_ROW`, otherwise non-null. If non-null, must be a finite value or
   * behaviour is undefined.
   * @param stream CUDA stream used for kernel launches and memory allocations
   * @param mr Memory resource used for memory allocations.
   *
   * @return A column containing the computed endpoints for the given window description.
   */
  template <typename OrderbyT, typename ScalarT = cudf::scalar_type_t<OrderbyT>>
  [[nodiscard]] std::unique_ptr<column> window_bounds(
    column_view const& orderby,
    direction direction,
    order order,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = make_numeric_column(
      data_type(type_to_id<size_type>()), orderby.size(), mask_state::UNALLOCATED, stream, mr);
    auto d_orderby = column_device_view::create(orderby, stream);
    auto d_begin   = d_orderby->begin<OrderbyT>();
    auto expand    = [&](auto&& grouping) {
      auto result_view = result->mutable_view();
      if constexpr (cuda::std::is_same_v<WindowType, unbounded>) {
        expand_unbounded(grouping, direction, orderby.size(), result_view, stream);
      } else if constexpr (cuda::std::is_same_v<WindowType, current_row>) {
        expand_current_row(
          grouping, direction, order, d_begin, orderby.size(), result_view, stream);
      } else {
        auto const* d_row_delta = static_cast<ScalarT const*>(row_delta)->data();
        expand_bounded(
          grouping, direction, order, d_begin, d_row_delta, orderby.size(), result_view, stream);
      }
    };

    if (grouping.has_value()) {
      if (orderby.has_nulls()) {
        expand(grouped_with_nulls{nulls_at_start,
                                  grouping->labels.data(),
                                  grouping->offsets.data(),
                                  grouping->nulls_per_group.data()});
      } else {
        expand(grouped{grouping->labels.data(), grouping->offsets.data()});
      }
    } else {
      if (orderby.has_nulls()) {
        expand(ungrouped_with_nulls{nulls_at_start, orderby.size(), orderby.null_count()});
      } else {
        expand(ungrouped{orderby.size()});
      }
    }
    return result;
  }

  /**
   * @brief Is the given type supported as an orderby column.
   *
   * @tparam The type of the elements of the orderby column.
   */
  template <typename OrderbyT>
  static constexpr bool is_supported()
  {
    return (cuda::std::is_same_v<OrderbyT, cudf::string_view> &&
            (cuda::std::is_same_v<WindowType, current_row> ||
             cuda::std::is_same_v<WindowType, unbounded>)) ||
           cudf::is_numeric_not_bool<OrderbyT>() || cudf::is_timestamp<OrderbyT>() ||
           cudf::is_fixed_point<OrderbyT>();
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_timestamp<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    direction direction,
    order order,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<typename OrderbyT::duration>;
    CUDF_EXPECTS(!row_delta || cudf::is_duration(row_delta->type()),
                 "Row delta must be a duration type.",
                 cudf::data_type_error);
    CUDF_EXPECTS(!row_delta || row_delta->type().id() == type_to_id<typename OrderbyT::duration>(),
                 "Row delta must have same the resolution as orderby.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, ScalarT>(
      orderby, direction, order, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_fixed_point<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    direction direction,
    order order,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta || (orderby.type().id() == row_delta->type().id()),
                 "Orderby column and row_delta must both be fixed point.",
                 cudf::data_type_error);
    // TODO: Push this requirement onto the caller and just check for
    // equal scales (avoids a kernel launch to rescale)
    CUDF_EXPECTS(!row_delta || row_delta->type().scale() >= orderby.type().scale(),
                 "row_delta must have at least as much scale as orderby column.",
                 cudf::data_type_error);
    if (row_delta && row_delta->type().scale() != orderby.type().scale()) {
      auto const value =
        static_cast<fixed_point_scalar<OrderbyT> const*>(row_delta)->fixed_point_value(stream);
      auto const new_scalar = cudf::fixed_point_scalar<OrderbyT>{
        value.rescaled(numeric::scale_type{orderby.type().scale()}),
        true,
        stream,
        cudf::get_current_device_resource_ref()};
      return window_bounds<OrderbyT>(
        orderby, direction, order, grouping, nulls_at_start, &new_scalar, stream, mr);
    }
    return window_bounds<OrderbyT>(
      orderby, direction, order, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_numeric_not_bool<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    direction direction,
    order order,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta || cudf::have_same_types(orderby, *row_delta),
                 "Orderby column and row_delta must have the same type.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT>(
      orderby, direction, order, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT,
            CUDF_ENABLE_IF(cuda::std::is_same_v<OrderbyT, cudf::string_view> &&
                           (cuda::std::is_same_v<WindowType, current_row> ||
                            cuda::std::is_same_v<WindowType, unbounded>))>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    direction direction,
    order order,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta,
                 "Not expecting window range to have value for string-based window calculation");
    return window_bounds<OrderbyT>(
      orderby, direction, order, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(!is_supported<OrderbyT>())>
  std::unique_ptr<column> operator()(column_view const&,
                                     direction,
                                     order,
                                     std::optional<preprocessed_group_info> const&,
                                     bool,
                                     scalar const*,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }
};
}  // namespace rolling
}  // namespace detail
}  // namespace cudf
