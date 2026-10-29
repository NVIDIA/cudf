/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file typed_statistics_chunk.cuh
 * @brief Templated wrapper to generalize statistics chunk reduction and aggregation
 * across different leaf column types
 */

#pragma once

#include "byte_array_view.cuh"
#include "statistics.cuh"
#include "statistics_type_identification.cuh"
#include "temp_storage_wrapper.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <math_constants.h>

namespace cudf {
namespace io {

/**
 * @brief Class used to get reference to members of unions related to statistics calculations
 */
class union_member {
  template <typename U, typename V>
  using reference_type = std::conditional_t<std::is_const_v<U>, V const&, V&>;

 public:
  template <typename T, typename U>
  using type = std::conditional_t<
    std::is_same_v<std::remove_cv_t<T>, string_view>,
    reference_type<U, string_stats>,
    std::conditional_t<std::is_same_v<std::remove_cv_t<T>, statistics::byte_array_view>,
                       reference_type<U, byte_array_stats>,
                       reference_type<U, T>>>;

  template <cuda::std::unsigned_integral T, typename U>
  __device__ static type<T, U> get(U& val)
  {
    return val.u_val;
  }

  template <cuda::std::signed_integral T, typename U>
  __device__ static type<T, U> get(U& val)
    requires(not cuda::std::same_as<T, __int128_t>)
  {
    return val.i_val;
  }

  template <cuda::std::same_as<__int128_t> T, typename U>
  __device__ static type<T, U> get(U& val)
  {
    return val.d128_val;
  }

  template <cuda::std::floating_point T, typename U>
  __device__ static type<T, U> get(U& val)
  {
    return val.fp_val;
  }

  template <cudf::concepts::string_view T, typename U>
  __device__ static type<T, U> get(U& val)
  {
    return val.str_val;
  }

  template <cuda::std::same_as<statistics::byte_array_view> T, typename U>
  __device__ static type<T, U> get(U& val)
  {
    return val.byte_val;
  }
};

namespace detail {

// Integral sums track overflow exactly via net wrap counts; floating point sums are plain
template <typename A>
using sum_type = cuda::std::
  conditional_t<cuda::std::is_floating_point_v<A>, A, reduction::detail::sum_overflow_result<A>>;
template <typename A>
using sum_op = cuda::std::conditional_t<cuda::std::is_floating_point_v<A>,
                                        cuda::std::plus<A>,
                                        reduction::detail::overflow_sum_op<A>>;
template <typename A>
using to_sum = cuda::std::conditional_t<cuda::std::is_floating_point_v<A>,
                                        cuda::std::identity,
                                        reduction::detail::to_sum_overflow<A>>;

}  // namespace detail

/**
 * @brief Templated structure used for merging and gathering of statistics chunks
 *
 * This uses the reduce function to compute the minimum, maximum and aggregate
 * values simultaneously.
 *
 * @tparam T The input type associated with the chunk
 * @tparam is_aggregation_supported Set to true if input type is meant to be aggregated
 */
template <typename T, bool is_aggregation_supported>
struct typed_statistics_chunk {};

template <typename T>
struct typed_statistics_chunk<T, true> {
  using E = typename detail::extrema_type<T>::type;
  using A = typename detail::aggregation_type<T>::type;
  using S = detail::sum_type<A>;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;
  S aggregate{};

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{true};      //!< Nonzero if all merged chunks have a valid sum
  uint8_t has_nan{false};     //!< Nonzero if a NaN was seen (floating point only)

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()), maximum_value(detail::maximum_identity<E>())
  {
  }

  __device__ void reduce(T const& elem)
  {
    non_nulls++;
    if constexpr (cuda::std::is_floating_point_v<T>) { has_nan |= cuda::std::isnan(elem); }
    minimum_value = cuda::std::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = cuda::std::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    aggregate     = detail::sum_op<A>{}(
      aggregate, detail::to_sum<A>{}(detail::aggregation_type<T>::convert(elem)));
    has_minmax = true;
  }

  __device__ void reduce(statistics_chunk const& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value = cuda::std::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = cuda::std::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    aggregate =
      detail::sum_op<A>{}(aggregate, detail::to_sum<A>{}(union_member::get<A>(chunk.sum)));
    has_sum &= chunk.has_sum;
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
    has_nan |= chunk.has_nan;
  }
};

template <typename T>
struct typed_statistics_chunk<T, false> {
  using E = typename detail::extrema_type<T>::type;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{false};     //!< Nonzero if sum is valid
  uint8_t has_nan{false};     //!< Nonzero if a NaN was seen (floating point only)

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()), maximum_value(detail::maximum_identity<E>())
  {
  }

  __device__ void reduce(T const& elem)
  {
    non_nulls++;
    if constexpr (cuda::std::is_floating_point_v<T>) { has_nan |= cuda::std::isnan(elem); }
    minimum_value = cuda::std::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = cuda::std::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    has_minmax    = true;
  }

  __device__ void reduce(statistics_chunk const& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value = cuda::std::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = cuda::std::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
    has_nan |= chunk.has_nan;
  }
};

/**
 * @brief Function to reduce members of a typed_statistics_chunk across a thread block
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @tparam block_size Dimension of the thread block
 * @param chunk The input typed_statistics_chunk
 * @param storage Temporary storage to be used by cub calls
 */
template <typename T, bool include_aggregate, int block_size>
__inline__ __device__ typed_statistics_chunk<T, include_aggregate> block_reduce(
  typed_statistics_chunk<T, include_aggregate>& chunk, detail::storage_wrapper<block_size>& storage)
{
  typed_statistics_chunk<T, include_aggregate> output_chunk = chunk;

  using E              = typename detail::extrema_type<T>::type;
  using extrema_reduce = cub::BlockReduce<E, block_size>;
  using count_reduce   = cub::BlockReduce<uint32_t, block_size>;

  output_chunk.minimum_value =
    extrema_reduce(storage.template get<E>()).Reduce(output_chunk.minimum_value, cuda::minimum{});
  __syncthreads();
  output_chunk.maximum_value =
    extrema_reduce(storage.template get<E>()).Reduce(output_chunk.maximum_value, cuda::maximum{});
  __syncthreads();
  output_chunk.non_nulls =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.non_nulls);
  __syncthreads();
  output_chunk.null_count =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.null_count);
  __syncthreads();
  output_chunk.has_minmax = __syncthreads_or(output_chunk.has_minmax);
  output_chunk.has_nan    = __syncthreads_or(output_chunk.has_nan);

  // FIXME : Is another syncthreads needed here?
  if constexpr (include_aggregate) {
    output_chunk.has_sum = __syncthreads_and(output_chunk.has_sum);
    if (output_chunk.has_minmax) {
      using A                = typename detail::aggregation_type<T>::type;
      using S                = detail::sum_type<A>;
      using aggregate_reduce = cub::BlockReduce<S, block_size>;
      output_chunk.aggregate = aggregate_reduce(storage.template get<S>())
                                 .Reduce(output_chunk.aggregate, detail::sum_op<A>{});
    }
  }
  return output_chunk;
}

/**
 * @brief Function to convert typed_statistics_chunk into statistics_chunk
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @param chunk The input typed_statistics_chunk
 */
template <typename T, bool include_aggregate>
__inline__ __device__ statistics_chunk
get_untyped_chunk(typed_statistics_chunk<T, include_aggregate> const& chunk)
{
  using E = typename detail::extrema_type<T>::type;
  statistics_chunk stat{};
  stat.non_nulls  = chunk.non_nulls;
  stat.null_count = chunk.null_count;
  stat.has_minmax = chunk.has_minmax;
  stat.has_nan    = chunk.has_nan;
  if (chunk.has_minmax) {
    if constexpr (std::is_floating_point_v<E>) {
      union_member::get<E>(stat.min_value) =
        (chunk.minimum_value != 0.0) ? chunk.minimum_value : CUDART_NEG_ZERO;
      union_member::get<E>(stat.max_value) =
        (chunk.maximum_value != 0.0) ? chunk.maximum_value : CUDART_ZERO;
    } else {
      union_member::get<E>(stat.min_value) = chunk.minimum_value;
      union_member::get<E>(stat.max_value) = chunk.maximum_value;
    }
  }
  if constexpr (include_aggregate) {
    // invalidate the sum if it overflowed
    using A = typename detail::aggregation_type<T>::type;
    if constexpr (cuda::std::is_floating_point_v<A>) {
      stat.has_sum                   = chunk.has_sum and cuda::std::isfinite(chunk.aggregate);
      union_member::get<A>(stat.sum) = chunk.aggregate;
    } else {
      stat.has_sum                   = chunk.has_sum and chunk.aggregate.wraps == 0;
      union_member::get<A>(stat.sum) = chunk.aggregate.sum;
    }
  }
  return stat;
}

}  // namespace io
}  // namespace cudf
