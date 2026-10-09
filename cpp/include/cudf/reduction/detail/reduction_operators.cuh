/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/transform_unary_functions.cuh>
#include <cudf/types.hpp>  //for CUDF_HOST_DEVICE

#include <cmath>

namespace cudf {
namespace reduction {
namespace detail {
// intermediate data structure to compute `var`, `std`
template <typename ResultType>
struct var_std {
  // Uses the pairwise approach of Chan, Golub, and LeVeque,
  // _Algorithms for computing the sample variance: analysis and
  // recommendations_ (1983)
  // https://doi.org/10.1080/00031305.1983.10483115
  // Also http://www.cs.yale.edu/publications/techreports/tr222.pdf
  // This is a modification of Youngs and Cramer's online approach.
  // We track the running mean, rather than the running sum so that columns whose sum
  // overflows but whose mean does not produce finite variance.
  ResultType running_mean;
  ResultType running_square_deviations;
  size_type count;

  CUDF_HOST_DEVICE inline var_std(ResultType mean = 0, ResultType m2 = 0, size_type n = 0)
    : running_mean(mean), running_square_deviations(m2), count(n){};

  using this_t = var_std<ResultType>;

  CUDF_HOST_DEVICE inline this_t operator+(this_t const& rhs) const
  {
    auto const m = this->count;
    auto const n = rhs.count;
    // Empty states are identities, including when the other state is non-finite.
    if (m == 0) { return rhs; }
    if (n == 0) { return *this; }
    auto const total   = m + n;
    auto const delta   = rhs.running_mean - running_mean;
    auto const delta_n = delta / static_cast<ResultType>(total);
    return {running_mean + delta_n * n,
            running_square_deviations + rhs.running_square_deviations + delta * delta_n * m * n,
            total};
  };
};

// transformer for `struct var_std` in order to compute `var`, `std`
template <typename ResultType>
struct transformer_var_std {
  using OutputType = var_std<ResultType>;

  CUDF_HOST_DEVICE inline OutputType operator()(ResultType const& value) { return {value, 0, 1}; };
};

// ------------------------------------------------------------------------
// Definitions of device struct for reduction operation
// all `op::xxx` must have `op` and `transformer`
// `op`  is used to compute the reduction at device
// `transformer` is used to convert elements for computing the reduction at device.
// By default `transformer` is static type conversion to ResultType.
// In some cases, it could be square or abs or complex operations
namespace op {
/**
 * @brief  Simple reduction operator CRTP Base class
 *
 * @tparam Derived operator with simple_op interface
 */
template <typename Derived>
struct simple_op {
  /**
   * @brief Get binary operator functor for reduction
   *
   * @return binary operator functor object
   */
  auto get_binary_op()
  {
    using binary_op = typename Derived::op;
    return binary_op{};
  }

  /**
   * @brief Get transformer functor for transforming input column
   * which inturn is used by reduction binary operator
   *
   * @tparam ResultType output type for element transformer
   *
   * @return element transformer functor object
   */
  template <typename ResultType>
  auto get_element_transformer()
  {
    using element_transformer = typename Derived::transformer<ResultType>;
    return element_transformer{};
  }

  /**
   * @brief Get transformer functor for transforming input column pair iterator
   * which is used by reduction binary operator
   *
   * @tparam ResultType output type for element transformer
   *
   * @return element transformer functor object
   */
  template <typename ResultType>
  auto get_null_replacing_element_transformer()
  {
    using element_transformer = typename Derived::transformer<ResultType>;
    return null_replacing_transformer<ResultType, element_transformer>{get_identity<ResultType>(),
                                                                       element_transformer{}};
  }

  /**
   * @brief get identity value of type `T` for binary reduction operator
   *
   * @tparam T data type of identity value
   *
   * @return identity value
   */
  template <typename T>
  constexpr T get_identity()
  {
    return Derived::op::template identity<T>();
  }
};

// `sum`, `product`, `sum_of_squares`, `min`, `max` are used at simple_reduction
// interface is defined by CRTP class simple_op

// operator for `sum`
struct sum : public simple_op<sum> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `product`
struct product : public simple_op<product> {
  using op = cudf::DeviceProduct;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `sum_of_squares`
struct sum_of_squares : public simple_op<sum_of_squares> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::transformer_squared<ResultType>;
};

// operator for `min`
struct min : public simple_op<min> {
  using op = cudf::DeviceMin;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `max`
struct max : public simple_op<max> {
  using op = cudf::DeviceMax;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `bit_and`
struct bit_and : public simple_op<bit_and> {
  using op = cudf::DeviceBitAnd;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `bit_or`
struct bit_or : public simple_op<bit_or> {
  using op = cudf::DeviceBitOr;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

// operator for `bit_xor`
struct bit_xor : public simple_op<bit_xor> {
  using op = cudf::DeviceBitXor;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;
};

/**
 * @brief  Compound reduction operator CRTP Base class
 * This template class defines the interface for compound operators
 * In addition to interface defined by simple_op CRTP, this class defines
 * interface for final result transformation.
 *
 * @tparam Derived compound operators derived from compound_op
 */
template <typename Derived>
struct compound_op : public simple_op<Derived> {
  /**
   * @copydoc simple_op<Derived>::template get_null_replacing_element_transformer<ResultType>()
   */
  template <typename ResultType>
  auto get_null_replacing_element_transformer()
  {
    using element_transformer = typename Derived::transformer<ResultType>;
    using OutputType          = typename Derived::intermediate<ResultType>::IntermediateType;
    return null_replacing_transformer<OutputType, element_transformer>{
      simple_op<Derived>::template get_identity<OutputType>(), element_transformer{}};
  }
  /**
   * @brief  computes the transformed result from result of simple operator.
   *
   * @tparam ResultType output type of compound reduction operator
   * @tparam IntermediateType output type of simple reduction operator
   * @param input output of simple reduction as input for result transformation
   * @param count validity count
   * @param ddof  `ddof` parameter used by variance and standard deviation
   *
   * @return transformed output result of compound operator
   */
  template <typename ResultType, typename IntermediateType>
  CUDF_HOST_DEVICE inline static ResultType compute_result(IntermediateType const& input,
                                                           cudf::size_type const& count,
                                                           cudf::size_type const& ddof)
  {
    // Enforced interface
    return Derived::template intermediate<ResultType>::compute_result(input, count, ddof);
  }
};

// `mean`, `variance`, `standard_deviation` are used at compound_reduction
// compound_reduction requires intermediate::IntermediateType and
// intermediate::compute_result IntermediateType is the intermediate data
// structure type of a single reduction call, it is also used as OutputType of
// cudf::reduction::detail::reduce at compound_reduction. compute_result
// computes the final ResultType from the IntermediateType.
// intermediate::compute_result method is enforced by CRTP base class compound_op

// operator for `mean`
struct mean : public compound_op<mean> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::detail::cast_fn<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType = ResultType;  // sum value

    // compute `mean` from intermediate type `IntermediateType`
    CUDF_HOST_DEVICE inline static ResultType compute_result(IntermediateType const& input,
                                                             cudf::size_type const& count,
                                                             cudf::size_type const& ddof)
    {
      return (input / count);
    };
  };
};

// operator for `variance`
struct variance : public compound_op<variance> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::reduction::detail::transformer_var_std<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType = var_std<ResultType>;  // with mean, M2, and count

    // compute `variance` from intermediate type `IntermediateType`
    CUDF_HOST_DEVICE inline static ResultType compute_result(IntermediateType const& input,
                                                             cudf::size_type const& count,
                                                             cudf::size_type const& ddof)
    {
      return input.running_square_deviations / (count - ddof);
    };
  };
};

// operator for `standard deviation`
struct standard_deviation : public compound_op<standard_deviation> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::reduction::detail::transformer_var_std<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType = var_std<ResultType>;  // with mean, M2, and count

    // compute `standard deviation` from intermediate type `IntermediateType`
    CUDF_HOST_DEVICE inline static ResultType compute_result(IntermediateType const& input,
                                                             cudf::size_type const& count,
                                                             cudf::size_type const& ddof)
    {
      using intermediateOp = variance::template intermediate<ResultType>;
      ResultType var       = intermediateOp::compute_result(input, count, ddof);

      return static_cast<ResultType>(std::sqrt(var));
    };
  };
};
}  // namespace op
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
