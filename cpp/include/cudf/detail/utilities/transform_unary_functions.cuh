/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief unary functions for cuda::transform_iterator
 * @file transform_unary_functions.cuh
 *
 * These are designed for using as AdaptableUnaryFunction
 * for cuda::transform_iterator.
 * For the detail of example cases,
 * @see iterator.cuh iterator_test.cu
 */

#pragma once

#include <cuda/iterator>
#include <cuda/std/utility>

namespace cudf {
/**
 * @brief Transforms non-null input using `Functor`, and for null, returns `null_replacement`.
 *
 * This functor argument is considered null if second value of functor argument pair is false.
 *
 * @tparam ResultType Output type of `Functor` and null replacement type.
 * @tparam Functor functor to transform first value of argument pair to ResultType.
 */
template <typename ResultType, typename Functor>
struct null_replacing_transformer {
  using type = ResultType;
  Functor f;
  type replacement;
  CUDF_HOST_DEVICE inline null_replacing_transformer(type null_replacement, Functor transformer)
    : f(transformer), replacement(null_replacement)
  {
  }

  template <typename ElementType>
  CUDF_HOST_DEVICE inline type operator()(cuda::std::pair<ElementType, bool> const& pair_value)
  {
    if (pair_value.second)
      return f(pair_value.first);
    else
      return replacement;
  }
};

/**
 * @brief intermediate struct to calculate mean and variance
 * This is an example case to output a struct from column input.
 *
 * this will be used to calculate and hold `sum of values`, 'sum of squares',
 * 'sum of valid count'.
 * Those will be used to compute `mean` (= sum / count)
 * and `variance` (= sum of squares / count - mean^2).
 *
 * @tparam ElementType  element data type of value and value_squared.
 */
template <typename ElementType>
struct meanvar {
  ElementType value;          /// the value
  ElementType value_squared;  /// the value of squared
  cudf::size_type count;      /// the count

  CUDF_HOST_DEVICE inline meanvar(ElementType _value         = 0,
                                  ElementType _value_squared = 0,
                                  cudf::size_type _count     = 0)
    : value(_value), value_squared(_value_squared), count(_count){};

  using this_t = cudf::meanvar<ElementType>;

  CUDF_HOST_DEVICE inline this_t operator+(this_t const& rhs) const
  {
    return this_t((this->value + rhs.value),
                  (this->value_squared + rhs.value_squared),
                  (this->count + rhs.count));
  };

  CUDF_HOST_DEVICE inline bool operator==(this_t const& rhs) const
  {
    return ((this->value == rhs.value) && (this->value_squared == rhs.value_squared) &&
            (this->count == rhs.count));
  };
};

// --------------------------------------------------------------------------
// transformers

/**
 * @brief Transforms a scalar by first casting to another type, and then squaring the result.
 *
 * This struct transforms the output value as
 * `value * value`.
 *
 * This will be used to compute "sum of squares".
 *
 * @tparam  ResultType  scalar data type of output
 */
template <typename ElementType>
struct transformer_squared {
  CUDF_HOST_DEVICE inline ElementType operator()(ElementType const& value)
  {
    return (value * value);
  };
};

/**
 * @brief Uses a scalar value to construct a `meanvar` object.
 * This transforms `cuda::std::pair<ElementType, bool>` into
 * `ResultType = meanvar<ElementType>` form.
 *
 * This struct transforms the value and the squared value and the count at once.
 *
 * @tparam  ElementType         scalar data type of input
 */
template <typename ElementType>
struct transformer_meanvar {
  using ResultType = meanvar<ElementType>;

  CUDF_HOST_DEVICE inline ResultType operator()(cuda::std::pair<ElementType, bool> const& pair)
  {
    ElementType v = pair.first;
    return meanvar<ElementType>(v, v * v, (pair.second) ? 1 : 0);
  };
};

}  // namespace cudf
