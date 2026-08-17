/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nested_types_extrema_utils.cuh"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction.cuh>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace reduction {
namespace simple {
namespace detail {
/**
 * @brief Reduction for 'sum', 'product', 'min', 'max', 'sum of squares', and bitwise AND/OR/XOR
 * which directly compute the reduction by a single step reduction call
 *
 * @tparam ElementType  the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param init Optional initial value of the reduction
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Output scalar in device memory
 */
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> simple_reduction(column_view const& col,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  // reduction by iterator
  auto dcol      = cudf::column_device_view::create(col, stream);
  auto simple_op = Op{};

  // Cast initial value
  std::optional<ResultType> const initial_value = [&] {
    if (init.has_value() && init.value().get().is_valid(stream)) {
      using ScalarType = cudf::scalar_type_t<ElementType>;
      auto input_value = static_cast<ScalarType const*>(&init.value().get())->value(stream);
      return std::optional<ResultType>(static_cast<ResultType>(input_value));
    } else {
      return std::optional<ResultType>(std::nullopt);
    }
  }();

  auto result = [&] {
    if (col.has_nulls()) {
      auto f  = simple_op.template get_null_replacing_element_transformer<ResultType>();
      auto it = thrust::make_transform_iterator(dcol->pair_begin<ElementType, true>(), f);
      return cudf::reduction::detail::reduce(it, col.size(), simple_op, initial_value, stream, mr);
    } else {
      auto f  = simple_op.template get_element_transformer<ResultType>();
      auto it = thrust::make_transform_iterator(dcol->begin<ElementType>(), f);
      return cudf::reduction::detail::reduce(it, col.size(), simple_op, initial_value, stream, mr);
    }
  }();

  // set scalar is valid
  result->set_valid_async(
    col.null_count() < col.size() && (!init.has_value() || init.value().get().is_valid(stream)),
    stream);
  return result;
}

/**
 * @brief Reduction for `sum`, `product`, `min` and `max` for decimal types
 *
 * @tparam DecimalXX  The `decimal32`, `decimal64` or `decimal128` type
 * @tparam Op         The operator of cudf::reduction::op::
 *
 * @param col         Input column of data to reduce
 * @param init        Optional initial value of the reduction
 * @param stream      Used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned scalar's device memory
 * @return            Output scalar in device memory
 */
template <typename DecimalXX, typename Op>
std::unique_ptr<scalar> fixed_point_reduction(
  column_view const& col,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  using Type = device_storage_type_t<DecimalXX>;

  auto result = simple_reduction<Type, Type, Op>(col, init, stream, mr);

  auto const scale = [&] {
    if (std::is_same_v<Op, cudf::reduction::detail::op::product>) {
      auto const valid_count = static_cast<int32_t>(col.size() - col.null_count());
      return numeric::scale_type{col.type().scale() * (valid_count + (init.has_value() ? 1 : 0))};
    } else if (std::is_same_v<Op, cudf::reduction::detail::op::sum_of_squares>) {
      return numeric::scale_type{col.type().scale() * 2};
    }
    return numeric::scale_type{col.type().scale()};
  }();

  auto const val = static_cast<cudf::scalar_type_t<Type>*>(result.get());
  auto result_scalar =
    cudf::make_fixed_point_scalar<DecimalXX>(val->value(stream), scale, stream, mr);
  result_scalar->set_valid_async(
    col.null_count() < col.size() && (!init.has_value() || init.value().get().is_valid(stream)),
    stream);
  return result_scalar;
}

/**
 * @brief Reduction for 'sum', 'product', 'sum of squares' for dictionary columns.
 *
 * @tparam ElementType  The key type of the input dictionary column.
 * @tparam ResultType   The output data-type for the resulting scalar
 * @tparam Op           The operator of cudf::reduction::op::
 *
 * @param col Input dictionary column of data to reduce
 * @param init Optional initial value of the reduction
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Output scalar in device memory
 */
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> dictionary_reduction(
  column_view const& col,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!init.has_value(), "Initial value not supported for dictionary reductions");

  auto dcol      = cudf::column_device_view::create(col, stream);
  auto simple_op = Op{};

  auto result = [&] {
    auto f = simple_op.template get_null_replacing_element_transformer<ResultType>();
    auto p =
      cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType>(*dcol, col.has_nulls());
    auto it = thrust::make_transform_iterator(p, f);
    return cudf::reduction::detail::reduce(it, col.size(), simple_op, {}, stream, mr);
  }();

  // set scalar is valid
  result->set_valid_async(
    col.null_count() < col.size() && (!init.has_value() || init.value().get().is_valid(stream)),
    stream);
  return result;
}

/**
 * @brief Convert a numeric scalar to another numeric scalar.
 *
 * The input value and validity are cast to the output scalar.
 *
 * @tparam InputType The type of the input scalar to copy from
 * @tparam OutputType The output scalar type to copy to
 */
template <typename InputType, typename OutputType>
struct assign_scalar_fn {
  __device__ void operator()()
  {
    d_output.set_value(static_cast<OutputType>(d_input.value()));
    d_output.set_valid(d_input.is_valid());
  }

  cudf::numeric_scalar_device_view<InputType> d_input;
  cudf::numeric_scalar_device_view<OutputType> d_output;
};

/**
 * @brief A type-dispatcher functor for converting a numeric scalar.
 *
 * The InputType is known and the dispatch is on the ResultType
 * which is the output numeric scalar type.
 *
 * @tparam InputType The scalar type to convert from
 */
template <typename InputType>
struct cast_numeric_scalar_fn {
 private:
  template <typename ResultType>
  static constexpr bool is_supported()
  {
    return cudf::is_convertible<InputType, ResultType>::value && cudf::is_numeric<ResultType>();
  }

 public:
  template <typename ResultType>
  std::unique_ptr<scalar> operator()(numeric_scalar<InputType>* input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported<ResultType>())
  {
    auto d_input  = cudf::get_scalar_device_view(*input);
    auto result   = std::make_unique<numeric_scalar<ResultType>>(ResultType{}, true, stream, mr);
    auto d_output = cudf::get_scalar_device_view(*result);
    cudf::detail::device_single_thread(assign_scalar_fn<InputType, ResultType>{d_input, d_output},
                                       stream);
    return result;
  }

  template <typename ResultType>
  std::unique_ptr<scalar> operator()(numeric_scalar<InputType>*,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not is_supported<ResultType>())
  {
    CUDF_FAIL("input data type is not convertible to output data type");
  }
};

/**
 * @brief Call reduce and return a scalar of type bool.
 *
 * This is used by operations `any()` and `all()`.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct bool_result_element_dispatcher {
  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_arithmetic_v<ElementType>)
  {
    return simple_reduction<ElementType, bool, Op>(col, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     std::optional<std::reference_wrapper<scalar const>>,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not std::is_arithmetic_v<ElementType>)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a scalar of type matching the input column.
 *
 * This is used by operations `min()` and `max()`, and bitwise operations.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct same_element_type_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !cudf::is_dictionary<ElementType>() && !std::is_same_v<ElementType, void>;
  }

  template <typename IndexType>
  std::unique_ptr<scalar> resolve_key(column_view const& keys,
                                      scalar const& keys_index,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
    requires(cudf::is_index_type<IndexType>())
  {
    auto& index = static_cast<numeric_scalar<IndexType> const&>(keys_index);
    return cudf::detail::get_element(keys, index.value(stream), stream, mr);
  }

  template <typename IndexType>
  std::unique_ptr<scalar> resolve_key(column_view const&,
                                      scalar const&,
                                      cuda::stream_ref,
                                      rmm::device_async_resource_ref)
    requires(!cudf::is_index_type<IndexType>())
  {
    CUDF_FAIL("index type expected for dictionary column");
  }

 public:
  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_nested<ElementType>() &&
             (std::is_same_v<Op, cudf::reduction::detail::op::min> ||
              std::is_same_v<Op, cudf::reduction::detail::op::max>))
  {
    CUDF_EXPECTS(!init.has_value(), "Initial value not supported for nested type reductions");

    if (input.is_empty()) { return cudf::make_empty_scalar_like(input, stream, mr); }

    // We will do reduction to find the ARGMIN/ARGMAX index, then return the element at that index.
    auto const binop_generator =
      cudf::reduction::detail::arg_minmax_binop_generator::create<Op>(input, stream);
    auto const binary_op = cudf::detail::cast_functor<size_type>(binop_generator.binop());
    auto const minmax_idx =
      thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<cudf::size_type>{0},
                     cuda::counting_iterator{input.size()},
                     size_type{0},
                     binary_op);

    return cudf::detail::get_element(input, minmax_idx, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported<ElementType>() && !cudf::is_nested<ElementType>() &&
             !cudf::is_fixed_point<ElementType>())
  {
    if (!cudf::is_dictionary(col.type())) {
      return simple_reduction<ElementType, ElementType, Op>(col, init, stream, mr);
    }
    auto mm = cudf::reduction::detail::minmax(col, stream, mr);
    return std::is_same_v<Op, cudf::reduction::detail::op::min> ? std::move(mm.first)
                                                                : std::move(mm.second);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<ElementType>())
  {
    return fixed_point_reduction<ElementType, Op>(col, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     std::optional<std::reference_wrapper<scalar const>>,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not is_supported<ElementType>())
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a scalar of the type specified.
 *
 * This is used by operations sum(), product(), and sum_of_squares().
 * It only supports numeric types. If the output type is not the
 * same as the input type, an extra cast operation may incur.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct element_type_dispatcher {
  /**
   * @brief Specialization for reducing numeric column types to any output type.
   */
  template <typename ElementType, typename OutputType>
  std::unique_ptr<scalar> reduce(column_view const& col,
                                 std::optional<std::reference_wrapper<scalar const>> init,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
  {
    return !cudf::is_dictionary(col.type())
             ? simple_reduction<ElementType, OutputType, Op>(col, init, stream, mr)
             : dictionary_reduction<ElementType, OutputType, Op>(col, init, stream, mr);
  }

  /**
   * @brief Called by the type-dispatcher to reduce the input column `col` using
   * the `Op` operation.
   *
   * @tparam ElementType The input column type or key type
   * @param col Input column (must be numeric)
   * @param output_type Requested type of the scalar result
   * @param init Optional initial value of the reduction
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     data_type const output_type,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<ElementType>())
  {
    if (output_type.id() == cudf::type_to_id<ElementType>()) {
      return reduce<ElementType, ElementType>(col, init, stream, mr);
    }

    using OutputType  = std::conditional_t<std::is_integral_v<ElementType>, int64_t, double>;
    auto const result = reduce<ElementType, OutputType>(col, init, stream, mr);

    // This will cast the result to the output_type.
    return cudf::type_dispatcher(output_type,
                                 cast_numeric_scalar_fn<OutputType>{},
                                 static_cast<numeric_scalar<OutputType>*>(result.get()),
                                 stream,
                                 mr);
  }

  /**
   * @brief Specialization for reducing fixed_point column types to fixed_point number
   */
  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     data_type const output_type,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<ElementType>())
  {
    CUDF_EXPECTS(output_type == col.type(), "Output type must be same as input column type.");
    return fixed_point_reduction<ElementType, Op>(col, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     data_type const,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not cudf::is_numeric<ElementType>() and not cudf::is_fixed_point<ElementType>())
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

}  // namespace detail
}  // namespace simple
}  // namespace reduction
}  // namespace cudf
