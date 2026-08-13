/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "counts.hpp"
#include "update_validity.hpp"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/reduction/detail/segmented_reduction.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <optional>
#include <type_traits>

namespace cudf {
namespace reduction {
namespace simple {
namespace detail {

/**
 * @brief Segment reduction for 'sum', 'product', 'min', 'max', 'sum of squares', etc
 * which directly compute the reduction by a single step reduction call.
 *
 * @tparam InputType    the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param init Optional initial value of the reduction
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType, typename ResultType, typename Op>
std::unique_ptr<column> simple_segmented_reduction(
  column_view const& col,
  device_span<size_type const> offsets,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto dcol               = cudf::column_device_view::create(col, stream);
  auto simple_op          = Op{};
  auto const num_segments = offsets.size() - 1;

  auto const binary_op = cudf::detail::cast_functor<ResultType>(simple_op.get_binary_op());

  // Cast initial value
  ResultType initial_value = [&] {
    if (init.has_value() && init.value().get().is_valid(stream)) {
      using ScalarType = cudf::scalar_type_t<InputType>;
      auto input_value = static_cast<ScalarType const*>(&init.value().get())->value(stream);
      return static_cast<ResultType>(input_value);
    } else {
      return simple_op.template get_identity<ResultType>();
    }
  }();

  auto const result_type =
    cudf::is_fixed_point(col.type()) ? col.type() : data_type{type_to_id<ResultType>()};
  auto result =
    make_fixed_width_column(result_type, num_segments, mask_state::UNALLOCATED, stream, mr);
  auto outit = result->mutable_view().template begin<ResultType>();

  if (col.has_nulls()) {
    auto f  = simple_op.template get_null_replacing_element_transformer<ResultType>();
    auto it = thrust::make_transform_iterator(dcol->pair_begin<InputType, true>(), f);
    cudf::reduction::detail::segmented_reduce(
      it, offsets.begin(), offsets.end(), outit, binary_op, initial_value, stream);
  } else {
    auto f  = simple_op.template get_element_transformer<ResultType>();
    auto it = thrust::make_transform_iterator(dcol->begin<InputType>(), f);
    cudf::reduction::detail::segmented_reduce(
      it, offsets.begin(), offsets.end(), outit, binary_op, initial_value, stream);
  }

  // Compute the output null mask
  cudf::reduction::detail::segmented_update_validity(
    *result, col, offsets, null_handling, init, stream, mr);

  return result;
}

template <typename T>
struct reduce_argminmax_fn {
  column_device_view const d_col;  // column data
  bool const arg_min;              // true if argmin, otherwise argmax
  null_policy null_handler;        // include or exclude nulls

  __device__ inline auto operator()(size_type const& lhs_idx, size_type const& rhs_idx) const
  {
    // CUB segmented reduce calls with OOB indices
    if (lhs_idx < 0 || lhs_idx >= d_col.size()) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= d_col.size()) { return lhs_idx; }
    if (d_col.is_null(lhs_idx)) { return null_handler == null_policy::INCLUDE ? lhs_idx : rhs_idx; }
    if (d_col.is_null(rhs_idx)) { return null_handler == null_policy::INCLUDE ? rhs_idx : lhs_idx; }
    auto const less = d_col.element<T>(lhs_idx) < d_col.element<T>(rhs_idx);
    return less == arg_min ? lhs_idx : rhs_idx;
  }
};

/**
 * @brief String segmented reduction for 'min', 'max'.
 *
 * This algorithm uses argmin/argmax as a custom comparator to build a gather
 * map, then builds the output.
 *
 * @tparam InputType    the input column data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(std::is_same_v<Op, cudf::reduction::detail::op::min> ||
                         std::is_same_v<Op, cudf::reduction::detail::op::max>)>
std::unique_ptr<column> string_segmented_reduction(column_view const& col,
                                                   device_span<size_type const> offsets,
                                                   null_policy null_handling,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  // Pass to simple_segmented_reduction, get indices to gather, perform gather here.
  auto device_col = cudf::column_device_view::create(col, stream);

  auto it                 = cuda::counting_iterator<cudf::size_type>{0};
  auto const num_segments = static_cast<size_type>(offsets.size()) - 1;

  bool constexpr is_argmin = std::is_same_v<Op, cudf::reduction::detail::op::min>;
  auto string_comparator   = reduce_argminmax_fn<InputType>{*device_col, is_argmin, null_handling};
  auto constexpr identity =
    is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL;

  auto gather_map = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_segments, mask_state::UNALLOCATED, stream, mr);

  auto gather_map_it = gather_map->mutable_view().begin<size_type>();

  cudf::reduction::detail::segmented_reduce(
    it, offsets.begin(), offsets.end(), gather_map_it, string_comparator, identity, stream);

  auto result = std::move(cudf::detail::gather(table_view{{col}},
                                               *gather_map,
                                               cudf::out_of_bounds_policy::NULLIFY,
                                               cudf::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr)
                            ->release()[0]);

  // Compute the output null mask
  cudf::reduction::detail::segmented_update_validity(
    *result, col, offsets, null_handling, std::nullopt, stream, mr);

  return result;
}

template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(!std::is_same_v<Op, cudf::reduction::detail::op::min>() &&
                         !std::is_same_v<Op, cudf::reduction::detail::op::max>())>
std::unique_ptr<column> string_segmented_reduction(column_view const& col,
                                                   device_span<size_type const> offsets,
                                                   null_policy null_handling,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("Segmented reduction on string column only supports min and max reduction.");
}

/**
 * @brief Specialization for fixed-point segmented reduction
 *
 * @tparam InputType    the input column data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param init Optional initial value for the reduction
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType, typename Op>
std::unique_ptr<column> fixed_point_segmented_reduction(
  column_view const& col,
  device_span<size_type const> offsets,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using RepType = device_storage_type_t<InputType>;
  auto result =
    simple_segmented_reduction<RepType, RepType, Op>(col, offsets, null_handling, init, stream, mr);
  auto const scale = [&] {
    if constexpr (std::is_same_v<Op, cudf::reduction::detail::op::product>) {
      // The product aggregation requires updating the scale of the fixed-point output column.
      // The output scale needs to be the maximum count of all segments multiplied by
      // the input scale value.
      rmm::device_uvector<size_type> const counts =
        cudf::reduction::detail::segmented_counts(col.null_mask(),
                                                  col.has_nulls(),
                                                  offsets,
                                                  null_policy::EXCLUDE,  // do not count nulls
                                                  stream,
                                                  cudf::get_current_device_resource_ref());

      auto const max_count =
        thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       counts.begin(),
                       counts.end(),
                       size_type{0},
                       cuda::maximum<size_type>{});

      auto const new_scale = numeric::scale_type{col.type().scale() * max_count};

      // adjust values in each segment to match the new scale
      auto const d_col = column_device_view::create(col, stream);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        d_col->begin<InputType>(),
                        d_col->end<InputType>(),
                        d_col->begin<InputType>(),
                        cuda::proclaim_return_type<InputType>(
                          [new_scale] __device__(auto fp) { return fp.rescaled(new_scale); }));
      return new_scale;
    }

    if constexpr (std::is_same_v<Op, cudf::reduction::detail::op::sum_of_squares>) {
      return numeric::scale_type{col.type().scale() * 2};
    }

    return numeric::scale_type{col.type().scale()};
  }();

  auto const size       = result->size();        // get these before
  auto const null_count = result->null_count();  // release() is called
  auto contents         = result->release();

  return std::make_unique<column>(data_type{type_to_id<InputType>(), scale},
                                  size,
                                  std::move(*(contents.data.release())),
                                  std::move(*(contents.null_mask.release())),
                                  null_count);
}

/**
 * @brief Call reduce and return a column of type bool.
 *
 * This is used by operations `any()` and `all()`.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct bool_result_column_dispatcher {
  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<ElementType>())
  {
    return simple_segmented_reduction<ElementType, bool, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref)
    requires(not cudf::is_numeric<ElementType>())
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a column of type matching the input column.
 *
 * This is used by operations `min()` and `max()`.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct same_column_type_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !(cudf::is_dictionary<ElementType>() || std::is_same_v<ElementType, cudf::list_view> ||
             std::is_same_v<ElementType, cudf::struct_view>);
  }

 public:
  template <typename ElementType,
            CUDF_ENABLE_IF(is_supported<ElementType>() &&
                           !std::is_same_v<ElementType, string_view> &&
                           !cudf::is_fixed_point<ElementType>())>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return simple_segmented_reduction<ElementType, ElementType, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType,
            CUDF_ENABLE_IF(is_supported<ElementType>() && std::is_same_v<ElementType, string_view>)>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(!init.has_value(), "Initial value not supported for strings");

    return string_segmented_reduction<ElementType, Op>(col, offsets, null_handling, stream, mr);
  }

  template <typename ElementType,
            CUDF_ENABLE_IF(is_supported<ElementType>() && cudf::is_fixed_point<ElementType>())>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return fixed_point_segmented_reduction<ElementType, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType, CUDF_ENABLE_IF(!is_supported<ElementType>())>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a column of the type specified.
 *
 * This is used by operations such as sum(), product(), sum_of_squares(), etc
 * It only supports numeric types. If the output type is not the
 * same as the input type, an extra cast operation may occur.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct column_type_dispatcher {
  /**
   * @brief Specialization for reducing floating-point column types to any output type.
   *
   * This is called when the output_type does not match the ElementType.
   * The input values are promoted to double (via transform-iterator) for the
   * reduce calculation. The result is then cast to the specified output_type.
   */
  template <typename ElementType>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         device_span<size_type const> offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
    requires(std::is_floating_point<ElementType>::value)
  {
    // Floats are computed in double precision and then cast to the output type
    auto result = simple_segmented_reduction<ElementType, double, Op>(
      col, offsets, null_handling, init, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Specialization for reducing integer column types to any output type.
   *
   * This is called when the output_type does not match the ElementType.
   * The input values are promoted to int64_t (via transform-iterator) for the
   * reduce calculation. The result is then cast to the specified output_type.
   *
   * For uint64_t case, the only reasonable output_type is also UINT64 and
   * this is not called when the input/output types match.
   */
  template <typename ElementType>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         device_span<size_type const> offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
    requires(std::is_integral<ElementType>::value)
  {
    // Integers are computed in int64 precision and then cast to the output type.
    auto result = simple_segmented_reduction<ElementType, int64_t, Op>(
      col, offsets, null_handling, init, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Called by the type-dispatcher to reduce the input column `col` using
   * the `Op` operation.
   *
   * @tparam ElementType The input column type or key type
   * @param col Input column (must be numeric)
   * @param offsets Indices to segment boundaries
   * @param output_type Requested type of the output column
   * @param null_handling How null entries are processed within each segment
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     data_type const output_type,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<ElementType>())
  {
    // If the output type matches the input type, then reduce using that type
    if (output_type.id() == cudf::type_to_id<ElementType>()) {
      return simple_segmented_reduction<ElementType, ElementType, Op>(
        col, offsets, null_handling, init, stream, mr);
    }
    // otherwise, reduce and map to output type
    return reduce_numeric<ElementType>(col, offsets, output_type, null_handling, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     data_type const output_type,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<ElementType>())
  {
    CUDF_EXPECTS(output_type == col.type(), "Output type must be same as input column type.");
    return fixed_point_segmented_reduction<ElementType, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     data_type const,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
                                     rmm::cuda_stream_view,
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
