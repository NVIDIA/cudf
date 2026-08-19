/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "counts.hpp"
#include "update_validity.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/reduction/detail/segmented_reduction.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <thrust/adjacent_difference.h>

namespace cudf {
namespace reduction {
namespace compound {
namespace detail {
/**
 * @brief Multi-step reduction for operations such as mean, variance, and standard deviation.
 *
 * @tparam InputType  the input column data-type
 * @tparam ResultType the output data-type
 * @tparam Op         the compound operator derived from `cudf::reduction::op::compound_op`
 *
 * @param col Input column view
 * @param offsets Indices identifying segments
 * @param null_handling Indicates if null elements should be included in the reduction
 * @param ddof Delta degrees of freedom used for standard deviation and variance.
 *             The divisor used is N - ddof, where N represents the number of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Segmented reduce result
 */
template <typename InputType, typename ResultType, typename Op>
std::unique_ptr<column> compound_segmented_reduction(column_view const& col,
                                                     device_span<size_type const> offsets,
                                                     null_policy null_handling,
                                                     size_type ddof,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto d_col              = cudf::column_device_view::create(col, stream);
  auto compound_op        = Op{};
  auto const num_segments = offsets.size() - 1;

  auto result = make_fixed_width_column(
    data_type{type_to_id<ResultType>()}, num_segments, mask_state::UNALLOCATED, stream, mr);
  auto out_itr = result->mutable_view().template begin<ResultType>();

  // Compute counts
  rmm::device_uvector<size_type> counts =
    cudf::reduction::detail::segmented_counts(col.null_mask(),
                                              col.has_nulls(),
                                              offsets,
                                              null_handling,
                                              stream,
                                              cudf::get_current_device_resource_ref());

  // Run segmented reduction
  if (col.has_nulls()) {
    auto nrt = compound_op.template get_null_replacing_element_transformer<ResultType>();
    auto itr = cuda::transform_iterator(d_col->pair_begin<InputType, true>(), nrt);
    cudf::reduction::detail::segmented_reduce(
      itr, offsets.begin(), offsets.end(), out_itr, compound_op, ddof, counts.data(), stream);
  } else {
    auto et  = compound_op.template get_element_transformer<ResultType>();
    auto itr = cuda::transform_iterator(d_col->begin<InputType>(), et);
    cudf::reduction::detail::segmented_reduce(
      itr, offsets.begin(), offsets.end(), out_itr, compound_op, ddof, counts.data(), stream);
  }

  // Compute the output null mask
  cudf::reduction::detail::segmented_update_validity(
    *result, col, offsets, null_handling, std::nullopt, stream, mr);

  return result;
};

template <typename ElementType, typename Op>
struct compound_float_output_dispatcher {
 private:
  template <typename ResultType>
  static constexpr bool is_supported_v()
  {
    return std::is_floating_point_v<ResultType>;
  }

 public:
  template <typename ResultType>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     size_type ddof,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported_v<ResultType>())
  {
    return compound_segmented_reduction<ElementType, ResultType, Op>(
      col, offsets, null_handling, ddof, stream, mr);
  }

  template <typename ResultType>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     null_policy,
                                     size_type,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not is_supported_v<ResultType>())
  {
    CUDF_FAIL("Unsupported output data type");
  }
};

template <typename Op>
struct compound_segmented_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported_v()
  {
    return std::is_arithmetic_v<ElementType>;
  }

 public:
  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     cudf::data_type const output_dtype,
                                     null_policy null_handling,
                                     size_type ddof,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported_v<ElementType>())
  {
    return cudf::type_dispatcher(output_dtype,
                                 compound_float_output_dispatcher<ElementType, Op>(),
                                 col,
                                 offsets,
                                 null_handling,
                                 ddof,
                                 stream,
                                 mr);
  }

  template <typename ElementType>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     cudf::data_type const,
                                     null_policy,
                                     size_type,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref)
    requires(not is_supported_v<ElementType>())
  {
    CUDF_FAIL("Compound operators are not supported for non-arithmetic types");
  }
};

}  // namespace detail
}  // namespace compound
}  // namespace reduction
}  // namespace cudf
