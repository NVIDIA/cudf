/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/quantiles.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::scalar> quantile(column_view const& col,
                                       double quantile_value,
                                       cudf::interpolation interpolation,
                                       cudf::data_type const output_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto current_mr = cudf::get_current_device_resource_ref();
  auto sorted_indices =
    cudf::detail::sorted_order(table_view{{col}}, {}, {null_order::AFTER}, stream, current_mr);
  auto valid_sorted_indices =
    cudf::detail::split(*sorted_indices, {col.size() - col.null_count()}, stream)[0];
  // only perform an exact quantile calculation for output-type FLOAT64
  // @see cudf::quantile for more details on this parameter
  auto exact   = output_type.id() == cudf::type_id::FLOAT64;
  auto col_ptr = cudf::detail::quantile(
    col, {quantile_value}, interpolation, valid_sorted_indices, exact, stream, current_mr);
  auto result = cudf::detail::get_element(*col_ptr, 0, stream, mr);
  if (result->type().id() == output_type.id()) { return result; }
  return cudf::type_dispatcher(output_type,
                               cudf::reduction::simple::detail::cast_numeric_scalar_fn<double>{},
                               static_cast<numeric_scalar<double>*>(result.get()),
                               stream,
                               mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
