/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

struct count_scalar_fn {
  template <typename T>
    requires(cudf::is_numeric_not_bool<T>())
  std::unique_ptr<cudf::scalar> operator()(size_type count,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto const value = static_cast<T>(count);
    return cudf::make_fixed_width_scalar<T>(value, stream, mr);
  }

  template <typename T>
    requires(not cudf::is_numeric_not_bool<T>())
  std::unique_ptr<cudf::scalar> operator()(size_type,
                                           cuda::stream_ref,
                                           rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("COUNT is not supported for boolean or non-numeric types", std::invalid_argument);
  }
};
}  // namespace

std::unique_ptr<cudf::scalar> count(column_view const& col,
                                    cudf::null_policy null_handling,
                                    cudf::data_type const output_dtype,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto const count = col.size() - (null_handling == null_policy::EXCLUDE ? col.null_count() : 0);
  return cudf::type_dispatcher(output_dtype, count_scalar_fn{}, count, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
