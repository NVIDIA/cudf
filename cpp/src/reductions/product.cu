/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::scalar> product(column_view const& col,
                                      cudf::data_type const output_dtype,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type(),
    simple::detail::element_type_dispatcher<op::product>{},
    col,
    output_dtype,
    init,
    stream,
    mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
