/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace reduction {
namespace detail {
std::unique_ptr<cudf::scalar> min(column_view const& col,
                                  data_type const output_dtype,
                                  std::optional<std::reference_wrapper<scalar const>> init,
                                  nan_semantics semantics,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  auto const input_type =
    cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type() : col.type();
  CUDF_EXPECTS(input_type == output_dtype, "min() operation requires matching output type");

  // TOTALLY_ORDERED only changes floating-point results, so only float and double are instantiated
  // with op::min_totally_ordered. A dictionary column is decoded first since the dictionary
  // reduction uses the IEEE 754 minimum.
  if (semantics == nan_semantics::TOTALLY_ORDERED &&
      (input_type.id() == type_id::FLOAT32 || input_type.id() == type_id::FLOAT64)) {
    auto const decoded =
      cudf::is_dictionary(col.type())
        ? cudf::dictionary::detail::decode(
            dictionary_column_view(col), stream, cudf::get_current_device_resource_ref())
        : nullptr;
    auto const input      = decoded ? decoded->view() : col;
    using ordered_reducer = simple::detail::same_element_type_dispatcher<op::min_totally_ordered>;
    return input_type.id() == type_id::FLOAT32
             ? ordered_reducer{}.template operator()<float>(input, init, stream, mr)
             : ordered_reducer{}.template operator()<double>(input, init, stream, mr);
  }

  using reducer = simple::detail::same_element_type_dispatcher<op::min>;
  return cudf::type_dispatcher(input_type, reducer{}, col, init, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
