/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_argminmax_impl.cuh"

#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::groupby::detail {

namespace {

struct argminmax_dispatcher {
  template <typename T>
  void operator()(cudf::device_span<cudf::size_type const> group_labels,
                  column_device_view const& values,
                  bool has_nulls,
                  bool arg_min,
                  cudf::size_type* output,
                  cuda::stream_ref stream) const
  {
    if constexpr (is_relationally_comparable<T, T>() && !is_nested<T>()) {
      launch_argminmax_reduction(group_labels,
                                 cudf::detail::element_argminmax_fn<T>{values, has_nulls, arg_min},
                                 output,
                                 stream);
    } else {
      CUDF_FAIL("Unsupported groupby reduction type-agg combination.");
    }
  }
};

}  // namespace

void launch_argminmax_reduction(cudf::device_span<cudf::size_type const> group_labels,
                                data_type value_type,
                                column_device_view const& values,
                                bool has_nulls,
                                bool arg_min,
                                cudf::size_type* output,
                                cuda::stream_ref stream)
{
  type_dispatcher(
    value_type, argminmax_dispatcher{}, group_labels, values, has_nulls, arg_min, output, stream);
}

}  // namespace cudf::groupby::detail
