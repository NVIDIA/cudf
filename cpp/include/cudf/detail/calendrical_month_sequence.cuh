/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/datetime_ops.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
struct calendrical_month_sequence_functor {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(size_type n,
                                           scalar const& input,
                                           size_type months,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
    requires(cudf::is_timestamp_t<T>::value)
  {
    // Return empty column if n = 0
    if (n == 0) return cudf::make_empty_column(input.type());

    auto const device_input =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(input)));
    auto output_column_type = cudf::data_type{cudf::type_to_id<T>()};
    auto output             = cudf::make_fixed_width_column(
      output_column_type, n, cudf::mask_state::UNALLOCATED, stream, mr);

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{n},
                      output->mutable_view().begin<T>(),
                      [initial = device_input, months] __device__(size_type i) {
                        return datetime::detail::add_calendrical_months_with_scale_back(
                          initial.value(), cuda::std::chrono::months{i * months});
                      });

    return output;
  }

  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!cudf::is_timestamp_t<T>::value)
  {
    CUDF_FAIL("Cannot make a date_range of a non-datetime type");
  }
};

}  // namespace detail
}  // namespace cudf
