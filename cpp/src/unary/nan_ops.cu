/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "true_if.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
struct nan_dispatcher {
  template <typename T, typename Predicate>
  std::unique_ptr<column> operator()(cudf::column_view const& input,
                                     Predicate predicate,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    auto input_device_view = column_device_view::create(input, stream);

    if (input.has_nulls()) {
      auto input_pair_iterator = make_pair_iterator<T, true>(*input_device_view);
      return true_if(input_pair_iterator,
                     input_pair_iterator + input.size(),
                     input.size(),
                     predicate,
                     stream,
                     mr);
    } else {
      auto input_pair_iterator = make_pair_iterator<T, false>(*input_device_view);
      return true_if(input_pair_iterator,
                     input_pair_iterator + input.size(),
                     input.size(),
                     predicate,
                     stream,
                     mr);
    }
  }

  template <typename T, typename Predicate>
  std::unique_ptr<column> operator()(cudf::column_view const& input,
                                     Predicate predicate,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(auto element_validity_pair) {
    return element_validity_pair.second and std::isnan(element_validity_pair.first);
  };

  return cudf::type_dispatcher(input.type(), nan_dispatcher{}, input, predicate, stream, mr);
}

std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto predicate = [] __device__(auto element_validity_pair) {
    return !element_validity_pair.second or !std::isnan(element_validity_pair.first);
  };

  return cudf::type_dispatcher(input.type(), nan_dispatcher{}, input, predicate, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_nan(input, stream, mr);
}

std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_not_nan(input, stream, mr);
}

}  // namespace cudf
