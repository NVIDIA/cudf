/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "m2_var_std.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail {

namespace {

// M2, VARIANCE, STD and COUNT_VALID aggregations always have fixed types, thus we hardcode them
// instead of using type dispatcher for faster compilation.
using M2Type       = double;
using VarianceType = double;
using StdType      = double;
using CountType    = int32_t;

void check_input_types(column_view const& m2, column_view const& count)
{
  CUDF_EXPECTS(m2.type().id() == type_to_id<M2Type>(),
               "Data type of M2 aggregation must be FLOAT64.",
               std::invalid_argument);
  CUDF_EXPECTS(count.type().id() == type_to_id<CountType>(),
               "Data type of COUNT_VALID aggregation must be INT32.",
               std::invalid_argument);
}

template <typename TargetType, typename TransformFunc>
std::unique_ptr<column> compute_variance_std(TransformFunc&& transform_fn,
                                             size_type size,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(
    data_type(type_to_id<TargetType>()), size, mask_state::UNALLOCATED, stream, mr);

  // Since we may have new null rows depending on the group count, we need to generate a new null
  // mask from scratch.
  rmm::device_uvector<bool> validity(size, stream);

  auto const out_it =
    cuda::make_zip_iterator(output->mutable_view().begin<TargetType>(), validity.begin());
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   out_it,
                   out_it + size,
                   transform_fn);

  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
  if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }

  return output;
}

}  // namespace

std::unique_ptr<column> compute_variance(column_view const& m2,
                                         column_view const& count,
                                         size_type ddof,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  check_input_types(m2, count);

  auto const transform_func =
    [m2 = m2.begin<M2Type>(), count = count.begin<CountType>(), ddof] __device__(
      size_type const idx) -> cuda::std::pair<VarianceType, bool> {
    auto const group_count = count[idx];
    auto const df          = group_count - ddof;
    if (group_count == 0 || df <= 0) { return {VarianceType{}, false}; }
    return {m2[idx] / df, true};
  };
  return compute_variance_std<VarianceType>(transform_func, m2.size(), stream, mr);
}

std::unique_ptr<column> compute_std(column_view const& m2,
                                    column_view const& count,
                                    size_type ddof,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  check_input_types(m2, count);

  auto const transform_func =
    [m2 = m2.begin<M2Type>(), count = count.begin<CountType>(), ddof] __device__(
      size_type const idx) -> cuda::std::pair<StdType, bool> {
    auto const group_count = count[idx];
    auto const df          = group_count - ddof;
    if (group_count == 0 || df <= 0) { return {StdType{}, false}; }
    return {cuda::std::sqrt(m2[idx] / df), true};
  };
  return compute_variance_std<StdType>(transform_func, m2.size(), stream, mr);
}

}  // namespace cudf::groupby::detail
