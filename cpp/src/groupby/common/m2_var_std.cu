/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "m2_var_std.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail {

namespace {

template <typename Source>
constexpr bool is_m2_supported()
{
  return is_numeric<Source>() && !is_fixed_point<Source>();
}

struct m2_functor {
  template <typename Source, typename... Args>
  void operator()(Args&&...)  //
    requires(!is_m2_supported<Source>())
  {
    CUDF_FAIL("Invalid source type for M2 aggregation.");
  }

  template <typename Target, typename SumSqrType, typename SumType, typename CountType>
  void evaluate(Target* target,
                SumSqrType const* sum_sqr,
                SumType const* sum,
                CountType const* count,
                size_type size,
                rmm::cuda_stream_view stream) const noexcept
  {
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     target,
                     target + size,
                     [sum_sqr, sum, count] __device__(size_type const idx) {
                       auto const group_count = count[idx];
                       if (group_count == 0) { return Target{}; }
                       auto const group_sum_sqr = static_cast<Target>(sum_sqr[idx]);
                       auto const group_sum     = static_cast<Target>(sum[idx]);
                       auto const result = group_sum_sqr - group_sum * group_sum / group_count;
                       return result;
                     });
  }

  template <typename Source>
  void operator()(mutable_column_view const& target,
                  column_view const& sum_sqr,
                  column_view const& sum,
                  column_view const& count,
                  rmm::cuda_stream_view stream) const noexcept  //
    requires(is_m2_supported<Source>())
  {
    using Target     = cudf::detail::target_type_t<Source, aggregation::M2>;
    using SumSqrType = cudf::detail::target_type_t<Source, aggregation::SUM_OF_SQUARES>;
    using SumType    = cudf::detail::target_type_t<Source, aggregation::SUM>;
    using CountType  = cudf::detail::target_type_t<Source, aggregation::COUNT_VALID>;

    // Separate the implementation into another function, which has fewer instantiations since
    // the data types (target/sum/count etc) are mostly the same.
    evaluate(target.begin<Target>(),
             sum_sqr.begin<SumSqrType>(),
             sum.begin<SumType>(),
             count.begin<CountType>(),
             target.size(),
             stream);
  }
};

}  // namespace

std::unique_ptr<column> compute_m2(data_type source_type,
                                   column_view const& sum_sqr,
                                   column_view const& sum,
                                   column_view const& count,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(cudf::detail::target_type(source_type, aggregation::M2),
                                    sum.size(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);
  type_dispatcher(source_type, m2_functor{}, output->mutable_view(), sum_sqr, sum, count, stream);
  return output;
}

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
                                             rmm::cuda_stream_view stream,
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
                                         rmm::cuda_stream_view stream,
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
                                    rmm::cuda_stream_view stream,
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
