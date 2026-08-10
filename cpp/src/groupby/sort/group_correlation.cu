/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/transform.h>

#include <type_traits>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <typename T>
constexpr bool is_double_convertible()
{
  return std::is_convertible_v<T, double> || std::is_constructible_v<double, T>;
}

struct is_double_convertible_impl {
  template <typename T>
  bool operator()()
  {
    return is_double_convertible<T>();
  }
};

/**
 * @brief Typecasts each element of the column to `CastType`
 */
template <typename CastType>
struct type_casted_accessor {
  template <typename Element>
  __device__ inline CastType operator()(cudf::size_type i, column_device_view const& col) const
  {
    if constexpr (column_device_view::has_element_accessor<Element>() and
                  std::is_convertible_v<Element, CastType>)
      return static_cast<CastType>(col.element<Element>(i));
    (void)i;
    (void)col;
    return {};
  }
};

template <typename ResultType>
struct covariance_transform {
  column_device_view const d_values_0, d_values_1;
  ResultType const *d_means_0, *d_means_1;
  size_type const* d_group_sizes;
  size_type const* d_group_labels;
  size_type ddof{1};  // TODO update based on bias.

  __device__ static ResultType value(column_device_view const& view, size_type i)
  {
    bool const is_dict = view.type().id() == type_id::DICTIONARY32;
    i                  = is_dict ? static_cast<size_type>(view.element<dictionary32>(i)) : i;
    auto values_col    = is_dict ? view.child(dictionary_column_view::keys_column_index) : view;
    return type_dispatcher(values_col.type(), type_casted_accessor<ResultType>{}, i, values_col);
  }

  __device__ ResultType operator()(size_type i)
  {
    if (d_values_0.is_null(i) or d_values_1.is_null(i)) return 0.0;

    // This has to be device dispatch because x and y type may differ
    auto const x = value(d_values_0, i);
    auto const y = value(d_values_1, i);

    size_type const group_idx  = d_group_labels[i];
    size_type const group_size = d_group_sizes[group_idx];

    // prevent divide by zero error
    if (group_size == 0 or group_size - ddof <= 0) return 0.0;

    ResultType const xmean = d_means_0[group_idx];
    ResultType const ymean = d_means_1[group_idx];
    return (x - xmean) * (y - ymean) / (group_size - ddof);
  }
};
}  // namespace

std::unique_ptr<column> group_covariance(column_view const& values_0,
                                         column_view const& values_1,
                                         cudf::device_span<size_type const> group_labels,
                                         size_type num_groups,
                                         column_view const& count,
                                         column_view const& mean_0,
                                         column_view const& mean_1,
                                         size_type min_periods,
                                         size_type ddof,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  using result_type = id_to_type<type_id::FLOAT64>;
  static_assert(
    std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::CORRELATION>,
                   result_type>);

  // check if each child type can be converted to float64.
  auto get_base_type = [](auto const& col) {
    return (col.type().id() == type_id::DICTIONARY32
              ? col.child(dictionary_column_view::keys_column_index)
              : col)
      .type();
  };
  bool const is_convertible =
    type_dispatcher(get_base_type(values_0), is_double_convertible_impl{}) and
    type_dispatcher(get_base_type(values_1), is_double_convertible_impl{});

  CUDF_EXPECTS(is_convertible,
               "Input to `group_correlation` must be columns of type convertible to float64.");

  auto mean0_ptr = mean_0.begin<result_type>();
  auto mean1_ptr = mean_1.begin<result_type>();

  auto d_values_0 = column_device_view::create(values_0, stream);
  auto d_values_1 = column_device_view::create(values_1, stream);
  covariance_transform<result_type> covariance_transform_op{*d_values_0,
                                                            *d_values_1,
                                                            mean0_ptr,
                                                            mean1_ptr,
                                                            count.data<size_type>(),
                                                            group_labels.data(),
                                                            ddof};

  auto result = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto d_result = result->mutable_view().begin<result_type>();

  auto corr_iter =
    cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, covariance_transform_op);

  cudf::detail::reduce_by_key_async(group_labels.begin(),
                                    group_labels.end(),
                                    corr_iter,
                                    cuda::make_discard_iterator(),
                                    d_result,
                                    cuda::std::plus<result_type>(),
                                    stream);

  auto is_null = [ddof, min_periods] __device__(size_type group_size) {
    return not(group_size == 0 or group_size - ddof <= 0 or group_size < min_periods);
  };
  auto [new_nullmask, null_count] =
    cudf::detail::valid_if(count.begin<size_type>(), count.end<size_type>(), is_null, stream, mr);
  if (null_count != 0) { result->set_null_mask(std::move(new_nullmask), null_count); }
  return result;
}

std::unique_ptr<column> group_correlation(column_view const& covariance,
                                          column_view const& stddev_0,
                                          column_view const& stddev_1,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  using result_type = id_to_type<type_id::FLOAT64>;
  CUDF_EXPECTS(covariance.type().id() == type_id::FLOAT64, "Covariance result must be FLOAT64");
  auto stddev0_ptr = stddev_0.begin<result_type>();
  auto stddev1_ptr = stddev_1.begin<result_type>();
  auto stddev_iter = cuda::make_zip_iterator(cuda::std::make_tuple(stddev0_ptr, stddev1_ptr));
  auto result      = make_numeric_column(covariance.type(),
                                    covariance.size(),
                                    cudf::detail::copy_bitmask(covariance, stream, mr),
                                    covariance.null_count(),
                                    stream,
                                    mr);
  auto d_result    = result->mutable_view().begin<result_type>();
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    covariance.begin<result_type>(),
                    covariance.end<result_type>(),
                    stddev_iter,
                    d_result,
                    [] __device__(auto const covariance, auto const stddev) {
                      return covariance / cuda::std::get<0>(stddev) / cuda::std::get<1>(stddev);
                    });

  result->set_null_count(covariance.null_count());

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
