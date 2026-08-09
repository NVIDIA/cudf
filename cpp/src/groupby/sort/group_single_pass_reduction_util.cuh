/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reductions/nested_types_extrema_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * @brief Value accessor for column which supports dictionary column too.
 *
 * This is similar to `value_accessor` in `column_device_view.cuh` but with support of dictionary
 * type.
 *
 * @tparam T Type of the underlying column. For dictionary column, type of the key column.
 */
template <typename T>
struct value_accessor {
  column_device_view const col;
  bool const is_dict;

  value_accessor(column_device_view const& col) : col(col), is_dict(cudf::is_dictionary(col.type()))
  {
  }

  __device__ T value(size_type i) const
  {
    if (is_dict) {
      auto keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(i).value()));
    } else {
      return col.element<T>(i);
    }
  }

  __device__ auto operator()(size_type i) const { return value(i); }
};

/**
 * @brief Null replaced value accessor for column which supports dictionary column too.
 * For null value, returns null `init` value
 *
 * @tparam SourceType Type of the underlying column. For dictionary column, type of the key column.
 * @tparam TargetType Type that is used for computation.
 */
template <typename SourceType, typename TargetType>
struct null_replaced_value_accessor : value_accessor<SourceType> {
  using super_t = value_accessor<SourceType>;

  TargetType const init;
  bool const has_nulls;

  null_replaced_value_accessor(column_device_view const& col,
                               TargetType const& init,
                               bool const has_nulls)
    : super_t(col), init(init), has_nulls(has_nulls)
  {
  }

  __device__ TargetType operator()(size_type i) const
  {
    return has_nulls && super_t::col.is_null_nocheck(i)
             ? init
             : static_cast<TargetType>(super_t::value(i));
  }
};

// Error case when no other overload or specialization is available
template <aggregation::Kind K, typename T, typename Enable = void>
struct group_reduction_functor {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported groupby reduction type-agg combination.");
  }
};

template <aggregation::Kind K>
struct group_reduction_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<cudf::size_type const> group_labels,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return group_reduction_functor<K, T>::invoke(values, num_groups, group_labels, stream, mr);
  }
};

/**
 * @brief Check if the given aggregation K with data type T is supported in groupby reduction.
 */
template <aggregation::Kind K, typename T>
static constexpr bool is_group_reduction_supported()
{
  switch (K) {
    case aggregation::SUM:
      return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
    case aggregation::PRODUCT: return cudf::detail::is_product_supported<T>();
    case aggregation::MIN:
    case aggregation::MAX: return cudf::is_fixed_width<T>() and is_relationally_comparable<T, T>();
    case aggregation::ARGMIN:
    case aggregation::ARGMAX: return is_relationally_comparable<T, T>() or cudf::is_nested<T>();
    default: return false;
  }
}

template <aggregation::Kind K, typename T>
struct group_reduction_functor<
  K,
  T,
  std::enable_if_t<is_group_reduction_supported<K, T>() && !cudf::is_nested<T>()>> {
  static std::unique_ptr<column> invoke(column_view const& values,
                                        size_type num_groups,
                                        cudf::device_span<cudf::size_type const> group_labels,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
  {
    using SourceDType = device_storage_type_t<T>;
    using ResultType  = cudf::detail::target_type_t<T, K>;
    using ResultDType = device_storage_type_t<ResultType>;

    auto result_type = is_fixed_point<ResultType>()
                         ? data_type{type_to_id<ResultType>(), values.type().scale()}
                         : data_type{type_to_id<ResultType>()};

    std::unique_ptr<column> result =
      make_fixed_width_column(result_type, num_groups, mask_state::UNALLOCATED, stream, mr);

    if (values.is_empty()) { return result; }

    // Perform segmented reduction.
    auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        group_labels.data(),
        group_labels.data() + group_labels.size(),
        inp_iter,
        cuda::make_discard_iterator(),
        out_iter,
        cuda::std::equal_to{},
        binop);
    };

    auto const d_values_ptr = column_device_view::create(values, stream);
    auto const result_begin = result->mutable_view().template begin<ResultDType>();

    if constexpr (K == aggregation::ARGMAX || K == aggregation::ARGMIN) {
      auto const count_iter = cuda::counting_iterator<ResultType>{0};
      auto const binop      = cudf::detail::element_argminmax_fn<T>{
        *d_values_ptr, values.has_nulls(), K == aggregation::ARGMIN};
      do_reduction(count_iter, result_begin, binop);
    } else {
      using OpType    = cudf::detail::corresponding_operator_t<K>;
      auto init       = OpType::template identity<ResultDType>();
      auto inp_values = cudf::detail::make_counting_transform_iterator(
        0,
        null_replaced_value_accessor<SourceDType, ResultDType>{
          *d_values_ptr, init, values.has_nulls()});
      do_reduction(inp_values, result_begin, OpType{});
    }

    if (values.has_nulls()) {
      rmm::device_uvector<bool> validity(num_groups, stream);
      do_reduction(cudf::detail::make_validity_iterator(*d_values_ptr),
                   validity.begin(),
                   cuda::std::logical_or{});

      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
      result->set_null_mask(std::move(null_mask), null_count);
    }
    return result;
  }
};

template <aggregation::Kind K, typename T>
struct group_reduction_functor<
  K,
  T,
  std::enable_if_t<is_group_reduction_supported<K, T>() && cudf::is_nested<T>()>> {
  static std::unique_ptr<column> invoke(column_view const& values,
                                        size_type num_groups,
                                        cudf::device_span<cudf::size_type const> group_labels,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
  {
    // This is be expected to be size_type.
    using ResultType = cudf::detail::target_type_t<T, K>;

    auto result = make_fixed_width_column(
      data_type{type_to_id<ResultType>()}, num_groups, mask_state::UNALLOCATED, stream, mr);

    if (values.is_empty()) { return result; }

    // Perform segmented reduction to find ARGMIN/ARGMAX.
    auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
      thrust::reduce_by_key(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        group_labels.data(),
        group_labels.data() + group_labels.size(),
        inp_iter,
        cuda::make_discard_iterator(),
        out_iter,
        cuda::std::equal_to{},
        binop);
    };

    auto const count_iter   = cuda::counting_iterator<ResultType>{0};
    auto const result_begin = result->mutable_view().template begin<ResultType>();
    auto const binop_generator =
      cudf::reduction::detail::arg_minmax_binop_generator::create<K>(values, stream);
    do_reduction(count_iter, result_begin, binop_generator.binop());

    if (values.has_nulls()) {
      // Generate bitmask for the output by segmented reduction of the input bitmask.
      auto const d_values_ptr = column_device_view::create(values, stream);
      auto validity           = rmm::device_uvector<bool>(num_groups, stream);
      do_reduction(cudf::detail::make_validity_iterator(*d_values_ptr),
                   validity.begin(),
                   cuda::std::logical_or{});

      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
      result->set_null_mask(std::move(null_mask), null_count);
    }

    return result;
  }
};

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
