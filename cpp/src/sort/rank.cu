/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

template <typename PermutationIteratorType, typename DeviceComparatorType>
struct unique_functor {
  unique_functor(PermutationIteratorType permute, DeviceComparatorType device_comparator)
    : _permute(permute), _device_comparator(device_comparator)
  {
  }

  auto __device__ operator()(size_type index) const noexcept
  {
    return static_cast<size_type>(index == 0 ||
                                  not _device_comparator(_permute[index], _permute[index - 1]));
  }

 private:
  PermutationIteratorType _permute;
  DeviceComparatorType _device_comparator;
};

// Assign rank from 1 to n unique values. Equal values get same rank value.
rmm::device_uvector<size_type> sorted_dense_rank(column_view input_col,
                                                 column_view sorted_order_view,
                                                 rmm::cuda_stream_view stream)
{
  auto const t_input    = table_view{{input_col}};
  auto const comparator = cudf::detail::row::equality::self_comparator{t_input, stream};

  auto const sorted_index_order = cuda::make_permutation_iterator(
    sorted_order_view.begin<size_type>(), cuda::counting_iterator<size_type>{0});

  auto const input_size = input_col.size();
  rmm::device_uvector<size_type> dense_rank_sorted(input_size, stream);

  auto const comparator_helper = [&](auto const device_comparator) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{input_size},
                      dense_rank_sorted.data(),
                      unique_functor<decltype(sorted_index_order), decltype(device_comparator)>{
                        sorted_index_order, device_comparator});
  };

  if (cudf::detail::has_nested_columns(t_input)) {
    auto const device_comparator =
      comparator.equal_to<true>(nullate::DYNAMIC{has_nested_nulls(t_input)});
    comparator_helper(device_comparator);
  } else {
    auto const device_comparator =
      comparator.equal_to<false>(nullate::DYNAMIC{has_nested_nulls(t_input)});
    comparator_helper(device_comparator);
  }

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         dense_rank_sorted.begin(),
                         dense_rank_sorted.end(),
                         dense_rank_sorted.data());

  return dense_rank_sorted;
}

/**
 * @brief Breaks the ties among equal value groups using binary operator and
 * transform this tied value to final rank.
 *
 * @param dense_rank_sorted dense rank of sorted input column (acts as key for value
 * groups).
 * @param tie_iter  iterator of rank to break ties among equal value groups.
 * @param sorted_order_view sorted order indices of input column
 * @param rank_iter output rank iterator
 * @param tie_breaker tie breaking operator. For example, maximum & minimum.
 * @param transformer transform after tie breaking (useful for average).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename TieType,
          typename outputIterator,
          typename TieBreaker,
          typename Transformer,
          typename TieIterator>
void tie_break_ranks_transform(cudf::device_span<size_type const> dense_rank_sorted,
                               TieIterator tie_iter,
                               column_view const& sorted_order_view,
                               outputIterator rank_iter,
                               TieBreaker tie_breaker,
                               Transformer transformer,
                               rmm::cuda_stream_view stream)
{
  auto const input_size = sorted_order_view.size();
  // algorithm: reduce_by_key(dense_rank, 1, n, reduction_tie_breaker)
  // reduction_tie_breaker = min, max, min_count
  rmm::device_uvector<TieType> tie_sorted(sorted_order_view.size(), stream);
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        dense_rank_sorted.begin(),
                        dense_rank_sorted.end(),
                        tie_iter,
                        cuda::make_discard_iterator(),
                        tie_sorted.begin(),
                        cuda::std::equal_to{},
                        tie_breaker);
  using TransformerReturnType =
    cuda::std::decay_t<cuda::std::invoke_result_t<Transformer, TieType>>;
  auto sorted_tied_rank = thrust::make_transform_iterator(
    dense_rank_sorted.begin(),
    cuda::proclaim_return_type<TransformerReturnType>(
      [tied_rank = tie_sorted.begin(), transformer] __device__(auto dense_pos) {
        return transformer(tied_rank[dense_pos - 1]);
      }));
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  sorted_tied_rank,
                  sorted_tied_rank + input_size,
                  sorted_order_view.begin<size_type>(),
                  rank_iter);
}

template <typename outputType>
void rank_first(column_view sorted_order_view,
                mutable_column_view rank_mutable_view,
                rmm::cuda_stream_view stream)
{
  // stable sort order ranking (no ties)
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  cuda::counting_iterator<size_type>{1},
                  cuda::counting_iterator<size_type>{rank_mutable_view.size() + 1},
                  sorted_order_view.begin<size_type>(),
                  rank_mutable_view.begin<outputType>());
}

template <typename outputType>
void rank_dense(cudf::device_span<size_type const> dense_rank_sorted,
                column_view sorted_order_view,
                mutable_column_view rank_mutable_view,
                rmm::cuda_stream_view stream)
{
  // All equal values have same rank and rank always increases by 1 between groups
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  dense_rank_sorted.begin(),
                  dense_rank_sorted.end(),
                  sorted_order_view.begin<size_type>(),
                  rank_mutable_view.begin<outputType>());
}

template <typename outputType>
void rank_min(cudf::device_span<size_type const> group_keys,
              column_view sorted_order_view,
              mutable_column_view rank_mutable_view,
              rmm::cuda_stream_view stream)
{
  // min of first in the group
  // All equal values have min of ranks among them.
  // algorithm: reduce_by_key(dense_rank, 1, n, min), scatter
  tie_break_ranks_transform<size_type>(group_keys,
                                       cuda::counting_iterator<size_type>{1},
                                       sorted_order_view,
                                       rank_mutable_view.begin<outputType>(),
                                       cuda::minimum{},
                                       cuda::std::identity{},
                                       stream);
}

template <typename outputType>
void rank_max(cudf::device_span<size_type const> group_keys,
              column_view sorted_order_view,
              mutable_column_view rank_mutable_view,
              rmm::cuda_stream_view stream)
{
  // max of first in the group
  // All equal values have max of ranks among them.
  // algorithm: reduce_by_key(dense_rank, 1, n, max), scatter
  tie_break_ranks_transform<size_type>(group_keys,
                                       cuda::counting_iterator<size_type>{1},
                                       sorted_order_view,
                                       rank_mutable_view.begin<outputType>(),
                                       cuda::maximum{},
                                       cuda::std::identity{},
                                       stream);
}

// Returns index, count
template <typename T>
struct index_counter {
  __device__ T operator()(size_type i) { return T{i, 1}; }
};

void rank_average(cudf::device_span<size_type const> group_keys,
                  column_view sorted_order_view,
                  mutable_column_view rank_mutable_view,
                  rmm::cuda_stream_view stream)
{
  // k, k+1, .. k+n-1
  // average = (n*k+ n*(n-1)/2)/n
  // average = k + (n-1)/2 = min + (count-1)/2
  // Calculate Min of ranks and Count of equal values
  // algorithm: reduce_by_key(dense_rank, 1, n, min_count)
  //            transform(min+(count-1)/2), scatter
  using MinCount = cuda::std::pair<size_type, size_type>;
  tie_break_ranks_transform<MinCount>(
    group_keys,
    // Use device functor with return type. Cannot use device lambda due to limitation.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-restrictions
    cudf::detail::make_counting_transform_iterator(1, index_counter<MinCount>{}),
    sorted_order_view,
    rank_mutable_view.begin<double>(),
    cuda::proclaim_return_type<MinCount>([] __device__(auto rank_count1, auto rank_count2) {
      return MinCount{cuda::std::min(rank_count1.first, rank_count2.first),
                      rank_count1.second + rank_count2.second};
    }),
    cuda::proclaim_return_type<double>([] __device__(MinCount minrank_count) {  // min+(count-1)/2
      return static_cast<double>(cuda::std::get<0>(minrank_count)) +
             (static_cast<double>(cuda::std::get<1>(minrank_count)) - 1) / 2.0;
    }),
    stream);
}

}  // anonymous namespace

std::unique_ptr<column> rank(column_view const& input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  data_type const output_type         = (percentage or method == rank_method::AVERAGE)
                                          ? data_type(type_id::FLOAT64)
                                          : data_type(type_to_id<size_type>());
  std::unique_ptr<column> rank_column = [&null_handling, &output_type, &input, &stream, &mr] {
    // na_option=keep assign NA to NA values
    if (null_handling == null_policy::EXCLUDE)
      return make_numeric_column(output_type,
                                 input.size(),
                                 detail::copy_bitmask(input, stream, mr),
                                 input.null_count(),
                                 stream,
                                 mr);
    else
      return make_numeric_column(output_type, input.size(), mask_state::UNALLOCATED, stream, mr);
  }();
  auto rank_mutable_view = rank_column->mutable_view();

  std::unique_ptr<column> sorted_order =
    (method == rank_method::FIRST)
      ? detail::stable_sorted_order(
          table_view{{input}}, {column_order}, {null_precedence}, stream, mr)
      : detail::sorted_order(table_view{{input}}, {column_order}, {null_precedence}, stream, mr);
  column_view sorted_order_view = sorted_order->view();

  // dense: All equal values have same rank and rank always increases by 1 between groups
  // acts as key for min, max, average to denote equal value groups
  rmm::device_uvector<size_type> const dense_rank_sorted =
    [&method, &input, &sorted_order_view, &stream] {
      if (method != rank_method::FIRST)
        return sorted_dense_rank(input, sorted_order_view, stream);
      else
        return rmm::device_uvector<size_type>(0, stream);
    }();

  if (output_type.id() == type_id::FLOAT64) {
    switch (method) {
      case rank_method::FIRST:
        rank_first<double>(sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::DENSE:
        rank_dense<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MIN:
        rank_min<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MAX:
        rank_max<double>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::AVERAGE:
        rank_average(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      default: CUDF_FAIL("Unexpected rank_method for rank()");
    }
  } else {
    switch (method) {
      case rank_method::FIRST:
        rank_first<size_type>(sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::DENSE:
        rank_dense<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MIN:
        rank_min<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::MAX:
        rank_max<size_type>(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      case rank_method::AVERAGE:
        rank_average(dense_rank_sorted, sorted_order_view, rank_mutable_view, stream);
        break;
      default: CUDF_FAIL("Unexpected rank_method for rank()");
    }
  }

  // pct inplace transform
  if (percentage) {
    auto rank_iter = rank_mutable_view.begin<double>();
    size_type const count =
      (null_handling == null_policy::EXCLUDE) ? input.size() - input.null_count() : input.size();
    auto drs            = dense_rank_sorted.data();
    bool const is_dense = (method == rank_method::DENSE);
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      rank_iter,
      rank_iter + input.size(),
      rank_iter,
      cuda::proclaim_return_type<double>([is_dense, drs, count] __device__(double r) -> double {
        return is_dense ? r / drs[count - 1] : r / count;
      }));
  }
  return rank_column;
}
}  // namespace detail

std::unique_ptr<column> rank(column_view const& input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rank(
    input, method, column_order, null_handling, null_precedence, percentage, stream, mr);
}
}  // namespace cudf
