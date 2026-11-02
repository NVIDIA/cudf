/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/label_bins.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/dispatchers.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

template <typename, inclusive>
struct InclusiveToComparator;

template <typename T>
struct InclusiveToComparator<T, inclusive::YES> {
  using type = cuda::std::less_equal<T>;
};

template <typename T>
struct InclusiveToComparator<T, inclusive::NO> {
  using type = cuda::std::less<T>;
};

template <typename T, inclusive i>
using InclusiveToComparator_t = typename InclusiveToComparator<T, i>::type;

// Helper to have single point of dispatch for inclusive enum
template <typename Func>
auto dispatch_inclusive(inclusive i, Func&& func)
{
  return dispatch_enum<inclusive::YES, inclusive::NO>(i, std::forward<Func>(func));
}

// Sentinel used to indicate that an input value should be placed in the null
// bin.
// NOTE: In theory if a user decided to specify 2^31 bins this would fail. We
// could make this an error in Python, but that is such a crazy edge case...
__device__ constexpr size_type NULL_VALUE{cuda::std::numeric_limits<size_type>::max()};

/*
 * Functor for finding bins using thrust::transform.
 *
 * This functor is stateful, in the sense that it stores (for read-only use)
 * pointers to the edge ranges on construction to enable natural use with
 * thrust::transform semantics.  To handle null values, this functor assumes
 * that the input iterators have already been shifted to exclude the range
 * containing nulls. The `edge_index_shift` parameter is used to return the
 * index of a value's bin accounting for this shift.
 */
template <typename T,
          typename RandomAccessIterator,
          typename LeftComparator,
          typename RightComparator>
struct bin_finder {
  bin_finder(RandomAccessIterator left_begin,
             RandomAccessIterator left_end,
             RandomAccessIterator right_begin)
    : m_left_begin(left_begin), m_left_end(left_end), m_right_begin(right_begin)
  {
  }

  __device__ size_type operator()(cuda::std::pair<T, bool> input_value) const
  {
    // Immediately return sentinel for null inputs.
    if (!input_value.second) return NULL_VALUE;

    T value    = input_value.first;
    auto bound = thrust::lower_bound(thrust::seq, m_left_begin, m_left_end, value, m_left_comp);

    // Exit early and return sentinel for values that lie below the interval.
    if (bound == m_left_begin) { return NULL_VALUE; }

    auto index = cuda::std::distance(m_left_begin, cuda::std::prev(bound));
    return (m_right_comp(value, m_right_begin[index])) ? index : NULL_VALUE;
  }

  RandomAccessIterator const
    m_left_begin{};  // The beginning of the range containing the left bin edges.
  RandomAccessIterator const m_left_end{};  // The end of the range containing the left bin edges.
  RandomAccessIterator const
    m_right_begin{};                   // The beginning of the range containing the right bin edges.
  LeftComparator const m_left_comp{};  // Comparator used for left edges.
  RightComparator const m_right_comp{};  // Comparator used for right edges.
};

// Functor to identify rows that should be filtered out based on the sentinel set by
// bin_finder::operator().
struct filter_null_sentinel {
  __device__ bool operator()(size_type i) { return i != NULL_VALUE; }
};

// Bin the input by the edges in left_edges and right_edges.
template <typename T, typename LeftComparator, typename RightComparator>
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   column_view const& right_edges,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  auto output_mutable_view = output->mutable_view();
  auto output_begin        = output_mutable_view.begin<size_type>();
  auto output_end          = output_mutable_view.end<size_type>();

  // These device column views are necessary for creating iterators that work
  // for columns of compound types. The column_view iterators fail for compound
  // types because they return raw pointers to the start of the data. The output
  // does not require these iterators because it's always a primitive type.
  auto input_device_view       = column_device_view::create(input, stream);
  auto left_edges_device_view  = column_device_view::create(left_edges, stream);
  auto right_edges_device_view = column_device_view::create(right_edges, stream);

  auto left_begin  = left_edges_device_view->begin<T>();
  auto left_end    = left_edges_device_view->end<T>();
  auto right_begin = right_edges_device_view->begin<T>();

  using RandomAccessIterator = decltype(left_edges_device_view->begin<T>());

  dispatch_bool(input.has_nulls(), [&](auto has_nulls) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input_device_view->pair_begin<T, has_nulls>(),
                      input_device_view->pair_end<T, has_nulls>(),
                      output_begin,
                      bin_finder<T, RandomAccessIterator, LeftComparator, RightComparator>(
                        left_begin, left_end, right_begin));
  });

  auto mask_and_count = valid_if(output_begin, output_end, filter_null_sentinel(), stream, mr);

  output->set_null_mask(std::move(mask_and_count.first), mask_and_count.second);
  return output;
}

template <typename T>
constexpr auto is_supported_bin_type()
{
  return cudf::is_relationally_comparable<T, T>() && cudf::is_equality_comparable<T, T>();
}

struct bin_type_dispatcher {
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(not detail::is_supported_bin_type<T>())
  {
    CUDF_FAIL("Type not support for cudf::bin");
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& left_edges,
                                     inclusive left_inclusive,
                                     column_view const& right_edges,
                                     inclusive right_inclusive,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(detail::is_supported_bin_type<T>())
  {
    return dispatch_inclusive(left_inclusive, [&](auto li) {
      return dispatch_inclusive(right_inclusive, [&](auto ri) {
        return label_bins<T, InclusiveToComparator_t<T, li>, InclusiveToComparator_t<T, ri>>(
          input, left_edges, right_edges, stream, mr);
      });
    });
  }
};

}  // anonymous namespace

/// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    cudf::have_same_types(input, left_edges) && cudf::have_same_types(input, right_edges),
    "The input and edge columns must have the same types.",
    cudf::data_type_error);
  CUDF_EXPECTS(left_edges.size() == right_edges.size(),
               "The left and right edge columns must be of the same length.");
  CUDF_EXPECTS(!left_edges.has_nulls() && !right_edges.has_nulls(),
               "The left and right edge columns cannot contain nulls.");

  // Handle empty inputs.
  if (input.is_empty()) { return make_empty_column(type_to_id<size_type>()); }

  return type_dispatcher<dispatch_storage_type>(input.type(),
                                                detail::bin_type_dispatcher{},
                                                input,
                                                left_edges,
                                                left_inclusive,
                                                right_edges,
                                                right_inclusive,
                                                stream,
                                                mr);
}

}  // namespace detail

/// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::label_bins(
    input, left_edges, left_inclusive, right_edges, right_inclusive, stream, mr);
}
}  // namespace cudf
