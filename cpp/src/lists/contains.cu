/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/detail/contains.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <type_traits>

namespace cudf::lists {
namespace {

/**
 * @brief A sentinel value used for marking that a given key has not been found in the search list.
 *
 * The value should be `-1` as indicated in the public API documentation.
 */
auto constexpr __device__ NOT_FOUND_SENTINEL = size_type{-1};

/**
 * @brief A sentinel value used for marking that a given output row should be null.
 *
 * This value should be different from `NOT_FOUND_SENTINEL`.
 */
auto constexpr __device__ NULL_SENTINEL = cuda::std::numeric_limits<size_type>::min();

/**
 * @brief Check if the given type is a supported non-nested type in `cudf::lists::contains`.
 */
template <typename Element>
static auto constexpr is_supported_non_nested_type()
{
  return cudf::is_fixed_width<Element>() || std::is_same_v<Element, cudf::string_view>;
}

/**
 * @brief Check if the given type is supported in `cudf::lists::contains`.
 */
struct is_supported_type_fn {
  template <typename Element>
  auto constexpr operator()()
  {
    return is_supported_non_nested_type<Element>() || cudf::is_nested<Element>();
  }
};

/**
 * @brief Return a pair of index iterators {begin, end} to loop through elements within a
 * list.
 *
 * Depending on the value of `forward`, a pair of forward or reverse iterators will be
 * returned, allowing to loop through elements in the list in first-to-last or last-to-first
 * order.
 *
 * Note that the element indices always restart to `0` at the first position in each list.
 *
 * @tparam forward A boolean value indicating whether we want to iterate elements in the list
 *         by forward or reverse order.
 * @param size The number of elements in the list.
 * @return A pair of {begin, end} iterators to iterate through the range `[0, size)`.
 */
template <bool forward>
__device__ auto element_index_pair_iter(size_type const size)
{
  auto const begin = cuda::counting_iterator<cudf::size_type>{0};
  auto const end   = cuda::counting_iterator{size};

  if constexpr (forward) {
    return cuda::std::pair{begin, end};
  } else {
    return cuda::std::pair{cuda::std::make_reverse_iterator(end),
                           cuda::std::make_reverse_iterator(begin)};
  }
}

/**
 * @brief Functor to perform searching for index of a key element in a given list, specialized
 * for nested types.
 */
template <typename KeyValidityIter, typename EqComparator>
struct search_list_fn {
  duplicate_find_option const find_option;
  KeyValidityIter const key_validity_iter;
  EqComparator const d_comp;

  search_list_fn(duplicate_find_option const find_option,
                 KeyValidityIter const key_validity_iter,
                 EqComparator const& d_comp)
    : find_option(find_option), key_validity_iter(key_validity_iter), d_comp(d_comp)
  {
  }

  __device__ size_type operator()(list_device_view const list) const
  {
    // A null list or null key will result in a null output row.
    if (list.is_null() || !key_validity_iter[list.row_index()]) { return NULL_SENTINEL; }

    return find_option == duplicate_find_option::FIND_FIRST ? search_list_op<true>(list)
                                                            : search_list_op<false>(list);
  }

 private:
  template <bool forward>
  __device__ inline size_type search_list_op(list_device_view const list) const
  {
    using cudf::detail::row::lhs_index_type;
    using cudf::detail::row::rhs_index_type;

    auto const [begin, end] = element_index_pair_iter<forward>(list.size());
    auto const found_iter =
      thrust::find_if(thrust::seq, begin, end, [&list, this] __device__(auto const idx) {
        return !list.is_null(idx) && d_comp(static_cast<lhs_index_type>(list.element_offset(idx)),
                                            static_cast<rhs_index_type>(list.row_index()));
      });
    // If the key is found, return its found position in the list from `found_iter`.
    return found_iter == end ? NOT_FOUND_SENTINEL : *found_iter;
  }
};

/**
 * @brief Function to search for index of key element(s) in the corresponding rows of a lists
 * column, specialized for nested types.
 */
template <typename InputIterator, typename OutputIterator, typename DeviceComp>
void index_of(InputIterator input_it,
              size_type num_rows,
              OutputIterator output_it,
              column_view const& child,
              column_view const& search_keys,
              duplicate_find_option find_option,
              DeviceComp d_comp,
              rmm::cuda_stream_view stream)
{
  auto const keys_dv_ptr       = column_device_view::create(search_keys, stream);
  auto const key_validity_iter = cudf::detail::make_validity_iterator<true>(*keys_dv_ptr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input_it,
                    input_it + num_rows,
                    output_it,
                    search_list_fn{find_option, key_validity_iter, d_comp});
}

/**
 * @brief Dispatch function to search for index of key element(s) in the corresponding rows of a
 * lists column.
 */
std::unique_ptr<column> dispatch_index_of(lists_column_view const& lists,
                                          column_view const& search_keys,
                                          duplicate_find_option find_option,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::type_dispatcher(search_keys.type(), is_supported_type_fn{}),
               "Unsupported type in `dispatch_index_of` function.");
  // Access the child column through `child()` method, not `get_sliced_child()`.
  // This is because slicing offset has already been taken into account during row
  // comparisons.
  auto const child = lists.child();

  CUDF_EXPECTS(cudf::have_same_types(child, search_keys),
               "Type/Scale of search key does not match list column element type.",
               cudf::data_type_error);
  CUDF_EXPECTS(search_keys.type().id() != type_id::EMPTY, "Type cannot be empty.");

  auto const search_keys_have_nulls = search_keys.has_nulls();

  auto const num_rows = lists.size();

  auto const lists_cdv_ptr = column_device_view::create(lists.parent(), stream);
  auto const input_it      = cudf::detail::make_counting_transform_iterator(
    size_type{0},
    cuda::proclaim_return_type<list_device_view>(
      [lists = cudf::lists_column_device_view{*lists_cdv_ptr}] __device__(auto const idx) {
        return list_device_view{lists, idx};
      }));

  auto out_positions = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const output_it = out_positions->mutable_view().template begin<size_type>();

  auto const keys_tview  = cudf::table_view{{search_keys}};
  auto const child_tview = cudf::table_view{{child}};
  auto const has_nulls   = has_nested_nulls(child_tview) || has_nested_nulls(keys_tview);
  auto const comparator =
    cudf::detail::row::equality::two_table_comparator(child_tview, keys_tview, stream);
  if (cudf::is_nested(search_keys.type())) {
    auto const d_comp = comparator.equal_to<true>(nullate::DYNAMIC{has_nulls});
    index_of(input_it, num_rows, output_it, child, search_keys, find_option, d_comp, stream);
  } else {
    auto const d_comp = comparator.equal_to<false>(nullate::DYNAMIC{has_nulls});
    index_of(input_it, num_rows, output_it, child, search_keys, find_option, d_comp, stream);
  }

  if (search_keys_have_nulls || lists.has_nulls()) {
    auto [null_mask, null_count] = cudf::detail::valid_if(
      output_it,
      output_it + num_rows,
      [] __device__(auto const idx) { return idx != NULL_SENTINEL; },
      stream,
      mr);
    out_positions->set_null_mask(std::move(null_mask), null_count);
  }
  return out_positions;
}

/**
 * @brief Converts key-positions vector (from `index_of()`) to a BOOL8 vector, indicating if
 * the search key(s) were found.
 */
std::unique_ptr<column> to_contains(std::unique_ptr<column>&& key_positions,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(key_positions->type().id() == type_to_id<size_type>(),
               "Expected input column of type cudf::size_type.");
  auto const positions_begin = key_positions->view().template begin<size_type>();
  auto result                = make_numeric_column(
    data_type{type_id::BOOL8}, key_positions->size(), mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    positions_begin,
                    positions_begin + key_positions->size(),
                    result->mutable_view().template begin<bool>(),
                    cuda::proclaim_return_type<bool>([] __device__(auto const i) {
                      // position == NOT_FOUND_SENTINEL: the list does not contain the search key.
                      return i != NOT_FOUND_SENTINEL;
                    }));

  auto const null_count                             = key_positions->null_count();
  [[maybe_unused]] auto [data, null_mask, children] = key_positions->release();
  result->set_null_mask(std::move(*null_mask.release()), null_count);

  return result;
}
}  // namespace

namespace detail {
std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  if (!search_key.is_valid(stream)) {
    return make_numeric_column(
      data_type{cudf::type_to_id<size_type>()},
      lists.size(),
      cudf::detail::create_null_mask(lists.size(), mask_state::ALL_NULL, stream, mr),
      lists.size(),
      stream,
      mr);
  }
  if (lists.size() == 0) {
    return make_numeric_column(
      data_type{type_to_id<size_type>()}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto search_key_col = cudf::make_column_from_scalar(search_key, lists.size(), stream, mr);
  return detail::index_of(lists, search_key_col->view(), find_option, stream, mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");
  return dispatch_index_of(lists, search_keys, find_option, stream, mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto key_indices = detail::index_of(lists,
                                      search_key,
                                      duplicate_find_option::FIND_FIRST,
                                      stream,
                                      cudf::get_current_device_resource_ref());
  return to_contains(std::move(key_indices), stream, mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");

  auto key_indices = detail::index_of(lists,
                                      search_keys,
                                      duplicate_find_option::FIND_FIRST,
                                      stream,
                                      cudf::get_current_device_resource_ref());
  return to_contains(std::move(key_indices), stream, mr);
}

std::unique_ptr<column> contains_nulls(lists_column_view const& lists,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const lists_cv      = lists.parent();
  auto output              = make_numeric_column(data_type{type_to_id<bool>()},
                                    lists.size(),
                                    cudf::detail::copy_bitmask(lists_cv, stream, mr),
                                    lists_cv.null_count(),
                                    stream,
                                    mr);
  auto const out_begin     = output->mutable_view().template begin<bool>();
  auto const lists_cdv_ptr = column_device_view::create(lists_cv, stream);

  thrust::tabulate(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    out_begin,
    out_begin + lists.size(),
    cuda::proclaim_return_type<bool>(
      [lists = cudf::lists_column_device_view{*lists_cdv_ptr}] __device__(auto const list_idx) {
        auto const list = list_device_view{lists, list_idx};
        return list.is_null() ||
               thrust::any_of(thrust::seq,
                              cuda::counting_iterator<cudf::size_type>{0},
                              cuda::counting_iterator{list.size()},
                              [&list](auto const idx) { return list.is_null(idx); });
      }));

  return output;
}

}  // namespace detail

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_key, stream, mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_keys, stream, mr);
}

std::unique_ptr<column> contains_nulls(lists_column_view const& lists,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_nulls(lists, stream, mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_key, find_option, stream, mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_keys, find_option, stream, mr);
}

}  // namespace cudf::lists
