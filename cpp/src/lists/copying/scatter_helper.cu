/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/detail/copying.hpp>
#include <cudf/lists/detail/scatter_helper.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief Constructs null mask for a scattered list's child column
 *
 * @param parent_list_vector Vector of unbound_list_view, for parent lists column
 * @param parent_list_offsets List column offsets for parent lists column
 * @param source_lists Source lists column for scatter operation
 * @param target_lists Target lists column for scatter operation
 * @param num_child_rows Number of rows in child column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate child column's null mask
 * @return std::pair<rmm::device_buffer, size_type> Child column's null mask and null row count
 */
std::pair<rmm::device_buffer, size_type> construct_child_nullmask(
  rmm::device_uvector<unbound_list_view> const& parent_list_vector,
  column_view const& parent_list_offsets,
  cudf::lists_column_device_view const& source_lists,
  cudf::lists_column_device_view const& target_lists,
  size_type num_child_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto is_valid_predicate = [d_list_vector  = parent_list_vector.begin(),
                             d_offsets      = parent_list_offsets.template data<size_type>(),
                             d_offsets_size = parent_list_offsets.size(),
                             source_lists,
                             target_lists] __device__(auto const& i) {
    auto list_start =
      thrust::upper_bound(thrust::seq, d_offsets, d_offsets + d_offsets_size, i) - 1;
    auto list_index    = list_start - d_offsets;
    auto element_index = i - *list_start;

    auto list_row = d_list_vector[list_index];
    return !list_row.bind_to_column(source_lists, target_lists).is_null(element_index);
  };

  return cudf::detail::valid_if(cuda::counting_iterator<size_type>{0},
                                cuda::counting_iterator<size_type>{num_child_rows},
                                is_valid_predicate,
                                stream,
                                mr);
}

/**
 * @brief (type_dispatch endpoint) Functor that constructs the child column result
 *        of `scatter()`ing a list column.
 *
 * The protocol is as follows:
 *
 * Inputs:
 *  1. list_vector:  A device_uvector of unbound_list_view, with each element
 *                   indicating the position, size, and which column the list
 *                   row came from.
 *  2. list_offsets: The offsets column for the (outer) lists column, each offset
 *                   marking the beginning of a list row.
 *  3. source_list:  The lists-column that is the source of the scatter().
 *  4. target_list:  The lists-column that is the target of the scatter().
 *
 * Output: A (possibly non-list) child column, which may be used in combination
 *         with list_offsets to fully construct the outer list.
 *
 * Example:
 *
 * Consider the following scatter operation of two `list<int>` columns:
 *
 * 1. Source:      [{9,9,9,9}, {8,8,8}], i.e.
 *    a. Child:    [9,9,9,9,8,8,8]
 *    b. Offsets:  [0,      4,    7]
 *
 * 2. Target:      [{1,1}, {2,2}, {3,3}], i.e.
 *    a. Child:    [1,1,2,2,3,3]
 *    b. Offsets:  [0,  2,  4,  6]
 *
 * 3. Scatter-map: [2, 0]
 *
 * 4. Expected output: [{8,8,8}, {2,2}, {9,9,9,9}], i.e.
 *    a. Child:        [8,8,8,2,2,9,9,9,9]  <--- THIS
 *    b. Offsets:      [0,    3,  5,     9]
 *
 * `list_child_constructor` constructs the Expected Child column indicated above.
 *
 * `list_child_constructor` expects to be called with the `Source`/`Target`
 * lists columns, along with the following:
 *
 * 1. list_vector: [ S[1](3), T[1](2), S[0](4) ]
 *    Each unbound_list_view (e.g. S[1](3)) indicates:
 *      a. Which column the row is bound to: S == Source, T == Target
 *      b. The list index. E.g. S[1] indicates the 2nd list row of the Source column.
 *      c. The row size.   E.g. S[1](3) indicates that the row has 3 elements.
 *
 * 2. list_offsets: [0, 3, 5, 9]
 *    The caller may construct this with an `inclusive_scan()` on `list_vector`
 *    element sizes.
 */
struct list_child_constructor {
 private:
  /**
   * @brief Determine whether the child column type is supported with scattering lists.
   *
   * @tparam T The data type of the child column of the list being scattered.
   */
  template <typename T>
  struct is_supported_child_type {
    static bool const value = cudf::is_fixed_width<T>() || std::is_same_v<T, string_view> ||
                              std::is_same_v<T, list_view> || std::is_same_v<T, struct_view>;
  };

 public:
  // SFINAE catch-all, for unsupported child column types.
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&... args)
    requires(!is_supported_child_type<T>::value)
  {
    CUDF_FAIL("list_child_constructor unsupported!");
  }

  /**
   * @brief Implementation for fixed_width child column types.
   */
  template <typename T>
  std::unique_ptr<column> operator()(rmm::device_uvector<unbound_list_view> const& list_vector,
                                     cudf::column_view const& list_offsets,
                                     cudf::lists_column_view const& source_lists_column_view,
                                     cudf::lists_column_view const& target_lists_column_view,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_width<T>())
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::lists_column_device_view(*target_column_device_view);

    auto const num_child_rows{
      cudf::detail::get_value<size_type>(list_offsets, list_offsets.size() - 1, stream)};

    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::pair(rmm::device_buffer{}, 0);

    auto child_column = cudf::make_fixed_width_column(source_lists_column_view.child().type(),
                                                      num_child_rows,
                                                      std::move(child_null_mask.first),
                                                      child_null_mask.second,
                                                      stream,
                                                      mr);

    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::counting_iterator{child_column->size()},
      child_column->mutable_view().begin<T>(),
      cuda::proclaim_return_type<T>([offset_begin  = list_offsets.begin<size_type>(),
                                     offset_size   = list_offsets.size(),
                                     d_list_vector = list_vector.begin(),
                                     source_lists,
                                     target_lists] __device__(auto index) {
        auto const list_index_iter =
          thrust::upper_bound(thrust::seq, offset_begin, offset_begin + offset_size, index);
        auto const list_index =
          static_cast<size_type>(cuda::std::distance(offset_begin, list_index_iter) - 1);
        auto const intra_index = static_cast<size_type>(index - offset_begin[list_index]);
        auto actual_list_row = d_list_vector[list_index].bind_to_column(source_lists, target_lists);
        return actual_list_row.template element<T>(intra_index);
      }));

    child_column->set_null_count(child_null_mask.second);

    return child_column;
  }

  /**
   * @brief Implementation for list child columns that contain strings.
   */
  template <typename T>
  std::unique_ptr<column> operator()(rmm::device_uvector<unbound_list_view> const& list_vector,
                                     cudf::column_view const& list_offsets,
                                     cudf::lists_column_view const& source_lists_column_view,
                                     cudf::lists_column_view const& target_lists_column_view,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(std::is_same_v<T, string_view>)
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::lists_column_device_view(*target_column_device_view);

    auto const num_child_rows{
      cudf::detail::get_value<size_type>(list_offsets, list_offsets.size() - 1, stream)};

    if (num_child_rows == 0) { return make_empty_column(type_id::STRING); }

    auto string_views = rmm::device_uvector<string_view>(num_child_rows, stream);

    auto const null_string_view = string_view{nullptr, 0};  // placeholder for factory function

    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator{static_cast<size_type>(string_views.size())},
      string_views.begin(),
      cuda::proclaim_return_type<string_view>([offset_begin  = list_offsets.begin<size_type>(),
                                               offset_size   = list_offsets.size(),
                                               d_list_vector = list_vector.begin(),
                                               source_lists,
                                               target_lists,
                                               null_string_view] __device__(auto index) {
        auto const list_index_iter =
          thrust::upper_bound(thrust::seq, offset_begin, offset_begin + offset_size, index);
        auto const list_index =
          static_cast<size_type>(cuda::std::distance(offset_begin, list_index_iter) - 1);
        auto const intra_index = static_cast<size_type>(index - offset_begin[list_index]);
        auto row_index         = d_list_vector[list_index].row_index();
        auto actual_list_row = d_list_vector[list_index].bind_to_column(source_lists, target_lists);
        auto lists_column    = actual_list_row.get_column();
        auto lists_offsets_ptr    = lists_column.offsets().template data<size_type>();
        auto child_strings_column = lists_column.child();
        auto strings_offset       = lists_offsets_ptr[row_index] + intra_index;

        if (child_strings_column.is_null(strings_offset)) { return null_string_view; }
        auto const d_str = child_strings_column.template element<string_view>(strings_offset);
        // ensure a string from an all-empty column is not mapped to the null placeholder
        auto const empty_string_view = string_view{};
        return d_str.empty() ? empty_string_view : d_str;
      }));

    // string_views should now have been populated with source and target references.
    auto sv_span = cudf::device_span<string_view const>(string_views);
    return cudf::make_strings_column(sv_span, null_string_view, stream, mr);
  }

  /**
   * @brief (Recursively) Constructs a child column that is itself a list column.
   */
  template <typename T>
  std::unique_ptr<column> operator()(rmm::device_uvector<unbound_list_view> const& list_vector,
                                     cudf::column_view const& list_offsets,
                                     cudf::lists_column_view const& source_lists_column_view,
                                     cudf::lists_column_view const& target_lists_column_view,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(std::is_same_v<T, list_view>)
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::lists_column_device_view(*target_column_device_view);

    auto const num_child_rows{
      cudf::detail::get_value<size_type>(list_offsets, list_offsets.size() - 1, stream)};

    if (num_child_rows == 0) {
      // make an empty lists column using the input child type
      return empty_like(source_lists_column_view.child());
    }

    auto child_list_views = rmm::device_uvector<unbound_list_view>(num_child_rows, stream, mr);

    // Convert from parent list_device_view instances to child list_device_views.
    // For instance, if a parent list_device_view has 3 elements, it should have 3 corresponding
    // child list_device_view instances.
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator{static_cast<size_type>(child_list_views.size())},
      child_list_views.begin(),
      cuda::proclaim_return_type<unbound_list_view>([offset_begin = list_offsets.begin<size_type>(),
                                                     offset_size  = list_offsets.size(),
                                                     d_list_vector = list_vector.begin(),
                                                     source_lists,
                                                     target_lists] __device__(auto index) {
        auto const list_index_iter =
          thrust::upper_bound(thrust::seq, offset_begin, offset_begin + offset_size, index);
        auto const list_index =
          static_cast<size_type>(cuda::std::distance(offset_begin, list_index_iter) - 1);
        auto const intra_index = static_cast<size_type>(index - offset_begin[list_index]);
        auto label             = d_list_vector[list_index].label();
        auto row_index         = d_list_vector[list_index].row_index();
        auto actual_list_row = d_list_vector[list_index].bind_to_column(source_lists, target_lists);
        auto lists_column    = actual_list_row.get_column();
        auto child_lists_column = lists_column.child();
        auto lists_offsets_ptr  = lists_column.offsets().template data<size_type>();
        auto child_lists_offsets_ptr =
          child_lists_column.child(lists_column_view::offsets_column_index)
            .template data<size_type>();
        auto child_row_index = lists_offsets_ptr[row_index] + intra_index;
        auto size =
          child_lists_offsets_ptr[child_row_index + 1] - child_lists_offsets_ptr[child_row_index];
        return unbound_list_view{label, child_row_index, size};
      }));

    // child_list_views should now have been populated, with source and target references.

    auto begin = thrust::make_transform_iterator(
      child_list_views.begin(),
      cuda::proclaim_return_type<size_type>([] __device__(auto const& row) { return row.size(); }));

    auto child_offsets = std::get<0>(
      cudf::detail::make_offsets_child_column(begin, begin + child_list_views.size(), stream, mr));

    auto child_column = cudf::type_dispatcher<dispatch_storage_type>(
      source_lists_column_view.child().child(1).type(),
      list_child_constructor{},
      child_list_views,
      child_offsets->view(),
      cudf::lists_column_view(source_lists_column_view.child()),
      cudf::lists_column_view(target_lists_column_view.child()),
      stream,
      mr);

    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::pair(rmm::device_buffer{}, 0);

    return cudf::make_lists_column(num_child_rows,
                                   std::move(child_offsets),
                                   std::move(child_column),
                                   child_null_mask.second,             // Null count
                                   std::move(child_null_mask.first));  // Null mask
  }

  /**
   * @brief (Recursively) constructs child columns that are structs.
   */
  template <typename T>
  std::unique_ptr<column> operator()(rmm::device_uvector<unbound_list_view> const& list_vector,
                                     cudf::column_view const& list_offsets,
                                     cudf::lists_column_view const& source_lists_column_view,
                                     cudf::lists_column_view const& target_lists_column_view,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(std::is_same_v<T, struct_view>)
  {
    auto const source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto const target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto const source_lists = cudf::lists_column_device_view(*source_column_device_view);
    auto const target_lists = cudf::lists_column_device_view(*target_column_device_view);

    auto const source_structs = source_lists_column_view.child();
    auto const target_structs = target_lists_column_view.child();

    auto const num_child_rows{
      cudf::detail::get_value<size_type>(list_offsets, list_offsets.size() - 1, stream)};

    auto const num_struct_members =
      std::distance(source_structs.child_begin(), source_structs.child_end());
    std::vector<std::unique_ptr<column>> child_columns;
    child_columns.reserve(num_struct_members);

    auto project_member_as_list_view = [](column_view const& structs_member,
                                          cudf::size_type const& structs_list_num_rows,
                                          column_view const& structs_list_offsets,
                                          bitmask_type const* structs_list_nullmask,
                                          cudf::size_type const& structs_list_null_count) {
      return lists_column_view(column_view(data_type{type_id::LIST},
                                           structs_list_num_rows,
                                           nullptr,
                                           structs_list_nullmask,
                                           structs_list_null_count,
                                           0,
                                           {structs_list_offsets, structs_member}));
    };

    auto const iter_source_member_as_list = thrust::make_transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0}, [&](auto child_idx) {
        return project_member_as_list_view(source_structs.child(child_idx),
                                           source_lists_column_view.size(),
                                           source_lists_column_view.offsets(),
                                           source_lists_column_view.null_mask(),
                                           source_lists_column_view.null_count());
      });

    auto const iter_target_member_as_list = thrust::make_transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0}, [&](auto child_idx) {
        return project_member_as_list_view(target_structs.child(child_idx),
                                           target_lists_column_view.size(),
                                           target_lists_column_view.offsets(),
                                           target_lists_column_view.null_mask(),
                                           target_lists_column_view.null_count());
      });

    std::transform(iter_source_member_as_list,
                   iter_source_member_as_list + num_struct_members,
                   iter_target_member_as_list,
                   std::back_inserter(child_columns),
                   [&](auto source_struct_member_list_view, auto target_struct_member_list_view) {
                     return cudf::type_dispatcher<dispatch_storage_type>(
                       source_struct_member_list_view.child().type(),
                       list_child_constructor{},
                       list_vector,
                       list_offsets,
                       source_struct_member_list_view,
                       target_struct_member_list_view,
                       stream,
                       mr);
                   });

    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::pair(rmm::device_buffer{}, 0);

    return cudf::make_structs_column(num_child_rows,
                                     std::move(child_columns),
                                     child_null_mask.second,
                                     std::move(child_null_mask.first),
                                     stream,
                                     mr);
  }
};

std::unique_ptr<column> build_lists_child_column_recursive(
  data_type child_column_type,
  rmm::device_uvector<unbound_list_view> const& list_vector,
  cudf::column_view const& list_offsets,
  cudf::lists_column_view const& source_lists_column_view,
  cudf::lists_column_view const& target_lists_column_view,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher<dispatch_storage_type>(child_column_type,
                                                      list_child_constructor{},
                                                      list_vector,
                                                      list_offsets,
                                                      source_lists_column_view,
                                                      target_lists_column_view,
                                                      stream,
                                                      mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
