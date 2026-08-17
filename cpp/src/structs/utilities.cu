/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <functional>
#include <numeric>

namespace cudf::structs::detail {

/**
 * @copydoc cudf::structs::detail::extract_ordered_struct_children
 */
std::vector<std::vector<column_view>> extract_ordered_struct_children(
  host_span<column_view const> struct_cols, cuda::stream_ref stream)
{
  auto const num_children = struct_cols[0].num_children();
  auto const num_cols     = static_cast<size_type>(struct_cols.size());

  std::vector<std::vector<column_view>> result;
  result.reserve(num_children);

  for (size_type child_index = 0; child_index < num_children; child_index++) {
    std::vector<column_view> children;
    children.reserve(num_cols);
    for (size_type col_index = 0; col_index < num_cols; col_index++) {
      structs_column_view const scv(struct_cols[col_index]);

      // all inputs must have the same # of children and they must all be of the
      // same type.
      CUDF_EXPECTS(struct_cols[0].num_children() == scv.num_children(),
                   "Mismatch in number of children during struct concatenate");
      CUDF_EXPECTS(struct_cols[0].child(child_index).type() == scv.child(child_index).type(),
                   "Mismatch in child types during struct concatenate");
      children.push_back(scv.get_sliced_child(child_index, stream));
    }

    result.push_back(std::move(children));
  }

  return result;
}

namespace {

/**
 * @brief Check whether the specified column is of type `STRUCT`.
 */
bool is_struct(cudf::column_view const& col) { return col.type().id() == type_id::STRUCT; }

}  // namespace

bool is_or_has_nested_lists(cudf::column_view const& col)
{
  auto is_list = [](cudf::column_view const& col) { return col.type().id() == type_id::LIST; };

  return is_list(col) || std::any_of(col.child_begin(), col.child_end(), is_or_has_nested_lists);
}

/**
 * @brief Flattens struct columns to constituent non-struct columns in the input table.
 *
 */
struct table_flattener {
  table_view input;
  std::vector<order> const& column_order;
  std::vector<null_order> const& null_precedence;
  column_nullability nullability;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  temporary_nullable_data nullable_data;
  std::vector<std::unique_ptr<column>> validity_as_column;
  std::vector<column_view> flat_columns;
  std::vector<order> flat_column_order;
  std::vector<null_order> flat_null_precedence;

  table_flattener(table_view const& input,
                  std::vector<order> const& column_order,
                  std::vector<null_order> const& null_precedence,
                  column_nullability nullability,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
    : column_order{column_order},
      null_precedence{null_precedence},
      nullability{nullability},
      stream{stream},
      mr{mr}
  {
    superimpose_nulls(input);
  }

  /**
   * @brief Pushes down nulls from struct columns to children, saves the resulting
   * column to `input`, and generated null masks to `superimposed_nullmasks`.
   */
  void superimpose_nulls(table_view const& input_table)
  {
    auto [table, tmp_nullable_data] = push_down_nulls(input_table, stream, mr);
    this->input                     = std::move(table);
    this->nullable_data             = std::move(tmp_nullable_data);
  }

  // Convert null_mask to BOOL8 columns and flatten the struct children in order.
  void flatten_struct_column(structs_column_view const& col,
                             order col_order,
                             null_order col_null_order)
  {
    // Even if it is not required to extract the bitmask to a separate column,
    // we should always do that if the structs column has any null element.
    //
    // In addition, we should check for null by calling to `has_nulls()`, not `nullable()`.
    // This is because when comparing structs columns, if one column has bitmask while the other
    // does not (and both columns do not have any null element) then flattening them using
    // `nullable()` will result in tables with different number of columns.
    //
    // Notice that, for comparing structs columns when one column has null while the other
    // doesn't, `nullability` must be passed in with value `column_nullability::FORCE` to make
    // sure the flattening results are tables having the same number of columns.

    if (nullability == column_nullability::FORCE || col.has_nulls()) {
      validity_as_column.push_back(cudf::detail::is_valid(col, stream, mr));
      if (col.has_nulls()) {
        // copy bitmask is needed only if the column has null
        validity_as_column.back()->set_null_mask(cudf::detail::copy_bitmask(col, stream, mr),
                                                 col.null_count());
      }
      flat_columns.push_back(validity_as_column.back()->view());
      if (not column_order.empty()) { flat_column_order.push_back(col_order); }  // doesn't matter.
      if (not null_precedence.empty()) { flat_null_precedence.push_back(col_null_order); }
    }
    for (decltype(col.num_children()) i = 0; i < col.num_children(); ++i) {
      auto const& child = col.get_sliced_child(i, stream);
      if (child.type().id() == type_id::STRUCT) {
        flatten_struct_column(structs_column_view{child}, col_order, col_null_order);
      } else {
        flat_columns.push_back(child);
        if (not column_order.empty()) flat_column_order.push_back(col_order);
        if (not null_precedence.empty()) flat_null_precedence.push_back(col_null_order);
      }
    }
  }
  // Note: possibly expand for flattening list columns too.

  /**
   * @copydoc flattened_table
   *
   * @return tuple with flattened table, flattened column order, flattened null precedence,
   * vector of boolean columns (struct validity).
   */
  auto operator()()
  {
    for (auto i = 0; i < input.num_columns(); ++i) {
      auto const& col = input.column(i);
      if (col.type().id() == type_id::STRUCT) {
        flatten_struct_column(structs_column_view{col},
                              (column_order.empty() ? order() : column_order[i]),
                              (null_precedence.empty() ? null_order() : null_precedence[i]));
      } else {
        flat_columns.push_back(col);
        if (not column_order.empty()) flat_column_order.push_back(column_order[i]);
        if (not null_precedence.empty()) flat_null_precedence.push_back(null_precedence[i]);
      }
    }

    return std::make_unique<flattened_table>(table_view{flat_columns},
                                             std::move(flat_column_order),
                                             std::move(flat_null_precedence),
                                             std::move(validity_as_column),
                                             std::move(nullable_data));
  }
};

std::unique_ptr<flattened_table> flatten_nested_columns(
  table_view const& input,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  column_nullability nullability,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const has_struct = std::any_of(input.begin(), input.end(), is_struct);
  if (not has_struct) {
    return std::make_unique<flattened_table>(input,
                                             column_order,
                                             null_precedence,
                                             std::vector<std::unique_ptr<column>>{},
                                             temporary_nullable_data{});
  }

  return table_flattener{input, column_order, null_precedence, nullability, stream, mr}();
}

namespace {

/**
 * @brief Superimpose the given null mask into the input column and its descendants. This function
 * does not enforce null consistency of the null masks of the descendant columns i.e. non-empty
 * nulls that appear in the descendant columns due to the null mask update are not purged.
 *
 * @param null_mask Null mask to be applied to the input column
 * @param null_count Null count in the given null mask
 * @param input Column to apply the null mask to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate new device memory
 * @return A new column with potentially new null mask
 */
std::unique_ptr<column> superimpose_nulls(bitmask_type const* null_mask,
                                          size_type null_count,
                                          std::unique_ptr<column>&& input,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  if (input->type().id() == cudf::type_id::EMPTY) {
    // EMPTY columns should not have a null mask,
    // so don't superimpose null mask on empty columns.
    return std::move(input);
  }

  auto const num_rows = input->size();

  if (!input->nullable()) {
    input->set_null_mask(cudf::detail::copy_bitmask(null_mask, 0, num_rows, stream, mr),
                         null_count);
  } else {
    auto current_mask = input->mutable_view().null_mask();
    std::vector<bitmask_type const*> masks{reinterpret_cast<bitmask_type const*>(null_mask),
                                           reinterpret_cast<bitmask_type const*>(current_mask)};
    std::vector<size_type> begin_bits{0, 0};
    auto const valid_count = cudf::detail::inplace_bitmask_and(
      device_span<bitmask_type>(current_mask, num_bitmask_words(num_rows)),
      masks,
      begin_bits,
      num_rows,
      stream);
    auto const new_null_count = num_rows - valid_count;
    input->set_null_count(new_null_count);
  }

  // If the input is also a struct, repeat for all its children. Otherwise just return.
  if (input->type().id() != cudf::type_id::STRUCT) { return std::move(input); }

  auto const new_null_count = input->null_count();  // this was just computed in the step above
  auto content              = input->release();

  // Recursively replace the children columns with new struct columns that have the updated null
  // mask
  CUDF_EXPECTS(std::all_of(content.children.begin(),
                           content.children.end(),
                           [&](auto const& child_col) { return num_rows == child_col->size(); }),
               "Child columns must have the same number of rows as the Struct column.");

  for (auto& child : content.children) {
    child = superimpose_nulls(static_cast<bitmask_type const*>(content.null_mask->data()),
                              new_null_count,
                              std::move(child),
                              stream,
                              mr);
  }
  return std::make_unique<column>(cudf::data_type{type_id::STRUCT},
                                  num_rows,
                                  rmm::device_buffer{},  // Empty data buffer. Structs hold no data.
                                  std::move(*content.null_mask),
                                  new_null_count,
                                  std::move(content.children));
}

/**
 * @brief Superimpose each given null mask onto the corresponding input column and its descendants.
 * This function does not enforce null consistency of the null masks of the descendant columns i.e.
 * non-empty nulls that appear in descendant columns due to the null mask update are not purged

 * The vector version of superimpose_nulls applies null masks to multiple columns at once by:
 *  1. First gathering all null masks in a flattened structure
 *  2. Applying the nulls in a segmented batch operation
 *  3. Then updating all the columns with their new null masks
 *
 * @param null_masks Vector of null masks to be applied to the input column
 * @param inputs Vector of input column to apply the null mask to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate new device memory
 * @return A new column with potentially new null mask
 */
std::vector<std::unique_ptr<column>> superimpose_nulls(
  host_span<bitmask_type const* const> null_masks,
  std::vector<std::unique_ptr<column>> inputs,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(null_masks.size() == inputs.size(),
               "The number of null masks to apply must match the number of input columns");

  if (inputs.empty()) { return inputs; }

  auto const num_rows = inputs[0]->size();
  std::vector<bitmask_type const*> sources;
  std::vector<size_type> segment_offsets;
  std::vector<bitmask_type const*> path;

  // This recursive function navigates the column hierarchy and for each path in the tree, it
  // collects all null masks that need to be combined for each column in the hierarchy
  std::function<void(column_view input)> populate_segmented_sources =
    [&populate_segmented_sources, &path, &sources, &segment_offsets](column_view input) -> void {
    if (input.type().id() != cudf::type_id::EMPTY) {
      // EMPTY columns should not have a null mask,
      // so don't superimpose null mask on empty columns.
      if (input.nullable()) {
        // Add this column's null mask to the current path
        path.push_back(input.null_mask());
      }

      // Add all null masks in current path to sources
      sources.insert(sources.end(), path.begin(), path.end());
      segment_offsets.push_back(path.size());

      // For struct columns, recursively process all children
      if (input.type().id() == cudf::type_id::STRUCT) {
        for (int i = 0; i < input.num_children(); i++) {
          populate_segmented_sources(input.child(i));
        }
      }

      // Backtrack: remove this column's mask from path when done
      if (input.nullable()) { path.pop_back(); }
    }
  };

  // Process each top-level column in the inputs vector
  for (size_t c = 0; c < null_masks.size(); c++) {
    // Start with the external null mask for this column
    path.push_back(null_masks[c]);

    // Collect all null masks for this column and its descendants
    populate_segmented_sources(inputs[c]->view());
    path.pop_back();
  }

  // Convert segment_offsets from segment sizes to cumulative offsets
  {
    auto total_sum = std::accumulate(segment_offsets.begin(), segment_offsets.end(), 0);
    std::exclusive_scan(segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin(), 0);
    segment_offsets.push_back(total_sum);
  }

  auto [result_null_masks, result_null_counts] =
    cudf::detail::segmented_bitmask_and(sources, segment_offsets, num_rows, stream, mr);

  // Create new struct column and its descendants with updated null masks
  // Recursively updates each column and its children with their new null masks
  std::function<int(int marker, column& input)> create_updated_column =
    [&create_updated_column, &result_null_masks, &result_null_counts](int marker,
                                                                      column& input) -> int {
    if (input.type().id() != cudf::type_id::EMPTY) {
      // EMPTY columns should not have a null mask,
      // so don't superimpose null mask on empty columns.

      // Update this column's null mask with the result from the batch operation
      input.set_null_mask(std::move(*(result_null_masks[marker])), result_null_counts[marker]);
      marker++;

      // For struct columns, recursively update all children
      if (input.type().id() == cudf::type_id::STRUCT) {
        for (int i = 0; i < input.num_children(); i++) {
          marker = create_updated_column(marker, input.child(i));
        }
      }
    }
    return marker;
  };

  // Apply the new null masks to all top-level columns
  auto marker = 0;
  for (size_t c = 0; c < inputs.size(); c++) {
    marker = create_updated_column(marker, *(inputs[c]));
  }
  return inputs;
}

/**
 * @brief Push down nulls from the given input column into its children columns without any
 * sanitization for non-empty nulls.
 *
 * @copydoc cudf::structs::detail::push_down_nulls
 */
std::pair<column_view, temporary_nullable_data> push_down_nulls_no_sanitize(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto ret_nullable_data = temporary_nullable_data{};
  if (input.type().id() != type_id::STRUCT) {
    // NOOP for non-STRUCT columns.
    return {input, std::move(ret_nullable_data)};
  }

  auto const structs_view = structs_column_view{input};

  // Function to rewrite child null mask.
  auto const child_with_new_mask = [&](auto const& child_idx) {
    auto child = structs_view.get_sliced_child(child_idx, stream);

    // If struct is not nullable, child null mask is retained. NOOP.
    if (not structs_view.nullable()) { return child; }

    auto parent_child_null_masks =
      std::vector<cudf::bitmask_type const*>{structs_view.null_mask(), child.null_mask()};

    auto [new_child_mask, null_count] = [&] {
      if (not child.nullable()) {
        // Adopt parent STRUCT's null mask.
        return std::pair{structs_view.null_mask(), 0};
      }

      // Both STRUCT and child are nullable. AND() for the child's new null mask.
      //
      // Note: ANDing only [offset(), offset()+size()) would not work. The null-mask produced thus
      // would start at offset=0. The column-view attempts to apply its offset() to both the _data
      // and the _null_mask(). It would be better to AND the bits from the beginning, and apply
      // offset() uniformly.
      // Alternatively, one could construct a big enough buffer, and use inplace_bitwise_and.
      auto [new_mask, null_count] = cudf::detail::bitmask_and(parent_child_null_masks,
                                                              std::vector<size_type>{0, 0},
                                                              child.offset() + child.size(),
                                                              stream,
                                                              mr);
      ret_nullable_data.new_null_masks.push_back(std::move(new_mask));
      return std::pair{
        reinterpret_cast<bitmask_type const*>(ret_nullable_data.new_null_masks.back().data()),
        null_count};
    }();

    return column_view(child.type(),
                       child.size(),
                       child.head(),
                       new_child_mask,
                       null_count,
                       child.offset(),
                       std::vector<column_view>{child.child_begin(), child.child_end()});
  };

  auto const child_begin =
    cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, child_with_new_mask);
  auto const child_end = child_begin + structs_view.num_children();
  auto ret_children    = std::vector<column_view>{};

  std::for_each(child_begin, child_end, [&](auto const& child) {
    auto [processed_child, child_nullable_data] = push_down_nulls_no_sanitize(child, stream, mr);
    ret_children.emplace_back(std::move(processed_child));
    ret_nullable_data.emplace_back(std::move(child_nullable_data));
  });

  // Make column view out of newly constructed column_views, and all the validity buffers.

  return std::pair{column_view(input.type(),
                               input.size(),
                               nullptr,
                               input.null_mask(),
                               input.null_count(),  // Alternatively, postpone.
                               input.offset(),
                               ret_children),
                   std::move(ret_nullable_data)};
}

}  // namespace

void temporary_nullable_data::emplace_back(temporary_nullable_data&& other)
{
  auto const move_append = [](auto& dst, auto& src) {
    dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
  };
  move_append(new_null_masks, other.new_null_masks);
  move_append(new_columns, other.new_columns);
}

/*
 * @copydoc cudf::structs::detail::superimpose_and_sanitize_nulls
 */
std::unique_ptr<column> superimpose_and_sanitize_nulls(bitmask_type const* null_mask,
                                                       size_type null_count,
                                                       std::unique_ptr<column>&& input,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  input = superimpose_nulls(null_mask, null_count, std::move(input), stream, mr);

  nvtxRangePushA("purging");
  if (auto const input_view = input->view(); cudf::detail::has_nonempty_nulls(input_view, stream)) {
    // We can't call `purge_nonempty_nulls` for individual child column(s) that need to be
    // sanitized. Instead, we have to call it from the top level column.
    // This is to make sure all the columns (top level + all children) have consistent offsets.
    // Otherwise, the sanitized children may have offsets that are different from the others and
    // also different from the parent column, causing data corruption.
    return cudf::detail::purge_nonempty_nulls(input_view, stream, mr);
  }
  nvtxRangePop();

  return std::move(input);
}

std::vector<std::unique_ptr<column>> superimpose_and_sanitize_nulls(
  host_span<bitmask_type const* const> null_masks,
  std::vector<std::unique_ptr<column>> inputs,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  inputs = superimpose_nulls(null_masks, std::move(inputs), stream, mr);

  std::vector<std::unique_ptr<column>> purged_columns;
  for (auto& input : inputs) {
    auto const input_view = input->view();
    auto const nullbool   = cudf::detail::has_nonempty_nulls(input_view, stream);
    if (nullbool) {
      purged_columns.emplace_back(cudf::detail::purge_nonempty_nulls(input_view, stream, mr));
    } else {
      purged_columns.emplace_back(std::move(input));
    }
  }

  return purged_columns;
}

std::vector<std::unique_ptr<column>> enforce_null_consistency(
  std::vector<std::unique_ptr<column>> columns,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Prepare containers to store information about struct columns
  std::vector<size_type> struct_column_positions;  // Stores positions of struct columns in output
  std::vector<bitmask_type const*> struct_root_masks;      // Stores null masks of struct columns
  std::vector<std::unique_ptr<column>> struct_child_cols;  // Stores child columns of all structs

  // Helper struct to store and manipulate struct column properties
  struct contents {
    size_type null_count                          = 0;        // Number of null values in the struct
    std::unique_ptr<rmm::device_buffer> null_mask = nullptr;  // Null mask buffer for the struct
    size_type num_children                        = 0;  // Number of child columns in this struct
    size_type num_elements                        = 0;  // Number of rows in the struct
  };

  std::vector<contents> struct_contents;  // Store properties of each struct column

  for (size_t i = 0; i < columns.size(); i++) {
    // Process only nullable struct columns
    if (columns[i]->type().id() == cudf::type_id::STRUCT && columns[i]->nullable()) {
      struct_column_positions.push_back(i);  // Record position of this struct column

      // Store the struct's properties
      struct_contents.push_back(contents{.null_count   = columns[i]->null_count(),
                                         .num_children = columns[i]->num_children(),
                                         .num_elements = columns[i]->size()});

      auto col_contents                = columns[i]->release();
      struct_contents.back().null_mask = std::move(col_contents.null_mask);

      // Add null mask pointers for each child of this struct
      struct_root_masks.insert(
        struct_root_masks.end(),
        col_contents.children.size(),
        static_cast<bitmask_type const*>(struct_contents.back().null_mask->data()));

      // Add all child columns from this struct
      for (auto& child : col_contents.children) {
        struct_child_cols.push_back(std::move(child));
      }
    }
  }

  // Apply parent struct nulls to all child columns (if there are any struct columns)
  if (!struct_root_masks.empty()) {
    struct_child_cols =
      superimpose_and_sanitize_nulls(struct_root_masks, std::move(struct_child_cols), stream, mr);
  }

  // Rebuild struct columns with the updated child columns
  auto offset = 0;
  for (size_t i = 0; i < struct_column_positions.size(); i++) {
    // Collect children for this specific struct
    std::vector<std::unique_ptr<column>> children;
    for (auto j = 0; j < struct_contents[i].num_children; j++) {
      children.emplace_back(std::move(struct_child_cols[offset + j]));
    }
    offset += struct_contents[i].num_children;

    // Replace the original struct column with a reconstructed one containing updated children
    columns[struct_column_positions[i]] =
      std::make_unique<column>(cudf::data_type{type_id::STRUCT},
                               struct_contents[i].num_elements,
                               rmm::device_buffer{},  // Empty data buffer. Structs hold no data.
                               std::move(*struct_contents[i].null_mask),
                               struct_contents[i].null_count,
                               std::move(children));
  }

  return columns;
}

std::pair<column_view, temporary_nullable_data> push_down_nulls(column_view const& input,
                                                                cuda::stream_ref stream,
                                                                rmm::device_async_resource_ref mr)
{
  auto output = push_down_nulls_no_sanitize(input, stream, mr);

  if (auto const output_view = output.first;
      cudf::detail::has_nonempty_nulls(output_view, stream)) {
    output.second.new_columns.emplace_back(
      cudf::detail::purge_nonempty_nulls(output_view, stream, mr));
    output.first = output.second.new_columns.back()->view();

    // Don't need the temp null mask anymore, as we will create a new column.
    // However, these null masks are still needed for `purge_nonempty_nulls` thus removing them
    // must be done after calling it.
    output.second.new_null_masks.clear();
  }

  return output;
}

std::pair<table_view, temporary_nullable_data> push_down_nulls(table_view const& table,
                                                               cuda::stream_ref stream,
                                                               rmm::device_async_resource_ref mr)
{
  auto processed_columns = std::vector<column_view>{};
  auto nullable_data     = temporary_nullable_data{};
  for (auto const& col : table) {
    auto [processed_col, col_nullable_data] = push_down_nulls(col, stream, mr);
    processed_columns.emplace_back(std::move(processed_col));
    nullable_data.emplace_back(std::move(col_nullable_data));
  }
  return {table_view{processed_columns}, std::move(nullable_data)};
}

bool contains_null_structs(column_view const& col)
{
  return (is_struct(col) && col.has_nulls()) ||
         std::any_of(col.child_begin(), col.child_end(), contains_null_structs);
}

}  // namespace cudf::structs::detail
