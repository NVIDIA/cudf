/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

// position of the element separator string (e.g. comma ',') within the separators column
constexpr size_type separator_index = 0;
// position of the enclosure strings (e.g. []) within the separators column
constexpr size_type left_brace_index  = 1;
constexpr size_type right_brace_index = 2;

/**
 * @brief Pending separator type for `stack_item`
 */
enum class item_separator : int8_t { NONE, ELEMENT, LIST };

/**
 * @brief Stack item used to manage nested lists.
 *
 * Each item includes the current range and the pending separator.
 */
struct alignas(8) stack_item {
  size_type left_idx;
  size_type right_idx;
  item_separator separator{item_separator::NONE};
};

/**
 * @brief Formatting lists functor.
 *
 * This formats the input list column into individual strings using the
 * specified separators and null-representation (na_rep) string.
 *
 * Recursion is simulated by using stack allocating per output string.
 */
struct format_lists_fn {
  column_device_view const d_input;
  column_device_view const d_separators;
  string_view const d_na_rep;
  stack_item* d_stack;
  size_type const max_depth;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ column_device_view get_nested_child(size_type idx)
  {
    auto current = d_input;
    while (idx > 0) {
      current = current.child(cudf::lists_column_view::child_column_index);
      --idx;
    }
    return current;
  }

  __device__ size_type write_separator(char*& d_output, size_type sep_idx = separator_index)
  {
    auto d_str = [&] {
      if (d_separators.size() > sep_idx) return d_separators.element<string_view>(sep_idx);
      if (sep_idx == left_brace_index) return string_view("[", 1);
      if (sep_idx == right_brace_index) return string_view("]", 1);
      return string_view(",", 1);
    }();
    if (d_output) d_output = copy_string(d_output, d_str);
    return d_str.size_bytes();
  }

  __device__ size_type write_na_rep(char*& d_output)
  {
    if (d_output) d_output = copy_string(d_output, d_na_rep);
    return d_na_rep.size_bytes();
  }

  __device__ size_type write_strings(column_device_view const& col,
                                     size_type left_idx,
                                     size_type right_idx,
                                     char* d_output)
  {
    size_type bytes = 0;
    for (size_type idx = left_idx; idx < right_idx; ++idx) {
      if (col.is_null(idx)) {
        bytes += write_na_rep(d_output);  // e.g. 'NULL'
      } else {
        auto d_str = col.element<string_view>(idx);
        if (d_output) d_output = copy_string(d_output, d_str);
        bytes += d_str.size_bytes();
      }
      if (idx + 1 < right_idx) {
        bytes += write_separator(d_output);  // e.g. comma ','
      }
    }
    return bytes;
  }

  __device__ void operator()(size_type idx)
  {
    size_type bytes = 0;
    char* d_output  = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // push first item to the stack
    auto item_stack         = d_stack + idx * max_depth;
    auto stack_idx          = size_type{0};
    item_stack[stack_idx++] = stack_item{idx, idx + 1};

    // process until stack is empty
    while (stack_idx > 0) {
      --stack_idx;  // pop from stack
      auto const item = item_stack[stack_idx];
      auto const view = get_nested_child(stack_idx);

      auto offsets   = view.child(cudf::lists_column_view::offsets_column_index);
      auto d_offsets = offsets.data<int32_t>() + view.offset();

      // add pending separator
      if (item.separator == item_separator::LIST) {
        bytes += write_separator(d_output, right_brace_index);
      } else if (item.separator == item_separator::ELEMENT) {
        bytes += write_separator(d_output, separator_index);
      }

      // loop through the child elements for the current view
      for (auto jdx = item.left_idx; jdx < item.right_idx; ++jdx) {
        auto const lhs = d_offsets[jdx];
        auto const rhs = d_offsets[jdx + 1];

        if (view.is_null(jdx)) {
          bytes += write_na_rep(d_output);  // e.g. 'NULL'
        } else if (lhs == rhs) {            // e.g. '[]'
          bytes += write_separator(d_output, left_brace_index);
          bytes += write_separator(d_output, right_brace_index);
        } else {
          auto child = view.child(cudf::lists_column_view::child_column_index);
          bytes += write_separator(d_output, left_brace_index);

          // if child is a list type, then recurse into it
          if (child.type().id() == type_id::LIST) {
            // push current state to the stack
            item_stack[stack_idx++] =
              stack_item{jdx + 1,
                         item.right_idx,
                         jdx + 1 < item.right_idx ? item_separator::ELEMENT : item_separator::LIST};
            // push child to the stack
            item_stack[stack_idx++] = stack_item{lhs, rhs};
            break;  // back to the stack (while-loop)
          }

          // otherwise, the child is a strings column;
          // write out the string elements
          auto const size = write_strings(child, lhs, rhs, d_output);
          bytes += size;
          if (d_output) d_output += size;

          bytes += write_separator(d_output, right_brace_index);
        }

        // write element separator (e.g. comma ',') if not at the end
        if (jdx + 1 < item.right_idx) { bytes += write_separator(d_output); }
      }
    }

    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

std::unique_ptr<column> format_list_column(lists_column_view const& input,
                                           string_scalar const& na_rep,
                                           strings_column_view const& separators,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(data_type{type_id::STRING});

  size_type depth = 1;  // count the depth to the strings column
  auto child_col  = input.child();
  while (child_col.type().id() == type_id::LIST) {
    child_col = cudf::lists_column_view(child_col).child();
    ++depth;
  }
  CUDF_EXPECTS(child_col.type().id() == type_id::STRING, "lists child must be a STRING column");

  CUDF_EXPECTS(separators.size() == 0 || separators.size() == 3,
               "Invalid number of separator strings");
  CUDF_EXPECTS(na_rep.is_valid(stream), "Null replacement string must be valid");

  // create stack memory for processing nested lists
  auto stack_buffer = rmm::device_uvector<stack_item>(input.size() * depth, stream);

  auto const d_input      = column_device_view::create(input.parent(), stream);
  auto const d_separators = column_device_view::create(separators.parent(), stream);
  auto const d_na_rep     = na_rep.value(stream);

  auto [offsets_column, chars] = make_strings_children(
    format_lists_fn{*d_input, *d_separators, d_na_rep, stack_buffer.data(), depth},
    input.size(),
    stream,
    mr);

  return make_strings_column(
    input.size(), std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

// external API

std::unique_ptr<column> format_list_column(lists_column_view const& input,
                                           string_scalar const& na_rep,
                                           strings_column_view const& separators,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::format_list_column(input, na_rep, separators, stream, mr);
}

}  // namespace strings
}  // namespace cudf
