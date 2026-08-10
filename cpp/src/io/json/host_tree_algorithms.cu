/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/parsing_utils.cuh"
#include "nested_json.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace cudf::io::json::detail {

/**
 * @brief Get the column indices for the values column for array of arrays rows
 *
 * @param row_array_children_level The level of the row array's children
 * @param d_tree The tree metadata
 * @param col_ids The column ids
 * @param num_columns The number of columns
 * @param stream The stream to use
 * @return The value columns' indices
 */
rmm::device_uvector<NodeIndexT> get_values_column_indices(TreeDepthT const row_array_children_level,
                                                          tree_meta_t const& d_tree,
                                                          device_span<NodeIndexT const> col_ids,
                                                          size_type const num_columns,
                                                          rmm::cuda_stream_view stream)
{
  auto [level2_nodes, level2_indices] = get_array_children_indices(
    row_array_children_level, d_tree.node_levels, d_tree.parent_node_ids, stream);
  auto col_id_location = cuda::make_permutation_iterator(col_ids.begin(), level2_nodes.begin());
  rmm::device_uvector<NodeIndexT> values_column_indices(num_columns, stream);
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  level2_indices.begin(),
                  level2_indices.end(),
                  col_id_location,
                  values_column_indices.begin());
  return values_column_indices;
}

/**
 * @brief Copies strings specified by pair of begin, end offsets to host vector of strings.
 *
 * @param input String device buffer
 * @param node_range_begin Begin offset of the strings
 * @param node_range_end End offset of the strings
 * @param stream CUDA stream
 * @return Vector of strings
 */
std::vector<std::string> copy_strings_to_host_sync(
  device_span<SymbolT const> input,
  device_span<SymbolOffsetT const> node_range_begin,
  device_span<SymbolOffsetT const> node_range_end,
  rmm::cuda_stream_view stream)
{
  auto const num_strings = node_range_begin.size();
  rmm::device_uvector<size_type> string_offsets(num_strings, stream);
  rmm::device_uvector<size_type> string_lengths(num_strings, stream);
  auto d_offset_pairs = thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_offset_pairs,
    d_offset_pairs + num_strings,
    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin()),
    [] __device__(auto const& offsets) {
      // Note: first character for non-field columns
      return cuda::std::make_tuple(
        static_cast<size_type>(cuda::std::get<0>(offsets)),
        static_cast<size_type>(cuda::std::get<1>(offsets) - cuda::std::get<0>(offsets)));
    });

  cudf::io::parse_options_view options_view{};
  options_view.quotechar  = '\0';  // no quotes
  options_view.keepquotes = true;
  auto d_offset_length_it =
    thrust::make_zip_iterator(string_offsets.begin(), string_lengths.begin());
  auto d_column_names = parse_data(input.data(),
                                   d_offset_length_it,
                                   num_strings,
                                   data_type{type_id::STRING},
                                   rmm::device_buffer{},
                                   0,
                                   options_view,
                                   stream,
                                   cudf::get_current_device_resource_ref());
  auto to_host        = [stream](auto const& col) {
    if (col.is_empty()) return std::vector<std::string>{};
    auto const scv     = cudf::strings_column_view(col);
    auto const h_chars = cudf::detail::make_host_vector_async<char>(
      cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
    // Offsets may be INT32 or INT64 depending on the char buffer size, so normalize through the
    // offsetalator rather than reading them as a typed pointer.
    auto d_offsets = rmm::device_uvector<int64_t>(scv.size() + 1, stream);
    thrust::copy_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cudf::detail::offsetalator_factory::make_input_iterator(scv.offsets(), scv.offset()),
      scv.size() + 1,
      d_offsets.begin());
    auto const h_offsets = cudf::detail::make_host_vector_async(d_offsets, stream);
    stream.synchronize();

    // build std::string vector from chars and offsets
    std::vector<std::string> host_data;
    host_data.reserve(col.size());
    std::transform(std::begin(h_offsets),
                   std::end(h_offsets) - 1,
                   std::begin(h_offsets) + 1,
                   std::back_inserter(host_data),
                   [&h_chars](auto start, auto end) {
                     return std::string(h_chars.data() + start, end - start);
                   });
    return host_data;
  };
  return to_host(d_column_names->view());
}

/**
 * @brief Checks if all strings in each string column in the tree are nulls.
 * For non-string columns, it's set as true. If any of rows in a string column is false, it's set as
 * false.
 *
 * @param input Input JSON string device data
 * @param d_column_tree column tree representation of JSON string
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param options Parsing options specifying the parsing behaviour
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Array of bytes where each byte indicate if it is all nulls string column.
 */
rmm::device_uvector<uint8_t> is_all_nulls_each_column(device_span<SymbolT const> input,
                                                      tree_meta_t const& d_column_tree,
                                                      tree_meta_t const& tree,
                                                      device_span<NodeIndexT const> col_ids,
                                                      cudf::io::json_reader_options const& options,
                                                      rmm::cuda_stream_view stream)
{
  auto const num_nodes = col_ids.size();
  auto const num_cols  = d_column_tree.node_categories.size();
  rmm::device_uvector<uint8_t> is_all_nulls(num_cols, stream);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               is_all_nulls.begin(),
               is_all_nulls.end(),
               true);

  auto parse_opt = parsing_options(options, stream);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    num_nodes,
    [options           = parse_opt.view(),
     data              = input.data(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     is_all_nulls      = is_all_nulls.begin()] __device__(size_type i) {
      auto const node_category = column_categories[col_ids[i]];
      if (node_category == NC_STR or node_category == NC_VAL) {
        auto const is_null_literal = serialized_trie_contains(
          options.trie_na,
          {data + range_begin[i], static_cast<size_t>(range_end[i] - range_begin[i])});
        if (!is_null_literal) is_all_nulls[col_ids[i]] = false;
      }
    });
  return is_all_nulls;
}

NodeIndexT get_row_array_parent_col_id(device_span<NodeIndexT const> col_ids,
                                       bool is_enabled_lines,
                                       rmm::cuda_stream_view stream)
{
  if (col_ids.empty()) { return parent_node_sentinel; }

  auto const list_node_index = is_enabled_lines ? 0 : 1;
  if (std::cmp_greater_equal(list_node_index, col_ids.size())) { return parent_node_sentinel; }
  auto const value = cudf::detail::make_host_vector(
    device_span<NodeIndexT const>{col_ids.data() + list_node_index, 1}, stream);

  return value[0];
}
/**
 * @brief Holds member data pointers of `d_json_column`
 *
 */
struct json_column_data {
  using row_offset_t = json_column::row_offset_t;
  row_offset_t* string_offsets;
  row_offset_t* string_lengths;
  row_offset_t* child_offsets;
  bitmask_type* validity;
};

using hashmap_of_device_columns =
  std::unordered_map<NodeIndexT, std::reference_wrapper<device_json_column>>;

struct build_tree_result {
  build_tree_result(cudf::detail::host_vector<bool> ignore_vals,
                    cudf::detail::host_vector<bool> is_mixed_pruned,
                    hashmap_of_device_columns columns,
                    cudf::detail::host_vector<bool> is_schema_mismatched,
                    cudf::detail::host_vector<NodeT> expected_types,
                    NodeIndexT named_level)
    : ignore_vals{std::move(ignore_vals)},
      is_mixed_pruned{std::move(is_mixed_pruned)},
      columns{std::move(columns)},
      is_schema_mismatched{std::move(is_schema_mismatched)},
      expected_types{std::move(expected_types)},
      named_level{named_level}
  {
  }

  explicit build_tree_result(rmm::cuda_stream_view stream)
    : build_tree_result{cudf::detail::make_host_vector<bool>(0, stream),
                        cudf::detail::make_host_vector<bool>(0, stream),
                        {},
                        cudf::detail::make_host_vector<bool>(0, stream),
                        cudf::detail::make_host_vector<NodeT>(0, stream),
                        parent_node_sentinel}
  {
  }

  cudf::detail::host_vector<bool> ignore_vals;
  cudf::detail::host_vector<bool> is_mixed_pruned;
  hashmap_of_device_columns columns;
  cudf::detail::host_vector<bool> is_schema_mismatched;
  cudf::detail::host_vector<NodeT> expected_types;
  NodeIndexT named_level;
};

[[nodiscard]] build_tree_result build_tree(device_json_column& root,
                                           host_span<uint8_t const> is_str_column_all_nulls,
                                           tree_meta_t& d_column_tree,
                                           device_span<NodeIndexT const> d_unique_col_ids,
                                           device_span<size_type const> d_max_row_offsets,
                                           std::vector<std::string> const& column_names,
                                           NodeIndexT row_array_parent_col_id,
                                           bool is_array_of_arrays,
                                           cudf::io::json_reader_options const& options,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

void scatter_offsets(tree_meta_t const& tree,
                     device_span<NodeIndexT const> col_ids,
                     device_span<size_type const> row_offsets,
                     device_span<size_type> node_ids,
                     device_span<size_type> sorted_col_ids,  // Reuse this for parent_col_ids
                     tree_meta_t const& d_column_tree,
                     host_span<bool const> ignore_vals,
                     host_span<bool const> is_mixed,
                     hashmap_of_device_columns const& columns,
                     rmm::cuda_stream_view stream);

std::map<std::string, schema_element> unified_schema(cudf::io::json_reader_options const& options)
{
  return std::visit(
    cudf::detail::visitor_overload{
      [](std::vector<data_type> const& user_dtypes) {
        std::map<std::string, schema_element> dnew;
        std::transform(cuda::counting_iterator<size_t>{0},
                       cuda::counting_iterator<size_t>{user_dtypes.size()},
                       std::inserter(dnew, dnew.end()),
                       [&user_dtypes](auto i) {
                         return std::pair(std::to_string(i), schema_element{user_dtypes[i]});
                       });
        return dnew;
      },
      [](std::map<std::string, data_type> const& user_dtypes) {
        std::map<std::string, schema_element> dnew;
        std::transform(user_dtypes.begin(),
                       user_dtypes.end(),
                       std::inserter(dnew, dnew.end()),
                       [](auto key_dtype) {
                         return std::pair(key_dtype.first, schema_element{key_dtype.second});
                       });
        return dnew;
      },
      [](std::map<std::string, schema_element> const& user_dtypes) { return user_dtypes; },
      [](schema_element const& user_dtypes) { return user_dtypes.child_types; }},
    options.get_dtypes());
}

/**
 * @brief Constructs `d_json_column` from node tree representation
 * Newly constructed columns are inserted into `root`'s children.
 * `root` must be a list type.
 *
 * @param input Input JSON string device data
 * @param tree Node tree representation of the JSON string
 * @param col_ids Column ids of the nodes in the tree
 * @param row_offsets Row offsets of the nodes in the tree
 * @param root Root node of the `d_json_column` tree
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param collect_schema_mismatch_rows Whether to collect row-level schema mismatch diagnostics
 * @param options Parsing options specifying the parsing behaviour
 * options affecting behaviour are
 *   is_enabled_lines: Whether the input is a line-delimited JSON
 *   is_enabled_mixed_types_as_string: Whether to enable reading mixed types as string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the device memory
 * of child_offets and validity members of `d_json_column`
 */
void make_device_json_column(device_span<SymbolT const> input,
                             tree_meta_t const& tree,
                             device_span<NodeIndexT const> col_ids,
                             device_span<size_type const> row_offsets,
                             device_json_column& root,
                             bool is_array_of_arrays,
                             bool collect_schema_mismatch_rows,
                             cudf::io::json_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  // make a copy
  auto sorted_col_ids = cudf::detail::make_device_uvector_async(
    col_ids, stream, cudf::get_current_device_resource_ref());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<NodeIndexT> node_ids(col_ids.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   node_ids.begin(),
                   node_ids.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    sorted_col_ids.begin(),
    sorted_col_ids.end(),
    node_ids.begin());

  NodeIndexT const row_array_parent_col_id =
    get_row_array_parent_col_id(col_ids, is_enabled_lines, stream);

  // 1. gather column information.
  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    reduce_to_column_tree(tree,
                          col_ids,
                          sorted_col_ids,
                          node_ids,
                          row_offsets,
                          is_array_of_arrays,
                          row_array_parent_col_id,
                          stream);

  auto num_columns                      = d_unique_col_ids.size();
  std::vector<std::string> column_names = copy_strings_to_host_sync(
    input, d_column_tree.node_range_begin, d_column_tree.node_range_end, stream);
  // array of arrays column names
  if (is_array_of_arrays) {
    auto const unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
    auto const column_parent_ids =
      cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
    TreeDepthT const row_array_children_level = is_enabled_lines ? 1 : 2;
    auto values_column_indices =
      get_values_column_indices(row_array_children_level, tree, col_ids, num_columns, stream);
    auto h_values_column_indices = cudf::detail::make_host_vector(values_column_indices, stream);
    std::transform(unique_col_ids.begin(),
                   unique_col_ids.end(),
                   column_names.cbegin(),
                   column_names.begin(),
                   [&h_values_column_indices, &column_parent_ids, row_array_parent_col_id](
                     auto col_id, auto name) mutable {
                     return column_parent_ids[col_id] == row_array_parent_col_id
                              ? std::to_string(h_values_column_indices[col_id])
                              : name;
                   });
  }

  auto const is_str_column_all_nulls = [&, &column_tree = d_column_tree]() {
    if (is_enabled_mixed_types_as_string) {
      return cudf::detail::make_std_vector(
        is_all_nulls_each_column(input, column_tree, tree, col_ids, options, stream), stream);
    }
    return std::vector<uint8_t>();
  }();
  auto build_result                = build_tree(root,
                                 is_str_column_all_nulls,
                                 d_column_tree,
                                 d_unique_col_ids,
                                 d_max_row_offsets,
                                 column_names,
                                 row_array_parent_col_id,
                                 is_array_of_arrays,
                                 options,
                                 stream,
                                 mr);
  auto const& ignore_vals          = build_result.ignore_vals;
  auto const& is_mixed_pruned      = build_result.is_mixed_pruned;
  auto const& is_schema_mismatched = build_result.is_schema_mismatched;
  auto const& expected_types       = build_result.expected_types;
  auto const named_level           = build_result.named_level;
  auto const& columns              = build_result.columns;
  if (ignore_vals.empty()) return;

  if (collect_schema_mismatch_rows) {
    auto const h_col_ids     = cudf::detail::make_pinned_vector_async(col_ids, stream);
    auto const h_row_offsets = cudf::detail::make_pinned_vector_async(row_offsets, stream);
    auto const h_parent_node_ids =
      cudf::detail::make_pinned_vector_async(tree.parent_node_ids, stream);
    auto const h_column_parent_ids =
      cudf::detail::make_pinned_vector_async(d_column_tree.parent_node_ids, stream);
    auto const h_node_categories =
      cudf::detail::make_pinned_vector_async(tree.node_categories, stream);
    stream.synchronize();

    std::map<std::string, std::set<size_type>> rows_by_top_level_column;
    for (size_type node_id = 0; node_id < static_cast<size_type>(h_col_ids.size()); ++node_id) {
      auto const mismatch_col_id = h_col_ids[node_id];
      if (mismatch_col_id < 0 ||
          static_cast<std::size_t>(mismatch_col_id) >= expected_types.size()) {
        continue;
      }

      auto const actual_category   = h_node_categories[node_id];
      auto const expected_category = expected_types[mismatch_col_id];
      auto const expected_nested_category_mismatch =
        (expected_category == NC_STRUCT || expected_category == NC_LIST) &&
        actual_category != NC_FN && actual_category != expected_category;
      if (!is_schema_mismatched[mismatch_col_id] && !expected_nested_category_mismatch) {
        continue;
      }

      auto top_node_id = static_cast<NodeIndexT>(node_id);
      while (top_node_id != parent_node_sentinel) {
        auto const top_col_id = h_col_ids[top_node_id];
        if (top_col_id >= 0 && static_cast<std::size_t>(top_col_id) < h_column_parent_ids.size() &&
            h_column_parent_ids[top_col_id] == named_level) {
          auto const& top_level_column_name = column_names[top_col_id];
          root.schema_mismatch_column_names.insert(top_level_column_name);
          rows_by_top_level_column[top_level_column_name].insert(h_row_offsets[top_node_id]);
          break;
        }
        top_node_id = h_parent_node_ids[top_node_id];
      }
    }

    root.rows_with_schema_mismatch.reserve(rows_by_top_level_column.size());
    for (auto const& [col_name, rows] : rows_by_top_level_column) {
      root.rows_with_schema_mismatch.push_back(
        schema_mismatch_rows{col_name, std::vector<size_type>{rows.begin(), rows.end()}});
    }
  }

  scatter_offsets(tree,
                  col_ids,
                  row_offsets,
                  node_ids,
                  sorted_col_ids,
                  d_column_tree,
                  ignore_vals,
                  is_mixed_pruned,
                  columns,
                  stream);
}

[[nodiscard]] build_tree_result build_tree(device_json_column& root,
                                           host_span<uint8_t const> is_str_column_all_nulls,
                                           tree_meta_t& d_column_tree,
                                           device_span<NodeIndexT const> d_unique_col_ids,
                                           device_span<size_type const> d_max_row_offsets,
                                           std::vector<std::string> const& column_names,
                                           NodeIndexT row_array_parent_col_id,
                                           bool is_array_of_arrays,
                                           cudf::io::json_reader_options const& options,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  bool const is_enabled_lines                 = options.is_enabled_lines();
  bool const is_enabled_mixed_types_as_string = options.is_enabled_mixed_types_as_string();
  auto unique_col_ids = cudf::detail::make_host_vector_async(d_unique_col_ids, stream);
  auto column_categories =
    cudf::detail::make_host_vector_async(d_column_tree.node_categories, stream);
  auto const column_parent_ids =
    cudf::detail::make_host_vector_async(d_column_tree.parent_node_ids, stream);
  auto column_range_beg =
    cudf::detail::make_host_vector_async(d_column_tree.node_range_begin, stream);
  auto const max_row_offsets = cudf::detail::make_host_vector_async(d_max_row_offsets, stream);
  auto num_columns           = d_unique_col_ids.size();
  stream.synchronize();

  auto to_json_col_type = [](auto category) {
    switch (category) {
      case NC_STRUCT: return json_col_t::StructColumn;
      case NC_LIST: return json_col_t::ListColumn;
      case NC_STR: [[fallthrough]];
      case NC_VAL: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    }
  };

  auto initialize_json_columns = [&](auto i, auto& col_ref, auto column_category) {
    auto& col = col_ref.get();
    if (col.type != json_col_t::Unknown) { return; }
    if (column_category == NC_ERR || column_category == NC_FN) {
      return;
    } else if (column_category == NC_VAL || column_category == NC_STR) {
      col.string_offsets.resize(max_row_offsets[i] + 1, stream);
      col.string_lengths.resize(max_row_offsets[i] + 1, stream);
      thrust::fill(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        thrust::make_zip_iterator(col.string_offsets.begin(), col.string_lengths.begin()),
        thrust::make_zip_iterator(col.string_offsets.end(), col.string_lengths.end()),
        cuda::std::make_tuple(0, 0));
    } else if (column_category == NC_LIST) {
      col.child_offsets.resize(max_row_offsets[i] + 2, stream);
      thrust::uninitialized_fill(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        col.child_offsets.begin(),
        col.child_offsets.end(),
        0);
    }
    col.num_rows = max_row_offsets[i] + 1;
    col.validity =
      cudf::detail::create_null_mask(col.num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    col.type = to_json_col_type(column_category);
  };

  // 2. generate nested columns tree and its device_memory
  // reorder unique_col_ids w.r.t. column_range_begin for order of column to be in field order.
  auto h_range_col_id_it =
    thrust::make_zip_iterator(column_range_beg.begin(), unique_col_ids.begin());
  std::sort(h_range_col_id_it, h_range_col_id_it + num_columns, [](auto const& a, auto const& b) {
    return cuda::std::get<0>(a) < cuda::std::get<0>(b);
  });
  // adjacency list construction
  std::map<NodeIndexT, std::vector<NodeIndexT>> adj;
  for (auto const this_col_id : unique_col_ids) {
    auto parent_col_id = column_parent_ids[this_col_id];
    adj[parent_col_id].push_back(this_col_id);
  }

  // Pruning
  auto is_pruned = cudf::detail::make_host_vector<bool>(num_columns, stream);
  std::fill_n(is_pruned.begin(), num_columns, options.is_enabled_prune_columns());

  // prune all children of a column, but not self.
  auto ignore_all_children = [&adj, &is_pruned](auto parent_col_id) {
    std::deque<NodeIndexT> offspring;
    if (adj.count(parent_col_id)) {
      for (auto const& child : adj[parent_col_id]) {
        offspring.push_back(child);
      }
    }
    while (!offspring.empty()) {
      auto this_id = offspring.front();
      offspring.pop_front();
      is_pruned[this_id] = true;
      if (adj.count(this_id)) {
        for (auto const& child : adj[this_id]) {
          offspring.push_back(child);
        }
      }
    }
  };

  // Pruning: iterate through schema and mark only those columns and enforce type.
  // NoPruning: iterate through schema and enforce type.

  if (adj[parent_node_sentinel].empty()) return build_tree_result{stream};  // for empty file
  CUDF_EXPECTS(adj[parent_node_sentinel].size() == 1, "Should be 1");
  auto expected_types = cudf::detail::make_host_vector<NodeT>(num_columns, stream);
  std::fill_n(expected_types.begin(), num_columns, NUM_NODE_CLASSES);

  auto lookup_names = [&column_names](auto const& child_ids, auto const& name) -> NodeIndexT {
    for (auto const& child_id : child_ids) {
      if (column_names[child_id] == name) return child_id;
    }
    return -1;
  };
  // Collected during mark_is_pruned: each col_id whose JSON-tree category did not match the
  // requested schema type (and was therefore pruned). After `construct_tree` returns we walk each
  // entry up to its top-level ancestor and add that ancestor's name to
  // `root.schema_mismatch_column_names`, which (when the caller invoked
  // `device_parse_nested_json_with_diagnostics`) feeds
  // `json_reader_diagnostics::top_level_columns_with_schema_mismatch`.
  auto mismatched_col_ids = std::vector<NodeIndexT>{};

  // recursive lambda on schema to mark columns as pruned.
  std::function<void(NodeIndexT root, schema_element const& schema)> mark_is_pruned;
  mark_is_pruned = [&is_pruned,
                    &mark_is_pruned,
                    &adj,
                    &lookup_names,
                    &column_categories,
                    &expected_types,
                    &ignore_all_children,
                    &mismatched_col_ids](NodeIndexT root, schema_element const& schema) -> void {
    if (root == -1) return;
    bool pass =
      (schema.type == data_type{type_id::STRUCT} and column_categories[root] == NC_STRUCT) or
      (schema.type == data_type{type_id::LIST} and column_categories[root] == NC_LIST) or
      (schema.type != data_type{type_id::STRUCT} and schema.type != data_type{type_id::LIST} and
       column_categories[root] != NC_FN);
    if (!pass) {
      // The JSON tree's actual category for this node disagrees with the requested schema type
      // (e.g. JSON has a scalar where schema expects a struct). Record the col_id so that after
      // `construct_tree` builds the device tree we can walk up to the top-level ancestor and
      // record the diagnostic on `device_json_column::schema_mismatch_column_names`.
      mismatched_col_ids.push_back(root);
      // ignore all children of this column and prune this column.
      is_pruned[root] = true;
      ignore_all_children(root);
      return;
    }
    is_pruned[root]    = false;
    auto expected_type = [](auto type, auto cat) {
      if (type == data_type{type_id::STRUCT} and cat == NC_STRUCT) return NC_STRUCT;
      if (type == data_type{type_id::LIST} and cat == NC_LIST) return NC_LIST;
      if (type != data_type{type_id::STRUCT} and type != data_type{type_id::LIST}) return NC_STR;
      return NC_ERR;
    }(schema.type, column_categories[root]);
    expected_types[root] = expected_type;  // forced type.
    // ignore children of nested columns, but not self.
    if (expected_type == NC_STR and
        (column_categories[root] == NC_STRUCT or column_categories[root] == NC_LIST))
      ignore_all_children(root);
    if (not(schema.type == data_type{type_id::STRUCT} or schema.type == data_type{type_id::LIST}))
      return;  // no children to mark for non-nested.
    auto child_ids = adj.count(root) ? adj[root] : std::vector<NodeIndexT>{};
    if (schema.type == data_type{type_id::STRUCT}) {
      for (auto const& key_pair : schema.child_types) {
        auto col_id = lookup_names(child_ids, key_pair.first);
        if (col_id == -1) continue;
        is_pruned[col_id] = false;
        for (auto const& child_id : adj[col_id])  // children of field (>1 if mixed)
          mark_is_pruned(child_id, key_pair.second);
      }
    } else if (schema.type == data_type{type_id::LIST}) {
      // partial solution for list children to have any name.
      auto this_list_child_name =
        schema.child_types.size() == 1 ? schema.child_types.begin()->first : list_child_name;
      if (schema.child_types.count(this_list_child_name) == 0) return;
      auto list_child = schema.child_types.at(this_list_child_name);
      for (auto const& child_id : child_ids)
        mark_is_pruned(child_id, list_child);
      // TODO: Store null map of non-target types for list children to mark list entry as null.
    }
  };
  if (is_array_of_arrays) {
    if (adj[adj[parent_node_sentinel][0]].empty()) return build_tree_result{stream};
    auto root_list_col_id =
      is_enabled_lines ? adj[parent_node_sentinel][0] : adj[adj[parent_node_sentinel][0]][0];
    // mark root and row array col_id as not pruned.
    if (!is_enabled_lines) {
      auto top_level_list_id       = adj[parent_node_sentinel][0];
      is_pruned[top_level_list_id] = false;
    }
    is_pruned[root_list_col_id] = false;
    std::visit(cudf::detail::visitor_overload{
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::vector<data_type> const& user_dtypes) -> void {
                   for (size_t i = 0; i < adj[root_list_col_id].size() && i < user_dtypes.size();
                        i++) {
                     NodeIndexT const first_child_id = adj[root_list_col_id][i];
                     auto const& name                = column_names[first_child_id];
                     auto value_id                   = std::stol(name);
                     if (value_id >= 0 and value_id < static_cast<long>(user_dtypes.size()))
                       mark_is_pruned(first_child_id, schema_element{user_dtypes[value_id]});
                     // Note: mixed type - forced type, will work here.
                   }
                 },
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::map<std::string, data_type> const& user_dtypes) -> void {
                   for (int first_child_id : adj[root_list_col_id]) {
                     auto const& name = column_names[first_child_id];
                     if (user_dtypes.count(name))
                       mark_is_pruned(first_child_id, schema_element{user_dtypes.at(name)});
                   }
                 },
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   std::map<std::string, schema_element> const& user_dtypes) -> void {
                   for (int first_child_id : adj[root_list_col_id]) {
                     auto const& name = column_names[first_child_id];
                     if (user_dtypes.count(name))
                       mark_is_pruned(first_child_id, user_dtypes.at(name));
                   }
                 },
                 [&root_list_col_id, &adj, &mark_is_pruned, &column_names](
                   schema_element const& user_dtypes) -> void {
                   for (int first_child_id : adj[root_list_col_id]) {
                     auto const& name = column_names[first_child_id];
                     if (user_dtypes.child_types.count(name) != 0)
                       mark_is_pruned(first_child_id, user_dtypes.child_types.at(name));
                   }
                 }},
               options.get_dtypes());
  } else {
    auto root_struct_col_id =
      is_enabled_lines
        ? adj[parent_node_sentinel][0]
        : (adj[adj[parent_node_sentinel][0]].empty() ? -1 : adj[adj[parent_node_sentinel][0]][0]);
    // mark root and row struct col_id as not pruned.
    if (!is_enabled_lines) {
      auto top_level_list_id       = adj[parent_node_sentinel][0];
      is_pruned[top_level_list_id] = false;
    }
    is_pruned[root_struct_col_id] = false;
    schema_element u_schema{data_type{type_id::STRUCT}};
    u_schema.child_types = unified_schema(options);
    std::visit(
      cudf::detail::visitor_overload{
        [&is_pruned, &root_struct_col_id, &adj, &mark_is_pruned](
          std::vector<data_type> const& user_dtypes) -> void {
          for (size_t i = 0; i < adj[root_struct_col_id].size() && i < user_dtypes.size(); i++) {
            NodeIndexT const first_field_id = adj[root_struct_col_id][i];
            is_pruned[first_field_id]       = false;
            for (auto const& child_id : adj[first_field_id])  // children of field (>1 if mixed)
              mark_is_pruned(child_id, schema_element{user_dtypes[i]});
          }
        },
        [&root_struct_col_id, &adj, &mark_is_pruned, &u_schema](
          std::map<std::string, data_type> const& user_dtypes) -> void {
          mark_is_pruned(root_struct_col_id, u_schema);
        },
        [&root_struct_col_id, &adj, &mark_is_pruned, &u_schema](
          std::map<std::string, schema_element> const& user_dtypes) -> void {
          mark_is_pruned(root_struct_col_id, u_schema);
        },
        [&root_struct_col_id, &adj, &mark_is_pruned, &u_schema](schema_element const& user_dtypes)
          -> void { mark_is_pruned(root_struct_col_id, u_schema); }},
      options.get_dtypes());
  }
  // Useful for array of arrays
  auto named_level =
    is_enabled_lines
      ? adj[parent_node_sentinel][0]
      : (adj[adj[parent_node_sentinel][0]].empty() ? -1 : adj[adj[parent_node_sentinel][0]][0]);
  auto add_top_level_schema_mismatch = [&](NodeIndexT col_id) {
    while (col_id != parent_node_sentinel and col_id != -1) {
      if (column_parent_ids[col_id] == named_level) {
        root.schema_mismatch_column_names.insert(column_names[col_id]);
        return;
      }
      col_id = column_parent_ids[col_id];
    }
  };

  // List children which are pruned mixed types, nullify parent list row.
  auto is_mixed_pruned = cudf::detail::make_host_vector<bool>(num_columns, stream);
  std::fill_n(is_mixed_pruned.begin(), num_columns, false);
  auto handle_mixed_types = [&column_categories,
                             &is_str_column_all_nulls,
                             &is_mixed_pruned,
                             &is_pruned,
                             &expected_types,
                             &is_enabled_mixed_types_as_string,
                             &ignore_all_children](std::vector<NodeIndexT>& child_ids) {
    // do these on unpruned columns only.
    // when mixed types is disabled, ignore string sibling of nested column.
    // when mixed types is disabled, and both list and struct columns are siblings, error out.
    // when mixed types is enabled, force string type on all columns

    // Remove pruned children (forced type will not clash here because other types are already
    // pruned)
    child_ids.erase(
      std::remove_if(child_ids.begin(),
                     child_ids.end(),
                     [&is_pruned](NodeIndexT child_id) { return is_pruned[child_id]; }),
      child_ids.end());
    // find string id, struct id, list id.
    NodeIndexT str_col_id{-1}, struct_col_id{-1}, list_col_id{-1};
    for (auto const& child_id : child_ids) {
      if (column_categories[child_id] == NC_VAL || column_categories[child_id] == NC_STR)
        str_col_id = child_id;
      else if (column_categories[child_id] == NC_STRUCT)
        struct_col_id = child_id;
      else if (column_categories[child_id] == NC_LIST)
        list_col_id = child_id;
    }
    // conditions for handling mixed types.
    if (is_enabled_mixed_types_as_string) {
      if (struct_col_id != -1 and list_col_id != -1) {
        expected_types[struct_col_id] = NC_STR;
        expected_types[list_col_id]   = NC_STR;
        // ignore children of nested columns.
        ignore_all_children(struct_col_id);
        ignore_all_children(list_col_id);
      }
      if ((struct_col_id != -1 or list_col_id != -1) and str_col_id != -1) {
        if (is_str_column_all_nulls[str_col_id])
          is_pruned[str_col_id] = true;
        else {
          // ignore children of nested columns.
          if (struct_col_id != -1) {
            expected_types[struct_col_id] = NC_STR;
            ignore_all_children(struct_col_id);
          }
          if (list_col_id != -1) {
            expected_types[list_col_id] = NC_STR;
            ignore_all_children(list_col_id);
          }
        }
      }
    } else {
      // if both are present, error out.
      CUDF_EXPECTS(struct_col_id == -1 or list_col_id == -1,
                   "A mix of lists and structs within the same column is not supported");
      // either one only: so ignore str column.
      if ((struct_col_id != -1 or list_col_id != -1) and str_col_id != -1) {
        is_pruned[str_col_id] = true;
      }
    }
  };

  using dev_ref = std::reference_wrapper<device_json_column>;
  std::unordered_map<NodeIndexT, dev_ref> columns;
  columns.try_emplace(parent_node_sentinel, std::ref(root));
  // convert adjaceny list to tree.
  dev_ref parent_ref = std::ref(root);
  // creates children column
  std::function<void(NodeIndexT, dev_ref)> construct_tree;
  construct_tree = [&](NodeIndexT root, dev_ref ref) -> void {
    if (is_pruned[root]) return;
    auto expected_category =
      expected_types[root] == NUM_NODE_CLASSES ? column_categories[root] : expected_types[root];
    initialize_json_columns(root, ref, expected_category);
    auto child_ids = adj.count(root) ? adj[root] : std::vector<NodeIndexT>{};
    if (expected_category == NC_STRUCT) {
      // find field column ids, and its children and create columns.
      for (auto const& field_id : child_ids) {
        auto const& name = column_names[field_id];
        if (is_pruned[field_id]) continue;
        auto inserted =
          ref.get().child_columns.try_emplace(name, device_json_column(stream, mr)).second;
        ref.get().column_order.emplace_back(name);
        CUDF_EXPECTS(inserted,
                     "struct child column insertion failed, duplicate column name in the parent");
        auto this_ref = std::ref(ref.get().child_columns.at(name));
        // Mixed type handling
        auto& value_col_ids = adj[field_id];
        handle_mixed_types(value_col_ids);
        if (value_col_ids.empty()) {
          // If no column is present, remove the uninitialized column.
          ref.get().child_columns.erase(name);
          ref.get().column_order.pop_back();
          continue;
        }
        for (auto const& child_id : value_col_ids)  // children of field (>1 if mixed)
        {
          if (is_pruned[child_id]) continue;
          columns.try_emplace(child_id, this_ref);
          construct_tree(child_id, this_ref);
        }
      }
    } else if (expected_category == NC_LIST) {
      // array of arrays interpreted as array of structs.
      if (is_array_of_arrays and root == named_level) {
        // create column names
        std::map<NodeIndexT, std::vector<NodeIndexT>> array_values;
        for (auto const& child_id : child_ids) {
          if (is_pruned[child_id]) continue;
          auto const& name = column_names[child_id];
          array_values[std::stoi(name)].push_back(child_id);
        }
        //
        for (auto const& value_id_pair : array_values) {
          auto [value_id, value_col_ids] = value_id_pair;
          auto name                      = std::to_string(value_id);
          auto inserted =
            ref.get().child_columns.try_emplace(name, device_json_column(stream, mr)).second;
          ref.get().column_order.emplace_back(name);
          CUDF_EXPECTS(inserted,
                       "list child column insertion failed, duplicate column name in the parent");
          auto this_ref = std::ref(ref.get().child_columns.at(name));
          handle_mixed_types(value_col_ids);
          if (value_col_ids.empty()) {
            // If no column is present, remove the uninitialized column.
            ref.get().child_columns.erase(name);
            ref.get().column_order.pop_back();
            continue;
          }
          for (auto const& child_id : value_col_ids)  // children of field (>1 if mixed)
          {
            if (is_pruned[child_id]) continue;
            columns.try_emplace(child_id, this_ref);
            construct_tree(child_id, this_ref);
          }
        }
      } else {
        if (child_ids.empty()) return;
        auto inserted =
          ref.get()
            .child_columns.try_emplace(list_child_name, device_json_column(stream, mr))
            .second;
        CUDF_EXPECTS(inserted,
                     "list child column insertion failed, duplicate column name in the parent");
        ref.get().column_order.emplace_back(list_child_name);
        auto this_ref = std::ref(ref.get().child_columns.at(list_child_name));
        if (options.is_enabled_experimental()) {
          for (auto const& child_id : child_ids) {
            if (is_pruned[child_id]) {
              // store this child_id for mixed_type nullify parent list_id.
              is_mixed_pruned[child_id] = is_pruned[child_id];
            }
          }
        }
        // Mixed type handling
        handle_mixed_types(child_ids);
        if (child_ids.empty()) {
          // If no column is present, remove the uninitialized column.
          ref.get().child_columns.erase(list_child_name);
        }
        for (auto const& child_id : child_ids) {
          if (is_pruned[child_id]) continue;
          columns.try_emplace(child_id, this_ref);
          construct_tree(child_id, this_ref);
        }
      }
    }
  };
  auto inserted = parent_ref.get()
                    .child_columns.try_emplace(list_child_name, device_json_column(stream, mr))
                    .second;
  CUDF_EXPECTS(inserted, "child column insertion failed, duplicate column name in the parent");
  parent_ref = std::ref(parent_ref.get().child_columns.at(list_child_name));
  columns.try_emplace(adj[parent_node_sentinel][0], parent_ref);
  construct_tree(adj[parent_node_sentinel][0], parent_ref);

  // Forced string type due to input schema and mixed type as string.
  for (size_t i = 0; i < expected_types.size(); i++) {
    if (expected_types[i] == NC_STR) {
      if (columns.count(i)) { columns.at(i).get().forced_as_string_column = true; }
    }
  }

  // For each schema-mismatched col_id (collected in `mark_is_pruned` above), record the name of
  // its top-level output column on `root.schema_mismatch_column_names`. The diagnostic is only
  // surfaced per top-level output column, so we do not need to mark intermediate ancestors'
  // `device_json_column`s.
  auto is_schema_mismatched = cudf::detail::make_host_vector<bool>(num_columns, stream);
  std::fill_n(is_schema_mismatched.begin(), num_columns, false);
  for (auto const mismatched_id : mismatched_col_ids) {
    if (mismatched_id >= 0 &&
        static_cast<std::size_t>(mismatched_id) < is_schema_mismatched.size()) {
      is_schema_mismatched[mismatched_id] = true;
    }
    add_top_level_schema_mismatch(mismatched_id);
  }
  std::transform(expected_types.cbegin(),
                 expected_types.cend(),
                 column_categories.cbegin(),
                 expected_types.begin(),
                 [](auto exp, auto cat) { return exp == NUM_NODE_CLASSES ? cat : exp; });
  cudf::detail::cuda_memcpy<NodeT>(d_column_tree.node_categories, expected_types, stream);

  return build_tree_result{std::move(is_pruned),
                           std::move(is_mixed_pruned),
                           std::move(columns),
                           std::move(is_schema_mismatched),
                           std::move(expected_types),
                           named_level};
}

void scatter_offsets(tree_meta_t const& tree,
                     device_span<NodeIndexT const> col_ids,
                     device_span<size_type const> row_offsets,
                     device_span<size_type> node_ids,
                     device_span<size_type> sorted_col_ids,  // Reuse this for parent_col_ids
                     tree_meta_t const& d_column_tree,
                     host_span<bool const> ignore_vals,
                     host_span<bool const> is_mixed_pruned,
                     hashmap_of_device_columns const& columns,
                     rmm::cuda_stream_view stream)
{
  auto const num_nodes   = col_ids.size();
  auto const num_columns = d_column_tree.node_categories.size();
  // move columns data to device.
  auto columns_data = cudf::detail::make_host_vector<json_column_data>(num_columns, stream);
  for (auto& [col_id, col_ref] : columns) {
    if (col_id == parent_node_sentinel) continue;
    auto& col            = col_ref.get();
    columns_data[col_id] = json_column_data{col.string_offsets.data(),
                                            col.string_lengths.data(),
                                            col.child_offsets.data(),
                                            static_cast<bitmask_type*>(col.validity.data())};
  }

  auto d_ignore_vals = cudf::detail::make_device_uvector_async(
    ignore_vals, stream, cudf::get_current_device_resource_ref());
  auto d_is_mixed_pruned = cudf::detail::make_device_uvector_async(
    is_mixed_pruned, stream, cudf::get_current_device_resource_ref());
  auto d_columns_data = cudf::detail::make_device_uvector_async(
    columns_data, stream, cudf::get_current_device_resource_ref());

  // 3. scatter string offsets to respective columns, set validity bits
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    num_nodes,
    [column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     row_offsets       = row_offsets.begin(),
     range_begin       = tree.node_range_begin.begin(),
     range_end         = tree.node_range_end.begin(),
     d_ignore_vals     = d_ignore_vals.begin(),
     d_columns_data    = d_columns_data.begin()] __device__(size_type i) {
      if (d_ignore_vals[col_ids[i]]) return;
      auto const node_category = column_categories[col_ids[i]];
      switch (node_category) {
        case NC_STRUCT:
          if (d_columns_data[col_ids[i]].validity)
            set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          break;
        case NC_LIST:
          if (d_columns_data[col_ids[i]].validity)
            set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          break;
        case NC_STR: [[fallthrough]];
        case NC_VAL:
          if (d_ignore_vals[col_ids[i]]) break;
          if (d_columns_data[col_ids[i]].validity)
            set_bit(d_columns_data[col_ids[i]].validity, row_offsets[i]);
          d_columns_data[col_ids[i]].string_offsets[row_offsets[i]] = range_begin[i];
          d_columns_data[col_ids[i]].string_lengths[row_offsets[i]] = range_end[i] - range_begin[i];
          break;
        default: break;
      }
    });

  // 4. scatter List offset
  // copy_if only node's whose parent is list, (node_id, parent_col_id)
  // stable_sort by parent_col_id of {node_id}.
  // For all unique parent_node_id of (i==0, i-1!=i), write start offset.
  //                                  (i==last, i+1!=i), write end offset.
  //    unique_copy_by_key {parent_node_id} {row_offset} to
  //    col[parent_col_id].child_offsets[row_offset[parent_node_id]]

  auto& parent_col_ids = sorted_col_ids;  // reuse sorted_col_ids
  auto parent_col_id   = thrust::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<NodeIndexT>(
      [col_ids         = col_ids.begin(),
       parent_node_ids = tree.parent_node_ids.begin()] __device__(size_type node_id) {
        return parent_node_ids[node_id] == parent_node_sentinel ? parent_node_sentinel
                                                                  : col_ids[parent_node_ids[node_id]];
      }));
  auto const list_children_end = cudf::detail::copy_if(
    thrust::make_zip_iterator(cuda::counting_iterator<size_type>{0}, parent_col_id),
    thrust::make_zip_iterator(cuda::counting_iterator<size_type>{0}, parent_col_id) + num_nodes,
    cuda::counting_iterator<size_type>{0},
    thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin()),
    [d_ignore_vals     = d_ignore_vals.begin(),
     parent_node_ids   = tree.parent_node_ids.begin(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin()] __device__(size_type node_id) -> bool {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             column_categories[col_ids[parent_node_id]] == NC_LIST and
             (!d_ignore_vals[col_ids[parent_node_id]]);
    },
    stream);
  // For children of list and in ignore_vals, find it's parent node id, and set corresponding
  // parent's null mask to null. Setting mixed type list rows to null.
  auto const num_list_children = cuda::std::distance(
    thrust::make_zip_iterator(node_ids.begin(), parent_col_ids.begin()), list_children_end);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    num_list_children,
    [node_ids          = node_ids.begin(),
     parent_node_ids   = tree.parent_node_ids.begin(),
     column_categories = d_column_tree.node_categories.begin(),
     col_ids           = col_ids.begin(),
     row_offsets       = row_offsets.begin(),
     d_is_mixed_pruned = d_is_mixed_pruned.begin(),
     d_ignore_vals     = d_ignore_vals.begin(),
     d_columns_data    = d_columns_data.begin()] __device__(size_type i) {
      auto const node_id        = node_ids[i];
      auto const parent_node_id = parent_node_ids[node_id];
      if (parent_node_id == parent_node_sentinel or d_ignore_vals[col_ids[parent_node_id]]) return;
      if (column_categories[col_ids[parent_node_id]] == NC_LIST and
          d_is_mixed_pruned[col_ids[node_id]]) {
        clear_bit(d_columns_data[col_ids[parent_node_id]].validity, row_offsets[parent_node_id]);
      }
    });

  thrust::stable_sort_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    parent_col_ids.begin(),
    parent_col_ids.begin() + num_list_children,
    node_ids.begin());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    num_list_children,
    [node_ids        = node_ids.begin(),
     parent_node_ids = tree.parent_node_ids.begin(),
     parent_col_ids  = parent_col_ids.begin(),
     row_offsets     = row_offsets.begin(),
     d_columns_data  = d_columns_data.begin(),
     num_list_children] __device__(size_type i) {
      auto const node_id        = node_ids[i];
      auto const parent_node_id = parent_node_ids[node_id];
      // scatter to list_offset
      if (i == 0 or parent_node_ids[node_ids[i - 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id]] =
          row_offsets[node_id];
      }
      // last value of list child_offset is its size.
      if (i == num_list_children - 1 or parent_node_ids[node_ids[i + 1]] != parent_node_id) {
        d_columns_data[parent_col_ids[i]].child_offsets[row_offsets[parent_node_id] + 1] =
          row_offsets[node_id] + 1;
      }
    });

  // 5. scan on offsets.
  for (auto& [id, col_ref] : columns) {
    auto& col = col_ref.get();
    if (col.type == json_col_t::StringColumn) {
      thrust::inclusive_scan(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        col.string_offsets.begin(),
        col.string_offsets.end(),
        col.string_offsets.begin(),
        cuda::maximum<json_column::row_offset_t>{});
    } else if (col.type == json_col_t::ListColumn) {
      thrust::inclusive_scan(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        col.child_offsets.begin(),
        col.child_offsets.end(),
        col.child_offsets.begin(),
        cuda::maximum<json_column::row_offset_t>{});
    }
  }
  stream.synchronize();
}

}  // namespace cudf::io::json::detail
