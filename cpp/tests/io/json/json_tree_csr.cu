/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/json/nested_json.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <string>

namespace cuio_json = cudf::io::json;

namespace {

struct h_tree_meta_t {
  std::vector<cuio_json::NodeT> node_categories;
  std::vector<cuio_json::NodeIndexT> parent_node_ids;
  std::vector<cuio_json::SymbolOffsetT> node_range_begin;
  std::vector<cuio_json::SymbolOffsetT> node_range_end;
};

struct h_column_tree {
  // position of nnzs
  std::vector<cuio_json::NodeIndexT> row_idx;
  std::vector<cuio_json::NodeIndexT> col_idx;
  // node properties
  std::vector<cuio_json::NodeT> categories;
  std::vector<cuio_json::NodeIndexT> column_ids;
};

// debug printing
template <typename T>
void print(cudf::host_span<T const> vec, std::string name)
{
  std::cout << name << " = ";
  for (auto e : vec) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
}

bool check_equality(cuio_json::tree_meta_t& d_a,
                    cudf::device_span<cudf::size_type const> d_a_max_row_offsets,
                    cuio_json::experimental::compressed_sparse_row& d_b_csr,
                    cuio_json::experimental::column_tree_properties& d_b_ctp,
                    cuda::stream_ref stream)
{
  // convert from tree_meta_t to column_tree_csr
  stream.sync();

  h_tree_meta_t a{cudf::detail::make_std_vector_async(d_a.node_categories, stream),
                  cudf::detail::make_std_vector_async(d_a.parent_node_ids, stream),
                  cudf::detail::make_std_vector_async(d_a.node_range_begin, stream),
                  cudf::detail::make_std_vector_async(d_a.node_range_end, stream)};

  h_column_tree b{cudf::detail::make_std_vector_async(d_b_csr.row_idx, stream),
                  cudf::detail::make_std_vector_async(d_b_csr.col_idx, stream),
                  cudf::detail::make_std_vector_async(d_b_ctp.categories, stream),
                  cudf::detail::make_std_vector_async(d_b_ctp.mapped_ids, stream)};

  auto a_max_row_offsets = cudf::detail::make_std_vector_async(d_a_max_row_offsets, stream);
  auto b_max_row_offsets = cudf::detail::make_std_vector_async(d_b_ctp.max_row_offsets, stream);

  stream.sync();

  auto num_nodes = a.parent_node_ids.size();
  if (num_nodes > 1) {
    if (b.row_idx.size() != num_nodes + 1) { return false; }

    for (auto pos = b.row_idx[0]; pos < b.row_idx[1]; pos++) {
      auto v = b.col_idx[pos];
      if (a.parent_node_ids[b.column_ids[v]] != b.column_ids[0]) { return false; }
    }
    for (size_t u = 1; u < num_nodes; u++) {
      auto v = b.col_idx[b.row_idx[u]];
      if (a.parent_node_ids[b.column_ids[u]] != b.column_ids[v]) { return false; }

      for (auto pos = b.row_idx[u] + 1; pos < b.row_idx[u + 1]; pos++) {
        v = b.col_idx[pos];
        if (a.parent_node_ids[b.column_ids[v]] != b.column_ids[u]) { return false; }
      }
    }
    for (size_t u = 0; u < num_nodes; u++) {
      if (a.node_categories[b.column_ids[u]] != b.categories[u]) { return false; }
    }
    for (size_t u = 0; u < num_nodes; u++) {
      if (a_max_row_offsets[b.column_ids[u]] != b_max_row_offsets[u]) { return false; }
    }
  } else if (num_nodes == 1) {
    if (b.row_idx.size() != num_nodes + 1) { return false; }

    if (b.row_idx[0] != 0 || b.row_idx[1] != 1) return false;
    if (!b.col_idx.empty()) return false;
    for (size_t u = 0; u < num_nodes; u++) {
      if (a.node_categories[b.column_ids[u]] != b.categories[u]) { return false; }
    }

    for (size_t u = 0; u < num_nodes; u++) {
      if (a_max_row_offsets[b.column_ids[u]] != b_max_row_offsets[u]) { return false; }
    }
  }
  return true;
}

void run_test(std::string const& input, bool enable_lines = true)
{
  auto const stream = cudf::get_default_stream();
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options options{};
  options.enable_lines(enable_lines);
  options.enable_mixed_types_as_string(true);

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());

  // Get the JSON's tree representation
  auto gpu_tree =
    cuio_json::detail::get_tree_representation(tokens_gpu,
                                               token_indices_gpu,
                                               options.is_enabled_mixed_types_as_string(),
                                               stream,
                                               cudf::get_current_device_resource_ref());

  bool const is_array_of_arrays = [&]() {
    std::array<cuio_json::node_t, 2> h_node_categories = {cuio_json::NC_ERR, cuio_json::NC_ERR};
    auto const size_to_copy = std::min(size_t{2}, gpu_tree.node_categories.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_node_categories.data(),
                                  gpu_tree.node_categories.data(),
                                  sizeof(cuio_json::node_t) * size_to_copy,
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
    if (options.is_enabled_lines()) return h_node_categories[0] == cuio_json::NC_LIST;
    return h_node_categories[0] == cuio_json::NC_LIST and
           h_node_categories[1] == cuio_json::NC_LIST;
  }();

  auto tup =
    cuio_json::detail::records_orient_tree_traversal(d_input,
                                                     gpu_tree,
                                                     is_array_of_arrays,
                                                     options.is_enabled_lines(),
                                                     false,
                                                     stream,
                                                     cudf::get_current_device_resource_ref());
  auto& gpu_col_id      = std::get<0>(tup);
  auto& gpu_row_offsets = std::get<1>(tup);

  auto const num_nodes = gpu_col_id.size();
  rmm::device_uvector<cudf::size_type> sorted_col_ids(gpu_col_id.size(), stream);  // make a copy
  thrust::copy(
    rmm::exec_policy_nosync(stream), gpu_col_id.begin(), gpu_col_id.end(), sorted_col_ids.begin());

  // sort by {col_id} on {node_ids} stable
  rmm::device_uvector<cudf::size_type> node_ids(gpu_col_id.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), node_ids.begin(), node_ids.end());
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             sorted_col_ids.begin(),
                             sorted_col_ids.end(),
                             node_ids.begin());

  cudf::size_type const row_array_parent_col_id = [&]() {
    cudf::size_type value      = cuio_json::parent_node_sentinel;
    auto const list_node_index = options.is_enabled_lines() ? 0 : 1;
    if (std::cmp_greater_equal(list_node_index, gpu_col_id.size())) { return value; }
    CUDF_CUDA_TRY(cudaMemcpyAsync(&value,
                                  gpu_col_id.data() + list_node_index,
                                  sizeof(cudf::size_type),
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
    return value;
  }();

  auto [d_column_tree, d_unique_col_ids, d_max_row_offsets] =
    cudf::io::json::detail::reduce_to_column_tree(gpu_tree,
                                                  gpu_col_id,
                                                  sorted_col_ids,
                                                  node_ids,
                                                  gpu_row_offsets,
                                                  is_array_of_arrays,
                                                  row_array_parent_col_id,
                                                  stream);

  auto [d_column_tree_csr, d_column_tree_properties] =
    cudf::io::json::experimental::detail::reduce_to_column_tree(gpu_tree,
                                                                gpu_col_id,
                                                                sorted_col_ids,
                                                                node_ids,
                                                                gpu_row_offsets,
                                                                is_array_of_arrays,
                                                                row_array_parent_col_id,
                                                                stream);

  auto iseq = check_equality(
    d_column_tree, d_max_row_offsets, d_column_tree_csr, d_column_tree_properties, stream);
  // assert equality between csr and meta formats
  ASSERT_TRUE(iseq);
}
}  // namespace

struct JsonColumnTreeTests : public cudf::test::BaseFixture {};

TEST_F(JsonColumnTreeTests, JSONL_Small)
{
  std::string const input =
    R"(  {}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }} )";  // Prepare input & output buffers
  run_test(input);
}

TEST_F(JsonColumnTreeTests, JSONL_Large)
{
  std::string const input =
    R"(  {}
    {}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }} )";
  run_test(input);
}

TEST_F(JsonColumnTreeTests, JSONL_ListofStruct)
{
  std::string const input = R"(
  { "Root": { "Key": [ { "EE": "A" } ] } }
  { "Root": { "Key": {  } } }
  { "Root": { "Key": [{ "YY": 1}] } }
  )";
  run_test(input);
}

TEST_F(JsonColumnTreeTests, JSONL_MissingEntries)
{
  std::string json_stringl = R"(
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true}
    {"a": 1, "b": {"0": "abc"          }, "c": false}
    {"a": 1, "b": {}}
    {"a": 1,                              "c": null}
    )";
  run_test(json_stringl);
}

TEST_F(JsonColumnTreeTests, JSONL_MoreMissingEntries)
{
  std::string json_stringl = R"(
    { "foo1": [1,2,3], "bar": 123 }
    { "foo2": { "a": 1 }, "bar": 456 }
    { "foo1": [1,2,3], "bar": 123 }
    { "foo2": { "a": 1 }, "bar": 456 }
    { "foo1": [1,2,3], "bar": 123 }
    { "foo2": { "a": 1 }, "bar": 456 }
    )";
  run_test(json_stringl);
}

TEST_F(JsonColumnTreeTests, JSONL_StillMoreMissingEntries)
{
  std::string json_stringl = R"(
    { "foo1": [1,2,3], "bar": 123 }
    { "foo2": { "a": 1 }, "bar": 456 }
    { "foo1": ["123","456"], "bar": 123 }
    { "foo2": { "b": 5 }, "car": 456 }
    { "foo1": [1,2,3], "bar": 123 }
    { "foo2": { "a": 1 }, "bar": 456 }
    )";
  run_test(json_stringl);
}

TEST_F(JsonColumnTreeTests, JSON_MissingEntries)
{
  std::string json_string = R"([
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true},
    {"a": 1, "b": {"0": "abc"          }, "c": false},
    {"a": 1, "b": {}},
    {"a": 1,                              "c": null}
    ])";
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSON_StructOfStructs)
{
  std::string json_string =
    R"([
    {},
    { "a": { "y" : 6, "z": [] }},
    { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }}
    ])";  // Prepare input & output buffers
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSONL_ArrayOfArrays_NestedList)
{
  std::string json_string =
    R"([123, [1,2,3]]
       [456, null,  { "a": 1 }])";
  run_test(json_string);
}

TEST_F(JsonColumnTreeTests, JSON_ArrayofArrays_NestedList)
{
  std::string json_string = R"([[[1,2,3], null, 123],
              [null, { "a": 1 }, 456 ]])";
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSON_CornerCase_Empty)
{
  std::string json_string = R"([])";
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSONL_CornerCase_List)
{
  std::string json_string = R"([123])";
  run_test(json_string, true);
}

TEST_F(JsonColumnTreeTests, JSON_CornerCase_EmptyNestedList)
{
  std::string json_string = R"([[[]]])";
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSON_CornerCase_EmptyNestedLists)
{
  std::string json_string = R"([[], [], []])";
  run_test(json_string, false);
}

TEST_F(JsonColumnTreeTests, JSONL_CornerCase_ListofLists)
{
  std::string json_string = R"([[1, 2, 3], [4, 5, null], []])";
  run_test(json_string, true);
}

TEST_F(JsonColumnTreeTests, JSONL_CornerCase_EmptyListOfLists)
{
  std::string json_string = R"([[]])";
  run_test(json_string, true);
}
