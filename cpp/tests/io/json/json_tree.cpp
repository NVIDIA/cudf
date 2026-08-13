/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/json/nested_json.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>

namespace cuio_json = cudf::io::json;

namespace {

// Host copy of tree_meta_t
struct tree_meta_t2 {
  std::vector<cuio_json::NodeT> node_categories;
  std::vector<cuio_json::NodeIndexT> parent_node_ids;
  std::vector<cuio_json::TreeDepthT> node_levels;
  std::vector<cuio_json::SymbolOffsetT> node_range_begin;
  std::vector<cuio_json::SymbolOffsetT> node_range_end;
};

tree_meta_t2 to_cpu_tree(cuio_json::tree_meta_t const& d_value, rmm::cuda_stream_view stream)
{
  tree_meta_t2 result{cudf::detail::make_std_vector_async(d_value.node_categories, stream),
                      cudf::detail::make_std_vector_async(d_value.parent_node_ids, stream),
                      cudf::detail::make_std_vector_async(d_value.node_levels, stream),
                      cudf::detail::make_std_vector_async(d_value.node_range_begin, stream),
                      cudf::detail::make_std_vector_async(d_value.node_range_end, stream)};
  stream.synchronize();
  return result;
}

// change this to non-zero and recompile to dump debug info to stdout
#define LIBCUDF_JSON_DEBUG_DUMP 0
#if LIBCUDF_JSON_DEBUG_DUMP
std::string get_node_string(std::size_t const node_id,
                            tree_meta_t2 const& tree_rep,
                            std::string const& json_input)
{
  auto node_to_str = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::NC_STRUCT: return "STRUCT";
      case cuio_json::NC_LIST: return "LIST";
      case cuio_json::NC_FN: return "FN";
      case cuio_json::NC_STR: return "STR";
      case cuio_json::NC_VAL: return "VAL";
      case cuio_json::NC_ERR: return "ERR";
      default: return "N/A";
    };
  };

  return "<" + std::to_string(node_id) + ":" + node_to_str(tree_rep.node_categories[node_id]) +
         ":[" + std::to_string(tree_rep.node_range_begin[node_id]) + ", " +
         std::to_string(tree_rep.node_range_end[node_id]) + ") '" +
         json_input.substr(tree_rep.node_range_begin[node_id],
                           tree_rep.node_range_end[node_id] - tree_rep.node_range_begin[node_id]) +
         "'>";
}

void print_tree_representation(std::string const& json_input, tree_meta_t2 const& tree_rep)
{
  for (std::size_t i = 0; i < tree_rep.node_categories.size(); i++) {
    auto parent_id = tree_rep.parent_node_ids[i];
    std::stack<std::size_t> path;
    path.push(i);
    while (parent_id != cuio_json::parent_node_sentinel) {
      path.push(parent_id);
      parent_id = tree_rep.parent_node_ids[parent_id];
    }

    while (path.size()) {
      auto const node_id = path.top();
      std::cout << get_node_string(node_id, tree_rep, json_input)
                << (path.size() > 1 ? " -> " : "");
      path.pop();
    }
    std::cout << "\n";
  }
}

// DEBUG prints
auto to_cat = [](auto v) -> std::string {
  switch (v) {
    case cuio_json::NC_STRUCT: return " S";
    case cuio_json::NC_LIST: return " L";
    case cuio_json::NC_STR: return " \"";
    case cuio_json::NC_VAL: return " V";
    case cuio_json::NC_FN: return " F";
    case cuio_json::NC_ERR: return "ER";
    default: return "UN";
  };
};
auto to_int    = [](auto v) { return std::to_string(static_cast<int>(v)); };
auto print_vec = [](auto const& cpu, auto const name, auto converter) {
  for (auto const& v : cpu)
    printf("%3s,", converter(v).c_str());
  std::cout << name << std::endl;
};
void print_tree(tree_meta_t2 const& cpu_tree)
{
  print_vec(cpu_tree.node_categories, "node_categories", to_cat);
  print_vec(cpu_tree.parent_node_ids, "parent_node_ids", to_int);
  print_vec(cpu_tree.node_levels, "node_levels", to_int);
  print_vec(cpu_tree.node_range_begin, "node_range_begin", to_int);
  print_vec(cpu_tree.node_range_end, "node_range_end", to_int);
}
void print_tree(cuio_json::tree_meta_t const& d_gpu_tree)
{
  auto const cpu_tree = to_cpu_tree(d_gpu_tree, cudf::get_default_stream());
  print_tree(cpu_tree);
}
#endif

template <typename T>
bool compare_vector(std::vector<T> const& cpu_vec,
                    std::vector<T> const& gpu_vec,
                    std::string const& name)
{
  EXPECT_EQ(cpu_vec.size(), gpu_vec.size());
  bool mismatch = false;
  if (!std::equal(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin())) {
    for (auto i = 0lu; i < cpu_vec.size(); i++) {
      mismatch |= (cpu_vec[i] != gpu_vec[i]);
    }
  }
  EXPECT_FALSE(mismatch);
  return mismatch;
}

template <typename T>
bool compare_vector(std::vector<T> const& cpu_vec,
                    rmm::device_uvector<T> const& d_vec,
                    std::string const& name)
{
  auto stream  = cudf::get_default_stream();
  auto gpu_vec = cudf::detail::make_std_vector(d_vec, stream);
  return compare_vector(cpu_vec, gpu_vec, name);
}

void compare_trees(tree_meta_t2 const& cpu_tree, cuio_json::tree_meta_t const& d_gpu_tree)
{
  auto cpu_num_nodes = cpu_tree.node_categories.size();
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_categories.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.parent_node_ids.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_levels.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_range_begin.size());
  EXPECT_EQ(cpu_num_nodes, d_gpu_tree.node_range_end.size());
  auto gpu_tree = to_cpu_tree(d_gpu_tree, cudf::get_default_stream());

#define COMPARE_MEMBER(member)                                                       \
  for (std::size_t i = 0; i < cpu_num_nodes; i++) {                                  \
    EXPECT_EQ(cpu_tree.member[i], gpu_tree.member[i]) << #member << "[" << i << "]"; \
  }
  COMPARE_MEMBER(node_categories);
  COMPARE_MEMBER(parent_node_ids);
  COMPARE_MEMBER(node_levels);
  COMPARE_MEMBER(node_range_begin);
  COMPARE_MEMBER(node_range_end);
#undef COMPARE_MEMBER

#if LIBCUDF_JSON_DEBUG_DUMP
  bool mismatch = false;
#define PRINT_VEC(vec, conv) print_vec(vec, #vec, conv);
#define PRINT_COMPARISON(vec, conv)                                                  \
  PRINT_VEC(cpu_tree.vec, conv);                                                     \
  PRINT_VEC(gpu_tree.vec, conv);                                                     \
  if (!std::equal(cpu_tree.vec.begin(), cpu_tree.vec.end(), gpu_tree.vec.begin())) { \
    for (auto i = 0lu; i < cpu_tree.vec.size(); i++) {                               \
      mismatch |= (gpu_tree.vec[i] != cpu_tree.vec[i]);                              \
      printf("%3s,", (gpu_tree.vec[i] == cpu_tree.vec[i] ? " " : "x"));              \
    }                                                                                \
    printf("\n");                                                                    \
  }
  for (int i = 0; i < int(cpu_num_nodes); i++)
    printf("%3d,", i);
  printf(" node_id\n");
  PRINT_COMPARISON(node_categories, to_cat);   // Works
  PRINT_COMPARISON(node_levels, to_int);       // Works
  PRINT_COMPARISON(node_range_begin, to_int);  // Works
  PRINT_COMPARISON(node_range_end, to_int);    // Works
  PRINT_COMPARISON(parent_node_ids, to_int);   // Works
  EXPECT_FALSE(mismatch);
#undef PRINT_VEC
#undef PRINT_COMPARISON
#endif
}

template <typename T>
auto translate_col_id(T const& col_id)
{
  using value_type = typename T::value_type;
  std::unordered_map<value_type, value_type> col_id_map;
  std::vector<value_type> new_col_ids(col_id.size());
  value_type unique_id = 0;
  for (auto id : col_id) {
    if (col_id_map.count(id) == 0) { col_id_map[id] = unique_id++; }
  }
  for (size_t i = 0; i < col_id.size(); i++) {
    new_col_ids[i] = col_id_map[col_id[i]];
  }
  return new_col_ids;
}

tree_meta_t2 get_tree_representation_cpu(
  cudf::device_span<cuio_json::PdaTokenT const> tokens_gpu,
  cudf::device_span<cuio_json::SymbolOffsetT const> token_indices_gpu1,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  constexpr bool include_quote_char = true;
  // Copy the JSON tokens to the host
  auto tokens        = cudf::detail::make_host_vector_async(tokens_gpu, stream);
  auto token_indices = cudf::detail::make_host_vector_async(token_indices_gpu1, stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

#if LIBCUDF_JSON_DEBUG_DUMP
  // DEBUG print
  [[maybe_unused]] auto to_token_str = [](cuio_json::PdaTokenT token) {
    switch (token) {
      case cuio_json::token_t::StructBegin: return " {";
      case cuio_json::token_t::StructEnd: return " }";
      case cuio_json::token_t::ListBegin: return " [";
      case cuio_json::token_t::ListEnd: return " ]";
      case cuio_json::token_t::FieldNameBegin: return "FB";
      case cuio_json::token_t::FieldNameEnd: return "FE";
      case cuio_json::token_t::StringBegin: return "SB";
      case cuio_json::token_t::StringEnd: return "SE";
      case cuio_json::token_t::ErrorBegin: return "er";
      case cuio_json::token_t::ValueBegin: return "VB";
      case cuio_json::token_t::ValueEnd: return "VE";
      case cuio_json::token_t::StructMemberBegin: return " <";
      case cuio_json::token_t::StructMemberEnd: return " >";
      case cuio_json::token_t::LineEnd: return ";";
      default: return ".";
    }
  };
  std::cout << "Tokens: \n";
  for (auto i = 0u; i < tokens.size(); i++) {
    std::cout << to_token_str(tokens[i]) << " ";
  }
  std::cout << std::endl;
#endif

  // Whether a token does represent a node in the tree representation
  auto is_node = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StructBegin:
      case cuio_json::token_t::ListBegin:
      case cuio_json::token_t::StringBegin:
      case cuio_json::token_t::ValueBegin:
      case cuio_json::token_t::FieldNameBegin:
      case cuio_json::token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // The node that a token represents
  auto token_to_node = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StructBegin: return cuio_json::NC_STRUCT;
      case cuio_json::token_t::ListBegin: return cuio_json::NC_LIST;
      case cuio_json::token_t::StringBegin: return cuio_json::NC_STR;
      case cuio_json::token_t::ValueBegin: return cuio_json::NC_STR;  // NC_VAL;
      case cuio_json::token_t::FieldNameBegin: return cuio_json::NC_FN;
      default: return cuio_json::NC_ERR;
    };
  };

  // Includes quote char for end-of-string token or Skips the quote char for beginning-of-field-name
  auto get_token_index = [](cuio_json::PdaTokenT const token,
                            cuio_json::SymbolOffsetT const token_index) {
    constexpr cuio_json::SymbolOffsetT quote_char_size = 1;
    switch (token) {
      // Strip off or include quote char for StringBegin
      case cuio_json::token_t::StringBegin:
        return token_index + (include_quote_char ? 0 : quote_char_size);
      // Strip off or Include trailing quote char for string values for StringEnd
      case cuio_json::token_t::StringEnd:
        return token_index + (include_quote_char ? quote_char_size : 0);
      // Strip off quote char included for FieldNameBegin
      case cuio_json::token_t::FieldNameBegin: return token_index + quote_char_size;
      default: return token_index;
    };
  };

  // Whether a token expects to be followed by its respective end-of-* token partner
  auto is_begin_of_section = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StringBegin:
      case cuio_json::token_t::ValueBegin:
      case cuio_json::token_t::FieldNameBegin: return true;
      default: return false;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StringBegin: return cuio_json::token_t::StringEnd;
      case cuio_json::token_t::ValueBegin: return cuio_json::token_t::ValueEnd;
      case cuio_json::token_t::FieldNameBegin: return cuio_json::token_t::FieldNameEnd;
      default: return cuio_json::token_t::ErrorBegin;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StructEnd:
      case cuio_json::token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [](cuio_json::PdaTokenT const token) {
    switch (token) {
      case cuio_json::token_t::StructBegin:
      case cuio_json::token_t::ListBegin: return true;
      default: return false;
    };
  };

  // The node id sitting on top of the stack becomes the node's parent
  // The full stack represents the path from the root to the current node
  std::stack<std::pair<cuio_json::NodeIndexT, bool>> parent_stack;

  constexpr bool field_name_node    = true;
  constexpr bool no_field_name_node = false;

  std::vector<cuio_json::NodeT> node_categories;
  std::vector<cuio_json::NodeIndexT> parent_node_ids;
  std::vector<cuio_json::TreeDepthT> node_levels;
  std::vector<cuio_json::SymbolOffsetT> node_range_begin;
  std::vector<cuio_json::SymbolOffsetT> node_range_end;

  std::size_t node_id = 0;
  for (std::size_t i = 0; i < tokens.size(); i++) {
    auto token = tokens[i];

    // The section from the original JSON input that this token demarcates
    std::size_t range_begin = get_token_index(token, token_indices[i]);
    std::size_t range_end   = range_begin + 1;

    // Identify this node's parent node id
    std::size_t parent_node_id =
      (parent_stack.size() > 0) ? parent_stack.top().first : cuio_json::parent_node_sentinel;

    // If this token is the beginning-of-{value, string, field name}, also consume the next end-of-*
    // token
    if (is_begin_of_section(token)) {
      if ((i + 1) < tokens.size() && end_of_partner(token) == tokens[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = get_token_index(tokens[i + 1], token_indices[i + 1]);
        // We can skip the subsequent end-of-* token
        i++;
      }
    }

    // Emit node if this token becomes a node in the tree
    if (is_node(token)) {
      node_categories.push_back(token_to_node(token));
      parent_node_ids.push_back(parent_node_id);
      node_levels.push_back(parent_stack.size());
      node_range_begin.push_back(range_begin);
      node_range_end.push_back(range_end);
    }

    // Modify the stack if needed
    if (token == cuio_json::token_t::FieldNameBegin) {
      parent_stack.emplace(node_id, field_name_node);
    } else {
      if (does_push(token)) {
        parent_stack.emplace(node_id, no_field_name_node);
      } else if (does_pop(token)) {
        CUDF_EXPECTS(parent_stack.size() >= 1, "Invalid JSON input.");
        parent_stack.pop();
      }

      // If what we're left with is a field name on top of stack, we need to pop it
      if (parent_stack.size() >= 1 && parent_stack.top().second == field_name_node) {
        parent_stack.pop();
      }
    }

    // Update node_id
    if (is_node(token)) { node_id++; }
  }

  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

std::tuple<std::vector<cuio_json::NodeIndexT>, std::vector<cudf::size_type>>
records_orient_tree_traversal_cpu(cudf::host_span<cuio_json::SymbolT const> input,
                                  tree_meta_t2 const& tree,
                                  bool is_array_of_arrays,
                                  bool is_enabled_lines,
                                  rmm::cuda_stream_view stream)
{
  std::vector<cuio_json::NodeIndexT> node_ids(tree.parent_node_ids.size());
  std::iota(node_ids.begin(), node_ids.end(), 0);

  cuio_json::NodeIndexT const row_array_children_level = is_enabled_lines ? 1 : 2;
  std::unordered_map<cuio_json::NodeIndexT, cuio_json::NodeIndexT> list_indices;
  if (is_array_of_arrays) {
    cuio_json::NodeIndexT parent_node = -1, child_index = 0;
    for (size_t i = 0; i < tree.node_levels.size(); i++) {
      if (tree.node_levels[i] == row_array_children_level) {
        if (tree.parent_node_ids[i] != parent_node) {
          parent_node = tree.parent_node_ids[i];
          child_index = 0;
        }
        list_indices[i] = child_index++;
      }
    }
  }

#if LIBCUDF_JSON_DEBUG_DUMP
  for (int i = 0; i < int(tree.node_range_begin.size()); i++) {
    printf("%3s ",
           std::string(input.data() + tree.node_range_begin[i],
                       tree.node_range_end[i] - tree.node_range_begin[i])
             .c_str());
  }
  printf(" (JSON)\n");
  print_vec(tree.node_categories, "node_categories", to_cat);
  print_vec(node_ids, "cpu.node_ids", to_int);
#endif

  // print_vec(tree.parent_node_ids, "tree.parent_node_ids (before)");
  constexpr cuio_json::NodeIndexT top_node = -1;
  // CPU version of the algorithm
  // Calculate row offsets too.
  auto hash_path = [&](auto node_id) {
    size_t seed = 0;
    while (node_id != top_node) {
      seed = cudf::hashing::detail::hash_combine(
        seed, std::hash<cuio_json::TreeDepthT>{}(tree.node_levels[node_id]));
      seed = cudf::hashing::detail::hash_combine(
        seed, std::hash<cuio_json::NodeT>{}(tree.node_categories[node_id]));
      if (tree.node_categories[node_id] == cuio_json::node_t::NC_FN) {
        auto field_name =
          std::string_view(input.data() + tree.node_range_begin[node_id],
                           tree.node_range_end[node_id] - tree.node_range_begin[node_id]);
        seed = cudf::hashing::detail::hash_combine(seed, std::hash<std::string_view>{}(field_name));
      }
      if (is_array_of_arrays and tree.node_levels[node_id] == row_array_children_level)
        seed = cudf::hashing::detail::hash_combine(seed, list_indices[node_id]);
      node_id = tree.parent_node_ids[node_id];
    }
    return seed;
  };
  auto equal_path = [&](auto node_id1, auto node_id2) {
    bool is_equal = true;
    while (is_equal and node_id1 != top_node and node_id2 != top_node) {
      is_equal &= tree.node_levels[node_id1] == tree.node_levels[node_id2];
      is_equal &= tree.node_categories[node_id1] == tree.node_categories[node_id2];
      if (is_equal and tree.node_categories[node_id1] == cuio_json::node_t::NC_FN) {
        auto field_name1 =
          std::string_view(input.data() + tree.node_range_begin[node_id1],
                           tree.node_range_end[node_id1] - tree.node_range_begin[node_id1]);
        auto field_name2 =
          std::string_view(input.data() + tree.node_range_begin[node_id2],
                           tree.node_range_end[node_id2] - tree.node_range_begin[node_id2]);
        is_equal &= field_name1 == field_name2;
      }
      if (is_array_of_arrays and is_equal and
          tree.node_levels[node_id1] == row_array_children_level) {
        is_equal &= list_indices[node_id1] == list_indices[node_id2];
      }
      node_id1 = tree.parent_node_ids[node_id1];
      node_id2 = tree.parent_node_ids[node_id2];
    }
    return is_equal and node_id1 == top_node and node_id2 == top_node;
  };
  std::unordered_map<cuio_json::NodeIndexT, int, decltype(hash_path), decltype(equal_path)>
    node_id_map(10, hash_path, equal_path);
  auto unique_col_id = 0;
  for (auto& node_idx : node_ids) {
    if (node_id_map.count(node_idx) == 0) {
      node_id_map[node_idx] = unique_col_id++;  // node_idx;
      node_idx              = node_id_map[node_idx];
    } else {
      node_idx = node_id_map[node_idx];
    }
  }
  // Translate parent_node_ids
  auto parent_col_ids(tree.parent_node_ids);
  for (auto& parent_node_id : parent_col_ids) {
    if (parent_node_id != top_node) parent_node_id = node_ids[parent_node_id];
  }

#if LIBCUDF_JSON_DEBUG_DUMP
  print_vec(node_ids, "cpu.node_ids (after)", to_int);
  print_vec(tree.parent_node_ids, "cpu.parent_node_ids (after)", to_int);
#endif

  // row_offsets
  std::vector<int> row_offsets(tree.parent_node_ids.size(), 0);
  std::unordered_map<int, int> col_id_current_offset;
  for (std::size_t i = 0; i < tree.parent_node_ids.size(); i++) {
    auto current_col_id = node_ids[i];
    auto parent_col_id  = parent_col_ids[i];
    auto parent_node_id = tree.parent_node_ids[i];
    if (parent_col_id == top_node) {
      // row_offsets[current_col_id] = 0; // JSON lines treats top node as list.
      col_id_current_offset[current_col_id]++;
      row_offsets[i] = col_id_current_offset[current_col_id] - 1;
    } else {
      if (tree.node_categories[parent_node_id] == cuio_json::node_t::NC_LIST and
          !(is_array_of_arrays and tree.node_levels[i] == row_array_children_level)) {
        col_id_current_offset[current_col_id]++;
        row_offsets[i] = col_id_current_offset[current_col_id] - 1;
      } else {
        row_offsets[i]                        = col_id_current_offset[parent_col_id] - 1;
        col_id_current_offset[current_col_id] = col_id_current_offset[parent_col_id];
      }
    }
  }

#if LIBCUDF_JSON_DEBUG_DUMP
  print_vec(row_offsets, "cpu.row_offsets (generated)", to_int);
#endif

  return {std::move(node_ids), std::move(row_offsets)};
}

}  // namespace

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {};

TEST_F(JsonTest, TreeRepresentation)
{
  auto const stream = cudf::get_default_stream();

  // Test input
  std::string const input = R"(  [{)"
                            R"("category": "reference",)"
                            R"("index:": [4,12,42],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "[Sayings of the Century]",)"
                            R"("price": 8.95)"
                            R"(},  )"
                            R"({)"
                            R"("category": "reference",)"
                            R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "{}[], <=semantic-symbols-string",)"
                            R"("price": 8.95)"
                            R"(}] )";
  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options const options{};

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    tokens_gpu, token_indices_gpu, false, stream, cudf::get_current_device_resource_ref());
  // host tree generation
  auto cpu_tree = get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  compare_trees(cpu_tree, gpu_tree);

#if LIBCUDF_JSON_DEBUG_DUMP
  print_tree_representation(input, cpu_tree);
#endif

  // Golden sample of node categories
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT, cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_STR,    cuio_json::NC_STR,
    cuio_json::NC_STR,  cuio_json::NC_FN,     cuio_json::NC_STR,    cuio_json::NC_FN,
    cuio_json::NC_STR,  cuio_json::NC_FN,     cuio_json::NC_STR,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_LIST,
    cuio_json::NC_STR,  cuio_json::NC_STRUCT, cuio_json::NC_STR,    cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_LIST,   cuio_json::NC_STRUCT, cuio_json::NC_STRUCT,
    cuio_json::NC_FN,   cuio_json::NC_STR,    cuio_json::NC_FN,     cuio_json::NC_STR,
    cuio_json::NC_FN,   cuio_json::NC_STR};

  // Golden sample of node ids
  // clang-format off
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0, 1, 2,
    1, 4, 5, 5,
    5, 1, 9, 1,
    11, 1, 13, 0,
    15, 16, 15, 18,
    19, 19, 19, 19,
    23, 24, 25, 25,
    15, 28, 15, 30,
    15, 32};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {0, 1, 2, 3, 2, 3, 4, 4, 4, 2, 3, 2,
                                                           3, 2, 3, 1, 2, 3, 2, 3, 4, 4, 4, 4,
                                                           5, 6, 7, 7, 2, 3, 2, 3, 2, 3};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {
    2,   3,   5,   16,  29,  38,  39,  41,  44,  49,  58,  72,  80,  108, 116, 124, 126,
    137, 150, 158, 159, 161, 164, 169, 171, 174, 175, 180, 189, 198, 212, 220, 255, 263};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {
    3,   4,   13,  27,  35,  39,  40,  43,  46,  55,  70,  77,  106, 113, 120, 125, 134,
    148, 155, 159, 160, 162, 168, 170, 172, 175, 176, 181, 195, 210, 217, 253, 260, 267};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), cpu_tree.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), cpu_tree.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), cpu_tree.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), cpu_tree.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), cpu_tree.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], cpu_tree.node_categories[i]) << "[" << i << "]";
    ASSERT_EQ(golden_parent_node_ids[i], cpu_tree.parent_node_ids[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_levels[i], cpu_tree.node_levels[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_range_begin[i], cpu_tree.node_range_begin[i]) << "[" << i << "]";
    ASSERT_EQ(golden_node_range_end[i], cpu_tree.node_range_end[i]) << "[" << i << "]";
  }
}

TEST_F(JsonTest, TreeRepresentation2)
{
  auto const stream = cudf::get_default_stream();
  // Test input: value end with comma, space, close-brace ", }"
  std::string const input =
    // 0         1         2         3         4         5         6         7         8         9
    // 0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
    R"([ {}, { "a": { "y" : 6, "z": [] }}, { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11)"
    "\n}}]";
  // Prepare input & output buffers
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options const options{};

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    tokens_gpu, token_indices_gpu, false, stream, cudf::get_current_device_resource_ref());
  // host tree generation
  auto cpu_tree = get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  compare_trees(cpu_tree, gpu_tree);

#if LIBCUDF_JSON_DEBUG_DUMP
  print_tree_representation(input, cpu_tree);
#endif

  // Golden sample of node categories
  // clang-format off
  std::vector<cuio_json::node_t> golden_node_categories = {
    cuio_json::NC_LIST, cuio_json::NC_STRUCT,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_STR, cuio_json::NC_FN,  cuio_json::NC_LIST,
    cuio_json::NC_STRUCT, cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_STR, cuio_json::NC_FN,  cuio_json::NC_STR,
                          cuio_json::NC_FN, cuio_json::NC_STRUCT,  cuio_json::NC_FN,  cuio_json::NC_STR, cuio_json::NC_FN,  cuio_json::NC_STR};

  // Golden sample of node ids
  std::vector<cuio_json::NodeIndexT> golden_parent_node_ids = {
    cuio_json::parent_node_sentinel, 0,
    0, 2,  3,  4,  5,  4, 7,
    0, 9, 10, 11, 12, 11, 14,
       9, 16, 17, 18, 17, 20};
  // clang-format on

  // Golden sample of node levels
  std::vector<cuio_json::TreeDepthT> golden_node_levels = {
    0, 1, 1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 4, 5, 4, 5, 2, 3, 4, 5, 4, 5,
  };

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_begin = {0,  2,  6,  9,  13, 16, 21, 25, 29, 36, 39,
                                                      44, 47, 52, 56, 60, 66, 71, 73, 77, 83, 87};

  // Golden sample of the character-ranges from the original input that each node demarcates
  std::vector<std::size_t> golden_node_range_end = {1,  3,  7,  10, 14, 17, 22, 26, 30, 37, 40,
                                                    45, 48, 53, 57, 61, 67, 72, 74, 79, 84, 89};

  // Check results against golden samples
  ASSERT_EQ(golden_node_categories.size(), cpu_tree.node_categories.size());
  ASSERT_EQ(golden_parent_node_ids.size(), cpu_tree.parent_node_ids.size());
  ASSERT_EQ(golden_node_levels.size(), cpu_tree.node_levels.size());
  ASSERT_EQ(golden_node_range_begin.size(), cpu_tree.node_range_begin.size());
  ASSERT_EQ(golden_node_range_end.size(), cpu_tree.node_range_end.size());

  for (std::size_t i = 0; i < golden_node_categories.size(); i++) {
    ASSERT_EQ(golden_node_categories[i], cpu_tree.node_categories[i]);
    ASSERT_EQ(golden_parent_node_ids[i], cpu_tree.parent_node_ids[i]);
    ASSERT_EQ(golden_node_levels[i], cpu_tree.node_levels[i]);
    ASSERT_EQ(golden_node_range_begin[i], cpu_tree.node_range_begin[i]);
    ASSERT_EQ(golden_node_range_end[i], cpu_tree.node_range_end[i]);
  }
}

TEST_F(JsonTest, TreeRepresentation3)
{
  auto const stream = cudf::get_default_stream();
  // Test input: Json lines with same TreeRepresentation2 input
  std::string const input =
    R"(  {}
 { "a": { "y" : 6, "z": [] }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }} )";  // Prepare input & output buffers
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  cudf::io::json_reader_options options{};
  options.enable_lines(true);

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());

  // Get the JSON's tree representation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    tokens_gpu, token_indices_gpu, false, stream, cudf::get_current_device_resource_ref());
  // host tree generation
  auto cpu_tree = get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  compare_trees(cpu_tree, gpu_tree);

#if LIBCUDF_JSON_DEBUG_DUMP
  print_tree_representation(input, cpu_tree);
#endif
}

TEST_F(JsonTest, TreeRepresentationError)
{
  auto const stream       = cudf::get_default_stream();
  std::string const input = R"([ {}, }{])";
  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};
  cudf::io::json_reader_options const options{};

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());

  // Get the JSON's tree representation
  // This JSON is invalid and will raise an exception.
  EXPECT_THROW(
    cuio_json::detail::get_tree_representation(
      tokens_gpu, token_indices_gpu, false, stream, cudf::get_current_device_resource_ref()),
    cudf::logic_error);
}

/**
 * @brief Test fixture for parametrized JSON tree traversal tests
 */
struct JsonTreeTraversalTest : public cudf::test::BaseFixture,
                               public testing::WithParamInterface<std::tuple<bool, std::string>> {};

//
std::vector<std::string> json_list = {
  "[]",
  "value",
  "\"string\"",
  "[1, 2, 3]",
  R"({"a": 1, "b": 2, "c": 3})",
  // input a: {x:i, y:i, z:[]}, b: {x:i, z:i}
  R"([ {}, { "a": { "y" : 6, "z": [] }}, { "a" : { "x" : 8, "y": 9}, "b" : {"x": 10, "z": 11}}])",
  // input a: {x:i, y:i, z:[]}, b: {x:i, z: {p: i, q: i}}
  R"([ {}, { "a": { "y" : 1, "z": [] }},
             { "a": { "x" : 2, "y": 3}, "b" : {"x": 4, "z": [ {"p": 1, "q": 2}]}},
             { "a": { "y" : 6, "z": [7, 8, 9]}, "b": {"x": 10, "z": [{}, {"q": 3}, {"p": 4}]}},
             { "a": { "z": [12, 13, 14, 15]}},
             { "a": { "z": [16], "x": 2}}
        ])",
  //^row offset a a.x a.y a.z   b b.x b.z
  //            1       1   1
  //            2   2   2       2   2   2                     b.z[] 0        b.z.p 0, b.z.q 0
  //            3       3   3   3   3   3   a.z[] 0, 1, 2     b.z[] 1, 2, 3  b.z.q 2, b.z.p 3
  //            4           4               a.z[] 3, 4, 5, 6
  //            5   5       5               a.z[] 7
  R"([[1, 2, 3], [4, 5, 6], [7, 8, 9]])",
  R"([[1, 2, 3], [4, 5], [7]])",
  R"([[1], [4, 5, 6], [7, 8]])",
};

std::vector<std::string> json_lines_list = {
  // Test input a: {x:i, y:i, z:[]}, b: {x:i, z:i} with JSON-lines
  "",
  R"(  {}
 { "a": { "y" : 6, "z": [] }}
 { "a": { "y" : 6, "z": [2, 3, 4, 5] }}
 { "a": { "z": [4], "y" : 6 }}
 { "a" : { "x" : 8, "y": 9 }, "b" : {"x": 10 , "z": 11 }} )",
  // empty list, row.
  R"( {"a" : [], "b" : {}}
 {"a" : []}
 {"b" : {}})",
  R"([1, 2, 3]
     [4, 5, 6])",
  R"([1]
     [4, [5], 6]
     [7, [8]])"};
INSTANTIATE_TEST_SUITE_P(Mixed_And_Records,
                         JsonTreeTraversalTest,
                         ::testing::Combine(::testing::Values(false),
                                            ::testing::ValuesIn(json_list)));
INSTANTIATE_TEST_SUITE_P(JsonLines,
                         JsonTreeTraversalTest,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::ValuesIn(json_lines_list)));

TEST_P(JsonTreeTraversalTest, CPUvsGPUTraversal)
{
  auto [json_lines, input] = GetParam();
  auto stream              = cudf::get_default_stream();
  cudf::io::json_reader_options options{};
  options.enable_lines(json_lines);

  // std::cout << json_lines << input << std::endl;
  cudf::string_scalar d_scalar(input, true, stream);
  auto d_input = cudf::device_span<cuio_json::SymbolT const>{d_scalar.data(),
                                                             static_cast<size_t>(d_scalar.size())};

  // Parse the JSON and get the token stream
  auto const [tokens_gpu, token_indices_gpu] = cudf::io::json::detail::get_token_stream(
    d_input, options, stream, cudf::get_current_device_resource_ref());
  // host tree generation
  auto cpu_tree = get_tree_representation_cpu(tokens_gpu, token_indices_gpu, options, stream);
  bool const is_array_of_arrays =
    (cpu_tree.node_categories.size() > 0 and
     cpu_tree.node_categories[0] == cudf::io::json::NC_LIST) and
    (json_lines or (cpu_tree.node_categories.size() > 1 and
                    cpu_tree.node_categories[1] == cudf::io::json::NC_LIST));
  // host tree traversal
  auto [cpu_col_id, cpu_row_offsets] =
    records_orient_tree_traversal_cpu(input, cpu_tree, is_array_of_arrays, json_lines, stream);
  // gpu tree generation
  auto gpu_tree = cuio_json::detail::get_tree_representation(
    tokens_gpu, token_indices_gpu, false, stream, cudf::get_current_device_resource_ref());

#if LIBCUDF_JSON_DEBUG_DUMP
  printf("BEFORE traversal (gpu_tree):\n");
  print_tree(gpu_tree);
#endif

  // gpu tree traversal
  auto [gpu_col_id, gpu_row_offsets] =
    cuio_json::detail::records_orient_tree_traversal(d_input,
                                                     gpu_tree,
                                                     is_array_of_arrays,
                                                     json_lines,
                                                     false,
                                                     stream,
                                                     cudf::get_current_device_resource_ref());
#if LIBCUDF_JSON_DEBUG_DUMP
  printf("AFTER  traversal (gpu_tree):\n");
  print_tree(gpu_tree);
#endif

  // convert to sequence because gpu col id might be have random id
  auto gpu_col_id_vec = cudf::detail::make_std_vector(gpu_col_id, stream);
  auto gpu_col_id2    = translate_col_id(gpu_col_id_vec);
  EXPECT_FALSE(compare_vector(cpu_col_id, gpu_col_id2, "col_id"));
  EXPECT_FALSE(compare_vector(cpu_row_offsets, gpu_row_offsets, "row_offsets"));
}
