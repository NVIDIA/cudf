/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/string_parsing.hpp"
#include "json_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <algorithm>
#include <iterator>
#include <type_traits>

using namespace cudf::test::iterators;

struct JSONTypeCastTest : public cudf::test::BaseFixture {};

auto default_json_options()
{
  auto parse_opts = cudf::io::parse_options{',', '\n', '\"', '.'};

  auto const stream     = cudf::get_default_stream();
  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);
  return parse_opts;
}

TEST_F(JSONTypeCastTest, String)
{
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  auto in_valids = null_at(4);
  std::vector<char const*> input_values{"this", "is", "null", "of", "", "strings", R"("null")"};
  cudf::test::strings_column_wrapper input(input_values.begin(), input_values.end(), in_valids);

  auto column                                     = cudf::strings_column_view(input);
  rmm::device_uvector<cudf::size_type> svs_length = string_offset_to_length(column, stream);

  auto null_mask_it = no_nulls();
  auto null_mask =
    std::get<0>(cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + column.size()));

  auto str_col = cudf::io::json::detail::parse_data(
    column.chars_begin(stream),
    cuda::make_zip_iterator(
      cuda::std::make_tuple(column.offsets().begin<cudf::size_type>(), svs_length.begin())),
    column.size(),
    type,
    std::move(null_mask),
    0,
    default_json_options().view(),
    stream,
    mr);

  auto out_valids = nulls_at({2, 4});
  std::vector<char const*> expected_values{"this", "is", "", "of", "", "strings", "null"};
  cudf::test::strings_column_wrapper expected(
    expected_values.begin(), expected_values.end(), out_valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(str_col->view(), expected);
}

TEST_F(JSONTypeCastTest, Int)
{
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();
  auto const type   = cudf::data_type{cudf::type_id::INT64};

  cudf::test::strings_column_wrapper data({"1", "null", "3", "true", "5", "false"});
  auto column                                     = cudf::strings_column_view(data);
  rmm::device_uvector<cudf::size_type> svs_length = string_offset_to_length(column, stream);

  auto null_mask_it = no_nulls();
  auto null_mask =
    std::get<0>(cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + column.size()));

  auto col = cudf::io::json::detail::parse_data(
    column.chars_begin(stream),
    cuda::make_zip_iterator(
      cuda::std::make_tuple(column.offsets().begin<cudf::size_type>(), svs_length.begin())),
    column.size(),
    type,
    std::move(null_mask),
    0,
    default_json_options().view(),
    stream,
    mr);

  auto expected =
    cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 1, 5, 0}, {1, 0, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col->view(), expected);
}

TEST_F(JSONTypeCastTest, StringEscapes)
{
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper data({
    R"("\uD83D\uDE80")",
    R"("\uff21\ud83d\ude80\uff21\uff21")",
    R"("invalid char being escaped escape char\-")",
    R"("too few hex digits \u12")",
    R"("too few hex digits for surrogate pair \uD83D\uDE")",
    R"("\u005C")",
    R"("\u27A9")",
    R"("escape with nothing to escape \")",
    R"("\"\\\/\b\f\n\r\t")",
  });
  auto column                                     = cudf::strings_column_view(data);
  rmm::device_uvector<cudf::size_type> svs_length = string_offset_to_length(column, stream);

  auto null_mask_it = no_nulls();
  auto null_mask =
    std::get<0>(cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + column.size()));

  auto col = cudf::io::json::detail::parse_data(
    column.chars_begin(stream),
    cuda::make_zip_iterator(
      cuda::std::make_tuple(column.offsets().begin<cudf::size_type>(), svs_length.begin())),
    column.size(),
    type,
    std::move(null_mask),
    0,
    default_json_options().view(),
    stream,
    mr);

  auto expected = cudf::test::strings_column_wrapper{
    {"🚀", "Ａ🚀ＡＡ", "", "", "", "\\", "➩", "", "\"\\/\b\f\n\r\t"},
    {true, true, false, false, false, true, true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col->view(), expected);
}

TEST_F(JSONTypeCastTest, ErrorNulls)
{
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  // error in decoding
  std::vector<char const*> input_values{R"("\"\a")",
                                        R"("\u")",
                                        R"("\u0")",
                                        R"("\u0b")",
                                        R"("\u00b")",
                                        R"("\u00bz")",
                                        R"("\t34567890123456\t9012345678901\ug0bc")",
                                        R"("\t34567890123456\t90123456789012\u0hbc")",
                                        R"("\t34567890123456\t90123456789012\u00ic")",
                                        R"("\t34567890123456\t9012345678901\")",
                                        R"("\t34567890123456\t90123456789012\")",
                                        R"(null)"};
  // Note: without quotes are copied without decoding
  cudf::test::strings_column_wrapper input(input_values.begin(), input_values.end());

  auto column        = cudf::strings_column_view(input);
  auto space_length  = 128;
  auto prepend_space = [&space_length](auto const& s) {
    if (s[0] == '"') return "\"" + std::string(space_length, ' ') + std::string(s + 1);
    return std::string(s);
  };
  std::vector<std::string> small_input;
  std::transform(
    input_values.begin(), input_values.end(), std::back_inserter(small_input), prepend_space);
  cudf::test::strings_column_wrapper small_col(small_input.begin(), small_input.end());

  std::vector<std::string> large_input;
  space_length = 128 * 128;
  std::transform(
    input_values.begin(), input_values.end(), std::back_inserter(large_input), prepend_space);
  cudf::test::strings_column_wrapper large_col(large_input.begin(), large_input.end());

  std::vector<char const*> expected_values{"", "", "", "", "", "", "", "", "", "", "", ""};
  cudf::test::strings_column_wrapper expected(
    expected_values.begin(), expected_values.end(), cudf::test::iterators::all_nulls());

  // single threads, warp, block.
  for (auto const& column :
       {column, cudf::strings_column_view(small_col), cudf::strings_column_view(large_col)}) {
    rmm::device_uvector<cudf::size_type> svs_length = string_offset_to_length(column, stream);

    auto null_mask_it = no_nulls();
    auto null_mask =
      std::get<0>(cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + column.size()));

    auto str_col = cudf::io::json::detail::parse_data(
      column.chars_begin(stream),
      cuda::make_zip_iterator(
        cuda::std::make_tuple(column.offsets().begin<cudf::size_type>(), svs_length.begin())),
      column.size(),
      type,
      std::move(null_mask),
      0,
      default_json_options().view(),
      stream,
      mr);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(str_col->view(), expected);
  }
}
