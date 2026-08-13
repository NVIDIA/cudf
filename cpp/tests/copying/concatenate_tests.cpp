/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <numeric>
#include <stdexcept>
#include <string>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using s_col_wrapper = cudf::test::strings_column_wrapper;

using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using column      = cudf::column;
using column_view = cudf::column_view;
using TView       = cudf::table_view;
using Table       = cudf::table;

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }

  TypedColumnTest(rmm::cuda_stream_view stream = cudf::get_default_stream())
    : data{_num_elements * sizeof(T), stream},
      mask{cudf::bitmask_allocation_size_bytes(_num_elements), stream}
  {
    auto typed_data = static_cast<char*>(data.data());
    auto typed_mask = static_cast<char*>(mask.data());
    std::vector<char> h_data(data.size());
    std::iota(h_data.begin(), h_data.end(), char{0});
    std::vector<char> h_mask(mask.size());
    std::iota(h_mask.begin(), h_mask.end(), char{0});
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(typed_data, h_data.data(), data.size(), cudaMemcpyDefault, stream.value()));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(typed_mask, h_mask.data(), mask.size(), cudaMemcpyDefault, stream.value()));
    _null_count = cudf::null_count(static_cast<cudf::bitmask_type*>(mask.data()), 0, _num_elements);
    stream.synchronize();
  }

  [[nodiscard]] cudf::size_type num_elements() const { return _num_elements; }
  [[nodiscard]] cudf::size_type null_count() const { return _null_count; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  cudf::size_type _null_count{};
  rmm::device_buffer all_valid_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_VALID)};
  rmm::device_buffer all_null_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_NULL)};
};

TYPED_TEST_SUITE(TypedColumnTest, cudf::test::Types<int32_t>);

TYPED_TEST(TypedColumnTest, ConcatenateEmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_first{};
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_second{};
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_third{};
  std::vector<column_view> columns_to_concat({empty_first, empty_second, empty_third});

  auto concat = cudf::concatenate(columns_to_concat);

  auto expected_type = cudf::column_view(empty_first).type();
  EXPECT_EQ(concat->size(), 0);
  EXPECT_EQ(concat->type(), expected_type);
}

TYPED_TEST(TypedColumnTest, ConcatenateNoColumns)
{
  std::vector<column_view> columns_to_concat{};
  EXPECT_THROW(cudf::concatenate(columns_to_concat), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, ConcatenateColumnView)
{
  column original{this->type(),
                  this->num_elements(),
                  std::move(this->data),
                  std::move(this->mask),
                  this->null_count()};
  std::vector<cudf::size_type> indices{0,
                                       this->num_elements() / 3,
                                       this->num_elements() / 3,
                                       this->num_elements() / 2,
                                       this->num_elements() / 2,
                                       this->num_elements()};
  std::vector<cudf::column_view> views = cudf::slice(original, indices);

  auto concatenated_col = cudf::concatenate(views);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, *concatenated_col);
}

struct StringColumnTest : public cudf::test::BaseFixture {};

TEST_F(StringColumnTest, ConcatenateColumnView)
{
  std::vector<char const*> h_strings{"aaa",
                                     "bb",
                                     "",
                                     "cccc",
                                     "d",
                                     "ééé",
                                     "ff",
                                     "gggg",
                                     "",
                                     "h",
                                     "iiii",
                                     "jjj",
                                     "k",
                                     "lllllll",
                                     "mmmmm",
                                     "n",
                                     "oo",
                                     "ppp"};
  cudf::test::strings_column_wrapper strings1(h_strings.data(), h_strings.data() + 6);
  cudf::test::strings_column_wrapper strings2(h_strings.data() + 6, h_strings.data() + 10);
  cudf::test::strings_column_wrapper strings3(h_strings.data() + 10,
                                              h_strings.data() + h_strings.size());

  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(strings1);
  strings_columns.push_back(strings2);
  strings_columns.push_back(strings3);

  auto results = cudf::concatenate(strings_columns);

  cudf::test::strings_column_wrapper expected(h_strings.begin(), h_strings.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringColumnTest, ConcatenateManyColumns)
{
  std::vector<char const*> h_strings{
    "aaa", "bb", "", "cccc", "d", "ééé", "ff", "gggg", "", "h", "iiii", "jjj"};

  std::vector<char const*> expected_strings;
  std::vector<cudf::test::strings_column_wrapper> wrappers;
  std::vector<cudf::column_view> strings_columns;
  for (int i = 0; i < 200; ++i) {
    wrappers.emplace_back(h_strings.data(), h_strings.data() + h_strings.size());
    strings_columns.push_back(wrappers[i]);
    expected_strings.insert(expected_strings.end(), h_strings.begin(), h_strings.end());
  }
  cudf::test::strings_column_wrapper expected(expected_strings.data(),
                                              expected_strings.data() + expected_strings.size());
  auto results = cudf::concatenate(strings_columns);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringColumnTest, ConcatenateTooLarge)
{
  if (cudf::strings::is_large_strings_enabled()) { return; }

  std::string big_str(1000000, 'a');  // 1 million bytes x 5 = 5 million bytes
  cudf::test::strings_column_wrapper input{big_str, big_str, big_str, big_str, big_str};
  std::vector<cudf::column_view> input_cols;
  // 5 millions bytes x 500 = 2.5GB > std::numeric_limits<size_type>::max()
  for (int i = 0; i < 500; ++i) {
    input_cols.push_back(input);
  }
  EXPECT_THROW(cudf::concatenate(input_cols), std::overflow_error);
}

struct TableTest : public cudf::test::BaseFixture {};

TEST_F(TableTest, ConcatenateTables)
{
  std::vector<char const*> h_strings{
    "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"};

  CVector cols_gold;
  column_wrapper<int8_t> col1_gold{{1, 2, 3, 4, 5, 6, 7, 8}};
  column_wrapper<int16_t> col2_gold{{1, 2, 3, 4, 5, 6, 7, 8}};
  s_col_wrapper col3_gold(h_strings.data(), h_strings.data() + h_strings.size());
  cols_gold.push_back(col1_gold.release());
  cols_gold.push_back(col2_gold.release());
  cols_gold.push_back(col3_gold.release());
  Table gold_table(std::move(cols_gold));

  CVector cols_table1;
  column_wrapper<int8_t> col1_table1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2_table1{{1, 2, 3, 4}};
  s_col_wrapper col3_table1(h_strings.data(), h_strings.data() + 4);
  cols_table1.push_back(col1_table1.release());
  cols_table1.push_back(col2_table1.release());
  cols_table1.push_back(col3_table1.release());
  Table t1(std::move(cols_table1));

  CVector cols_table2;
  column_wrapper<int8_t> col1_table2{{5, 6, 7, 8}};
  column_wrapper<int16_t> col2_table2{{5, 6, 7, 8}};
  s_col_wrapper col3_table2(h_strings.data() + 4, h_strings.data() + h_strings.size());
  cols_table2.push_back(col1_table2.release());
  cols_table2.push_back(col2_table2.release());
  cols_table2.push_back(col3_table2.release());
  Table t2(std::move(cols_table2));

  auto concat_table = cudf::concatenate(std::vector<TView>({t1, t2}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*concat_table, gold_table);
}

TEST_F(TableTest, ConcatenateZeroColumnTables)
{
  TView t1{std::vector<column_view>{}, 7};
  TView t2{std::vector<column_view>{}, 5};
  auto concat_table = cudf::concatenate(std::vector<TView>({t1, t2}));
  EXPECT_EQ(concat_table->num_columns(), 0);
  EXPECT_EQ(concat_table->num_rows(), 12);
}

TEST_F(TableTest, ConcatenateTablesWithOffsets)
{
  column_wrapper<int32_t> col1_1{{5, 4, 3, 5, 8, 5, 6}};
  cudf::test::strings_column_wrapper col2_1(
    {"dada", "egg", "avocado", "dada", "kite", "dog", "ln"});
  cudf::table_view table_view_in1{{col1_1, col2_1}};

  column_wrapper<int32_t> col1_2{{5, 8, 5, 6, 15, 14, 13}};
  cudf::test::strings_column_wrapper col2_2(
    {"dada", "kite", "dog", "ln", "dado", "greg", "spinach"});
  cudf::table_view table_view_in2{{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 = cudf::split(table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 = cudf::split(table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[0]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 4, 3, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "egg", "avocado", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 5, 8, 5}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"});
    cudf::table_view table_view_exp{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp);
  }
}

TEST_F(TableTest, ConcatenateTablesWithOffsetsAndNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1_1{
    {5, 4, 3, 5, 8, 5, 6}, {false, true, true, true, true, true, true}};
  cudf::test::strings_column_wrapper col2_1({"dada", "egg", "avocado", "dada", "kite", "dog", "ln"},
                                            {true, true, true, false, true, true, true});
  cudf::table_view table_view_in1{{col1_1, col2_1}};

  cudf::test::fixed_width_column_wrapper<int32_t> col1_2{
    {5, 8, 5, 6, 15, 14, 13}, {true, true, true, true, true, true, false}};
  cudf::test::strings_column_wrapper col2_2(
    {"dada", "kite", "dog", "ln", "dado", "greg", "spinach"},
    {true, false, true, true, true, true, true});
  cudf::table_view table_view_in2{{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 = cudf::split(table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 = cudf::split(table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{
      {5, 8, 5, 6, 6, 15, 14, 13}, {true, true, true, true, true, true, true, false}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"},
      {false, true, true, true, true, true, true, true});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{5, 8, 5, 6, 5, 8, 5};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"},
                                              {false, true, true, true, true, false, true});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
}

struct OverflowTest : public cudf::test::BaseFixture {};

TEST_F(OverflowTest, OverflowTest)
{
  // should concatenate up to size_type::max rows.
  {
    // 5 x size + size_last adds to size_type::max
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(250) * 1024 * 1024);
    constexpr auto size_last = static_cast<cudf::size_type>(836763647);

    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);
    auto many_chars_last =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size_last);

    cudf::table_view tbl({*many_chars});
    cudf::table_view tbl_last({*many_chars_last});
    std::vector<cudf::table_view> table_views_to_concat({tbl, tbl, tbl, tbl, tbl, tbl_last});
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);
    EXPECT_NO_THROW(cudf::get_default_stream().synchronize());
    ASSERT_EQ(concatenated_tables->num_rows(), std::numeric_limits<cudf::size_type>::max());
  }

  // primitive column
  {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 char columns of size 1 billion each
    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);

    cudf::table_view tbl({*many_chars});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({tbl, tbl, tbl, tbl, tbl, tbl})),
                 std::overflow_error);
  }

  // string column, overflow on chars
  if (!cudf::strings::is_large_strings_enabled()) {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 string columns of with 1 billion chars in each
    auto offsets    = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, size};
    auto many_chars = rmm::device_uvector<char>(size, cudf::get_default_stream());
    auto col        = cudf::make_strings_column(
      1, offsets.release(), many_chars.release(), 0, rmm::device_buffer{});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({tbl, tbl, tbl, tbl, tbl, tbl})),
                 std::overflow_error);
  }

  // string column, overflow on offsets (rows)
  {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 string columns 1 billion rows each
    auto many_offsets =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, size + 1);
    auto chars = rmm::device_uvector<char>(3, cudf::get_default_stream());
    auto col   = cudf::make_strings_column(
      size, std::move(many_offsets), chars.release(), 0, rmm::device_buffer{});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({tbl, tbl, tbl, tbl, tbl, tbl})),
                 std::overflow_error);
  }

  // list<struct>, structs too long
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(512) * 1024 * 1024);

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    auto struct_col =
      cudf::make_structs_column(inner_size, std::move(children), 0, rmm::device_buffer{});

    // list
    auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, inner_size};
    auto col =
      cudf::make_lists_column(1, offsets.release(), std::move(struct_col), 0, rmm::device_buffer{});

    cudf::table_view tbl({*col});
    auto tables =
      std::vector<cudf::table_view>({tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl});
    EXPECT_THROW(cudf::concatenate(tables), std::overflow_error);
  }

  // struct<int, list>, list child too long
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(512) * 1024 * 1024);
    constexpr cudf::size_type size = 3;

    // list
    auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, inner_size};
    auto many_chars =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size);
    auto list_col =
      cudf::make_lists_column(3, offsets.release(), std::move(many_chars), 0, rmm::device_buffer{});

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, size));
    children.push_back(std::move(list_col));
    auto col = cudf::make_structs_column(size, std::move(children), 0, rmm::device_buffer{});

    cudf::table_view tbl({*col});
    auto tables =
      std::vector<cudf::table_view>({tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl});
    EXPECT_THROW(cudf::concatenate(tables), std::overflow_error);
  }
}

TEST_F(OverflowTest, Presliced)
{
  // primitive column
  {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 4 char columns of size ~1/2 billion each
    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);
    auto sliced     = cudf::split(*many_chars, {511 * 1024 * 1024});

    // 511 * 1024 * 1024, should succeed
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a}));

    // 513 * 1024 * 1024, should fail
    cudf::table_view b({sliced[1]});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b})),
                 std::overflow_error);
  }

  // struct<int8> column
  {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 4 char columns of size ~1/2 billion each
    std::vector<std::unique_ptr<column>> children;
    children.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size));
    auto struct_col = cudf::make_structs_column(size, std::move(children), 0, rmm::device_buffer{});

    auto sliced = cudf::split(*struct_col, {511 * 1024 * 1024});

    // 511 * 1024 * 1024, should succeed
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a}));

    // 513 * 1024 * 1024, should fail
    cudf::table_view b({sliced[1]});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b})),
                 std::overflow_error);
  }

  // strings, overflow on chars
  if (!cudf::strings::is_large_strings_enabled()) {
    constexpr cudf::size_type total_chars_size = 1024 * 1024 * 1024;
    constexpr cudf::size_type string_size      = 64;
    constexpr cudf::size_type num_rows         = total_chars_size / string_size;

    // try and concatenate 4 string columns of with ~1/2 billion chars in each
    auto offset_gen = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type index) { return index * string_size; });
    cudf::test::fixed_width_column_wrapper<int> offsets(offset_gen, offset_gen + num_rows + 1);
    auto many_chars = rmm::device_uvector<char>(total_chars_size, cudf::get_default_stream());
    auto col        = cudf::make_strings_column(
      num_rows, offsets.release(), many_chars.release(), 0, rmm::device_buffer{});

    auto sliced = cudf::split(*col, {(num_rows / 2) - 1});

    // (num_rows / 2) - 1 should succeed
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a}));

    // (num_rows / 2) + 1 should fail
    cudf::table_view b({sliced[1]});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b})),
                 std::overflow_error);
  }

  // strings, overflow on offsets
  {
    constexpr cudf::size_type total_chars_size = 1024 * 1024 * 1024;
    constexpr cudf::size_type string_size      = 1;
    constexpr cudf::size_type num_rows         = total_chars_size / string_size;

    // try and concatenate 4 string columns of with ~1/2 billion chars in each
    auto offsets    = cudf::sequence(num_rows + 1,
                                  cudf::numeric_scalar<cudf::size_type>(0),
                                  cudf::numeric_scalar<cudf::size_type>(string_size));
    auto many_chars = rmm::device_uvector<char>(total_chars_size, cudf::get_default_stream());
    auto col        = cudf::make_strings_column(
      num_rows, std::move(offsets), many_chars.release(), 0, rmm::device_buffer{});

    // should pass (with 2 rows to spare)
    // leaving this disabled as it typically runs out of memory on a T4
    /*
    {
      auto sliced = cudf::split(*col, {(num_rows / 2) + 1});
      cudf::table_view b({sliced[1]});
      cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b}));
    }
    */

    // should fail by 1 row (2,147,483,647 rows which is fine, but that requires 2,147,483,648
    // offsets, which is not)
    {
      auto a = cudf::split(*col, {(num_rows / 2)});
      cudf::table_view ta({a[1]});

      auto b = cudf::split(*col, {(num_rows / 2) + 1});
      cudf::table_view tb({b[1]});

      EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({ta, ta, ta, tb})),
                   std::overflow_error);
    }
  }

  // list<struct>, structs too long
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    auto struct_col =
      cudf::make_structs_column(inner_size, std::move(children), 0, rmm::device_buffer{});

    // list
    constexpr cudf::size_type list_size = inner_size / 4;
    auto offsets                        = cudf::test::fixed_width_column_wrapper<int>{
      0, list_size, list_size * 2, list_size * 3, inner_size};
    auto col =
      cudf::make_lists_column(4, offsets.release(), std::move(struct_col), 0, rmm::device_buffer{});

    auto sliced = cudf::split(*col, {2});
    cudf::table_view tbl({sliced[1]});
    auto tables = std::vector<cudf::table_view>({tbl, tbl, tbl, tbl});
    EXPECT_THROW(cudf::concatenate(tables), std::overflow_error);
  }

  // list<struct>, overflow on offsets
  {
    constexpr cudf::size_type inner_size = 1024 * 1024 * 1024;
    constexpr cudf::size_type list_size  = 1;
    constexpr cudf::size_type num_rows   = inner_size / list_size;

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    auto struct_col =
      cudf::make_structs_column(inner_size, std::move(children), 0, rmm::device_buffer{});

    // try and concatenate 4 struct columns of with ~1/2 billion elements in each
    auto offsets = cudf::sequence(num_rows + 1,
                                  cudf::numeric_scalar<cudf::size_type>(0),
                                  cudf::numeric_scalar<cudf::size_type>(list_size));

    auto col = cudf::make_lists_column(
      num_rows, std::move(offsets), std::move(struct_col), 0, rmm::device_buffer{});

    // should pass (with 2 rows to spare)
    // leaving this disabled as it typically runs out of memory on a T4
    /*
    {
      auto sliced = cudf::split(*col, {(num_rows / 2) + 1});
      cudf::table_view b({sliced[1]});
      cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b}));
    }
    */

    // should fail by 1 row (2,147,483,647 rows which is fine, but that requires 2,147,483,648
    // offsets, which is not)
    {
      auto a = cudf::split(*col, {(num_rows / 2)});
      cudf::table_view ta({a[1]});

      auto b = cudf::split(*col, {(num_rows / 2) + 1});
      cudf::table_view tb({b[1]});

      EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({ta, ta, ta, tb})),
                   std::overflow_error);
    }
  }

  // struct<int8, list>, list child elements too long
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);
    constexpr cudf::size_type num_rows  = 4;
    constexpr cudf::size_type list_size = inner_size / num_rows;

    // list
    auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
      0, list_size, (list_size * 2) - 1, list_size * 3, inner_size};
    auto many_chars =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size);
    auto list_col = cudf::make_lists_column(
      num_rows, offsets.release(), std::move(many_chars), 0, rmm::device_buffer{});

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, num_rows));
    children.push_back(std::move(list_col));
    auto struct_col =
      cudf::make_structs_column(num_rows, std::move(children), 0, rmm::device_buffer{});

    auto sliced = cudf::split(*struct_col, {2});

    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a}));

    cudf::table_view b({sliced[1]});
    EXPECT_THROW(cudf::concatenate(std::vector<cudf::table_view>({b, b, b, b})),
                 std::overflow_error);
  }
}

TEST_F(OverflowTest, BigColumnsSmallSlices)
{
  // test : many small slices of large columns. the idea is to make sure
  // that we are respecting the offset/sizes of the slices and not attempting to
  // concatenate the entire large initial columns

  // primitive column
  {
    constexpr auto size = static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);
    auto sliced     = cudf::slice(*many_chars, {16, 32});

    // 192 total rows
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a, a, a, a, a, a, a, a, a}));
  }

  // strings column
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);
    constexpr cudf::size_type num_rows    = 1024;
    constexpr cudf::size_type string_size = inner_size / num_rows;

    auto offsets    = cudf::sequence(num_rows + 1,
                                  cudf::numeric_scalar<cudf::size_type>(0),
                                  cudf::numeric_scalar<cudf::size_type>(string_size));
    auto many_chars = rmm::device_uvector<char>(inner_size, cudf::get_default_stream());
    auto col        = cudf::make_strings_column(
      num_rows, std::move(offsets), many_chars.release(), 0, rmm::device_buffer{});

    auto sliced = cudf::slice(*col, {16, 32});

    // 192 outer rows
    // 201,326,592 inner rows
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a, a, a, a, a, a, a, a, a}));
  }

  // list<int8> column
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);
    constexpr cudf::size_type num_rows  = 1024;
    constexpr cudf::size_type list_size = inner_size / num_rows;

    auto offsets = cudf::sequence(num_rows + 1,
                                  cudf::numeric_scalar<cudf::size_type>(0),
                                  cudf::numeric_scalar<cudf::size_type>(list_size));
    auto many_chars =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size);
    auto col = cudf::make_lists_column(
      num_rows, std::move(offsets), std::move(many_chars), 0, rmm::device_buffer{});

    auto sliced = cudf::slice(*col, {16, 32});

    // 192 outer rows
    // 201,326,592 inner rows
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a, a, a, a, a, a, a, a, a}));
  }

  // struct<int8, list>
  {
    constexpr auto inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);
    constexpr cudf::size_type num_rows  = 1024;
    constexpr cudf::size_type list_size = inner_size / num_rows;

    auto offsets = cudf::sequence(num_rows + 1,
                                  cudf::numeric_scalar<cudf::size_type>(0),
                                  cudf::numeric_scalar<cudf::size_type>(list_size));
    auto many_chars =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size);
    auto list_col = cudf::make_lists_column(
      num_rows, std::move(offsets), std::move(many_chars), 0, rmm::device_buffer{});

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, num_rows));
    children.push_back(std::move(list_col));
    auto struct_col =
      cudf::make_structs_column(num_rows, std::move(children), 0, rmm::device_buffer{});

    auto sliced = cudf::slice(*struct_col, {16, 32});

    // 192 outer rows
    // 201,326,592 inner rows
    cudf::table_view a({sliced[0]});
    cudf::concatenate(std::vector<cudf::table_view>({a, a, a, a, a, a, a, a, a, a, a, a}));
  }
}

struct StructsColumnTest : public cudf::test::BaseFixture {};

TEST_F(StructsColumnTest, ConcatenateStructs)
{
  auto count_iter = cuda::counting_iterator<int>{0};

  // 1. String "names" column.
  std::vector<std::vector<std::string>> names(
    {{"Vimes", "Carrot"}, {"Angua", "Cheery"}, {}, {"Detritus", "Slant"}});
  std::vector<std::vector<bool>> names_validity({{true, true}, {true, true}, {}, {true, true}});
  std::vector<cudf::test::strings_column_wrapper> name_cols;
  std::transform(count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
    return cudf::test::strings_column_wrapper(
      names[i].begin(), names[i].end(), names_validity[i].begin());
  });

  // 2. Numeric "ages" column.
  std::vector<std::vector<int>> ages({{5, 10}, {15, 20}, {}, {25, 30}});
  std::vector<std::vector<bool>> ages_validity({{true, true}, {true, true}, {}, {false, true}});
  std::vector<cudf::test::fixed_width_column_wrapper<int>> age_cols;
  std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
    return cudf::test::fixed_width_column_wrapper<int>(
      ages[i].begin(), ages[i].end(), ages_validity[i].begin());
  });

  // 3. Boolean "is_human" column.
  std::vector<std::vector<bool>> is_human({{true, true}, {false, false}, {}, {false, false}});
  std::vector<std::vector<bool>> is_human_validity({{true, true}, {true, false}, {}, {true, true}});
  std::vector<cudf::test::fixed_width_column_wrapper<bool>> is_human_cols;
  std::transform(
    count_iter, count_iter + is_human.size(), std::back_inserter(is_human_cols), [&](int i) {
      return cudf::test::fixed_width_column_wrapper<bool>(
        is_human[i].begin(), is_human[i].end(), is_human_validity[i].begin());
    });

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;
  auto name_col_vec =
    std::vector<column_view>({name_cols[0], name_cols[1], name_cols[2], name_cols[3]});
  auto age_col_vec = std::vector<column_view>({age_cols[0], age_cols[1], age_cols[2], age_cols[3]});
  auto is_human_col_vec = std::vector<column_view>(
    {is_human_cols[0], is_human_cols[1], is_human_cols[2], is_human_cols[3]});
  expected_children.push_back(cudf::concatenate(name_col_vec));
  expected_children.push_back(cudf::concatenate(age_col_vec));
  expected_children.push_back(cudf::concatenate(is_human_col_vec));
  std::vector<bool> struct_validity({true, false, true, true, true, false});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto expected =
    make_structs_column(6, std::move(expected_children), null_count, std::move(null_mask));

  // concatenate as structs
  std::vector<cudf::test::structs_column_wrapper> src;
  src.push_back(cudf::test::structs_column_wrapper({name_cols[0], age_cols[0], is_human_cols[0]},
                                                   {true, false}));
  src.push_back(cudf::test::structs_column_wrapper({name_cols[1], age_cols[1], is_human_cols[1]},
                                                   {true, true}));
  src.push_back(
    cudf::test::structs_column_wrapper({name_cols[2], age_cols[2], is_human_cols[2]}, {}));
  src.push_back(cudf::test::structs_column_wrapper({name_cols[3], age_cols[3], is_human_cols[3]},
                                                   {true, false}));

  // concatenate
  auto result = cudf::concatenate(std::vector<column_view>({src[0], src[1], src[2], src[3]}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TEST_F(StructsColumnTest, ConcatenateEmptyStructs)
{
  auto expected = cudf::make_structs_column(10, {}, 0, rmm::device_buffer());
  auto first    = cudf::make_structs_column(5, {}, 0, rmm::device_buffer());
  auto second   = cudf::make_structs_column(2, {}, 0, rmm::device_buffer());
  auto third    = cudf::make_structs_column(0, {}, 0, rmm::device_buffer());
  auto fourth   = cudf::make_structs_column(3, {}, 0, rmm::device_buffer());

  // concatenate
  auto result = cudf::concatenate(std::vector<column_view>({*first, *second, *third, *fourth}));
  ASSERT_EQ(result->size(), expected->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TEST_F(StructsColumnTest, ConcatenateSplitStructs)
{
  auto count_iter = cuda::counting_iterator<int>{0};

  std::vector<int> splits({2});

  // 1. String "names" column.
  std::vector<std::vector<std::string>> names(
    {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
     {"Bill", "Bob", "Sam", "Fred", "Tom"}});
  std::vector<std::vector<bool>> names_validity(
    {{true, true, true, true, true, true}, {false, true, false, true, false}});
  std::vector<cudf::test::strings_column_wrapper> name_cols;
  std::transform(count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
    return cudf::test::strings_column_wrapper(
      names[i].begin(), names[i].end(), names_validity[i].begin());
  });

  // 2. Numeric "ages" column.
  std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {11, 16, 17, 41, 42}});
  std::vector<std::vector<bool>> ages_validity(
    {{true, true, true, true, false, true}, {true, true, true, false, false}});
  std::vector<cudf::test::fixed_width_column_wrapper<int>> age_cols;
  std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
    return cudf::test::fixed_width_column_wrapper<int>(
      ages[i].begin(), ages[i].end(), ages_validity[i].begin());
  });

  // 3. Boolean "is_human" column.
  std::vector<std::vector<bool>> is_human(
    {{true, true, false, false, false, false}, {true, true, true, false, true}});
  std::vector<std::vector<bool>> is_human_validity(
    {{true, true, true, false, true, true}, {false, false, false, true, true}});
  std::vector<cudf::test::fixed_width_column_wrapper<bool>> is_human_cols;
  std::transform(
    count_iter, count_iter + is_human.size(), std::back_inserter(is_human_cols), [&](int i) {
      return cudf::test::fixed_width_column_wrapper<bool>(
        is_human[i].begin(), is_human[i].end(), is_human_validity[i].begin());
    });

  // split the columns, keep the one on the end
  std::vector<column_view> split_names_cols(
    {cudf::split(name_cols[0], splits)[1], cudf::split(name_cols[1], splits)[1]});
  std::vector<column_view> split_ages_cols(
    {cudf::split(age_cols[0], splits)[1], cudf::split(age_cols[1], splits)[1]});
  std::vector<column_view> split_is_human_cols(
    {cudf::split(is_human_cols[0], splits)[1], cudf::split(is_human_cols[1], splits)[1]});

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;
  auto expected_names = std::vector<column_view>({split_names_cols[0], split_names_cols[1]});
  auto expected_ages  = std::vector<column_view>({split_ages_cols[0], split_ages_cols[1]});
  auto expected_is_human =
    std::vector<column_view>({split_is_human_cols[0], split_is_human_cols[1]});
  expected_children.push_back(cudf::concatenate(expected_names));
  expected_children.push_back(cudf::concatenate(expected_ages));
  expected_children.push_back(cudf::concatenate(expected_is_human));
  auto expected = make_structs_column(7, std::move(expected_children), 0, rmm::device_buffer{});

  // concatenate as structs
  std::vector<cudf::test::structs_column_wrapper> src;
  for (size_t idx = 0; idx < split_names_cols.size(); idx++) {
    std::vector<std::unique_ptr<column>> inputs;
    inputs.push_back(std::make_unique<column>(split_names_cols[idx]));
    inputs.push_back(std::make_unique<column>(split_ages_cols[idx]));
    inputs.push_back(std::make_unique<column>(split_is_human_cols[idx]));
    src.emplace_back(std::move(inputs));
  }

  // concatenate

  auto result = cudf::concatenate(std::vector<column_view>({src[0], src[1]}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TEST_F(StructsColumnTest, ConcatenateStructsNested)
{
  // includes Struct<Struct> and Struct<List>

  auto count_iter = cuda::counting_iterator<int>{0};

  // inner structs
  std::vector<cudf::test::structs_column_wrapper> inner_structs;
  {
    // 1. String "names" column.
    std::vector<std::vector<std::string>> names(
      {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
       {"Bill", "Bob", "Sam", "Fred", "Tom"}});
    std::vector<std::vector<bool>> names_validity(
      {{true, true, true, true, true, true}, {false, true, false, true, false}});
    std::vector<cudf::test::strings_column_wrapper> name_cols;
    std::transform(
      count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
        return cudf::test::strings_column_wrapper(
          names[i].begin(), names[i].end(), names_validity[i].begin());
      });

    // 2. Numeric "ages" column.
    std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {11, 16, 17, 41, 42}});
    std::vector<std::vector<bool>> ages_validity(
      {{true, true, true, true, false, true}, {true, true, true, false, false}});
    std::vector<cudf::test::fixed_width_column_wrapper<int>> age_cols;
    std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
      return cudf::test::fixed_width_column_wrapper<int>(
        ages[i].begin(), ages[i].end(), ages_validity[i].begin());
    });

    for (size_t idx = 0; idx < names.size(); idx++) {
      std::vector<std::unique_ptr<column>> children;
      children.push_back(name_cols[idx].release());
      children.push_back(age_cols[idx].release());
      inner_structs.emplace_back(std::move(children));
    }
  }

  // inner lists
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  std::vector<cudf::test::lists_column_wrapper<cudf::string_view>> inner_lists;
  {
    inner_lists.push_back(cudf::test::lists_column_wrapper<cudf::string_view>{
      {"abc", "d"}, {"ef", "ghi", "j"}, {"klm", "no"}, LCW{}, LCW{"whee"}, {"xyz", "ab", "g"}});

    inner_lists.push_back(cudf::test::lists_column_wrapper<cudf::string_view>{
      {"er", "hyj"}, {"", "", "uvw"}, LCW{}, LCW{"oipq", "te"}, LCW{"yay", "bonk"}});
  }

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;

  expected_children.push_back(
    cudf::concatenate(std::vector<column_view>({inner_structs[0], inner_structs[1]})));
  expected_children.push_back(
    cudf::concatenate(std::vector<column_view>({inner_lists[0], inner_lists[1]})));
  auto expected = make_structs_column(11, std::move(expected_children), 0, rmm::device_buffer{});

  // concatenate as structs
  std::vector<cudf::test::structs_column_wrapper> src;
  for (size_t idx = 0; idx < inner_structs.size(); idx++) {
    std::vector<std::unique_ptr<column>> inputs;
    inputs.push_back(std::make_unique<column>(inner_structs[idx]));
    inputs.push_back(std::make_unique<column>(inner_lists[idx]));
    src.emplace_back(std::move(inputs));
  }

  // concatenate
  auto result = cudf::concatenate(std::vector<column_view>({src[0], src[1]}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

struct ListsColumnTest : public cudf::test::BaseFixture {};

TEST_F(ListsColumnTest, ConcatenateLists)
{
  {
    cudf::test::lists_column_wrapper<int> a{0, 1, 2, 3};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7, 8, 9, 10};
    cudf::test::lists_column_wrapper<int> expected{{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9, 10}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1, 1}, {2, 3}, {4, 5}};
    cudf::test::lists_column_wrapper<int> b{{6}, {8, 9, 9, 9}, {10, 11}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1, 1}, {2, 3}, {4, 5}, {6}, {8, 9, 9, 9}, {10, 11}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1}, {2, 3, 4, 5}, {6, 7, 8}};
    cudf::test::lists_column_wrapper<int> b{{9}, {10, 11}, {12, 13, 14, 15}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1}, {2, 3, 4, 5}, {6, 7, 8}, {9}, {10, 11}, {12, 13, 14, 15}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateEmptyLists)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a;
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a, b, c;
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate(std::vector<column_view>({a, b, c, d}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}}, b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b, c, d}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{1, 2};
    cudf::test::lists_column_wrapper<int> b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{1, 2}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b, c, d}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateListsWithNulls)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // nulls in the leaves
  {
    cudf::test::lists_column_wrapper<int> a{{{0, 1, 2, 3}, valids}};
    cudf::test::lists_column_wrapper<int> b{{{4, 6, 7}, valids}};
    cudf::test::lists_column_wrapper<int> expected{{{0, 1, 2, 3}, valids}, {{4, 6, 7}, valids}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedLists)
{
  {
    cudf::test::lists_column_wrapper<int> a{{{0, 1}, {2}}, {{4, 5, 6, 7, 8, 9, 10}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {{8, 9, 10}, {11, 12}}};
    cudf::test::lists_column_wrapper<int> expected{
      {{0, 1}, {2}}, {{4, 5, 6, 7, 8, 9, 10}}, {{6, 7}}, {{8, 9, 10}, {11, 12}}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{0, 1, 2}, {3, 4}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{10}, {11, 12}}, {{13, 14, 15, 16}, {15, 16}}, {{17, 18}, {19, 20}}},
      {{{50}, {51, 52}}, {{54}, {55, 16}}, {{57, 18}, {59, 60}}}};

    cudf::test::lists_column_wrapper<int> b{
      {{{21, 22}, {23, 24}}, {{25}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}}, {{39, 40}}},
      {{{71, 72}, {74}}, {{75, 76, 77, 78}, {77, 78}}, {{79, 80, 81}}}};

    cudf::test::lists_column_wrapper<int> expected{
      {{{0, 1, 2}, {3, 4}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{10}, {11, 12}}, {{13, 14, 15, 16}, {15, 16}}, {{17, 18}, {19, 20}}},
      {{{50}, {51, 52}}, {{54}, {55, 16}}, {{57, 18}, {59, 60}}},
      {{{21, 22}, {23, 24}}, {{25}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}}, {{39, 40}}},
      {{{71, 72}, {74}}, {{75, 76, 77, 78}, {77, 78}}, {{79, 80, 81}}}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedEmptyLists)
{
  using T = int;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;
  {
    cudf::test::lists_column_wrapper<T> a{{LCW{}}, {{0, 1}, {2, 3}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {LCW{}, {11, 12}}};
    cudf::test::lists_column_wrapper<int> expected{
      {LCW{}}, {{0, 1}, {2, 3}}, {{6, 7}}, {LCW{}, {11, 12}}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{0, 1, 2}, LCW{}}, {{5}, {6, 7}}, {{8, 9}}},
      {{LCW{}}, {{17, 18}, {19, 20}}},
      {{LCW{}}},
      {{{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}}};

    cudf::test::lists_column_wrapper<int> b{
      {{{21, 22}, {23, 24}}, {LCW{}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{LCW{}}}};

    cudf::test::lists_column_wrapper<int> expected{
      {{{0, 1, 2}, LCW{}}, {{5}, {6, 7}}, {{8, 9}}},
      {{LCW{}}, {{17, 18}, {19, 20}}},
      {{LCW{}}},
      {{{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}},
      {{{21, 22}, {23, 24}}, {LCW{}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{LCW{}}}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedListsWithNulls)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // nulls in the lists
  {
    cudf::test::lists_column_wrapper<int> a{{{{0, 1}, {2, 3}}, valids}};
    cudf::test::lists_column_wrapper<int> b{{{{4}, {6, 7}}, valids}};

    cudf::test::lists_column_wrapper<int> expected{{{{0, 1}, {2, 3}}, valids},
                                                   {{{4}, {6, 7}}, valids}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  // nulls in the lists -and- the values
  {
    cudf::test::lists_column_wrapper<int> a{{{{{0}, valids}, {2, 3}}, valids}, {{4, 5}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {{{{8, 9, 10}, valids}, {11, 12}}, valids}};
    cudf::test::lists_column_wrapper<int> expected{{{{{0}, valids}, {2, 3}}, valids},
                                                   {{4, 5}},
                                                   {{6, 7}},
                                                   {{{{8, 9, 10}, valids}, {11, 12}}, valids}};

    auto result = cudf::concatenate(std::vector<column_view>({a, b}));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateMismatchedHierarchies)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a{{{{LCW{}}}}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};

    EXPECT_THROW(cudf::concatenate(std::vector<column_view>({a, b, c})), cudf::data_type_error);
  }

  {
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<int> a{{{{{LCW{}}}}, valids.begin()}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};

    EXPECT_THROW(cudf::concatenate(std::vector<column_view>({a, b, c})), cudf::data_type_error);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{{{LCW{}}}}};
    cudf::test::lists_column_wrapper<int> b{1, 2, 3};
    cudf::test::lists_column_wrapper<int> c{{3, 4, 5}};

    EXPECT_THROW(cudf::concatenate(std::vector<column_view>({a, b, c})), cudf::data_type_error);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{{1, 2, 3}}};
    cudf::test::lists_column_wrapper<int> b{{4, 5}};

    EXPECT_THROW(cudf::concatenate(std::vector<column_view>({a, b})), cudf::data_type_error);
  }
}

TEST_F(ListsColumnTest, SlicedColumns)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  {
    cudf::test::lists_column_wrapper<int> a{{{1, 1, 1}, {2, 2}, {3, 3}},
                                            {{4, 4, 4}, {5, 5}, {6, 6}},
                                            {{7, 7, 7}, {8, 8}, {9, 9}},
                                            {{10, 10, 10}, {11, 11}, {12, 12}}};
    auto split_a = cudf::split(a, {2});

    cudf::test::lists_column_wrapper<int> b{{{-1, -1, -1, -1}, {-2}},
                                            {{-3, -3, -3, -3}, {-4}},
                                            {{-5, -5, -5, -5}, {-6}},
                                            {{-7, -7, -7, -7}, {-8}}};
    auto split_b = cudf::split(b, {2});

    cudf::test::lists_column_wrapper<int> expected0{{{1, 1, 1}, {2, 2}, {3, 3}},
                                                    {{4, 4, 4}, {5, 5}, {6, 6}},
                                                    {{-1, -1, -1, -1}, {-2}},
                                                    {{-3, -3, -3, -3}, {-4}}};

    auto result0 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{{{1, 1, 1}, {2, 2}, {3, 3}},
                                                    {{4, 4, 4}, {5, 5}, {6, 6}},
                                                    {{-5, -5, -5, -5}, {-6}},
                                                    {{-7, -7, -7, -7}, {-8}}};

    auto result1 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{-1, -1, -1, -1}, {-2}},
      {{-3, -3, -3, -3}, {-4}},
    };

    auto result2 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{{{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{10, 10, 10}, {11, 11}, {12, 12}},
                                                    {{-5, -5, -5, -5}, {-6}},
                                                    {{-7, -7, -7, -7}, {-8}}};

    auto result3 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result3, expected3);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}}};
    auto split_a = cudf::split(a, {2});

    cudf::test::lists_column_wrapper<int> b{
      {{LCW{}}},
      {LCW{}, {LCW{}}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };
    auto split_b = cudf::split(b, {2});

    cudf::test::lists_column_wrapper<int> expected0{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {{LCW{}}},
      {LCW{}, {LCW{}}}};

    auto result0 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[0]}));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };

    auto result1 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}},
      {{LCW{}}},
      {LCW{}, {LCW{}}}};

    auto result2 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };

    auto result3 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result3, expected3);
  }
}

TEST_F(ListsColumnTest, SlicedColumnsWithNulls)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  {
    cudf::test::lists_column_wrapper<int> a{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                            {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                            {{7, 7, 7}, {8, 8}, {9, 9}},
                                            {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids}};
    auto split_a = cudf::split(a, {3});

    cudf::test::lists_column_wrapper<int> b{{{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
                                            {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
                                            {{{{-5, -5, -5, -5}, valids}, {-6}}, valids},
                                            {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};
    auto split_b = cudf::split(b, {3});

    cudf::test::lists_column_wrapper<int> expected0{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                                    {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                                    {{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
                                                    {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
                                                    {{{{-5, -5, -5, -5}, valids}, {-6}}, valids}};

    auto result0 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                                    {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                                    {{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};

    auto result1 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids},
      {{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
      {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
      {{{{-5, -5, -5, -5}, valids}, {-6}}, valids}};

    auto result2 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids},
      {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};

    auto result3 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result3, expected3);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}}};
    auto split_a = cudf::split(a, {3});

    cudf::test::lists_column_wrapper<int> b{
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
      {{LCW{}}},
    };
    auto split_b = cudf::split(b, {3});

    cudf::test::lists_column_wrapper<int> expected0{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
    };

    auto result0 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {{LCW{}}},
    };

    auto result1 = cudf::concatenate(std::vector<column_view>({split_a[0], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}},
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
    };

    auto result2 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[0]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}},
      {{LCW{}}},
    };

    auto result3 = cudf::concatenate(std::vector<column_view>({split_a[1], split_b[1]}));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result3, expected3);
  }
}

TEST_F(ListsColumnTest, ListOfStructs)
{
  auto count_iter = cuda::counting_iterator<int>{0};

  // inner structs
  std::vector<cudf::test::structs_column_wrapper> inner_structs;
  {
    // 1. String "names" column.
    std::vector<std::vector<std::string>> names(
      {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
       {},
       {},
       {"Bill", "Bob", "Sam", "Fred", "Tom"}});
    std::vector<std::vector<bool>> names_validity(
      {{true, true, true, true, true, true}, {}, {}, {false, true, false, true, false}});
    std::vector<cudf::test::strings_column_wrapper> name_cols;
    std::transform(
      count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
        return cudf::test::strings_column_wrapper(
          names[i].begin(), names[i].end(), names_validity[i].begin());
      });

    // 2. Numeric "ages" column.
    std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {}, {}, {11, 16, 17, 41, 42}});
    std::vector<std::vector<bool>> ages_validity(
      {{true, true, true, true, false, true}, {}, {}, {true, true, true, false, false}});
    std::vector<cudf::test::fixed_width_column_wrapper<int>> age_cols;
    std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
      return cudf::test::fixed_width_column_wrapper<int>(
        ages[i].begin(), ages[i].end(), ages_validity[i].begin());
    });

    for (size_t idx = 0; idx < names.size(); idx++) {
      std::vector<std::unique_ptr<column>> children;
      children.push_back(name_cols[idx].release());
      children.push_back(age_cols[idx].release());
      inner_structs.emplace_back(std::move(children));
    }
  }

  // build expected output
  auto struct_views = std::vector<column_view>(
    {inner_structs[0], inner_structs[1], inner_structs[2], inner_structs[3]});
  auto expected_child = cudf::concatenate(struct_views);
  cudf::test::fixed_width_column_wrapper<int> offsets_w{0, 1, 1, 1, 1, 4, 6, 6, 6, 10, 11};
  auto expected =
    make_lists_column(10, offsets_w.release(), std::move(expected_child), 0, rmm::device_buffer{});

  // lists
  std::vector<cudf::test::fixed_width_column_wrapper<int>> offsets;
  offsets.push_back({0, 1, 1, 1, 1, 4, 6, 6});
  offsets.push_back({0});
  offsets.push_back({0});
  offsets.push_back({0, 0, 4, 5});

  // concatenate as lists
  std::vector<std::unique_ptr<column>> src;
  for (size_t idx = 0; idx < inner_structs.size(); idx++) {
    int size = static_cast<column_view>(offsets[idx]).size() - 1;
    src.push_back(make_lists_column(
      size, offsets[idx].release(), inner_structs[idx].release(), 0, rmm::device_buffer{}));
  }

  // concatenate
  auto result = cudf::concatenate(std::vector<column_view>({*src[0], *src[1], *src[2], *src[3]}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

struct FixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestAllReps, FixedPointConcatentate)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<decimalXX>;

  auto begin =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return decimalXX{i}; });
  auto const vec = std::vector<decimalXX>(begin, begin + 1000);

  auto const a = fw_wrapper(vec.begin(), /***/ vec.begin() + 300);
  auto const b = fw_wrapper(vec.begin() + 300, vec.begin() + 700);
  auto const c = fw_wrapper(vec.begin() + 700, vec.end());

  auto const results  = cudf::concatenate(std::vector<cudf::column_view>{a, b, c});
  auto const expected = fw_wrapper(vec.begin(), vec.end());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(FixedPointTest, FixedPointConcatentate)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto begin     = cuda::counting_iterator<int32_t>{0};
  auto const vec = std::vector<int32_t>(begin, begin + 1000);

  auto const a = fp_wrapper(vec.begin(), /***/ vec.begin() + 300, scale_type{-2});
  auto const b = fp_wrapper(vec.begin() + 300, vec.begin() + 700, scale_type{-2});
  auto const c = fp_wrapper(vec.begin() + 700, vec.end(), /*****/ scale_type{-2});

  auto const results  = cudf::concatenate(std::vector<cudf::column_view>{a, b, c});
  auto const expected = fp_wrapper(vec.begin(), vec.end(), scale_type{-2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(FixedPointTest, FixedPointScaleMismatch)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto begin     = cuda::counting_iterator<int32_t>{0};
  auto const vec = std::vector<int32_t>(begin, begin + 1000);

  auto const a = fp_wrapper(vec.begin(), /***/ vec.begin() + 300, scale_type{-1});
  auto const b = fp_wrapper(vec.begin() + 300, vec.begin() + 700, scale_type{-2});
  auto const c = fp_wrapper(vec.begin() + 700, vec.end(), /*****/ scale_type{-3});

  EXPECT_THROW(cudf::concatenate(std::vector<cudf::column_view>{a, b, c}), cudf::data_type_error);
}

struct DictionaryConcatTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryConcatTest, StringsKeys)
{
  cudf::test::strings_column_wrapper strings(
    {"eee", "aaa", "ddd", "bbb", "", "", "ccc", "ccc", "ccc", "eee", "aaa"},
    {true, true, true, true, false, true, true, true, true, true, true});
  auto dictionary = cudf::dictionary::encode(strings);

  std::vector<cudf::size_type> splits{0, 2, 2, 5, 5, 7, 7, 7, 7, 11};
  std::vector<cudf::column_view> views = cudf::slice(dictionary->view(), splits);
  // concatenate should recreate the original column
  auto result  = cudf::concatenate(views);
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, strings);
}

template <typename T>
struct DictionaryConcatTestFW : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(DictionaryConcatTestFW, cudf::test::FixedWidthTypes);

TYPED_TEST(DictionaryConcatTestFW, FixedWidthKeys)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> original(
    {20, 10, 0, 5, 15, 15, 10, 5, 20}, {1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto dictionary = cudf::dictionary::encode(original);
  std::vector<cudf::size_type> splits{0, 3, 3, 5, 5, 9};
  std::vector<cudf::column_view> views = cudf::slice(dictionary->view(), splits);
  // concatenated result should equal the original column
  auto result  = cudf::concatenate(views);
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, original);
}

TEST_F(DictionaryConcatTest, ErrorsTest)
{
  cudf::test::strings_column_wrapper strings({"aaa", "ddd", "bbb"});
  auto dictionary1 = cudf::dictionary::encode(strings);
  cudf::test::fixed_width_column_wrapper<int32_t> integers({10, 30, 20});
  auto dictionary2 = cudf::dictionary::encode(integers);
  std::vector<cudf::column_view> views({dictionary1->view(), dictionary2->view()});
  EXPECT_THROW(cudf::concatenate(views), cudf::data_type_error);
  std::vector<cudf::column_view> empty;
  EXPECT_THROW(cudf::concatenate(empty), cudf::logic_error);
}

struct EmptyColumnTest : public cudf::test::BaseFixture {};

TEST_F(EmptyColumnTest, SimpleTest)
{
  std::vector<cudf::column> columns;
  constexpr auto num_copies = 10;
  constexpr auto num_rows   = 10;
  for (auto i = 0; i < num_copies; ++i) {
    columns.emplace_back(cudf::data_type(cudf::type_id::EMPTY),
                         num_rows,
                         rmm::device_buffer{},
                         rmm::device_buffer{},
                         0);
  }

  // Create views from columns
  std::vector<cudf::column_view> views;
  for (auto& col : columns) {
    views.push_back(col.view());
  }
  auto result = cudf::concatenate(views);

  ASSERT_EQ(result->size(), num_copies * num_rows);
  ASSERT_EQ(result->type().id(), cudf::type_id::EMPTY);
}

struct TableOfEmptyColumnsTest : public cudf::test::BaseFixture {};

TEST_F(TableOfEmptyColumnsTest, SimpleTest)
{
  std::vector<cudf::table> tables;
  constexpr auto num_copies  = 10;
  constexpr auto num_rows    = 10;
  constexpr auto num_columns = 10;
  for (auto i = 0; i < num_copies; ++i) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (auto j = 0; j < num_columns; ++j) {
      columns.push_back(std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::EMPTY),
                                                       num_rows,
                                                       rmm::device_buffer{},
                                                       rmm::device_buffer{},
                                                       0));
    }
    tables.emplace_back(std::move(columns));
  }

  // Create views from columns
  std::vector<cudf::table_view> views;
  for (auto& tbl : tables) {
    views.push_back(tbl.view());
  }
  auto result = cudf::concatenate(views);

  ASSERT_EQ(result->num_rows(), num_copies * num_rows);
  ASSERT_EQ(result->num_columns(), num_columns);
  for (auto i = 0; i < num_columns; ++i) {
    ASSERT_EQ(result->get_column(i).type().id(), cudf::type_id::EMPTY);
  }
}
