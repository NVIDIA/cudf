/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/copying/slice_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>

// Size of the serialized table header that precedes the column entries in the
// packed metadata buffer: version + num_columns + num_rows + pad, four 4-byte fields.
// Must match `serialized_table_header` in cpp/src/copying/pack.cpp.
auto constexpr metadata_header_size = 4 * sizeof(cudf::size_type);

struct PackUnpackTest : public cudf::test::BaseFixture {
  void verify_column_metadata(cudf::column_view const& col,
                              cudf::packed_metadata_view::column_view const& meta)
  {
    EXPECT_EQ(meta.type(), col.type());
    EXPECT_EQ(meta.num_rows(), col.size());
    EXPECT_EQ(meta.null_count(), col.null_count());
    EXPECT_EQ(meta.num_children(), col.num_children());
    for (cudf::size_type i = 0; i < col.num_children(); i++) {
      verify_column_metadata(col.child(i), meta.child(i));
    }
  }

  void verify_metadata(cudf::table_view const& t, cudf::packed_columns const& packed)
  {
    auto view = cudf::packed_metadata_view(*packed.metadata);
    EXPECT_EQ(view.num_columns(), t.num_columns());
    EXPECT_EQ(view.num_rows(), t.num_rows());
    for (cudf::size_type i = 0; i < t.num_columns(); i++) {
      verify_column_metadata(t.column(i), view.column(i));
    }
  }

  void run_test(cudf::table_view const& t)
  {
    // verify pack/unpack works
    auto packed   = cudf::pack(t);
    auto unpacked = cudf::unpack(packed);
    CUDF_TEST_EXPECT_TABLES_EQUAL(t, unpacked);

    // verify packed_metadata_view matches the unpacked table (which reflects
    // the compacted sizes stored in the packed metadata, not the original sliced sizes)
    if (!packed.metadata->empty()) { verify_metadata(unpacked, packed); }

    // verify packed_size returns the correct size
    EXPECT_EQ(cudf::packed_size(t), packed.gpu_data->size());

    // verify pack_metadata itself works
    auto metadata = cudf::pack_metadata(
      unpacked, reinterpret_cast<uint8_t const*>(packed.gpu_data->data()), packed.gpu_data->size());
    EXPECT_EQ(metadata.size(), packed.metadata->size());
    EXPECT_EQ(
      std::equal(metadata.data(), metadata.data() + metadata.size(), packed.metadata->data()),
      true);
  }
  void run_test(std::vector<cudf::column_view> const& t) { run_test(cudf::table_view{t}); }
};

TEST_F(PackUnpackTest, SingleColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, SingleColumnFixedWidthNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col1({1, 2, 3, 4, 5, 6, 7});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, MultiColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> col2({7, 8, 6, 5, 4, 3, 2},
                                                     {true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<double> col3({8, 4, 2, 0, 7, 1, 3},
                                                      {false, true, true, true, true, true, true});

  this->run_test({col1, col2, col3});
}

TEST_F(PackUnpackTest, MultiColumnWithStrings)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::strings_column_wrapper col2({"Lorem", "ipsum", "dolor", "sit", "amet", "ort", "ral"},
                                          {true, false, true, true, true, false, true});
  cudf::test::strings_column_wrapper col3({"", "this", "is", "a", "column", "of", "strings"});

  this->run_test({col1, col2, col3});
}
// clang-format on

TEST_F(PackUnpackTest, EmptyColumns)
{
  {
    auto empty_string = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }

  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }

  {
    cudf::test::fixed_width_column_wrapper<int> col0;
    cudf::test::dictionary_column_wrapper<int> col1;
    cudf::test::strings_column_wrapper col2;
    cudf::test::lists_column_wrapper<int> col3;
    cudf::test::structs_column_wrapper col4({});

    cudf::table_view src_table({col0, col1, col2, col3, col4});
    this->run_test(src_table);
  }
}

std::vector<std::unique_ptr<cudf::column>> generate_lists(bool include_validity)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  if (include_validity) {
    auto valids = cudf::test::iterators::valids_at_multiples_of(2);
    cudf::test::lists_column_wrapper<int> list0{{1, 2, 3},
                                                {4, 5},
                                                {6},
                                                {{7, 8}, valids},
                                                {9, 10, 11},
                                                LCW{},
                                                LCW{},
                                                {{-1, -2, -3, -4, -5}, valids},
                                                {{100, -200}, valids}};

    cudf::test::lists_column_wrapper<int> list1{{{{1, 2, 3}, valids}, {4, 5}},
                                                {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                                {LCW{6}},
                                                {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                                {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                                {LCW{}},
                                                {LCW{-10}, {-100, -200}},
                                                {{-10, -200}, LCW{}, {8, 9}},
                                                {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

    std::vector<std::unique_ptr<cudf::column>> out;
    out.push_back(list0.release());
    out.push_back(list1.release());
    return out;
  }

  cudf::test::lists_column_wrapper<int> list0{
    {1, 2, 3}, {4, 5}, {6}, {7, 8}, {9, 10, 11}, LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-100, -200}};

  cudf::test::lists_column_wrapper<int> list1{{{1, 2, 3}, {4, 5}},
                                              {LCW{}, LCW{}, {7, 8}, LCW{}},
                                              {LCW{6}},
                                              {{7, 8}, {9, 10, 11}, LCW{}},
                                              {LCW{}, {-1, -2, -3, -4, -5}},
                                              {LCW{}},
                                              {{-10}, {-100, -200}},
                                              {{-10, -200}, LCW{}, {8, 9}},
                                              {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(list0.release());
  out.push_back(list1.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_structs(bool include_validity)
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin())
      : cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end());

  // 3. Boolean "is_human" column.
  std::vector<bool> is_human{true, true, false, false, false, false, true, true, true};
  std::vector<bool> is_human_validity{true, true, true, false, true, true, true, true, false};
  auto is_human_col =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<bool>(
          is_human.begin(), is_human.end(), is_human_validity.begin())
      : cudf::test::fixed_width_column_wrapper<bool>(is_human.begin(), is_human.end());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    include_validity
      ? cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col},
                                           struct_validity.begin())
      : cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col});

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_struct_of_list()
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  std::vector<bool> list_validity{true, true, true, true, true, false, true, false, true};
  cudf::test::lists_column_wrapper<cudf::string_view> list(
    {{{"abc", "d", "edf"}, {"jjj"}},
     {{"dgaer", "-7"}, LCW{}},
     {LCW{}},
     {{"qwerty"}, {"ral", "ort", "tal"}, {"five", "six"}},
     {LCW{}, LCW{}, {"eight", "nine"}},
     {LCW{}},
     {{"fun"}, {"a", "bc", "def", "ghij", "klmno", "pqrstu"}},
     {{"seven", "zz"}, LCW{}, {"xyzzy"}},
     {LCW{"negative 3", "  ", "cleveland"}}},
    list_validity.begin());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_list_of_struct()
{
  // 1. String "names" column.
  std::vector<std::string> names{"Vimes",
                                 "Carrot",
                                 "Angua",
                                 "Cheery",
                                 "Detritus",
                                 "Slant",
                                 "Fred",
                                 "Todd",
                                 "Kevin",
                                 "Abc",
                                 "Def",
                                 "Xyz",
                                 "Five",
                                 "Seventeen",
                                 "Dol",
                                 "Est"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102, -1, -2, -3, -4, -5, -6, -7};
  std::vector<bool> ages_validity = {true,
                                     true,
                                     true,
                                     true,
                                     false,
                                     true,
                                     false,
                                     false,
                                     true,
                                     false,
                                     false,
                                     false,
                                     false,
                                     true,
                                     true,
                                     true};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 false,
                                                 false,
                                                 true,
                                                 false,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column}, struct_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{true, true, true, true, true, false, true, false, true};

  cudf::test::fixed_width_column_wrapper<int> offsets{0, 1, 4, 5, 7, 7, 10, 13, 14, 16};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_validity.begin(), list_validity.begin() + 9);
  auto list = [&] {
    auto tmp = cudf::make_lists_column(
      9, offsets.release(), struct_column.release(), null_count, std::move(null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(std::move(list));
  return out;
}

TEST_F(PackUnpackTest, Lists)
{
  // lists
  {
    auto cols = generate_lists(false);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // lists with validity
  {
    auto cols = generate_lists(true);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, Structs)
{
  // structs
  {
    auto cols = generate_structs(false);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // structs with validity
  {
    auto cols = generate_structs(true);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, NestedTypes)
{
  // build one big table containing, lists, structs, structs<list>, list<struct>
  std::vector<cudf::column_view> col_views;

  auto lists = generate_lists(true);
  std::transform(
    lists.begin(),
    lists.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto structs = generate_structs(true);
  std::transform(
    structs.begin(),
    structs.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto struct_of_list = generate_struct_of_list();
  std::transform(
    struct_of_list.begin(),
    struct_of_list.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto list_of_struct = generate_list_of_struct();
  std::transform(
    list_of_struct.begin(),
    list_of_struct.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  cudf::table_view src_table(col_views);
  this->run_test(src_table);
}

TEST_F(PackUnpackTest, NestedEmpty)
{
  // this produces an empty strings column with no children,
  // nested inside a list
  {
    auto empty_string = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty strings column with children that have no data,
  // nested inside a list
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty struct column with children that have no data,
  // nested inside a list
  {
    cudf::test::fixed_width_column_wrapper<int> ints{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> floats{4, 3, 2, 1, 0};
    auto struct_column = cudf::test::structs_column_wrapper({ints, floats});
    auto empty_struct  = cudf::empty_like(struct_column);
    auto offsets       = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list          = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_struct), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, NestedSliced)
{
  // list
  {
    auto valids = cudf::test::iterators::valids_at_multiples_of(2);

    using LCW = cudf::test::lists_column_wrapper<int>;

    cudf::test::lists_column_wrapper<int> col0{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{6, 12}},
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    cudf::test::strings_column_wrapper col1{
      "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred"};
    cudf::test::fixed_width_column_wrapper<float> col2{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::make_unique<cudf::column>(col2));
    children.push_back(std::make_unique<cudf::column>(col0));
    children.push_back(std::make_unique<cudf::column>(col1));
    auto col3 = cudf::make_structs_column(
      static_cast<cudf::column_view>(col0).size(), std::move(children), 0, rmm::device_buffer{});

    cudf::table_view t({col0, col1, col2, *col3});
    this->run_test(t);
  }

  // struct
  {
    cudf::test::fixed_width_column_wrapper<int> a{0, 1, 2, 3, 4, 5, 6, 7};
    cudf::test::fixed_width_column_wrapper<float> b{
      {0, -1, -2, -3, -4, -5, -6, -7}, {true, true, true, false, false, false, false, true}};
    cudf::test::strings_column_wrapper c{{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"},
                                         {false, false, true, true, true, true, true, true}};
    std::vector<bool> list_validity{true, false, true, false, true, false, true, true};
    cudf::test::lists_column_wrapper<int16_t> d{
      {{0, 1}, {2, 3, 4}, {5, 6}, {7}, {8, 9, 10}, {11, 12}, {}, {15, 16, 17}},
      list_validity.begin()};
    cudf::test::fixed_width_column_wrapper<int> _a{10, 20, 30, 40, 50, 60, 70, 80};
    cudf::test::fixed_width_column_wrapper<float> _b{-10, -20, -30, -40, -50, -60, -70, -80};
    cudf::test::strings_column_wrapper _c{"aa", "", "ccc", "dddd", "eeeee", "f", "gg", "hhh"};
    cudf::test::structs_column_wrapper e({_a, _b, _c},
                                         {true, true, true, false, true, true, true, false});
    cudf::test::structs_column_wrapper s({a, b, c, d, e},
                                         {true, true, false, true, true, true, true, true});

    auto split = cudf::split(s, {2, 5});

    this->run_test(cudf::table_view({split[0]}));
    this->run_test(cudf::table_view({split[1]}));
    this->run_test(cudf::table_view({split[2]}));
  }
}

TEST_F(PackUnpackTest, EmptyTable)
{
  // no columns
  {
    cudf::table_view t;
    this->run_test(t);
  }

  // no rows
  {
    cudf::test::fixed_width_column_wrapper<int> a;
    cudf::test::strings_column_wrapper b;
    cudf::test::lists_column_wrapper<float> c;
    cudf::table_view t({a, b, c});
    this->run_test(t);
  }
}

TEST_F(PackUnpackTest, ZeroColumnsWithRows)
{
  // A zero-column table with rows survives a pack/unpack round-trip.
  cudf::table_view t{std::vector<cudf::column_view>{}, 7};
  auto unpacked = cudf::unpack(cudf::pack(t));
  EXPECT_EQ(unpacked.num_columns(), 0);
  EXPECT_EQ(unpacked.num_rows(), 7);

  // A genuinely empty (0, 0) table round-trips to (0, 0).
  cudf::table_view empty{std::vector<cudf::column_view>{}, 0};
  auto unpacked_empty = cudf::unpack(cudf::pack(empty));
  EXPECT_EQ(unpacked_empty.num_columns(), 0);
  EXPECT_EQ(unpacked_empty.num_rows(), 0);
}

TEST_F(PackUnpackTest, SlicedEmpty)
{
  // empty sliced column. this is specifically testing the corner case:
  // - a sliced column of size 0
  // - having children that are of size > 0
  //
  cudf::test::strings_column_wrapper a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<int> b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper _a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<float> _b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> _c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::structs_column_wrapper d({_a, _b, _c});

  cudf::table_view t({a, b, c, d});

  auto sliced   = cudf::split(t, {0});
  auto packed   = cudf::pack(t);
  auto unpacked = cudf::unpack(packed);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t, unpacked);
}

TEST_F(PackUnpackTest, LongOffsets)
{
  auto str = make_long_offsets_string_column();
  cudf::table_view tbl({*str});
  this->run_test(tbl);
}

TEST_F(PackUnpackTest, DISABLED_LongOffsetsAndChars)
{
  auto str = make_long_offsets_and_chars_string_column();
  cudf::table_view tbl({*str});
  this->run_test(tbl);
}

TEST_F(PackUnpackTest, MetadataViewRejectsNonMultipleSize)
{
  // Pack a valid table, then present its metadata with one byte chopped off
  // so the size is not a multiple of the serialized entry size.
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));
  ASSERT_GT(packed.metadata->size(), 1);
  auto truncated = std::span<uint8_t const>(packed.metadata->data(), packed.metadata->size() - 1);
  EXPECT_THROW(cudf::packed_metadata_view{truncated}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsTruncatedBuffer)
{
  // Pack a multi-column table, then lop off one serialized entry so
  // the header claims more columns than the buffer actually contains.
  cudf::test::fixed_width_column_wrapper<int> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col2{4.0f, 5.0f, 6.0f};
  cudf::test::fixed_width_column_wrapper<double> col3{7.0, 8.0, 9.0};
  auto packed = cudf::pack(cudf::table_view({col1, col2, col3}));

  // Metadata has a table header plus 3 column entries. Remove the last entry so
  // the header still says "3 columns" but only 2 column entries remain.
  auto const entry_size     = (packed.metadata->size() - metadata_header_size) / 3;
  auto const truncated_size = packed.metadata->size() - entry_size;

  auto truncated = std::span<uint8_t const>(packed.metadata->data(), truncated_size);
  EXPECT_THROW(cudf::packed_metadata_view{truncated}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsTooLongBuffer)
{
  // Pack a valid table, then extend the buffer by one entry so the tree
  // doesn't consume the entire buffer.
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto const entry_size = packed.metadata->size() - metadata_header_size;  // 1 column entry
  auto extended         = *packed.metadata;
  extended.resize(packed.metadata->size() + entry_size, 0);

  EXPECT_THROW(cudf::packed_metadata_view{extended}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsCorruptedChildCount)
{
  // Pack a table with a struct column, then corrupt the struct's num_children
  // field so traversal would read past the buffer.
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));

  // The metadata layout is: [header, struct, ints_child, floats_child].
  // The struct entry is the first column entry. We corrupt its num_children from 2 to
  // something larger so the tree claims more entries than exist.
  auto corrupted = *packed.metadata;

  // The num_children field is the
  // second-to-last 4-byte value in each entry (before the trailing pad).
  auto const entry_size = (corrupted.size() - metadata_header_size) / 3;  // 3 column entries
  auto const num_children_offset = metadata_header_size                   // skip table header
                                   + entry_size - 2 * sizeof(int32_t);    // num_children in struct
  cudf::size_type bad_children = 10;
  std::memcpy(corrupted.data() + num_children_offset, &bad_children, sizeof(bad_children));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsNegativeColumnCount)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted = *packed.metadata;
  // num_columns follows the leading version field in the header.
  auto constexpr num_columns_offset = sizeof(std::int32_t);
  cudf::size_type const negative    = -1;
  std::memcpy(corrupted.data() + num_columns_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

// num_rows follows the leading version and num_columns fields in the header.
auto constexpr num_rows_offset = 2 * sizeof(std::int32_t);

TEST_F(PackUnpackTest, MetadataRejectsNegativeRowCount)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted                 = *packed.metadata;
  cudf::size_type const negative = -1;
  std::memcpy(corrupted.data() + num_rows_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsRowCountInconsistentWithColumns)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  // The column has 3 rows; a header row count that disagrees must be rejected.
  auto corrupted              = *packed.metadata;
  cudf::size_type const wrong = 99;
  std::memcpy(corrupted.data() + num_rows_offset, &wrong, sizeof(wrong));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsRowCountInconsistentAcrossColumns)
{
  cudf::test::fixed_width_column_wrapper<int> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int> col2{4, 5, 6};
  auto packed = cudf::pack(cudf::table_view({col1, col2}));

  // Leave the header and column 0 at 3 rows but corrupt column 1's size to 4. Validating only
  // the first column would miss this; every top-level column must match the recorded row count.
  auto corrupted        = *packed.metadata;
  auto const entry_size = (corrupted.size() - metadata_header_size) / 2;  // 2 column entries
  // size is the first field after the 8-byte data_type (two int32s) in each entry.
  auto const col1_size_offset = metadata_header_size + entry_size + 2 * sizeof(std::int32_t);
  cudf::size_type const wrong = 4;
  std::memcpy(corrupted.data() + col1_size_offset, &wrong, sizeof(wrong));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsUnsupportedVersion)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted = *packed.metadata;
  // The version is the leading value of the header.
  std::int32_t const unknown_version = 999;
  std::memcpy(corrupted.data(), &unknown_version, sizeof(unknown_version));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsNegativeChildCount)
{
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));

  auto corrupted        = *packed.metadata;
  auto const entry_size = (corrupted.size() - metadata_header_size) / 3;  // 3 column entries
  auto const num_children_offset = metadata_header_size + entry_size - 2 * sizeof(int32_t);
  cudf::size_type const negative = -1;
  std::memcpy(corrupted.data() + num_children_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewColumnIndexOutOfRange)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));
  auto view   = cudf::packed_metadata_view(*packed.metadata);

  EXPECT_THROW(std::ignore = view.column(-1), std::out_of_range);
  EXPECT_THROW(std::ignore = view.column(view.num_columns()), std::out_of_range);
}

TEST_F(PackUnpackTest, MetadataViewChildIndexOutOfRange)
{
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));
  auto view       = cudf::packed_metadata_view(*packed.metadata);
  auto col_meta   = view.column(0);

  EXPECT_THROW(std::ignore = col_meta.child(-1), std::out_of_range);
  EXPECT_THROW(std::ignore = col_meta.child(col_meta.num_children()), std::out_of_range);
}
