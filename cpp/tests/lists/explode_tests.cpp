/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/lists/explode.hpp>

using FCW = cudf::test::fixed_width_column_wrapper<int32_t>;
using LCW = cudf::test::lists_column_wrapper<int32_t>;

class ExplodeTest : public cudf::test::BaseFixture {};

class ExplodeOuterTest : public cudf::test::BaseFixture {};

template <typename T>
class ExplodeTypedTest : public cudf::test::BaseFixture {};

template <typename T>
class ExplodeOuterTypedTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ExplodeTypedTest, cudf::test::FixedPointTypes);

TYPED_TEST_SUITE(ExplodeOuterTypedTest, cudf::test::FixedPointTypes);

TEST_F(ExplodeTest, Empty)
{
  cudf::table_view t({LCW{}, FCW{}});

  auto ret = cudf::explode(t, 0);

  cudf::table_view expected({FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  auto pos_ret = cudf::explode_position(t, 0);

  cudf::table_view pos_expected({FCW{}, FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NonList)
{
  cudf::table_view t({FCW{100, 200, 300}, FCW{100, 200, 300}});

  EXPECT_THROW(cudf::explode(t, 1), cudf::logic_error);
  EXPECT_THROW(cudf::explode_position(t, 1), cudf::logic_error);
}

TEST_F(ExplodeTest, Basics)
{
  //    a                   b                  c
  //    100                [1, 2, 7]           string0
  //    200                [5, 6]              string1
  //    300                [0, 3]              string2

  FCW a{100, 200, 300};
  LCW b{LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}};
  cudf::test::strings_column_wrapper c{"string0", "string1", "string2"};

  FCW expected_a{100, 100, 100, 200, 200, 300, 300};
  FCW expected_b{1, 2, 7, 5, 6, 0, 3};
  cudf::test::strings_column_wrapper expected_c{
    "string0", "string0", "string0", "string1", "string1", "string2", "string2"};

  cudf::table_view t({a, b, c});
  cudf::table_view expected({expected_a, expected_b, expected_c});

  auto ret = cudf::explode(t, 1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_a, expected_pos_col, expected_b, expected_c});

  auto pos_ret = cudf::explode_position(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, SingleNull)
{
  //    a                   b
  //    null                100
  //    [5, 6]              200
  //    []                  300
  //    [0, 3]              400

  constexpr auto null = 0;

  auto first_invalid = cudf::test::iterators::null_at(0);

  LCW a({LCW{null}, LCW{5, 6}, LCW{}, LCW{0, 3}}, first_invalid);
  FCW b({100, 200, 300, 400});

  FCW expected_a{5, 6, 0, 3};
  FCW expected_b{200, 200, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, Nulls)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    null                200
  //    [0, 3]              300

  constexpr auto null = 0;

  auto valids       = cudf::test::iterators::valids_at_multiples_of(2);
  auto always_valid = cudf::test::iterators::no_nulls();

  LCW a({LCW{1, 2, 7}, LCW{null}, LCW{0, 3}}, valids);
  FCW b({100, 200, 300}, valids);

  FCW expected_a({1, 2, 7, 0, 3});
  FCW expected_b({100, 100, 100, 300, 300}, always_valid);

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInList)
{
  //    a                   b
  //    [1, null, 7]        100
  //    [5, null, 0, null]  200
  //    []                  300
  //    [0, null, 8]        400

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a{
    LCW({1, null, 7}, valids), LCW({5, null, 0, null}, valids), LCW{}, LCW({0, null, 8}, valids)};
  FCW b{100, 200, 300, 400};

  FCW expected_a({1, null, 7, 5, null, 0, null, 0, null, 8},
                 {true, false, true, true, false, true, false, true, false, true});
  FCW expected_b{100, 100, 100, 200, 200, 200, 200, 400, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2, 3, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, Nested)
{
  //    a                      b
  //    [[1, 2], [7, 6, 5]]    100
  //    [[5, 6]]               200
  //    [[0, 3],[],[5],[2, 1]] 300

  LCW a{LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}}};
  FCW b{100, 200, 300};

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}};
  FCW expected_b{100, 100, 200, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2, 3};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NestedNulls)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    null                null
  //    [[0, 3],[5],[2, 1]] 300

  constexpr auto null = 0;

  auto valids       = cudf::test::iterators::valids_at_multiples_of(2);
  auto always_valid = cudf::test::iterators::no_nulls();

  LCW a({LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{null}}, LCW{LCW{0, 3}, LCW{5}, LCW{2, 1}}}, valids);
  FCW b({100, null, 300}, valids);

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{0, 3}, LCW{5}, LCW{2, 1}};
  FCW expected_b({100, 100, 300, 300, 300}, always_valid);

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInNested)
{
  //    a                      b
  //    [[1, null], [7, 6, 5]] 100
  //    [[5, 6]]               200
  //    [[0, 3],[5],[2, null]] 300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, null}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}});
  FCW b({100, 200, 300});

  LCW expected_a{
    LCW({1, null}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, null}, valids)};
  FCW expected_b{100, 100, 200, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NullsInNestedDoubleExplode)
{
  //    a                          b
  //    [[1, null], [], [7, 6, 5]] 100
  //    [[5, 6]]                   200
  //    [[0, 3],[5],[2, null]]     300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a{LCW{LCW({1, null}, valids), LCW{}, LCW{7, 6, 5}},
        LCW{LCW{5, 6}},
        LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}};
  FCW b{100, 200, 300};

  FCW expected_a({1, null, 7, 6, 5, 5, 6, 0, 3, 5, 2, null},
                 {true, false, true, true, true, true, true, true, true, true, true, false});
  FCW expected_b{100, 100, 100, 100, 100, 200, 200, 300, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto first_explode_ret = cudf::explode(t, 0);
  auto ret               = cudf::explode(first_explode_ret->view(), 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(first_explode_ret->view(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, NestedStructs)
{
  //    a                      b
  //    [[1, null], [7, 6, 5]] {100, "100"}
  //    [[5, 6]]               {200, "200"}
  //    [[0, 3],[5],[2, null]] {300, "300"}

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, null}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}});
  FCW b1({100, 200, 300});
  cudf::test::strings_column_wrapper b2{"100", "200", "300"};
  cudf::test::structs_column_wrapper b({b1, b2});

  LCW expected_a{
    LCW({1, null}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, null}, valids)};
  FCW expected_b1{100, 100, 200, 300, 300, 300};
  cudf::test::strings_column_wrapper expected_b2{"100", "100", "200", "300", "300", "300"};
  cudf::test::structs_column_wrapper expected_b({expected_b1, expected_b2});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, ListOfStructsWithEmpties)
{
  //  a           b
  //  [{1}}]      "a"
  //  [{null}]    "b"
  //  [null]      "c"
  //  []          "d"
  //  null        "e"

  constexpr auto null = 0;

  // row 0.  1 struct that contains a single int
  cudf::test::fixed_width_column_wrapper<int32_t> i0{1};
  std::vector<std::unique_ptr<cudf::column>> s0_cols;
  s0_cols.push_back(i0.release());
  cudf::test::structs_column_wrapper s0(std::move(s0_cols));
  cudf::test::fixed_width_column_wrapper<int32_t> off0{0, 1};
  auto row0 = cudf::make_lists_column(1, off0.release(), s0.release(), 0, rmm::device_buffer{});

  // row 1.  1 struct that contains a null value
  cudf::test::fixed_width_column_wrapper<int32_t> i1{{1}, {false}};
  std::vector<std::unique_ptr<cudf::column>> s1_cols;
  s1_cols.push_back(i1.release());
  cudf::test::structs_column_wrapper s1(std::move(s1_cols));
  cudf::test::fixed_width_column_wrapper<int32_t> off1{0, 1};
  auto row1 = cudf::make_lists_column(1, off1.release(), s1.release(), 0, rmm::device_buffer{});

  // row 2.  1 null struct
  cudf::test::fixed_width_column_wrapper<int32_t> i2{0};
  std::vector<std::unique_ptr<cudf::column>> s2_cols;
  s2_cols.push_back(i2.release());
  std::vector<bool> r2_valids{false};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(r2_valids.begin(), r2_valids.end());
  auto s2 = cudf::make_structs_column(1, std::move(s2_cols), null_count, std::move(null_mask));
  cudf::test::fixed_width_column_wrapper<int32_t> off2{0, 1};
  auto row2 = cudf::make_lists_column(1, off2.release(), std::move(s2), 0, rmm::device_buffer{});

  // row 3.  empty list.
  cudf::test::fixed_width_column_wrapper<int32_t> i3{};
  std::vector<std::unique_ptr<cudf::column>> s3_cols;
  s3_cols.push_back(i3.release());
  auto s3 = cudf::make_structs_column(0, std::move(s3_cols), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<int32_t> off3{0, 0};
  auto row3 = cudf::make_lists_column(1, off3.release(), std::move(s3), 0, rmm::device_buffer{});

  // row 4.  null list
  cudf::test::fixed_width_column_wrapper<int32_t> i4{};
  std::vector<std::unique_ptr<cudf::column>> s4_cols;
  s4_cols.push_back(i4.release());
  auto s4 = cudf::make_structs_column(0, std::move(s4_cols), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<int32_t> off4{0, 0};
  std::vector<bool> r4_valids{false};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(r4_valids.begin(), r4_valids.end());
  auto row4 =
    cudf::make_lists_column(1, off4.release(), std::move(s4), null_count, std::move(null_mask));

  // concatenated
  auto final_col =
    cudf::concatenate(std::vector<cudf::column_view>({*row0, *row1, *row2, *row3, *row4}));
  auto s = cudf::test::strings_column_wrapper({"a", "b", "c", "d", "e"}).release();

  cudf::table_view t({final_col->view(), s->view()});

  auto ret = cudf::explode(t, 0);
  auto expected_numeric_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, null, null}, {true, false, false}};

  auto expected_a =
    cudf::test::structs_column_wrapper{{expected_numeric_col}, {true, true, false}}.release();
  auto expected_b = cudf::test::strings_column_wrapper({"a", "b", "c"}).release();

  cudf::table_view expected({expected_a->view(), expected_b->view()});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
  FCW expected_pos_col{0, 0, 0};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b->view()});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TYPED_TEST(ExplodeTypedTest, ListOfStructs)
{
  //  a                        b
  //  [{70, "70"}, {75, "75"}] 100
  //  [{50, "50"}, {55, "55"}] 200
  //  [{35, "35"}, {45, "45"}] 300
  //  [{25, "25"}, {30, "30"}] 400
  //  [{15, "15"}, {20, "20"}] 500

  auto numeric_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  cudf::test::strings_column_wrapper string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};
  auto struct_col = cudf::test::structs_column_wrapper{{numeric_col, string_col}}.release();
  auto a =
    cudf::make_lists_column(5, FCW{0, 2, 4, 6, 8, 10}.release(), std::move(struct_col), 0, {});

  FCW b{100, 200, 300, 400, 500};

  cudf::table_view t({a->view(), b});
  auto ret = cudf::explode(t, 0);

  auto expected_numeric_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  cudf::test::strings_column_wrapper expected_string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};

  auto expected_a =
    cudf::test::structs_column_wrapper{{expected_numeric_col, expected_string_col}}.release();
  FCW expected_b{100, 100, 200, 200, 300, 300, 400, 400, 500, 500};

  cudf::table_view expected({expected_a->view(), expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b});

  auto pos_ret = cudf::explode_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeTest, SlicedList)
{
  //    a                              b
  //    [[1, null],[7, 6, 5]]          100
  //    [[5, 6]]                       200
  //    [[0, 3],[5],[2, null]]         300
  //    [[8, 3],[],[4, null, 1, null]] 400
  //    [[2, 3, 4],[9, 8]]             500

  //    slicing the top 2 rows and the bottom row off

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, 2}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, 1}, valids)},
         LCW{LCW{8, 3}, LCW{}, LCW({4, 3, 1, 2}, valids)},
         LCW{LCW{2, 3, 4}, LCW{9, 8}}});
  FCW b({100, 200, 300, 400, 500});

  LCW expected_a{
    LCW{0, 3}, LCW{5}, LCW({2, null}, valids), LCW{8, 3}, LCW{}, LCW({4, null, 1, null}, valids)};
  FCW expected_b{300, 300, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  auto sliced_t = cudf::slice(t, {2, 4});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(sliced_t[0], 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_position(sliced_t[0], 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Empty)
{
  LCW a{};
  FCW b{};

  cudf::table_view t({LCW{}, FCW{}});

  auto ret = cudf::explode_outer(t, 0);

  FCW expected_a{};
  FCW expected_b{};
  cudf::table_view expected({FCW{}, FCW{}});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
}

TEST_F(ExplodeOuterTest, NonList)
{
  cudf::table_view t({FCW{100, 200, 300}, FCW{100, 200, 300}});

  EXPECT_THROW(cudf::explode_outer(t, 1), cudf::logic_error);
  EXPECT_THROW(cudf::explode_outer_position(t, 1), cudf::logic_error);
}

TEST_F(ExplodeOuterTest, Basics)
{
  //    a                   b                  c
  //    100                [1, 2, 7]           string0
  //    200                [5, 6]              string1
  //    300                [0, 3]              string2

  FCW a{100, 200, 300};
  LCW b{LCW{1, 2, 7}, LCW{5, 6}, LCW{0, 3}};
  cudf::test::strings_column_wrapper c{"string0", "string1", "string2"};

  FCW expected_a{100, 100, 100, 200, 200, 300, 300};
  FCW expected_b{1, 2, 7, 5, 6, 0, 3};
  cudf::test::strings_column_wrapper expected_c{
    "string0", "string0", "string0", "string1", "string1", "string2", "string2"};

  cudf::table_view t({a, b, c});
  cudf::table_view expected({expected_a, expected_b, expected_c});

  auto ret = cudf::explode_outer(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_a, expected_pos_col, expected_b, expected_c});

  auto pos_ret = cudf::explode_outer_position(t, 1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, SingleNull)
{
  //    a              b
  //    null           100
  //    [5, 6]         200
  //    []             300
  //    [0, 3]         400

  constexpr auto null = 0;

  auto first_invalid = cudf::test::iterators::null_at(0);

  LCW a({LCW{null}, LCW{5, 6}, LCW{}, LCW{0, 3}}, first_invalid);
  FCW b({100, 200, 300, 400});

  FCW expected_a{{null, 5, 6, 0, 0, 3}, {false, true, true, false, true, true}};
  FCW expected_b{100, 200, 200, 300, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 0, 1, 0, 0, 1}, {false, true, true, false, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});
  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Nulls)
{
  //    a                   b
  //    [1, 2, 7]           100
  //    null                null
  //    [0, 3]              300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{1, 2, 7}, LCW{null}, LCW{0, 3}}, valids);
  FCW b({100, null, 300}, valids);

  FCW expected_a({1, 2, 7, null, 0, 3}, {true, true, true, false, true, true});
  FCW expected_b({100, 100, 100, null, 300, 300}, {true, true, true, false, true, true});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 2, 0, 0, 1}, {true, true, true, false, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, AllNulls)
{
  //    a            b
  //    null         100
  //    null         200
  //    null         300

  constexpr auto null = 0;

  auto non_valid = cudf::test::iterators::all_nulls();

  LCW a({LCW{null}, LCW{null}, LCW{null}}, non_valid);
  FCW b({100, 200, 300});

  FCW expected_a({null, null, null}, {false, false, false});
  FCW expected_b({100, 200, 300});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 0, 0}, {false, false, false}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, SequentialNulls)
{
  //    a               b
  //    [1, 2, null]    100
  //    [3, 4]          200
  //    []              300
  //    []              400
  //    [5, 6, 7]       500

  constexpr auto null = 0;

  auto third_invalid = cudf::test::iterators::null_at(2);

  LCW a{LCW({1, 2, null}, third_invalid), LCW{3, 4}, LCW{}, LCW{}, LCW{5, 6, 7}};
  FCW b{100, 200, 300, 400, 500};

  FCW expected_a({1, 2, null, 3, 4, null, null, 5, 6, 7},
                 {true, true, false, true, true, false, false, true, true, true});
  FCW expected_b({100, 100, 100, 200, 200, 300, 400, 500, 500, 500});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 2, 0, 1, 0, 0, 0, 1, 2},
                       {true, true, true, true, true, false, false, true, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, MoreEmptyThanData)
{
  //    a               b
  //    [1, 2]          100
  //    []              200
  //    []              300
  //    []              400
  //    []              500
  //    [3]             600

  constexpr auto null = 0;

  LCW a{LCW{1, 2}, LCW{}, LCW{}, LCW{}, LCW{}, LCW{3}};
  FCW b{100, 200, 300, 400, 500, 600};

  FCW expected_a({1, 2, null, null, null, null, 3}, {true, true, false, false, false, false, true});
  FCW expected_b({100, 100, 200, 300, 400, 500, 600});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 0, 0, 0, 0, 0}, {true, true, false, false, false, false, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, TrailingEmptys)
{
  //    a               b
  //    [1, 2]          100
  //    []              200
  //    []              300
  //    []              400
  //    []              500

  constexpr auto null = 0;

  LCW a{LCW{1, 2}, LCW{}, LCW{}, LCW{}, LCW{}};
  FCW b{100, 200, 300, 400, 500};

  FCW expected_a({1, 2, null, null, null, null}, {true, true, false, false, false, false});
  FCW expected_b({100, 100, 200, 300, 400, 500});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 0, 0, 0, 0}, {true, true, false, false, false, false}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, LeadingNulls)
{
  //    a               b
  //    null            100
  //    null            200
  //    null            300
  //    null            400
  //    [1, 2]          500

  constexpr auto null = 0;

  auto valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 4; });

  LCW a({LCW{null}, LCW{null}, LCW{null}, LCW{null}, LCW{1, 2}}, valids);
  FCW b{100, 200, 300, 400, 500};

  FCW expected_a({null, null, null, null, 1, 2}, {false, false, false, false, true, true});
  FCW expected_b({100, 200, 300, 400, 500, 500});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 0, 0, 0, 0, 1}, {false, false, false, false, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInList)
{
  //    a                   b
  //    [1, null, 7]        100
  //    [5, null, 0, null]  200
  //    []                  300
  //    [0, null, 8]        400

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a{
    LCW({1, null, 7}, valids), LCW({5, null, 0, null}, valids), LCW{}, LCW({0, null, 8}, valids)};
  FCW b{100, 200, 300, 400};

  FCW expected_a({1, null, 7, 5, null, 0, null, null, 0, null, 8},
                 {true, false, true, true, false, true, false, false, true, false, true});
  FCW expected_b{100, 100, 100, 200, 200, 200, 200, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 2, 0, 1, 2, 3, 0, 0, 1, 2},
                       {true, true, true, true, true, true, true, false, true, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, Nested)
{
  //    a                      b
  //    [[1, 2], [7, 6, 5]]    100
  //    [[5, 6]]               200
  //    [[0, 3],[],[5],[2, 1]] 300

  LCW a{LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{5, 6}}, LCW{LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}}};
  FCW b{100, 200, 300};

  LCW expected_a{LCW{1, 2}, LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{}, LCW{5}, LCW{2, 1}};
  FCW expected_b{100, 100, 200, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2, 3};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NestedNulls)
{
  //    a                   b
  //    [[1, 2], [7, 6, 5]] 100
  //    [[5, 6]]            200
  //    [[0, 3],[5],[2, 1]] 300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW{1, 2}, LCW{7, 6, 5}}, LCW{LCW{null}}, LCW{LCW{0, 3}, LCW{5}, LCW{2, 1}}}, valids);
  FCW b({100, 200, 300});

  auto expected_valids = cudf::test::iterators::null_at(2);
  LCW expected_a({LCW{1, 2}, LCW{7, 6, 5}, LCW{null}, LCW{0, 3}, LCW{5}, LCW{2, 1}},
                 expected_valids);
  FCW expected_b({100, 100, 200, 300, 300, 300});
  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{{0, 1, 0, 0, 1, 2}, {true, true, false, true, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInNested)
{
  //    a                      b
  //    [[1, null], [7, 6, 5]] 100
  //    [[5, 6]]               200
  //    [[0, 3],[5],[2, null]] 300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, null}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}});
  FCW b({100, 200, 300});

  LCW expected_a{
    LCW({1, null}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, null}, valids)};
  FCW expected_b{100, 100, 200, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NullsInNestedDoubleExplode)
{
  //    a                          b
  //    [[1, null], [], [7, 6, 5]] 100
  //    [[5, 6]]                   200
  //    [[0, 3],[5],[2, null]]     300

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a{LCW{LCW({1, null}, valids), LCW{}, LCW{7, 6, 5}},
        LCW{LCW{5, 6}},
        LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}};
  FCW b{100, 200, 300};

  FCW expected_a({1, null, null, 7, 6, 5, 5, 6, 0, 3, 5, 2, null},
                 {true, false, false, true, true, true, true, true, true, true, true, true, false});
  FCW expected_b{100, 100, 100, 100, 100, 100, 200, 200, 300, 300, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto first_explode_ret = cudf::explode_outer(t, 0);
  auto ret               = cudf::explode_outer(first_explode_ret->view(), 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{
    {0, 1, 0, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1},
    {true, true, false, true, true, true, true, true, true, true, true, true, true}};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(first_explode_ret->view(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, NestedStructs)
{
  //    a                      b
  //    [[1, null], [7, 6, 5]] {100, "100"}
  //    [[5, 6]]               {200, "200"}
  //    [[0, 3],[5],[2, null]] {300, "300"}

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, null}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)}});
  FCW b1({100, 200, 300});
  cudf::test::strings_column_wrapper b2{"100", "200", "300"};
  cudf::test::structs_column_wrapper b({b1, b2});

  LCW expected_a{
    LCW({1, null}, valids), LCW{7, 6, 5}, LCW{5, 6}, LCW{0, 3}, LCW{5}, LCW({2, null}, valids)};
  FCW expected_b1{100, 100, 200, 300, 300, 300};
  cudf::test::strings_column_wrapper expected_b2{"100", "100", "200", "300", "300", "300"};
  cudf::test::structs_column_wrapper expected_b({expected_b1, expected_b2});

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, ListOfStructsWithEmpties)
{
  //  a           b
  //  [{1}}]      "a"
  //  [{null}]    "b"
  //  [null]      "c"
  //  []          "d"
  //  null        "e"

  constexpr auto null = 0;

  // row 0.  1 struct that contains a single int
  cudf::test::fixed_width_column_wrapper<int32_t> i0{1};
  std::vector<std::unique_ptr<cudf::column>> s0_cols;
  s0_cols.push_back(i0.release());
  cudf::test::structs_column_wrapper s0(std::move(s0_cols));
  cudf::test::fixed_width_column_wrapper<int32_t> off0{0, 1};
  auto row0 = cudf::make_lists_column(1, off0.release(), s0.release(), 0, rmm::device_buffer{});

  // row 1.  1 struct that contains a null value
  cudf::test::fixed_width_column_wrapper<int32_t> i1{{1}, {false}};
  std::vector<std::unique_ptr<cudf::column>> s1_cols;
  s1_cols.push_back(i1.release());
  cudf::test::structs_column_wrapper s1(std::move(s1_cols));
  cudf::test::fixed_width_column_wrapper<int32_t> off1{0, 1};
  auto row1 = cudf::make_lists_column(1, off1.release(), s1.release(), 0, rmm::device_buffer{});

  // row 2.  1 null struct
  cudf::test::fixed_width_column_wrapper<int32_t> i2{0};
  std::vector<std::unique_ptr<cudf::column>> s2_cols;
  s2_cols.push_back(i2.release());
  std::vector<bool> r2_valids{false};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(r2_valids.begin(), r2_valids.end());
  auto s2 = cudf::make_structs_column(1, std::move(s2_cols), null_count, std::move(null_mask));
  cudf::test::fixed_width_column_wrapper<int32_t> off2{0, 1};
  auto row2 = cudf::make_lists_column(1, off2.release(), std::move(s2), 0, rmm::device_buffer{});

  // row 3.  empty list.
  cudf::test::fixed_width_column_wrapper<int32_t> i3{};
  std::vector<std::unique_ptr<cudf::column>> s3_cols;
  s3_cols.push_back(i3.release());
  auto s3 = cudf::make_structs_column(0, std::move(s3_cols), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<int32_t> off3{0, 0};
  auto row3 = cudf::make_lists_column(1, off3.release(), std::move(s3), 0, rmm::device_buffer{});

  // row 4.  null list
  cudf::test::fixed_width_column_wrapper<int32_t> i4{};
  std::vector<std::unique_ptr<cudf::column>> s4_cols;
  s4_cols.push_back(i4.release());
  auto s4 = cudf::make_structs_column(0, std::move(s4_cols), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<int32_t> off4{0, 0};
  std::vector<bool> r4_valids{false};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(r4_valids.begin(), r4_valids.end());
  auto row4 =
    cudf::make_lists_column(1, off4.release(), std::move(s4), null_count, std::move(null_mask));

  // concatenated
  auto final_col =
    cudf::concatenate(std::vector<cudf::column_view>({*row0, *row1, *row2, *row3, *row4}));
  auto s = cudf::test::strings_column_wrapper({"a", "b", "c", "d", "e"}).release();

  cudf::table_view t({final_col->view(), s->view()});

  auto ret = cudf::explode_outer(t, 0);

  auto expected_numeric_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, null, null, null, null}, {true, false, false, false, false}};

  auto expected_a =
    cudf::test::structs_column_wrapper{{expected_numeric_col}, {true, true, false, false, false}}
      .release();
  auto expected_b = cudf::test::strings_column_wrapper({"a", "b", "c", "d", "e"}).release();

  cudf::table_view expected({expected_a->view(), expected_b->view()});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
  FCW expected_pos_col{{0, 0, 0, null, null}, {true, true, true, false, false}};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b->view()});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TYPED_TEST(ExplodeOuterTypedTest, ListOfStructs)
{
  //  a                        b
  //  [{70, "70"}, {75, "75"}] 100
  //  [{50, "50"}, {55, "55"}] 200
  //  [{35, "35"}, {45, "45"}] 300
  //  [{25, "25"}, {30, "30"}] 400
  //  [{15, "15"}, {20, "20"}] 500

  auto numeric_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  cudf::test::strings_column_wrapper string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};
  auto struct_col = cudf::test::structs_column_wrapper{{numeric_col, string_col}}.release();
  auto a =
    cudf::make_lists_column(5, FCW{0, 2, 4, 6, 8, 10}.release(), std::move(struct_col), 0, {});

  FCW b{100, 200, 300, 400, 500};

  cudf::table_view t({a->view(), b});
  auto ret = cudf::explode_outer(t, 0);

  auto expected_numeric_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
    {70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
  cudf::test::strings_column_wrapper expected_string_col{
    "70", "75", "50", "55", "35", "45", "25", "30", "15", "20"};

  auto expected_a =
    cudf::test::structs_column_wrapper{{expected_numeric_col, expected_string_col}}.release();
  FCW expected_b{100, 100, 200, 200, 300, 300, 400, 400, 500, 500};

  cudf::table_view expected({expected_a->view(), expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  cudf::table_view pos_expected({expected_pos_col, expected_a->view(), expected_b});

  auto pos_ret = cudf::explode_outer_position(t, 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}

TEST_F(ExplodeOuterTest, SlicedList)
{
  //    a                              b
  //    [[1, null],[7, 6, 5]]          100
  //    [[5, 6]]                       200
  //    [[0, 3],[5],[2, null]]         300
  //    [[8, 3],[],[4, null, 1, null]] 400
  //    [[2, 3, 4],[9, 8]]             500

  //    slicing the top 2 rows and the bottom row off

  constexpr auto null = 0;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  LCW a({LCW{LCW({1, null}, valids), LCW{7, 6, 5}},
         LCW{LCW{5, 6}},
         LCW{LCW{0, 3}, LCW{5}, LCW({2, null}, valids)},
         LCW{LCW{8, 3}, LCW{}, LCW({4, null, 1, null}, valids)},
         LCW{LCW{2, 3, 4}, LCW{9, 8}}});
  FCW b({100, 200, 300, 400, 500});

  LCW expected_a{
    LCW{0, 3}, LCW{5}, LCW({2, null}, valids), LCW{8, 3}, LCW{}, LCW({4, null, 1, null}, valids)};
  FCW expected_b{300, 300, 300, 400, 400, 400};

  cudf::table_view t({a, b});
  auto sliced_t = cudf::slice(t, {2, 4});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode_outer(sliced_t[0], 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);

  FCW expected_pos_col{0, 1, 2, 0, 1, 2};
  cudf::table_view pos_expected({expected_pos_col, expected_a, expected_b});

  auto pos_ret = cudf::explode_outer_position(sliced_t[0], 0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(pos_ret->view(), pos_expected);
}
