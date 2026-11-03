/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/merge.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class MergeStringTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(MergeStringTest, cudf::test::FixedWidthTypes);

TYPED_TEST(MergeStringTest, Merge1StringKeyColumns)
{
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"});
  cudf::size_type inputRows1 = static_cast<cudf::column_view const&>(leftColWrap1).size();

  auto sequence0 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row;
  });

  fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type> leftColWrap2(
    sequence0, sequence0 + inputRows1);

  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"});
  cudf::size_type inputRows2 = static_cast<cudf::column_view const&>(rightColWrap1).size();
  fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type> rightColWrap2(
    sequence0, sequence0 + inputRows2);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable =
                    cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  strings_column_wrapper expectedDataWrap1({"ab",
                                            "ac",
                                            "bc",
                                            "bd",
                                            "cd",
                                            "ce",
                                            "de",
                                            "df",
                                            "ef",
                                            "eg",
                                            "fg",
                                            "fh",
                                            "gh",
                                            "gi",
                                            "hi",
                                            "hj"});

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row / 2;
  });
  fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type> expectedDataWrap2(
    seq_out2, seq_out2 + outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

// rename test <TestName> as DISABLED_<TestName> to disable:
// Example: TYPED_TEST(MergeStringTest, DISABLED_Merge2StringKeyColumns)
//
TYPED_TEST(MergeStringTest, Merge2StringKeyColumns)
{
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"});
  strings_column_wrapper leftColWrap3({"zy", "yx", "xw", "wv", "vu", "ut", "ts", "sr"});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(leftColWrap3).size());

  auto sequence_l = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 1;
    else
      return 2 * row;
  });

  fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type> leftColWrap2(
    sequence_l, sequence_l + inputRows);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2, leftColWrap3}};

  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap1).size());

  auto sequence_r = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return 2 * row + 1;
  });
  fixed_width_column_wrapper<TypeParam, typename decltype(sequence_r)::value_type> rightColWrap2(
    sequence_r, sequence_r + inputRows);

  strings_column_wrapper rightColWrap3({"zx", "yw", "xv", "wu", "vt", "us", "tr", "sp"});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap3).size());

  cudf::table_view right_view{{rightColWrap1, rightColWrap2, rightColWrap3}};

  std::vector<cudf::size_type> key_cols{0, 2};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable =
                    cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
  strings_column_wrapper expectedDataWrap1({"ab",
                                            "ac",
                                            "bc",
                                            "bd",
                                            "cd",
                                            "ce",
                                            "de",
                                            "df",
                                            "ef",
                                            "eg",
                                            "fg",
                                            "fh",
                                            "gh",
                                            "gi",
                                            "hi",
                                            "hj"});

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });
  fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type> expectedDataWrap2(
    seq_out2, seq_out2 + outputRows);

  strings_column_wrapper expectedDataWrap3({"zy",
                                            "zx",
                                            "yx",
                                            "yw",
                                            "xw",
                                            "xv",
                                            "wv",
                                            "wu",
                                            "vu",
                                            "vt",
                                            "ut",
                                            "us",
                                            "ts",
                                            "tr",
                                            "sr",
                                            "sp"});

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
  auto expected_column_view3{static_cast<cudf::column_view const&>(expectedDataWrap3)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};
  auto output_column_view3{p_outputTable->view().column(2)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view3, output_column_view3);
}

TYPED_TEST(MergeStringTest, Merge1StringKeyNullColumns)
{
  // data: "ab", "bc", "cd", "de" | valid: 1 1 1 0
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"},
                                      {true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  auto sequence0 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row;
  });

  fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type> leftColWrap2(
    sequence0, sequence0 + inputRows);
  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};

  // data: "ac", "bd", "ce", "df" | valid: 1 1 1 0
  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"},
                                       {true, true, true, true, true, true, true, false});
  fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type> rightColWrap2(
    sequence0, sequence0 + inputRows);

  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  std::unique_ptr<cudf::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable =
                    cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  // data: "ab", "ac", "bc", "bd", "cd", "ce", "de", "df" | valid: 1 1 1 1 1 1 0 0
  strings_column_wrapper expectedDataWrap1({"ab",
                                            "ac",
                                            "bc",
                                            "bd",
                                            "cd",
                                            "ce",
                                            "de",
                                            "df",
                                            "ef",
                                            "eg",
                                            "fg",
                                            "fh",
                                            "gh",
                                            "gi",
                                            "hi",
                                            "hj"},
                                           {true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            false,
                                            false});
  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row / 2;
  });
  fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type> expectedDataWrap2(
    seq_out2, seq_out2 + outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeStringTest, Merge2StringKeyNullColumns)
{
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"},
                                      {true, true, true, true, true, true, true, false});
  strings_column_wrapper leftColWrap3({"zy", "yx", "xw", "wv", "vu", "ut", "ts", "sr"},
                                      {true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(leftColWrap3).size());

  auto sequence_l = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 1;
    else
      return 2 * row;
  });

  fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type> leftColWrap2(
    sequence_l, sequence_l + inputRows);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2, leftColWrap3}};

  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"},
                                       {true, true, true, true, true, true, true, false});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap1).size());

  auto sequence_r = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return 2 * row + 1;
  });
  fixed_width_column_wrapper<TypeParam, typename decltype(sequence_r)::value_type> rightColWrap2(
    sequence_r, sequence_r + inputRows);

  strings_column_wrapper rightColWrap3({"zx", "yw", "xv", "wu", "vt", "us", "tr", "sp"},
                                       {true, true, true, true, true, true, true, false});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap3).size());

  cudf::table_view right_view{{rightColWrap1, rightColWrap2, rightColWrap3}};

  std::vector<cudf::size_type> key_cols{0, 2};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  std::unique_ptr<cudf::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable =
                    cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
  strings_column_wrapper expectedDataWrap1({"ab",
                                            "ac",
                                            "bc",
                                            "bd",
                                            "cd",
                                            "ce",
                                            "de",
                                            "df",
                                            "ef",
                                            "eg",
                                            "fg",
                                            "fh",
                                            "gh",
                                            "gi",
                                            "hi",
                                            "hj"},
                                           {true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            false,
                                            false});

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type> expectedDataWrap2(
    seq_out2, seq_out2 + outputRows);

  strings_column_wrapper expectedDataWrap3({"zy",
                                            "zx",
                                            "yx",
                                            "yw",
                                            "xw",
                                            "xv",
                                            "wv",
                                            "wu",
                                            "vu",
                                            "vt",
                                            "ut",
                                            "us",
                                            "ts",
                                            "tr",
                                            "sr",
                                            "sp"},
                                           {true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            true,
                                            false,
                                            false});

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
  auto expected_column_view3{static_cast<cudf::column_view const&>(expectedDataWrap3)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};
  auto output_column_view3{p_outputTable->view().column(2)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view3, output_column_view3);
}
