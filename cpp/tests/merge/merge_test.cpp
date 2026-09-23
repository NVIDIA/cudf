/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>

#include <vector>

template <typename T>
class MergeTest_ : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(MergeTest_, cudf::test::FixedWidthTypes);

TYPED_TEST(MergeTest_, MergeIsZeroWhenShouldNotBeZero)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1({1, 2, 3, 4, 5});
  cudf::test::fixed_width_column_wrapper<TypeParam> rightColWrap1{};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order;
  column_order.push_back(cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(column_order.size(), cudf::null_order::AFTER);

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};
  cudf::table_view expected{{leftColWrap1}};

  auto result = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);

  int expected_len = 5;
  ASSERT_EQ(result->num_rows(), expected_len);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

TYPED_TEST(MergeTest_, MismatchedNumColumns)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1({0, 1, 2, 3});
  columnFactoryT rightColWrap1({0, 1, 2, 3});
  columnFactoryT rightColWrap2({0, 1, 2, 3});

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, MismatchedColumnDypes)
{
  cudf::test::fixed_width_column_wrapper<int32_t> leftColWrap1{{0, 1, 2, 3}};
  cudf::test::fixed_width_column_wrapper<double> rightColWrap1{{0, 1, 2, 3}};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, EmptyKeyColumns)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1({0, 1, 2, 3});
  columnFactoryT rightColWrap1({0, 1, 2, 3});

  std::vector<cudf::size_type> key_cols{};  // empty! this should trigger exception
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, TooManyKeyColumns)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1{0, 1, 2, 3};
  columnFactoryT rightColWrap1{0, 1, 2, 3};

  std::vector<cudf::size_type> key_cols{
    0, 1};  // more keys than columns: this should trigger exception
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, EmptyOrderTypes)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1{0, 1, 2, 3};
  columnFactoryT rightColWrap1{0, 1, 2, 3};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{};  // empty! this should trigger exception
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, TooManyOrderTypes)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1{0, 1, 2, 3};
  columnFactoryT rightColWrap1{0, 1, 2, 3};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING,
    cudf::order::DESCENDING};  // more order types than columns: this should trigger exception
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, MismatchedKeyColumnsAndOrderTypes)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  columnFactoryT leftColWrap1{0, 1, 2, 3};
  columnFactoryT leftColWrap2{0, 1, 2, 3};
  columnFactoryT rightColWrap1{0, 1, 2, 3};
  columnFactoryT rightColWrap2{0, 1, 2, 3};

  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  EXPECT_THROW(cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence),
               cudf::logic_error);
}

TYPED_TEST(MergeTest_, NoInputTables)
{
  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(p_outputTable = cudf::merge({}, {}, {}, {}));
  EXPECT_EQ(p_outputTable->num_columns(), 0);
}

TYPED_TEST(MergeTest_, SingleTableInput)
{
  cudf::size_type inputRows = 40;

  auto sequence = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type>
    colWrap1(sequence, sequence + inputRows);

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{colWrap1}};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(p_outputTable =
                         cudf::merge({left_view}, key_cols, column_order, null_precedence));

  auto input_column_view{left_view.column(0)};
  auto output_column_view{p_outputTable->view().column(0)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_column_view, output_column_view);
}

TYPED_TEST(MergeTest_, MergeTwoEmptyTables)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  columnFactoryT leftColWrap1{};
  columnFactoryT rightColWrap1{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  CUDF_TEST_EXPECT_TABLES_EQUAL(left_view, p_outputTable->view());
}

TYPED_TEST(MergeTest_, MergeWithEmptyColumn)
{
  using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;

  cudf::size_type inputRows = 40;

  auto sequence = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type>
    leftColWrap1(sequence, sequence + inputRows);
  columnFactoryT rightColWrap1{};  // wrapper of empty column <- this might require a (sequence,
                                   // sequence) generator

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type>
    expectedDataWrap1(
      sequence,
      sequence +
        outputRows);  //<- confirmed I can reuse a sequence, wo/ creating overlapping columns!

  auto expected_column_view{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto output_column_view{p_outputTable->view().column(0)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view, output_column_view);
}

TYPED_TEST(MergeTest_, Merge1KeyColumns)
{
  cudf::size_type inputRows = 40;

  auto sequence0 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row;
  });

  auto sequence1 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 1;
    else
      return 2 * row;
  });

  auto sequence2 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return 2 * row + 1;
  });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    leftColWrap1(sequence1, sequence1 + inputRows);
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type>
    leftColWrap2(sequence0, sequence0 + inputRows);

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence2)::value_type>
    rightColWrap1(sequence2, sequence2 + inputRows);
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type>
    rightColWrap2(
      sequence0,
      sequence0 +
        inputRows);  //<- confirmed I can reuse a sequence, wo/ creating overlapping columns!

  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  auto seq_out1 = cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row >= outputRows / 2) ? 1 : 0;
    } else
      return row;
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out1)::value_type>
    expectedDataWrap1(seq_out1, seq_out1 + outputRows);

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row / 2;
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type>
    expectedDataWrap2(seq_out2, seq_out2 + outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeTest_, Merge2KeyColumns)
{
  cudf::size_type inputRows = 40;

  auto sequence1 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row >= inputRows / 2) ? 1 : 0;
    } else
      return row;
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    leftColWrap1(sequence1, sequence1 + inputRows);

  auto sequence2 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (inputRows / 4)) % 2 == 0) ? 1 : 0;
    } else {
      return row * 2;
    }
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence2)::value_type>
    leftColWrap2(sequence2, sequence2 + inputRows);

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    rightColWrap1(sequence1, sequence1 + inputRows);

  auto sequence3 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (inputRows / 4)) % 2 == 0) ? 1 : 0;
    } else
      return (2 * row + 1);
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence3)::value_type>
    rightColWrap2(sequence3, sequence3 + inputRows);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  auto seq_out1 = cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row >= outputRows / 2) ? 1 : 0;
    } else
      return (row / 2);
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out1)::value_type>
    expectedDataWrap1(seq_out1, seq_out1 + outputRows);

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (outputRows / 4)) % 2 == 0) ? 1 : 0;
    } else {
      return (row % 2 == 0 ? row + 1 : row - 1);
    }
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type>
    expectedDataWrap2(seq_out2, seq_out2 + outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeTest_, Merge1KeyNullColumns)
{
  cudf::size_type inputRows = 40;

  // data: 0  2  4  6 | valid: 1 1 1 0
  auto sequence1       = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return 0;  // <- no shortcut to this can avoid compiler errors
    } else {
      return row * 2;
    }
  });
  auto valid_sequence1 = cudf::detail::make_counting_transform_iterator(
    0, [inputRows](auto row) { return (row < inputRows - 1); });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    leftColWrap1(sequence1, sequence1 + inputRows, valid_sequence1);

  // data: 1  3  5  7 | valid: 1 1 1 0
  auto sequence2 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return 1;
    } else
      return (2 * row + 1);
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence2)::value_type>
    rightColWrap1(sequence2,
                  sequence2 + inputRows,
                  valid_sequence1);  // <- recycle valid_seq1, confirmed okay...

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};

  /*Note: default behavior semantics for null_precedence has changed
   *      wrt legacy code:
   *
   * in legacy code missing (default) nulls argument
   * meant nulls are greatest; i.e., null_order::AFTER (not null_order::BEFORE)
   *
   * While new semantics is (see row_operators.cuh: row_lexicographic_comparator::operator() ):
   * null_order null_precedence = _null_precedence == nullptr ?
   *                  null_order::BEFORE: _null_precedence[i];
   *
   * hence missing (default) value meant nulls are smallest
   * null_order::BEFORE (not  null_order::AFTER) (!)
   */
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  cudf::table_view left_view{{leftColWrap1}};
  cudf::table_view right_view{{rightColWrap1}};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
  cudf::size_type const column1TotalNulls =
    a_left_tbl_cview.null_count() + a_right_tbl_cview.null_count();

  // data: 0 1 2 3 4 5 6 7 | valid: 1 1 1 1 1 1 0 0
  auto seq_out1 =
    cudf::detail::make_counting_transform_iterator(0, [outputRows, column1TotalNulls](auto row) {
      if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
        return (row >= (outputRows - column1TotalNulls) / 2) ? 1 : 0;
      } else
        return (row);
    });
  auto valid_seq_out = cudf::detail::make_counting_transform_iterator(
    0,
    [outputRows, column1TotalNulls](auto row) { return (row < (outputRows - column1TotalNulls)); });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out1)::value_type>
    expectedDataWrap1(seq_out1, seq_out1 + outputRows, valid_seq_out);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto output_column_view1{p_outputTable->view().column(0)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
}

TYPED_TEST(MergeTest_, Merge2KeyNullColumns)
{
  cudf::size_type inputRows = 40;

  // data: 0 1 2 3 | valid: 1 1 1 1
  auto sequence1 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row >= inputRows / 2) ? 1 : 0;
    } else
      return (row);
  });
  auto valid_sequence1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto row) { return true; });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    leftColWrap1(sequence1,
                 sequence1 + inputRows,
                 valid_sequence1);  // if left out: valid_sequence defaults to `false`;

  // data: 0 2 4 6 | valid: 1 1 1 1
  auto sequence2 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (inputRows / 4)) % 2 == 0) ? 1 : 0;
    } else {
      return row * 2;
    }
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence2)::value_type>
    leftColWrap2(sequence2, sequence2 + inputRows, valid_sequence1);

  // data: 0 1 2 3 | valid: 1 1 1 1
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>
    rightColWrap1(sequence1,
                  sequence1 + inputRows,
                  valid_sequence1);  // if left out: valid_sequence defaults to `false`;

  // data: 0 1 2 3 | valid: 0 0 0 0
  auto sequence3 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (inputRows / 4)) % 2 == 0) ? 1 : 0;
    } else
      return (row);
  });
  auto valid_sequence0 =
    cudf::detail::make_counting_transform_iterator(0, [](auto row) { return false; });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence3)::value_type>
    rightColWrap2(sequence3, sequence3 + inputRows, valid_sequence0);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::AFTER};

  std::unique_ptr<cudf::table> p_outputTable;
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  cudf::size_type const outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  // data: 0 0 1 1 2 2 3 3 | valid: 1 1 1 1 1 1 1 1
  auto seq_out1 = cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row >= outputRows / 2) ? 1 : 0;
    } else
      return (row / 2);
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out1)::value_type>
    expectedDataWrap1(seq_out1, seq_out1 + outputRows, valid_sequence1);

  // data: 0 0 2 1 4 2 6 3 | valid: 0 1 0 1 0 1 0 1
  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return ((row / (outputRows / 8)) % 2 == 0) ? 1 : 0;
    } else {
      return (row % 2 != 0 ? 2 * (row / 2) : (row / 2));
    }
  });
  auto valid_sequence_out =
    cudf::detail::make_counting_transform_iterator(0, [outputRows](auto row) {
      if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
        return ((row / (outputRows / 4)) % 2 == 1) ? 1 : 0;
      } else {
        return (row % 2 != 0) ? 1 : 0;
      }
    });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type>
    expectedDataWrap2(seq_out2, seq_out2 + outputRows, valid_sequence_out);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeTest_, NMerge1KeyColumns)
{
  cudf::size_type inputRows = 64;

  auto sequence0 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 0;
    else
      return row;
  });

  auto sequence1 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 1;
    else
      return inputRows - row;
  });

  constexpr int num_tables = 63;
  using PairT0 =
    cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence0)::value_type>;
  using PairT1 =
    cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence1)::value_type>;
  std::vector<std::pair<PairT0, PairT1>> facts{};
  std::vector<cudf::table_view> tables{};
  for (int i = 0; i < num_tables; ++i) {
    facts.emplace_back(std::pair(PairT0(sequence0, sequence0 + inputRows),
                                 PairT1(sequence1, sequence1 + inputRows)));
    tables.push_back(cudf::table_view{{facts.back().first, facts.back().second}});
  }
  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable = cudf::merge(tables, key_cols, column_order, null_precedence));

  cudf::size_type const outputRows = inputRows * num_tables;

  auto seq_out1 = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (0);
    } else
      return (row / num_tables);
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out1)::value_type>
    expectedDataWrap1(seq_out1, seq_out1 + outputRows);

  auto seq_out2 = cudf::detail::make_counting_transform_iterator(0, [inputRows](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)
      return 1;
    else
      return inputRows - row / num_tables;
  });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(seq_out2)::value_type>
    expectedDataWrap2(seq_out2, seq_out2 + outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
}

class MergeTest : public cudf::test::BaseFixture {};

TEST_F(MergeTest, KeysWithNulls)
{
  cudf::size_type nrows = 13200;  // Ensures that thrust::merge uses more than one tile/block
  auto data_iter        = cuda::counting_iterator<int32_t>{0};
  auto valids1          = cudf::test::iterators::nulls_at_multiples_of(10);
  cudf::test::fixed_width_column_wrapper<int32_t> data1(data_iter, data_iter + nrows, valids1);
  auto valids2 = cudf::test::iterators::nulls_at_multiples_of(15);
  cudf::test::fixed_width_column_wrapper<int32_t> data2(data_iter, data_iter + nrows, valids2);
  auto all_data = cudf::concatenate(std::vector<cudf::column_view>{{data1, data2}});

  std::vector<cudf::order> column_orders{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedences{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  for (auto co : column_orders)
    for (auto np : null_precedences) {
      std::vector<cudf::order> column_order{co};
      std::vector<cudf::null_order> null_precedence{np};
      auto sorted1 =
        cudf::sort(cudf::table_view({data1}), column_order, null_precedence)->release();
      auto col1 = sorted1.front()->view();
      auto sorted2 =
        cudf::sort(cudf::table_view({data2}), column_order, null_precedence)->release();
      auto col2 = sorted2.front()->view();

      auto result = cudf::merge(
        {cudf::table_view({col1}), cudf::table_view({col2})}, {0}, column_order, null_precedence);
      auto sorted_all =
        cudf::sort(cudf::table_view({all_data->view()}), column_order, null_precedence);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_all->view().column(0), result->view().column(0));
    }
}

TEST_F(MergeTest, Structs)
{
  cudf::test::fixed_width_column_wrapper<int> t0_col0{0, 2, 4, 6, 8};
  cudf::test::strings_column_wrapper t0_scol0{"abc", "def", "ghi", "jkl", "mno"};
  cudf::test::fixed_width_column_wrapper<float> t0_scol1{1, 2, 3, 4, 5};
  cudf::test::structs_column_wrapper t0_col1({t0_scol0, t0_scol1});

  cudf::test::fixed_width_column_wrapper<int> t1_col0{1, 3, 5, 7, 9};
  cudf::test::strings_column_wrapper t1_scol0{"pqr", "stu", "vwx", "yzz", "000"};
  cudf::test::fixed_width_column_wrapper<float> t1_scol1{-1, -2, -3, -4, -5};
  cudf::test::structs_column_wrapper t1_col1({t1_scol0, t1_scol1});

  cudf::table_view t0({t0_col0, t0_col1});
  cudf::table_view t1({t1_col0, t1_col1});

  auto result = cudf::merge({t0, t1}, {0}, {cudf::order::ASCENDING});

  cudf::test::fixed_width_column_wrapper<int> e_col0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::strings_column_wrapper e_scol0{
    "abc", "pqr", "def", "stu", "ghi", "vwx", "jkl", "yzz", "mno", "000"};
  cudf::test::fixed_width_column_wrapper<float> e_scol1{1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
  cudf::test::structs_column_wrapper e_col1({e_scol0, e_scol1});

  cudf::table_view expected({e_col0, e_col1});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *result);
}

TEST_F(MergeTest, StructsWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int> t0_col0{0, 2, 4, 6, 8};
  cudf::test::strings_column_wrapper t0_scol0{{"abc", "def", "", "", "mno"}, {1, 1, 0, 0, 1}};
  cudf::test::fixed_width_column_wrapper<float> t0_scol1{{1, 2, 3, 4, 5}, {0, 1, 0, 0, 1}};
  cudf::test::structs_column_wrapper t0_col1({t0_scol0, t0_scol1}, {1, 0, 1, 0, 0});

  cudf::test::fixed_width_column_wrapper<int> t1_col0{1, 3, 5, 7, 9};
  cudf::test::strings_column_wrapper t1_scol0{"pqr", "stu", "vwx", "yzz", "000"};
  cudf::test::fixed_width_column_wrapper<float> t1_scol1{{-1, -2, -3, -4, -5}, {1, 1, 1, 0, 0}};
  cudf::test::structs_column_wrapper t1_col1({t1_scol0, t1_scol1}, {1, 1, 1, 1, 0});

  cudf::table_view t0({t0_col0, t0_col1});
  cudf::table_view t1({t1_col0, t1_col1});

  auto result = cudf::merge({t0, t1}, {0}, {cudf::order::ASCENDING});

  cudf::test::fixed_width_column_wrapper<int> e_col0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::strings_column_wrapper e_scol0{
    {"abc", "pqr", "", "stu", "", "vwx", "", "yzz", "", "000"}, {1, 1, 0, 1, 0, 1, 0, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<float> e_scol1{{1, -1, 2, -2, 3, -3, 4, -4, 5, -5},
                                                        {0, 1, 0, 1, 0, 1, 0, 0, 0, 0}};
  cudf::test::structs_column_wrapper e_col1({e_scol0, e_scol1}, {1, 1, 0, 1, 1, 1, 0, 1, 0, 0});

  cudf::table_view expected({e_col0, e_col1});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *result);
}

TEST_F(MergeTest, StructsNested)
{
  cudf::test::fixed_width_column_wrapper<int> t0_col0{8, 6, 4, 2, 0};
  cudf::test::strings_column_wrapper t0_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> t0_scol1{5, 4, 3, 2, 1};
  cudf::test::strings_column_wrapper t0_sscol0{"5555", "4444", "333", "22", "1"};
  cudf::test::fixed_width_column_wrapper<float> t0_sscol1{50, 40, 30, 20, 10};
  cudf::test::structs_column_wrapper t0_scol2({t0_sscol0, t0_sscol1});
  cudf::test::structs_column_wrapper t0_col1({t0_scol0, t0_scol1, t0_scol2});

  cudf::test::fixed_width_column_wrapper<int> t1_col0{9, 7, 5, 3, 1};
  cudf::test::strings_column_wrapper t1_scol0{"000", "yzz", "vwx", "stu", "pqr"};
  cudf::test::fixed_width_column_wrapper<float> t1_scol1{-5, -4, -3, -2, -1};
  cudf::test::strings_column_wrapper t1_sscol0{"-5555", "-4444", "-333", "-22", "-1"};
  cudf::test::fixed_width_column_wrapper<float> t1_sscol1{-50, -40, -30, -20, -10};
  cudf::test::structs_column_wrapper t1_scol2({t1_sscol0, t1_sscol1});
  cudf::test::structs_column_wrapper t1_col1({t1_scol0, t1_scol1, t1_scol2});

  cudf::table_view t0({t0_col0, t0_col1});
  cudf::table_view t1({t1_col0, t1_col1});

  auto result = cudf::merge({t0, t1}, {0}, {cudf::order::DESCENDING});

  cudf::test::fixed_width_column_wrapper<int> e_col0{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::test::strings_column_wrapper e_scol0{
    "000", "mno", "yzz", "jkl", "vwx", "ghi", "stu", "def", "pqr", "abc"};
  cudf::test::fixed_width_column_wrapper<float> e_scol1{-5, 5, -4, 4, -3, 3, -2, 2, -1, 1};
  cudf::test::strings_column_wrapper e_sscol0{
    "-5555", "5555", "-4444", "4444", "-333", "333", "-22", "22", "-1", "1"};
  cudf::test::fixed_width_column_wrapper<float> e_sscol1{
    -50, 50, -40, 40, -30, 30, -20, 20, -10, 10};
  cudf::test::structs_column_wrapper e_scol2({e_sscol0, e_sscol1});
  cudf::test::structs_column_wrapper e_col1({e_scol0, e_scol1, e_scol2});

  cudf::table_view expected({e_col0, e_col1});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *result);
}

TEST_F(MergeTest, StructsNestedWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int> t0_col0{8, 6, 4, 2, 0};
  cudf::test::strings_column_wrapper t0_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> t0_scol1{{5, 4, 3, 2, 1}, {1, 1, 0, 1, 1}};
  cudf::test::strings_column_wrapper t0_sscol0{{"5555", "", "333", "22", ""}, {1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<float> t0_sscol1{50, 40, 30, 20, 10};
  cudf::test::structs_column_wrapper t0_scol2({t0_sscol0, t0_sscol1}, {0, 0, 1, 1, 1});
  cudf::test::structs_column_wrapper t0_col1({t0_scol0, t0_scol1, t0_scol2}, {0, 0, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<int> t1_col0{9, 7, 5, 3, 1};
  cudf::test::strings_column_wrapper t1_scol0{"000", "yzz", "vwx", "stu", "pqr"};
  cudf::test::fixed_width_column_wrapper<float> t1_scol1{{-5, -4, -3, -2, -1}, {1, 1, 1, 0, 1}};
  cudf::test::strings_column_wrapper t1_sscol0{{"-5555", "-4444", "-333", "-22", "-1"},
                                               {1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<float> t1_sscol1{-50, -40, -30, -20, -10};
  cudf::test::structs_column_wrapper t1_scol2({t1_sscol0, t1_sscol1}, {1, 1, 1, 1, 0});
  cudf::test::structs_column_wrapper t1_col1({t1_scol0, t1_scol1, t1_scol2});

  cudf::table_view t0({t0_col0, t0_col1});
  cudf::table_view t1({t1_col0, t1_col1});

  auto result = cudf::merge({t0, t1}, {0}, {cudf::order::DESCENDING});

  cudf::test::fixed_width_column_wrapper<int> e_col0{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::test::strings_column_wrapper e_scol0{
    "000", "mno", "yzz", "jkl", "vwx", "ghi", "stu", "def", "pqr", "abc"};
  cudf::test::fixed_width_column_wrapper<float> e_scol1{{-5, 5, -4, 4, -3, 3, -2, 2, -1, 1},
                                                        {1, 1, 1, 1, 1, 0, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper e_sscol0{
    {"-5555", "", "-4444", "", "-333", "333", "-22", "22", "", ""}, {1, 0, 1, 0, 1, 1, 1, 1, 0, 0}};
  cudf::test::fixed_width_column_wrapper<float> e_sscol1{
    -50, 50, -40, 40, -30, 30, -20, 20, -10, 10};
  cudf::test::structs_column_wrapper e_scol2({e_sscol0, e_sscol1}, {1, 0, 1, 0, 1, 1, 1, 1, 0, 1});
  cudf::test::structs_column_wrapper e_col1({e_scol0, e_scol1, e_scol2},
                                            {1, 0, 1, 0, 1, 1, 1, 1, 1, 1});

  cudf::table_view expected({e_col0, e_col1});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *result);
}

using lcw = cudf::test::lists_column_wrapper<int32_t>;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

TEST_F(MergeTest, Lists)
{
  auto col1 = lcw{lcw{1}, lcw{3}, lcw{5}, lcw{7}};
  auto col2 = lcw{lcw{2}, lcw{4}, lcw{6}, lcw{8}};

  auto tbl1 = cudf::table_view{{col1}};
  auto tbl2 = cudf::table_view{{col2}};

  auto result = cudf::merge({tbl1, tbl2}, {0}, {cudf::order::ASCENDING});

  auto expected_col = lcw{lcw{1}, lcw{2}, lcw{3}, lcw{4}, lcw{5}, lcw{6}, lcw{7}, lcw{8}};
  auto expected_tbl = cudf::table_view{{expected_col}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl, *result);
}

TEST_F(MergeTest, NestedListsWithNulls)
{
  auto col1 = lcw{{lcw{lcw{1}}, lcw{lcw{3}}, lcw{lcw{5}}, lcw{lcw{7}}}, null_at(3)};
  auto col2 = lcw{{lcw{lcw{2}}, lcw{lcw{4}}, lcw{lcw{6}}, lcw{lcw{8}}}, null_at(3)};

  auto tbl1 = cudf::table_view{{col1}};
  auto tbl2 = cudf::table_view{{col2}};

  auto result = cudf::merge({tbl1, tbl2}, {0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});

  auto expected_col = lcw{{lcw{lcw{1}},
                           lcw{lcw{2}},
                           lcw{lcw{3}},
                           lcw{lcw{4}},
                           lcw{lcw{5}},
                           lcw{lcw{6}},
                           lcw{lcw{7}},
                           lcw{lcw{8}}},
                          nulls_at({6, 7})};
  auto expected_tbl = cudf::table_view{{expected_col}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl, *result);
}

TEST_F(MergeTest, NestedListsofStructs)
{
  // [ {1},    {2},   {3} ]
  // [ {5}                ]
  // [ {7},    {8}        ]
  // [ {10}               ]
  auto const col1 = [] {
    auto const get_structs = [] {
      auto child0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 5, 7, 8, 10};
      return cudf::test::structs_column_wrapper{{child0}};
    };
    return cudf::make_lists_column(
      4,
      cudf::test::fixed_width_column_wrapper<int32_t>{0, 3, 4, 6, 7}.release(),
      get_structs().release(),
      0,
      {});
  }();

  // [ {4}                ]
  // [ {6}                ]
  // [ {9}                ]
  // [ {11}               ]
  auto const col2 = [] {
    auto const get_structs = [] {
      auto child0 = cudf::test::fixed_width_column_wrapper<int32_t>{4, 6, 9, 11};
      return cudf::test::structs_column_wrapper{{child0}};
    };
    return cudf::make_lists_column(
      4,
      cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4}.release(),
      get_structs().release(),
      0,
      {});
  }();

  auto tbl1 = cudf::table_view{{*col1}};
  auto tbl2 = cudf::table_view{{*col2}};

  auto result = cudf::merge({tbl1, tbl2}, {0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});

  // [ {1},    {2},   {3} ]
  // [ {4}                ]
  // [ {5}                ]
  // [ {6}                ]
  // [ {7},    {8}        ]
  // [ {9}                ]
  // [ {10}               ]
  // [ {11}               ]
  auto const expected_col = [] {
    auto const get_structs = [] {
      auto child0 =
        cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      return cudf::test::structs_column_wrapper{{child0}};
    };
    return cudf::make_lists_column(
      8,
      cudf::test::fixed_width_column_wrapper<int32_t>{0, 3, 4, 5, 6, 8, 9, 10, 11}.release(),
      get_structs().release(),
      0,
      {});
  }();
  auto expected_tbl = cudf::table_view{{*expected_col}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl, *result);
}

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TYPED_TEST_SUITE(FixedPointTestAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestAllReps, FixedPointMerge)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const a       = fp_wrapper<RepType>{{4, 22, 33, 44, 55}, scale_type{-1}};
  auto const b       = fp_wrapper<RepType>{{5, 7, 10}, scale_type{-1}};
  auto const table_a = cudf::table_view(std::vector<cudf::column_view>{a});
  auto const table_b = cudf::table_view(std::vector<cudf::column_view>{b});
  auto const tables  = std::vector<cudf::table_view>{table_a, table_b};

  auto const key_cols = std::vector<cudf::size_type>{0};
  auto const order    = std::vector<cudf::order>{cudf::order::ASCENDING};

  auto const exp       = fp_wrapper<RepType>{{4, 5, 7, 10, 22, 33, 44, 55}, scale_type{-1}};
  auto const exp_table = cudf::table_view(std::vector<cudf::column_view>{exp});

  auto const result = cudf::merge(tables, key_cols, order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exp_table.column(0), result->view().column(0));
}

CUDF_TEST_PROGRAM_MAIN()
