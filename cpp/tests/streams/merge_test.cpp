/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

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

  auto result = cudf::merge({left_view, right_view},
                            key_cols,
                            column_order,
                            null_precedence,
                            cudf::test::get_default_stream());

  int expected_len = 5;
  ASSERT_EQ(result->num_rows(), expected_len);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
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
  CUDF_EXPECT_NO_THROW(
    p_outputTable = cudf::merge(
      {left_view}, key_cols, column_order, null_precedence, cudf::test::get_default_stream()));

  auto input_column_view{left_view.column(0)};
  auto output_column_view{p_outputTable->view().column(0)};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_column_view, output_column_view);
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
  auto all_data = cudf::concatenate(std::vector<cudf::column_view>{{data1, data2}},
                                    cudf::test::get_default_stream());

  std::vector<cudf::order> column_orders{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedences{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  for (auto co : column_orders)
    for (auto np : null_precedences) {
      std::vector<cudf::order> column_order{co};
      std::vector<cudf::null_order> null_precedence{np};
      auto sorted1 = cudf::sort(cudf::table_view({data1}),
                                column_order,
                                null_precedence,
                                cudf::test::get_default_stream())
                       ->release();
      auto col1    = sorted1.front()->view();
      auto sorted2 = cudf::sort(cudf::table_view({data2}),
                                column_order,
                                null_precedence,
                                cudf::test::get_default_stream())
                       ->release();
      auto col2 = sorted2.front()->view();

      auto result     = cudf::merge({cudf::table_view({col1}), cudf::table_view({col2})},
                                    {0},
                                column_order,
                                null_precedence,
                                cudf::test::get_default_stream());
      auto sorted_all = cudf::sort(cudf::table_view({all_data->view()}),
                                   column_order,
                                   null_precedence,
                                   cudf::test::get_default_stream());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_all->view().column(0), result->view().column(0));
    }
}

CUDF_TEST_PROGRAM_MAIN()
