/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator_tests_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

template <typename T>
struct TypedTableViewTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(TypedTableViewTest, NumericTypesNotBool);

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream);

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorTwoTables)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 4, 3}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const lhs          = cudf::table_view{{col1}};
  auto const rhs          = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 1, 0, 1}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::detail::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorSameTable)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const input_table  = cudf::table_view{{col1}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got      = self_comparison(
    input_table, column_order, cudf::detail::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got =
    self_comparison(input_table,
                    column_order,
                    cudf::detail::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

TYPED_TEST(TypedTableViewTest, TestSortSameTableFromTwoTables)
{
  using data_col   = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int32s_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto const col1      = data_col{5, 2, 7, 1, 3};
  auto const col2      = data_col{};  // empty
  auto const lhs       = cudf::table_view{{col1}};
  auto const empty_rhs = cudf::table_view{{col2}};

  auto const stream = cudf::get_default_stream();
  auto const test_sort =
    [stream](
      auto const& preprocessed, auto const& input, auto const& comparator, auto const& expected) {
      auto const order = sorted_order(
        preprocessed, input.num_rows(), cudf::has_nested_columns(input), comparator, stream);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, order->view());
    };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{3, 1, 4, 0, 2};
    test_sort(preprocessed_lhs,
              lhs,
              cudf::detail::row::lexicographic::physical_element_comparator{},
              expected_lhs);
    test_sort(preprocessed_lhs,
              lhs,
              cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
              expected_lhs);

    auto const expected_empty_rhs = int32s_col{};
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::detail::row::lexicographic::physical_element_comparator{},
              expected_empty_rhs);
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
              expected_empty_rhs);
  };

  // Generate preprocessed data for both lhs and lhs at the same time.
  // Switching order of lhs and rhs tables then sorting them using their preprocessed data should
  // produce exactly the same result.
  {
    auto const [preprocessed_lhs, preprocessed_empty_rhs] =
      cudf::detail::row::lexicographic::preprocessed_table::create(
        lhs, empty_rhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
  {
    auto const [preprocessed_empty_rhs, preprocessed_lhs] =
      cudf::detail::row::lexicographic::preprocessed_table::create(
        empty_rhs, lhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
}

TYPED_TEST(TypedTableViewTest, TestSortSameTableFromTwoTablesWithListsOfStructs)
{
  using data_col    = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using strings_col = cudf::test::strings_column_wrapper;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const col1 = [] {
    auto const get_structs = [] {
      auto child0 = data_col{0, 3, 0, 2};
      auto child1 = strings_col{"a", "c", "a", "b"};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(
      2, int32s_col{0, 2, 4}.release(), get_structs().release(), 0, {});
  }();
  auto const col2 = [] {
    auto const get_structs = [] {
      auto child0 = data_col{};
      auto child1 = strings_col{};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(0, int32s_col{}.release(), get_structs().release(), 0, {});
  }();

  auto const column_order = std::vector{cudf::order::ASCENDING};
  auto const lhs          = cudf::table_view{{*col1}};
  auto const empty_rhs    = cudf::table_view{{*col2}};

  auto const stream = cudf::get_default_stream();
  auto const test_sort =
    [stream](
      auto const& preprocessed, auto const& input, auto const& comparator, auto const& expected) {
      auto const order = sorted_order(
        preprocessed, input.num_rows(), cudf::has_nested_columns(input), comparator, stream);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, order->view());
    };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{1, 0};
    test_sort(preprocessed_lhs,
              lhs,
              cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
              expected_lhs);

    auto const expected_empty_rhs = int32s_col{};
    test_sort(preprocessed_empty_rhs,
              empty_rhs,
              cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
              expected_empty_rhs);

    EXPECT_THROW(test_sort(preprocessed_lhs,
                           lhs,
                           cudf::detail::row::lexicographic::physical_element_comparator{},
                           expected_lhs),
                 cudf::logic_error);
    EXPECT_THROW(test_sort(preprocessed_empty_rhs,
                           empty_rhs,
                           cudf::detail::row::lexicographic::physical_element_comparator{},
                           expected_empty_rhs),
                 cudf::logic_error);
  };

  // Generate preprocessed data for both lhs and lhs at the same time.
  // Switching order of lhs and rhs tables then sorting them using their preprocessed data should
  // produce exactly the same result.
  {
    auto const [preprocessed_lhs, preprocessed_empty_rhs] =
      cudf::detail::row::lexicographic::preprocessed_table::create(
        lhs, empty_rhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
  {
    auto const [preprocessed_empty_rhs, preprocessed_lhs] =
      cudf::detail::row::lexicographic::preprocessed_table::create(
        empty_rhs, lhs, std::vector{cudf::order::ASCENDING}, {}, stream);
    test_sort_two_tables(preprocessed_lhs, preprocessed_empty_rhs);
  }
}

template <typename T>
struct NaNTableViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NaNTableViewTest, cudf::test::FloatingPointTypes);

TYPED_TEST(NaNTableViewTest, TestLexicographicalComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::detail::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 1, 0, 0}};
  auto const sorting_got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorting_expected, sorting_got->view());
}

TYPED_TEST(NaNTableViewTest, TestEqualityComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 1}};
  auto const got      = two_table_equality(
    lhs, rhs, column_order, cudf::detail::row::equality::physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const nan_equal_expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 0, 0, 1}};
  auto const nan_equal_got      = two_table_equality(
    lhs, rhs, column_order, cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(nan_equal_expected, nan_equal_got->view());
}

struct RowOperatorTest : public cudf::test::BaseFixture {};

TEST_F(RowOperatorTest, TestTwoTableComparatorColumnCountCheck)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto left_col1         = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}};
  auto left_col2         = cudf::test::fixed_width_column_wrapper<int32_t>{{3, 4}};
  auto const left_table  = cudf::table_view{{left_col1, left_col2}};
  auto right_col         = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}};
  auto const right_table = cudf::table_view{{right_col}};

  auto left_preprocessed =
    cudf::detail::row::equality::preprocessed_table::create(left_table, stream);
  auto right_preprocessed =
    cudf::detail::row::equality::preprocessed_table::create(right_table, stream);

  EXPECT_THROW(
    cudf::detail::row::equality::two_table_comparator(left_preprocessed, right_preprocessed),
    std::invalid_argument);
}

TEST_F(RowOperatorTest, TestCheckShapeCompatibility)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto left_col1_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}};
  auto left_col2_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{3, 4}};
  auto const left_table  = cudf::table_view{{left_col1_2, left_col2_2}};
  auto right_col_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}};
  auto const right_table = cudf::table_view{{right_col_2}};

  EXPECT_THROW(cudf::detail::row::equality::two_table_comparator(left_table, right_table, stream),
               std::invalid_argument);

  auto int_col           = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}};
  auto const int_table   = cudf::table_view{{int_col}};
  auto float_col         = cudf::test::fixed_width_column_wrapper<float>{{1.0f, 2.0f}};
  auto const float_table = cudf::table_view{{float_col}};

  EXPECT_THROW(cudf::detail::row::equality::two_table_comparator(int_table, float_table, stream),
               std::invalid_argument);

  auto str_col             = cudf::test::strings_column_wrapper({"hello", "world"});
  auto const string_table  = cudf::table_view{{str_col}};
  auto num_col             = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  auto const numeric_table = cudf::table_view{{num_col}};

  EXPECT_THROW(
    cudf::detail::row::equality::two_table_comparator(string_table, numeric_table, stream),
    std::invalid_argument);
}

TEST_F(RowOperatorTest, TestRowHasher64BitHash)
{
  auto const col   = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 123456789}};
  auto const input = cudf::table_view{{col}};

  auto const stream       = cudf::get_default_stream();
  auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher   = cudf::detail::row::hash::row_hasher{preprocessed};
  auto const hasher =
    row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});

  auto results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0}};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{results}.begin<std::uint64_t>(),
                    hasher);

  // Expected values match cuCollections xxhash_64 reference implementation
  // https://github.com/NVIDIA/cuCollections/blob/4f03dcccb3a944594c693aa8cebc89302bbd8e20/tests/utility/hash_test.cu#L134-L137
  auto const expected = cudf::test::fixed_width_column_wrapper<std::uint64_t>{
    {4246796580750024372ul, 15516826743637085169ul, 9462334144942111946ul}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results, expected);
}

TEST_F(RowOperatorTest, TestPrimitiveRowHasher64BitHash)
{
  auto const col   = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 123456789}};
  auto const input = cudf::table_view{{col}};

  auto const stream  = cudf::get_default_stream();
  auto const d_input = cudf::table_device_view::create(input, stream);

  auto const hasher = cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::XXHash_64>(
    cudf::nullate::DYNAMIC{false}, *d_input, static_cast<std::uint64_t>(cudf::DEFAULT_HASH_SEED));

  auto results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0}};

  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{results}.begin<std::uint64_t>(),
                    hasher);

  // Expected values match cuCollections xxhash_64 reference implementation
  // https://github.com/NVIDIA/cuCollections/blob/4f03dcccb3a944594c693aa8cebc89302bbd8e20/tests/utility/hash_test.cu#L134-L137
  auto const expected = cudf::test::fixed_width_column_wrapper<std::uint64_t>{
    {4246796580750024372ul, 15516826743637085169ul, 9462334144942111946ul}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results, expected);
}

TEST_F(RowOperatorTest, TestRowHasherDictionaryColumn)
{
  // Dictionary and equivalent string column should produce identical hashes.
  // This also verifies same logical values get same hashes (e.g., "baz" at rows 0 and 2).
  auto const dict_col =
    cudf::test::dictionary_column_wrapper<std::string>({"baz", "foo", "baz", "bar", "foo"});
  auto const str_col = cudf::test::strings_column_wrapper({"baz", "foo", "baz", "bar", "foo"});

  auto const stream = cudf::get_default_stream();
  auto const dict_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{dict_col}}, stream);
  auto const str_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{str_col}}, stream);

  auto const dict_hasher =
    dict_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});
  auto const str_hasher =
    str_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});

  auto dict_results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}};
  auto str_results  = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}};

  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{dict_results}.begin<std::uint64_t>(),
                    dict_hasher);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{str_results}.begin<std::uint64_t>(),
                    str_hasher);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dict_results, str_results);
}

TEST_F(RowOperatorTest, TestRowHasherDictionaryColumnWithNulls)
{
  auto const dict_col =
    cudf::test::dictionary_column_wrapper<int64_t>({100, 200, 300, 100, 200}, {1, 0, 1, 0, 1});
  auto const int_col =
    cudf::test::fixed_width_column_wrapper<int64_t>({100, 200, 300, 100, 200}, {1, 0, 1, 0, 1});

  auto const stream = cudf::get_default_stream();
  auto const dict_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{dict_col}}, stream);
  auto const int_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{int_col}}, stream);

  auto const dict_hasher =
    dict_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{true});
  auto const int_hasher =
    int_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{true});

  auto dict_results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}};
  auto int_results  = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}};

  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{dict_results}.begin<std::uint64_t>(),
                    dict_hasher);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{int_results}.begin<std::uint64_t>(),
                    int_hasher);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dict_results, int_results);
}
