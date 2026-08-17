/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
struct TypedTableViewTest : public cudf::test::BaseFixtureWithHarness {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(TypedTableViewTest, NumericTypesNotBool);

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator,
                                              rmm::cuda_stream_view stream,
                                              cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator,
                                                   rmm::cuda_stream_view stream,
                                                   cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator,
                                                 rmm::cuda_stream_view stream,
                                                 cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorTwoTables)
{
  using T = TypeParam;

  // TODO: lexicographic row operators still allocate from the current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}, stream, mr};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 4, 3}, stream, mr};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const lhs          = cudf::table_view{{col1}};
  auto const rhs          = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 1, 0, 1}, stream, mr};
  auto const got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::physical_element_comparator{},
                         stream,
                         mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

  auto const sorting_got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
                         stream,
                         mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, sorting_got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorSameTable)
{
  using T = TypeParam;

  // TODO: lexicographic row operators still allocate from the current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}, stream, mr};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const input_table  = cudf::table_view{{col1}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}, stream, mr};
  auto const got      = self_comparison(input_table,
                                   column_order,
                                   cudf::detail::row::lexicographic::physical_element_comparator{},
                                   stream,
                                   mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

  auto const sorting_got =
    self_comparison(input_table,
                    column_order,
                    cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
                    stream,
                    mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, sorting_got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(TypedTableViewTest, TestSortSameTableFromTwoTables)
{
  using data_col   = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

  // TODO: lexicographic row operators still allocate from the current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1      = data_col{{5, 2, 7, 1, 3}, stream, mr};
  auto const col2      = data_col{};  // empty
  auto const lhs       = cudf::table_view{{col1}};
  auto const empty_rhs = cudf::table_view{{col2}};

  auto const test_sort =
    [stream, mr](
      auto const& preprocessed, auto const& input, auto const& comparator, auto const& expected) {
      auto const order = sorted_order(
        preprocessed, input.num_rows(), cudf::has_nested_columns(input), comparator, stream, mr);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(
        expected, order->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{{3, 1, 4, 0, 2}, stream, mr};
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
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;
  using structs_col = cudf::test::structs_column_wrapper;

  // TODO: lexicographic row operators still allocate from the current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1 = [stream, mr] {
    auto const get_structs = [stream, mr] {
      auto child0 = data_col{{0, 3, 0, 2}, stream, mr};
      auto child1 = strings_col{{"a", "c", "a", "b"}, stream, mr};
      return structs_col{{child0, child1}, {}, stream, mr};
    };
    return cudf::make_lists_column(
      2, int32s_col{{0, 2, 4}, stream, mr}.release(), get_structs().release(), 0, {});
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

  auto const test_sort =
    [stream, mr](
      auto const& preprocessed, auto const& input, auto const& comparator, auto const& expected) {
      auto const order = sorted_order(
        preprocessed, input.num_rows(), cudf::has_nested_columns(input), comparator, stream, mr);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(
        expected, order->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    };

  auto const test_sort_two_tables = [&](auto const& preprocessed_lhs,
                                        auto const& preprocessed_empty_rhs) {
    auto const expected_lhs = int32s_col{{1, 0}, stream, mr};
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
struct NaNTableViewTest : public cudf::test::BaseFixtureWithHarness {};

TYPED_TEST_SUITE(NaNTableViewTest, cudf::test::FloatingPointTypes);

TYPED_TEST(NaNTableViewTest, TestLexicographicalComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  // TODO: lexicographic row operators still allocate from the current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1 =
    cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}, stream, mr};
  auto const col2 =
    cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}, stream, mr};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}, stream, mr};
  auto const got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::physical_element_comparator{},
                         stream,
                         mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

  auto const sorting_expected =
    cudf::test::fixed_width_column_wrapper<bool>{{0, 1, 0, 0}, stream, mr};
  auto const sorting_got =
    two_table_comparison(lhs,
                         rhs,
                         column_order,
                         cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
                         stream,
                         mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    sorting_expected, sorting_got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(NaNTableViewTest, TestEqualityComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col1 =
    cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}, stream, mr};
  auto const col2 =
    cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}, stream, mr};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 1}, stream, mr};
  auto const got      = two_table_equality(lhs,
                                      rhs,
                                      column_order,
                                      cudf::detail::row::equality::physical_equality_comparator{},
                                      stream,
                                      mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, got->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

  auto const nan_equal_expected =
    cudf::test::fixed_width_column_wrapper<bool>{{1, 0, 0, 1}, stream, mr};
  auto const nan_equal_got =
    two_table_equality(lhs,
                       rhs,
                       column_order,
                       cudf::detail::row::equality::nan_equal_physical_equality_comparator{},
                       stream,
                       mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(nan_equal_expected,
                                 nan_equal_got->view(),
                                 cudf::test::debug_output_level::FIRST_ERROR,
                                 stream,
                                 mr);
}

struct RowOperatorTest : public cudf::test::BaseFixtureWithHarness {};

TEST_F(RowOperatorTest, TestTwoTableComparatorColumnCountCheck)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto left_col1         = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, stream, mr};
  auto left_col2         = cudf::test::fixed_width_column_wrapper<int32_t>{{3, 4}, stream, mr};
  auto const left_table  = cudf::table_view{{left_col1, left_col2}};
  auto right_col         = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, stream, mr};
  auto const right_table = cudf::table_view{{right_col}};

  auto left_preprocessed = cudf::detail::row::equality::preprocessed_table::create(
    left_table, stream, mr.get_temporary_mr());
  auto right_preprocessed = cudf::detail::row::equality::preprocessed_table::create(
    right_table, stream, mr.get_temporary_mr());

  EXPECT_THROW(
    cudf::detail::row::equality::two_table_comparator(left_preprocessed, right_preprocessed),
    std::invalid_argument);
}

TEST_F(RowOperatorTest, TestCheckShapeCompatibility)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto left_col1_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, stream, mr};
  auto left_col2_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{3, 4}, stream, mr};
  auto const left_table  = cudf::table_view{{left_col1_2, left_col2_2}};
  auto right_col_2       = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, stream, mr};
  auto const right_table = cudf::table_view{{right_col_2}};

  EXPECT_THROW(cudf::detail::row::equality::two_table_comparator(
                 left_table, right_table, stream, mr.get_temporary_mr()),
               std::invalid_argument);

  auto int_col           = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, stream, mr};
  auto const int_table   = cudf::table_view{{int_col}};
  auto float_col         = cudf::test::fixed_width_column_wrapper<float>{{1.0f, 2.0f}, stream, mr};
  auto const float_table = cudf::table_view{{float_col}};

  EXPECT_THROW(cudf::detail::row::equality::two_table_comparator(
                 int_table, float_table, stream, mr.get_temporary_mr()),
               std::invalid_argument);

  auto str_col             = cudf::test::strings_column_wrapper({"hello", "world"}, stream, mr);
  auto const string_table  = cudf::table_view{{str_col}};
  auto num_col             = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2}, stream, mr);
  auto const numeric_table = cudf::table_view{{num_col}};

  EXPECT_THROW(cudf::detail::row::equality::two_table_comparator(
                 string_table, numeric_table, stream, mr.get_temporary_mr()),
               std::invalid_argument);
}

TEST_F(RowOperatorTest, TestRowHasher64BitHash)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 123456789}, stream, mr};
  auto const input = cudf::table_view{{col}};

  auto const preprocessed =
    cudf::detail::row::hash::preprocessed_table::create(input, stream, mr.get_temporary_mr());
  auto const row_hasher = cudf::detail::row::hash::row_hasher{preprocessed};
  auto const hasher =
    row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});

  auto results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0}, stream, mr};
  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{results}.begin<std::uint64_t>(),
                    hasher);

  // Expected values match cuCollections xxhash_64 reference implementation
  // https://github.com/NVIDIA/cuCollections/blob/4f03dcccb3a944594c693aa8cebc89302bbd8e20/tests/utility/hash_test.cu#L134-L137
  auto const expected = cudf::test::fixed_width_column_wrapper<std::uint64_t>{
    {4246796580750024372ul, 15516826743637085169ul, 9462334144942111946ul}, stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(RowOperatorTest, TestPrimitiveRowHasher64BitHash)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const col = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 42, 123456789}, stream, mr};
  auto const input = cudf::table_view{{col}};

  auto const d_input = cudf::table_device_view::create(input, stream, mr.get_temporary_mr());

  auto const hasher = cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::XXHash_64>(
    cudf::nullate::DYNAMIC{false}, *d_input, static_cast<std::uint64_t>(cudf::DEFAULT_HASH_SEED));

  auto results = cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0}, stream, mr};

  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{results}.begin<std::uint64_t>(),
                    hasher);

  // Expected values match cuCollections xxhash_64 reference implementation
  // https://github.com/NVIDIA/cuCollections/blob/4f03dcccb3a944594c693aa8cebc89302bbd8e20/tests/utility/hash_test.cu#L134-L137
  auto const expected = cudf::test::fixed_width_column_wrapper<std::uint64_t>{
    {4246796580750024372ul, 15516826743637085169ul, 9462334144942111946ul}, stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(RowOperatorTest, TestRowHasherDictionaryColumn)
{
  // TODO: dictionary encoding gathers the keys, and gather still allocates temporaries from the
  // current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // Dictionary and equivalent string column should produce identical hashes.
  // This also verifies same logical values get same hashes (e.g., "baz" at rows 0 and 2).
  auto const dict_col = cudf::test::dictionary_column_wrapper<std::string>(
    {"baz", "foo", "baz", "bar", "foo"}, stream, mr);
  auto const str_col =
    cudf::test::strings_column_wrapper({"baz", "foo", "baz", "bar", "foo"}, stream, mr);

  auto const dict_row_hasher = cudf::detail::row::hash::row_hasher(
    cudf::table_view{{dict_col}}, stream, mr.get_temporary_mr());
  auto const str_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{str_col}}, stream, mr.get_temporary_mr());

  auto const dict_hasher =
    dict_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});
  auto const str_hasher =
    str_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{false});

  auto dict_results =
    cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}, stream, mr};
  auto str_results =
    cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}, stream, mr};

  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{dict_results}.begin<std::uint64_t>(),
                    dict_hasher);
  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{str_results}.begin<std::uint64_t>(),
                    str_hasher);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dict_results, str_results, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(RowOperatorTest, TestRowHasherDictionaryColumnWithNulls)
{
  // TODO: dictionary encoding gathers the keys, and gather still allocates temporaries from the
  // current device resource.

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const dict_col = cudf::test::dictionary_column_wrapper<int64_t>(
    {100, 200, 300, 100, 200}, {1, 0, 1, 0, 1}, stream, mr);
  auto const int_col = cudf::test::fixed_width_column_wrapper<int64_t>(
    {100, 200, 300, 100, 200}, {1, 0, 1, 0, 1}, stream, mr);

  auto const dict_row_hasher = cudf::detail::row::hash::row_hasher(
    cudf::table_view{{dict_col}}, stream, mr.get_temporary_mr());
  auto const int_row_hasher =
    cudf::detail::row::hash::row_hasher(cudf::table_view{{int_col}}, stream, mr.get_temporary_mr());

  auto const dict_hasher =
    dict_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{true});
  auto const int_hasher =
    int_row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(cudf::nullate::DYNAMIC{true});

  auto dict_results =
    cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}, stream, mr};
  auto int_results =
    cudf::test::fixed_width_column_wrapper<std::uint64_t>{{0, 0, 0, 0, 0}, stream, mr};

  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{dict_results}.begin<std::uint64_t>(),
                    dict_hasher);
  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{5},
                    cudf::mutable_column_view{int_results}.begin<std::uint64_t>(),
                    int_hasher);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dict_results, int_results, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}
