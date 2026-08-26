/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using gather_map_t      = std::vector<cudf::size_type>;
using offsets           = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs           = cudf::test::structs_column_wrapper;
using strings           = cudf::test::strings_column_wrapper;
using bools             = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

// Test validity iterator utilities.
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

template <typename T>
using numerics = cudf::test::fixed_width_column_wrapper<T, int32_t>;

template <typename T>
using lists = cudf::test::lists_column_wrapper<T, int32_t>;

auto constexpr null_index = std::numeric_limits<cudf::size_type>::max();

struct StructGatherTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedStructGatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedStructGatherTest, cudf::test::FixedWidthTypes);

namespace {
template <typename ElementTo, typename SourceElementT = ElementTo>
struct column_wrapper_constructor {
  template <typename ValueIter, typename ValidityIter>
  auto operator()(ValueIter begin, ValueIter end, ValidityIter validity_begin) const
  {
    return cudf::test::fixed_width_column_wrapper<ElementTo, SourceElementT>{
      begin, end, validity_begin};
  }
};

template <>
struct column_wrapper_constructor<std::string, std::string> {
  template <typename ValueIter, typename ValidityIter>
  cudf::test::strings_column_wrapper operator()(ValueIter begin,
                                                ValueIter end,
                                                ValidityIter validity_begin) const
  {
    return cudf::test::strings_column_wrapper{begin, end, validity_begin};
  }
};

template <typename ElementTo,
          typename SourceElementT     = ElementTo,
          typename InputValidityIter  = decltype(null_at(0)),
          typename StructValidityIter = InputValidityIter>
auto get_expected_column(std::vector<SourceElementT> const& input_values,
                         InputValidityIter input_validity,
                         StructValidityIter struct_validity,
                         std::vector<int32_t> const& gather_map)
{
  auto is_valid =  // Validity predicate.
    [&input_values, &input_validity, &struct_validity, &gather_map](auto gather_index) {
      assert(
        (gather_index >= 0 && gather_index < static_cast<cudf::size_type>(gather_map.size())) &&
        "Gather-index out of range.");

      auto i = gather_map[gather_index];  // Index into input_values.

      return (i >= 0 && i < static_cast<int>(input_values.size())) && struct_validity[i] &&
             input_validity[i];
    };

  auto expected_row_count = gather_map.size();
  auto gather_iter        = cudf::detail::make_counting_transform_iterator(
    0, [is_valid, &input_values, &gather_map](auto i) {
      return is_valid(i) ? input_values[gather_map[i]] : SourceElementT{};
    });

  return column_wrapper_constructor<ElementTo, SourceElementT>()(
    gather_iter,
    gather_iter + expected_row_count,
    cudf::detail::make_counting_transform_iterator(0, is_valid));
}

auto do_gather(cudf::column_view const& input, gather_map_t const& gather_map)
{
  auto result = cudf::gather(cudf::table_view{{input}},
                             offsets(gather_map.begin(), gather_map.end()),
                             cudf::out_of_bounds_policy::NULLIFY);
  return std::move(result->release()[0]);
}
}  // namespace

TYPED_TEST(TypedStructGatherTest, TestSimpleStructGather)
{
  // Testing gather() on struct<string, numeric, bool>.

  // 1. String "names" column.
  auto const names =
    std::vector<std::string>{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"};
  auto const names_validity = no_nulls();

  // 2. Numeric "ages" column.
  auto const ages          = std::vector<int32_t>{5, 10, 15, 20, 25, 30};
  auto const ages_validity = null_at(4);

  // 3. Boolean "is_human" column.
  auto const is_human          = {true, true, false, false, false, false};
  auto const is_human_validity = null_at(3);

  // Assemble struct column.
  auto const struct_validity = null_at(5);
  auto const struct_column   = [&] {
    auto names_member    = ::strings(names.begin(), names.end(), names_validity);
    auto ages_member     = ::numerics<TypeParam>(ages.begin(), ages.end(), ages_validity);
    auto is_human_member = ::bools(is_human.begin(), is_human.end(), is_human_validity);
    return structs{{names_member, ages_member, is_human_member}, struct_validity};
  }();

  // Gather to new struct column.
  auto const gather_map = gather_map_t{null_index, 4, 3, 2, 1};

  auto const output = do_gather(struct_column, gather_map);

  auto const expected_output = [&] {
    auto names_member =
      get_expected_column<std::string>(names, names_validity, struct_validity, gather_map);
    auto ages_member =
      get_expected_column<TypeParam, int32_t>(ages, ages_validity, struct_validity, gather_map);
    auto is_human_member =
      get_expected_column<bool>(std::vector<bool>(is_human.begin(), is_human.end()),
                                is_human_validity,
                                struct_validity,
                                gather_map);
    return structs{{names_member, ages_member, is_human_member}, null_at(0)};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output->view(), expected_output);
}

TYPED_TEST(TypedStructGatherTest, TestSlicedStructsColumnGatherNoNulls)
{
  auto const structs_original = [] {
    auto child1 =
      cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto child2 = cudf::test::strings_column_wrapper{
      "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const expected = [] {
    auto child1 = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{6, 10, 8};
    auto child2 = cudf::test::strings_column_wrapper{"Six", "Ten", "Eight"};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const structs    = cudf::slice(structs_original, {4, 10})[0];
  auto const gather_map = cudf::test::fixed_width_column_wrapper<int32_t>{1, 5, 3};
  auto const result     = cudf::gather(cudf::table_view{{structs}}, gather_map)->get_column(0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view(), expected);
}

TYPED_TEST(TypedStructGatherTest, TestSlicedStructsColumnGatherWithNulls)
{
  auto constexpr null = int32_t{0};  // null at child
  auto constexpr XXX  = int32_t{0};  // null at parent

  auto const structs_original = [] {
    auto child1 = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{
      {null, XXX, 3, null, null, 6, XXX, null, null, 10}, nulls_at({0, 3, 4, 7, 8})};
    auto child2 = cudf::test::strings_column_wrapper{{"One",
                                                      "" /*NULL at both parent and child*/,
                                                      "Three",
                                                      "" /*NULL*/,
                                                      "Five",
                                                      "" /*NULL*/,
                                                      "" /*NULL at parent*/,
                                                      "" /*NULL*/,
                                                      "Nine",
                                                      "" /*NULL*/},
                                                     nulls_at({1, 3, 5, 7, 9})};
    return cudf::test::structs_column_wrapper{{child1, child2}, nulls_at({1, 6})};
  }();

  auto const expected = [] {
    auto child1 =
      cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>{{6, 10, null, XXX}, null_at(2)};
    auto child2 =
      cudf::test::strings_column_wrapper{{
                                           "" /*NULL*/, "" /*NULL*/, "Nine", "" /*NULL at parent*/
                                         },
                                         nulls_at({0, 1})};
    return cudf::test::structs_column_wrapper{{child1, child2}, null_at(3)};
  }();

  auto const structs    = cudf::slice(structs_original, {4, 10})[0];
  auto const gather_map = cudf::test::fixed_width_column_wrapper<int32_t>{1, 5, 4, 2};
  auto const result     = cudf::gather(cudf::table_view{{structs}}, gather_map)->get_column(0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view(), expected);
}

TYPED_TEST(TypedStructGatherTest, TestNullifyOnNonNullInput)
{
  // Test that the null masks of the struct output (and its children) are set correctly,
  // for an input struct column whose members are not nullable.

  // 1. String "names" column.
  auto const names =
    std::vector<std::string>{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"};

  // 2. Numeric "ages" column.
  auto const ages = std::vector<int32_t>{5, 10, 15, 20, 25, 30};

  // 3. Boolean "is_human" column.
  auto const is_human = {true, true, false, false, false, false};

  // Assemble struct column.
  auto const struct_column = [&] {
    auto names_member    = ::strings(names.begin(), names.end());
    auto ages_member     = ::numerics<TypeParam>(ages.begin(), ages.end());
    auto is_human_member = ::bools(is_human.begin(), is_human.end());
    return structs({names_member, ages_member, is_human_member});
  }();

  // Gather to new struct column.
  auto const gather_map = gather_map_t{null_index, 4, 3, 2, 1};

  auto const output = do_gather(struct_column, gather_map);

  auto const expected_output = [&] {
    auto names_member = get_expected_column<std::string>(names, no_nulls(), no_nulls(), gather_map);
    auto ages_member =
      get_expected_column<TypeParam, int32_t>(ages, no_nulls(), no_nulls(), gather_map);
    auto is_human_member = get_expected_column<bool>(
      std::vector<bool>(is_human.begin(), is_human.end()), no_nulls(), no_nulls(), gather_map);
    return cudf::test::structs_column_wrapper{{names_member, ages_member, is_human_member},
                                              null_at(0)};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output->view(), expected_output);
}

TYPED_TEST(TypedStructGatherTest, TestGatherStructOfLists)
{
  // Testing gather() on struct<list<numeric>>

  auto lists_column_exemplar = [] {
    return lists<TypeParam>{
      {{5}, {10, 15}, {20, 25, 30}, {35, 40, 45, 50}, {55, 60, 65}, {70, 75}, {80}, {}, {}},
      nulls_at({0, 3, 6, 9})};
  };

  // Assemble struct column.
  auto const structs_column = [&] {
    auto lists_column = lists_column_exemplar();
    return cudf::test::structs_column_wrapper{{lists_column}};
  }();

  // Gather to new struct column.
  auto const gather_map = gather_map_t{null_index, 4, 3, 2, 1, 7, 3};

  auto const gathered_structs = do_gather(structs_column, gather_map);

  // Verify that the gathered struct column's list member presents as if
  // it had itself been gathered individually.

  auto const expected_gathered_list_column = [&] {
    auto const list_column_before_gathering = lists_column_exemplar();
    return do_gather(list_column_before_gathering, gather_map);
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_gathered_list_column->view(),
                                      gathered_structs->view().child(0));
}

TYPED_TEST(TypedStructGatherTest, TestGatherStructOfListsOfLists)
{
  // Testing gather() on struct<list<list<numeric>>>

  auto const lists_column_exemplar = []() {
    return lists<TypeParam>{{{{5, 5}},
                             {{10, 15}},
                             {{20, 25}, {30}},
                             {{35, 40}, {45, 50}},
                             {{55}, {60, 65}},
                             {{70, 75}},
                             {{80, 80}},
                             {},
                             {}},
                            nulls_at({0, 3, 6, 9})};
  };

  auto const structs_column = [&] {
    auto lists_column = lists_column_exemplar();
    return cudf::test::structs_column_wrapper{{lists_column}};
  }();

  // Gather to new struct column.
  auto const gather_map = gather_map_t{null_index, 4, 3, 2, 1, 7, 3};

  auto const gathered_structs = do_gather(structs_column, gather_map);

  // Verify that the gathered struct column's list member presents as if
  // it had itself been gathered individually.

  auto const expected_gathered_list_column = [&] {
    auto const list_column_before_gathering = lists_column_exemplar();
    return do_gather(list_column_before_gathering, gather_map);
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_gathered_list_column->view(),
                                      gathered_structs->view().child(0));
}

TYPED_TEST(TypedStructGatherTest, TestGatherStructOfStructs)
{
  // Testing gather() on struct<struct<numeric>>

  auto const numeric_column_exemplar = []() {
    return numerics<TypeParam>{{5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75},
                               nulls_at({0, 3, 6, 9, 12, 15})};
  };

  auto const struct_of_structs_column = [&] {
    auto numeric_column = numeric_column_exemplar();
    auto structs_column = cudf::test::structs_column_wrapper{{numeric_column}};
    return cudf::test::structs_column_wrapper{{structs_column}};
  }();

  // Gather to new struct column.
  auto const gather_map       = gather_map_t{null_index, 4, 3, 2, 1, 7, 3};
  auto const gathered_structs = do_gather(struct_of_structs_column, gather_map);

  // Verify that the underlying numeric column presents as if
  // it had itself been gathered individually.

  auto const expected_gathered_column = [&] {
    auto const numeric_column_before_gathering = numeric_column_exemplar();
    return do_gather(numeric_column_before_gathering, gather_map);
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_gathered_column->view(),
                                      gathered_structs->view().child(0).child(0));
}

TYPED_TEST(TypedStructGatherTest, TestGatherStructOfListOfStructs)
{
  // Testing gather() on struct<list<struct<numeric>>>

  auto const struct_of_list_of_structs = [&] {
    auto numeric_column =
      numerics<TypeParam>{{5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75}};
    auto structs_column         = structs{{numeric_column}}.release();
    auto list_of_structs_column = cudf::make_lists_column(
      7, offsets{0, 2, 4, 6, 8, 10, 12, 14}.release(), std::move(structs_column), 0, {});

    std::vector<std::unique_ptr<cudf::column>> vector_of_columns;
    vector_of_columns.push_back(std::move(list_of_structs_column));
    return structs{std::move(vector_of_columns)};
  }();

  // Gather to new struct column.
  auto const gather_map       = gather_map_t{null_index, 4, 3, 2, 1};
  auto const gathered_structs = do_gather(struct_of_list_of_structs, gather_map);

  // Construct expected gather result.

  auto expected_gather_result = [&] {
    auto expected_numeric_col = numerics<TypeParam>{{70, 75, 50, 55, 35, 45, 25, 30, 15, 20}};
    auto expected_struct_col  = structs{{expected_numeric_col}}.release();
    auto expected_list_of_structs_column = cudf::make_lists_column(
      5, offsets{0, 2, 4, 6, 8, 10}.release(), std::move(expected_struct_col), 0, {});
    std::vector<std::unique_ptr<cudf::column>> expected_vector_of_columns;
    expected_vector_of_columns.push_back(std::move(expected_list_of_structs_column));
    return structs{std::move(expected_vector_of_columns), {false, true, true, true, true}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_gather_result, gathered_structs->view());
}

TYPED_TEST(TypedStructGatherTest, TestGatherStructOfStructsWithValidity)
{
  // Testing gather() on struct<struct<numeric>>

  using validity_iter_t = decltype(nulls_at({0}));

  // Factory to construct numeric column with configurable null-mask.
  auto const numeric_column_exemplar = [](validity_iter_t validity) {
    return numerics<TypeParam>{{5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75}, validity};
  };

  // Construct struct-of-struct-of-numerics.
  auto struct_of_structs_column = [&] {
    // Every 3rd element is null.
    auto numeric_column = numeric_column_exemplar(nulls_at({0, 3, 6, 9, 12, 15}));
    // 12th element is null.
    auto structs_column = cudf::test::structs_column_wrapper{{numeric_column}, nulls_at({11})};
    return cudf::test::structs_column_wrapper{{structs_column}};
  }();

  // Gather to new struct column.
  auto const gather_map       = gather_map_t{null_index, 4, 3, 2, 1, 7, 3};
  auto const gathered_structs = do_gather(struct_of_structs_column, gather_map);

  // Verify that the underlying numeric column presents as if
  // it had itself been gathered individually.

  auto const expected_gathered_column = [&] {
    // Every 3rd element *and* the 12th element are null.
    auto const final_validity                  = nulls_at({0, 3, 6, 9, 11, 12, 15});
    auto const numeric_column_before_gathering = numeric_column_exemplar(final_validity);
    return do_gather(numeric_column_before_gathering, gather_map);
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_gathered_column->view(),
                                      gathered_structs->view().child(0).child(0));
}

TYPED_TEST(TypedStructGatherTest, TestEmptyGather)
{
  auto const struct_column = [&] {
    auto ages = numerics<TypeParam>{{5, 10, 15, 20, 25, 30}, null_at(4)};
    return cudf::test::structs_column_wrapper{{ages}, null_at(5)};
  }();

  auto const empty_gather_map = gather_map_t{};
  auto const gathered_structs = do_gather(struct_column, empty_gather_map);

  // Expect empty struct column gathered.
  auto const expected_empty_column = [&] {
    auto expected_empty_numerics = numerics<TypeParam>{};
    return cudf::test::structs_column_wrapper{{expected_empty_numerics}};
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_empty_column, gathered_structs->view());
}
