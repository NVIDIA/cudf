/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <stdexcept>

using namespace cudf::test::iterators;

template <typename T>
struct FixedWidthGetValueTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedWidthGetValueTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FixedWidthGetValueTest, BasicGet)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6});
  auto s = cudf::get_element(col, 0);

  using ScalarType = cudf::scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(9), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetFromNullable)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = cudf::get_element(col, 1);

  using ScalarType = cudf::scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetNull)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = cudf::get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(FixedWidthGetValueTest, IndexOutOfBounds)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});

  // Test for out of bounds indexes in both directions.
  EXPECT_THROW(cudf::get_element(col, -1), std::out_of_range);
  EXPECT_THROW(cudf::get_element(col, 4), std::out_of_range);
}

struct StringGetValueTest : public cudf::test::BaseFixture {};

TEST_F(StringGetValueTest, BasicGet)
{
  cudf::test::strings_column_wrapper col{"this", "is", "a", "test"};
  auto s = cudf::get_element(col, 3);

  auto typed_s = static_cast<cudf::string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("test", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetEmpty)
{
  cudf::test::strings_column_wrapper col{"this", "is", "", "test"};
  auto s = cudf::get_element(col, 2);

  auto typed_s = static_cast<cudf::string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetFromNullable)
{
  cudf::test::strings_column_wrapper col({"", "is", "", "test"}, {false, true, false, true});
  auto s = cudf::get_element(col, 1);

  auto typed_s = static_cast<cudf::string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("is", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetNull)
{
  cudf::test::strings_column_wrapper col({"", "is", "", "test"}, {false, true, false, true});
  auto s = cudf::get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

template <typename T>
struct DictionaryGetValueTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(DictionaryGetValueTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(DictionaryGetValueTest, BasicGet)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> indices{0, 0, 1, 2, 1, 3, 3, 2};
  auto col = cudf::make_dictionary_column(keys, indices);

  auto s = cudf::get_element(*col, 2);

  using ScalarType = cudf::scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(7), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetFromNullable)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> indices(
    {0, 0, 1, 2, 1, 3, 3, 2}, {false, true, false, true, true, true, false, false});
  auto col = cudf::make_dictionary_column(keys, indices);

  auto s = cudf::get_element(*col, 3);

  using ScalarType = cudf::scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetNull)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> indices(
    {0, 0, 1, 2, 1, 3, 3, 2}, {false, true, false, true, true, true, false, false});
  auto col = cudf::make_dictionary_column(keys, indices);

  auto s = cudf::get_element(*col, 2);

  EXPECT_FALSE(s->is_valid());
}

/*
 * Lists test grid:
 * Dim1 nestedness:          {Nested, Non-nested}
 * Dim2 validity, emptiness: {Null element, Non-null non-empty list, Non-null empty list}
 * Dim3 leaf data type:      {Fixed-width, string, struct}
 */

template <typename T>
struct ListGetFixedWidthValueTest : public cudf::test::BaseFixture {
  auto odds_valid() { return nulls_at_multiples_of(2); }
  auto nth_valid(cudf::size_type x)
  {
    return cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return x == i; });
  }
};

TYPED_TEST_SUITE(ListGetFixedWidthValueTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW({1, 2, 34}, this->odds_valid()), LCW{}, LCW{1}, LCW{}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_data({1, 2, 34}, this->odds_valid());
  cudf::size_type index = 0;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_data{};
  cudf::size_type index = 1;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  LCW col({LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}}, this->odds_valid());
  cudf::size_type index = 2;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_FALSE(s->is_valid());
  // Test preserve column hierarchy
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(typed_s->view(), FCW{});
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{LCW{42}, LCW{10}};

  cudf::size_type index = 3;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  std::vector<cudf::valid_type> valid{0, 1, 1};
  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW({LCW{42}, LCW{10}, LCW({1, 3, 2}, this->nth_valid(1))}, valid.begin())
  };
  // clang-format on
  LCW expected_data({LCW{42}, LCW{10}, LCW({1, 3, 2}, this->nth_valid(1))}, valid.begin());
  cudf::size_type index = 3;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{};
  cudf::size_type index = 1;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW      = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  std::vector<cudf::valid_type> valid{1, 0, 1, 0};
  // clang-format off
  LCW col(
    {
      LCW{LCW{1, 2}, LCW{34}},
      LCW{},
      LCW{LCW{1}},
      LCW{LCW{42}, LCW{10}}
    }, valid.begin());
  // clang-format on
  cudf::size_type index = 1;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  auto expected_data =
    cudf::make_lists_column(0, offset_t{}.release(), FCW{}.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  // Test preserve column hierarchy
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(typed_s->view(), *expected_data);
}

struct ListGetStringValueTest : public cudf::test::BaseFixture {
  auto odds_valid() { return nulls_at_multiples_of(2); }
  auto nth_valid(cudf::size_type x)
  {
    return cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return x == i; });
  }
};

TEST_F(ListGetStringValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW col{LCW({"aaa", "Héllo"}, this->odds_valid()), LCW{}, LCW{""}, LCW{"42"}};
  cudf::test::strings_column_wrapper expected_data({"", "Héllo"}, this->odds_valid());
  cudf::size_type index = 0;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW col{LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}};
  cudf::test::strings_column_wrapper expected_data{};
  cudf::size_type index = 1;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view>;
  using StringCW = cudf::test::strings_column_wrapper;

  std::vector<cudf::valid_type> valid{1, 0, 0, 1};
  LCW col({LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}}, valid.begin());
  cudf::size_type index = 2;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(typed_s->view(), StringCW{});
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{},
    LCW{LCW{""}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))},
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data{LCW{""}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))};
  cudf::size_type index = 2;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  std::vector<cudf::valid_type> valid{0, 1, 1};
  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{},
    LCW({LCW{""}, LCW{"cc"}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))}, valid.begin()),
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data({LCW{""}, LCW{"cc"}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))},
                    valid.begin());
  cudf::size_type index = 2;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{LCW{""}},
    LCW{LCW{"42"}, LCW{"21"}},
    LCW{}
  };
  // clang-format on
  LCW expected_data{};
  cudf::size_type index = 3;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. `expected_data` leaf string column does not
  // allocate offset and byte array, but `typed_s` does.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<cudf::string_view>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using StringCW = cudf::test::strings_column_wrapper;

  std::vector<cudf::valid_type> valid{0, 0, 1, 1};
  // clang-format off
  LCW col(
    {
      LCW{LCW{"aaa", "Héllo"}},
      LCW{LCW{""}},
      LCW{LCW{"42"}, LCW{"21"}},
      LCW{}
    }, valid.begin());
  // clang-format on
  cudf::size_type index = 0;

  auto s       = cudf::get_element(col, index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  auto expected_data =
    cudf::make_lists_column(0, offset_t{}.release(), StringCW{}.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

/**
 * @brief Some shared helper functions used by lists of structs test.
 */
template <typename T>
struct ListGetStructValueTest : public cudf::test::BaseFixture {
  using SCW        = cudf::test::structs_column_wrapper;
  using LCWinner_t = cudf::test::lists_column_wrapper<T, int32_t>;

  /**
   * @brief Create a lists column
   *
   * @note Different from `cudf::make_lists_column`, this allows setting the `null_mask`
   * in `initializer_list`. However this is an expensive function because it repeatedly
   * calls `cudf::set_null_mask` for each row.
   */
  std::unique_ptr<cudf::column> make_test_lists_column(
    cudf::size_type num_lists,
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets,
    std::unique_ptr<cudf::column> child,
    std::initializer_list<cudf::valid_type> null_mask)
  {
    cudf::size_type null_count = num_lists - std::accumulate(null_mask.begin(), null_mask.end(), 0);
    auto d_null_mask           = cudf::create_null_mask(
      num_lists, null_count == 0 ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL);
    if (null_count > 0) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{num_lists},
                    [&](auto i) {
                      if (*(null_mask.begin() + i)) {
                        cudf::set_null_mask(
                          static_cast<cudf::bitmask_type*>(d_null_mask.data()), i, i + 1, true);
                      }
                    });
    }
    return cudf::make_lists_column(
      num_lists, offsets.release(), std::move(child), null_count, std::move(d_null_mask));
  }

  /**
   * @brief Create a structs column that contains 3 fields: int, string, List<int>
   */
  template <typename MaskIterator>
  SCW make_test_structs_column(cudf::test::fixed_width_column_wrapper<T> field1,
                               cudf::test::strings_column_wrapper field2,
                               cudf::test::lists_column_wrapper<T, int32_t> field3,
                               MaskIterator mask)
  {
    return SCW{{field1, field2, field3}, mask};
  }

  /**
   * @brief Concatenate structs columns, allow specifying inputs in `initializer_list`
   */
  std::unique_ptr<cudf::column> concat(std::initializer_list<SCW> rows)
  {
    std::vector<cudf::column_view> views;
    std::transform(rows.begin(), rows.end(), std::back_inserter(views), [](auto& r) {
      return cudf::column_view(r);
    });
    return cudf::concatenate(views);
  }

  /**
   * @brief Test data setup: row 0 of structs column
   */
  SCW row0()
  {
    // {int: 1, string: NULL, list: NULL}
    return this->make_test_structs_column({{1}, {1}},
                                          cudf::test::strings_column_wrapper({""}, {false}),
                                          LCWinner_t({{}}, all_nulls()),
                                          no_nulls());
  }

  /**
   * @brief Test data setup: row 1 of structs column
   */
  SCW row1()
  {
    // NULL
    return this->make_test_structs_column({-1}, {""}, LCWinner_t{-1}, all_nulls());
  }

  /**
   * @brief Test data setup: row 2 of structs column
   */
  SCW row2()
  {
    // {int: 3, string: "xyz", list: [3, 8, 4]}
    return this->make_test_structs_column({{3}, {1}},
                                          cudf::test::strings_column_wrapper({"xyz"}, {true}),
                                          LCWinner_t({{3, 8, 4}}, no_nulls()),
                                          no_nulls());
  }

  /**
   * @brief Test data setup: a 3-row structs column
   */
  std::unique_ptr<cudf::column> leaf_data()
  {
    // 3 rows:
    // {int: 1, string: NULL, list: NULL}
    // NULL
    // {int: 3, string: "xyz", list: [3, 8, 4]}
    return this->concat({row0(), row1(), row2()});
  }
};

TYPED_TEST_SUITE(ListGetStructValueTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullNonEmpty)
{
  // 2-rows
  // [{1, NULL, NULL}, NULL]
  // [{3, "xyz", [3, 8, 4]}] <- cudf::get_element(1)

  auto list_column      = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  cudf::size_type index = 1;
  auto expected_data    = this->row2();

  auto s       = cudf::get_element(list_column->view(), index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The nested list column in struct allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullNonEmpty2)
{
  // 2-rows
  // [{1, NULL, NULL}, NULL] <- cudf::get_element(0)
  // [{3, "xyz", [3, 8, 4]}]

  auto list_column      = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  cudf::size_type index = 0;
  auto expected_data    = this->concat({this->row0(), this->row1()});

  auto s       = cudf::get_element(list_column->view(), index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullEmpty)
{
  // 3-rows
  // [{1, NULL, NULL}, NULL]
  // [{3, "xyz", [3, 8, 4]}]
  // []                      <- cudf::get_element(2)

  auto list_column = this->make_test_lists_column(3, {0, 2, 3, 3}, this->leaf_data(), {1, 1, 1});
  cudf::size_type index = 2;
  // For well-formed list column, an empty list still holds the complete structure of
  // a 0-length structs column
  auto expected_data = this->make_test_structs_column({}, {}, {}, no_nulls());

  auto s       = cudf::get_element(list_column->view(), index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The nested list column in struct allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNull)
{
  // 2-rows
  // NULL                    <- cudf::get_element(0)
  // [{3, "xyz", [3, 8, 4]}]

  using valid_t = std::vector<cudf::valid_type>;

  auto list_column      = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {0, 1});
  cudf::size_type index = 0;

  auto s       = cudf::get_element(list_column->view(), index);
  auto typed_s = static_cast<cudf::list_scalar const*>(s.get());

  auto expected_data = this->make_test_structs_column({}, {}, {}, valid_t{}.begin());

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(typed_s->view(), expected_data);
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL], [{3, "xyz", [3, 8, 4]}]] <- cudf::get_element(0)
  // []

  auto list_column   = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto expected_data = std::make_unique<cudf::column>(*list_column);

  auto list_column_nested =
    this->make_test_lists_column(2, {0, 2, 2}, std::move(list_column), {1, 1});

  cudf::size_type index = 0;
  auto s                = cudf::get_element(list_column_nested->view(), index);
  auto typed_s          = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty2)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL]] <- cudf::get_element(0)
  // [[{3, "xyz", [3, 8, 4]}]]

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(2, {0, 1, 2}, std::move(list_column), {1, 1});

  auto expected_data =
    this->make_test_lists_column(1, {0, 2}, this->concat({this->row0(), this->row1()}), {1});

  cudf::size_type index = 0;
  auto s                = cudf::get_element(list_column_nested->view(), index);
  auto typed_s          = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty3)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL]]
  // [[{3, "xyz", [3, 8, 4]}]] <- cudf::get_element(1)

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(2, {0, 1, 2}, std::move(list_column), {1, 1});

  auto expected_data = this->make_test_lists_column(1, {0, 1}, this->row2().release(), {1});

  cudf::size_type index = 1;
  auto s                = cudf::get_element(list_column_nested->view(), index);
  auto typed_s          = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. For `get_element`, the nested list column in struct
  // allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullEmpty)
{
  // 3-rows
  // [[{1, NULL, NULL}, NULL]]
  // []                        <- cudf::get_element(1)
  // [[{3, "xyz", [3, 8, 4]}]]

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(3, {0, 1, 1, 2}, std::move(list_column), {1, 1, 1});

  auto expected_data = this->make_test_lists_column(
    0, {0}, this->make_test_structs_column({}, {}, {}, no_nulls()).release(), {});

  cudf::size_type index = 1;
  auto s                = cudf::get_element(list_column_nested->view(), index);
  auto typed_s          = static_cast<cudf::list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The sliced version still has the array for fields
  // allocated.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNull)
{
  // 3-rows
  // [[{1, NULL, NULL}, NULL]]
  // []
  // NULL                      <- cudf::get_element(2)

  using valid_t  = std::vector<cudf::valid_type>;
  using offset_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(3, {0, 1, 1, 2}, std::move(list_column), {1, 1, 0});

  cudf::size_type index = 2;
  auto s                = cudf::get_element(list_column_nested->view(), index);
  auto typed_s          = static_cast<cudf::list_scalar const*>(s.get());

  auto nested = this->make_test_structs_column({}, {}, {}, valid_t{}.begin());
  auto expected_data =
    cudf::make_lists_column(0, offset_t{}.release(), nested.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

struct StructGetValueTest : public cudf::test::BaseFixture {};
template <typename T>
struct StructGetValueTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(StructGetValueTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(StructGetValueTestTyped, mixed_types_valid)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // col fields
  cudf::test::fixed_width_column_wrapper<TypeParam> f1{1, 2, 3};
  cudf::test::strings_column_wrapper f2{"aa", "bbb", "c"};
  cudf::test::dictionary_column_wrapper<TypeParam, int32_t> f3{42, 42, 24};
  LCW f4{LCW{8, 8, 8}, LCW{9, 9}, LCW{10}};

  cudf::test::structs_column_wrapper col{f1, f2, f3, f4};

  cudf::size_type index = 2;
  auto s                = cudf::get_element(col, index);
  auto typed_s          = static_cast<cudf::struct_scalar const*>(s.get());

  // expect fields
  cudf::test::fixed_width_column_wrapper<TypeParam> ef1{3};
  cudf::test::strings_column_wrapper ef2{"c"};
  cudf::test::dictionary_column_wrapper<TypeParam, int32_t> ef3{24};
  LCW ef4{LCW{10}};

  // keys need to match so dictionaries can be compared
  auto def3 = cudf::dictionary::set_keys(cudf::dictionary_column_view(ef3),
                                         cudf::dictionary_column_view(f3).keys());

  cudf::table_view expect_data{{ef1, ef2, def3->view(), ef4}};

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect_data, typed_s->view());
}

TYPED_TEST(StructGetValueTestTyped, mixed_types_valid_with_nulls)
{
  using LCW             = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using validity_mask_t = std::vector<cudf::valid_type>;

  // col fields
  cudf::test::fixed_width_column_wrapper<TypeParam> f1({1, 2, 3}, {true, false, true});
  cudf::test::strings_column_wrapper f2({"", "", "c"}, {false, false, true});
  cudf::test::dictionary_column_wrapper<TypeParam, uint32_t> f3(
    {42, 42, 24}, validity_mask_t{true, true, true}.begin());
  LCW f4({LCW{8, 8, 8}, LCW{9, 9}, LCW{10}}, validity_mask_t{false, false, false}.begin());

  cudf::test::structs_column_wrapper col{f1, f2, f3, f4};

  cudf::size_type index = 1;
  auto s                = cudf::get_element(col, index);
  auto typed_s          = static_cast<cudf::struct_scalar const*>(s.get());

  // expect fields
  cudf::test::fixed_width_column_wrapper<TypeParam> ef1({-1}, {false});
  cudf::test::strings_column_wrapper ef2({""}, {false});

  cudf::test::dictionary_column_wrapper<TypeParam, uint32_t> x({42}, {true});
  cudf::dictionary_column_view dict_col(x);
  cudf::test::fixed_width_column_wrapper<TypeParam> new_key{24};
  auto ef3 = cudf::dictionary::add_keys(dict_col, new_key);

  LCW ef4({LCW{10}}, validity_mask_t{false}.begin());

  cudf::table_view expect_data{{ef1, ef2, *ef3, ef4}};

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect_data, typed_s->view());
}

TYPED_TEST(StructGetValueTestTyped, mixed_types_invalid)
{
  using LCW             = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using validity_mask_t = std::vector<cudf::valid_type>;

  // col fields
  cudf::test::fixed_width_column_wrapper<TypeParam> f1{1, 2, 3};
  cudf::test::strings_column_wrapper f2{"aa", "bbb", "c"};
  cudf::test::dictionary_column_wrapper<TypeParam, uint32_t> f3{42, 42, 24};
  LCW f4{LCW{8, 8, 8}, LCW{9, 9}, LCW{10}};

  cudf::test::structs_column_wrapper col({f1, f2, f3, f4},
                                         validity_mask_t{false, true, true}.begin());

  cudf::size_type index = 0;
  auto s                = cudf::get_element(col, index);
  auto typed_s          = static_cast<cudf::struct_scalar const*>(s.get());

  EXPECT_FALSE(typed_s->is_valid());

  // expect to preserve types along column hierarchy.
  EXPECT_EQ(typed_s->view().column(0).type().id(), cudf::type_to_id<TypeParam>());
  EXPECT_EQ(typed_s->view().column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(typed_s->view().column(2).type().id(), cudf::type_id::DICTIONARY32);
  EXPECT_EQ(typed_s->view().column(2).child(1).type().id(), cudf::type_to_id<TypeParam>());
  EXPECT_EQ(typed_s->view().column(3).type().id(), cudf::type_id::LIST);
  EXPECT_EQ(typed_s->view().column(3).child(1).type().id(), cudf::type_to_id<TypeParam>());
}

TEST_F(StructGetValueTest, multi_level_nested)
{
  using LCW             = cudf::test::lists_column_wrapper<int32_t, int32_t>;
  using validity_mask_t = std::vector<cudf::valid_type>;

  // col fields
  LCW l3({LCW{1, 1, 1}, LCW{2, 2}, LCW{3}}, validity_mask_t{false, true, true}.begin());
  cudf::test::structs_column_wrapper l2{l3};
  auto l1 = cudf::make_lists_column(1,
                                    cudf::test::fixed_width_column_wrapper<int32_t>{0, 3}.release(),
                                    l2.release(),
                                    0,
                                    cudf::create_null_mask(1, cudf::mask_state::UNALLOCATED));
  std::vector<std::unique_ptr<cudf::column>> l0_fields;
  l0_fields.emplace_back(std::move(l1));
  cudf::test::structs_column_wrapper l0(std::move(l0_fields));

  cudf::size_type index = 0;
  auto s                = cudf::get_element(l0, index);
  auto typed_s          = static_cast<cudf::struct_scalar const*>(s.get());

  // Expect fields
  cudf::column_view cv = cudf::column_view(l0);
  cudf::table_view fields(std::vector<cudf::column_view>(cv.child_begin(), cv.child_end()));

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(fields, typed_s->view());
}
