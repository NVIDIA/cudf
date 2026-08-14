/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>

#include <cuda/iterator>

template <typename T>
struct FixedWidthColumnWrapperTest : public cudf::test::BaseFixture,
                                     cudf::test::UniformRandomGenerator<cudf::size_type> {
  FixedWidthColumnWrapperTest() : cudf::test::UniformRandomGenerator<cudf::size_type>{1000, 5000} {}

  auto size() { return this->generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(FixedWidthColumnWrapperTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthColumnWrapperTest, EmptyIterator)
{
  auto sequence = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}
TYPED_TEST(FixedWidthColumnWrapperTest, EmptyList)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col{};
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableIteratorConstructor)
{
  auto sequence = cuda::counting_iterator{0};

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableListConstructor)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5});

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllValid)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_valid = cudf::test::iterators::no_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllNull)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_null = cudf::test::iterators::all_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), size);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {p{1, false}, p{2, false}, p{3, false}, p{4, false}, p{5, false}});
  cudf::column_view view = col;

  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> match_col({1, 2, 3, 4, 5}, odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({p{1, odd_valid[0]},
                                                                  p{2, odd_valid[1]},
                                                                  p{3, odd_valid[2]},
                                                                  p{4, odd_valid[3]},
                                                                  p{5, odd_valid[4]}});
  cudf::column_view view = col;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view,
                                 match_view,
                                 cudf::test::debug_output_level::FIRST_ERROR,
                                 this->stream(),
                                 this->resources());
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

template <typename T>
struct StringsColumnWrapperTest : public cudf::test::BaseFixture,
                                  cudf::test::UniformRandomGenerator<cudf::size_type> {
  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(StringsColumnWrapperTest, cudf::test::StringTypes);

TYPED_TEST(StringsColumnWrapperTest, EmptyList)
{
  cudf::test::strings_column_wrapper col;
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col(
    {p{"a", false}, p{"string", false}, p{"test", false}, p{"for", false}, p{"nulls", false}});
  cudf::strings_column_view view = cudf::column_view(col);

  constexpr auto count = 5;
  EXPECT_EQ(view.size(), count);
  EXPECT_EQ(view.offsets().size(), count + 1);
  // all null entries results in no data allocated to chars
  EXPECT_EQ(nullptr, view.parent().head());
  EXPECT_NE(nullptr, view.offsets().head());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::strings_column_wrapper match_col({"a", "string", "", "test", "for", "nulls"},
                                               odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col({p{"a", odd_valid[0]},
                                          p{"string", odd_valid[1]},
                                          p{"", odd_valid[2]},
                                          p{"test", odd_valid[3]},
                                          p{"for", odd_valid[4]},
                                          p{"nulls", odd_valid[5]}});
  cudf::column_view view = col;

  // TODO: has_nonempty_nulls (via count_if/transform_reduce) still allocates temporaries from the
  // current device resource for strings columns.
  this->enable_current_device_resource_use();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view,
                                 match_view,
                                 cudf::test::debug_output_level::FIRST_ERROR,
                                 this->stream(),
                                 this->resources());
}
