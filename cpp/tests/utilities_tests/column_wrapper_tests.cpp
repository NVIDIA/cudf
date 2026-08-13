/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>

#include <cuda/iterator>

using cudf::test::expect_output_uses_distinct_resources;
using cudf::test::temporary_allocation_expectation;

namespace {
auto const uses_temporary = cudf::test::memory_resource_expectations{
  cudf::test::output_allocation_expectation::EXACT, temporary_allocation_expectation::SOME};
}  // namespace

TEST(FixedPointColumnWrapperMemoryResourceTest, DistinctOutputAndTemporaryResources)
{
  auto stream         = cudf::test::get_default_stream();
  auto const elements = std::vector<int32_t>{1, 2, 3, 4};
  auto const validity = std::vector<bool>{true, false, true, false};
  auto const scale    = numeric::scale_type{-2};

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), scale, stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      {1, 2, 3, 4}, scale, stream, mr.get_output_mr());
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), scale, stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      {1, 2, 3, 4}, {true, false, true, false}, scale, stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      {1, 2, 3, 4}, validity.begin(), scale, stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), {true, false, true, false}, scale, stream, mr);
  });
}

TEST(StringsColumnWrapperMemoryResourceTest, DistinctOutputAndTemporaryResources)
{
  auto stream         = cudf::test::get_default_stream();
  auto const strings  = std::vector<std::string>{"", "alpha", "beta", "gamma"};
  auto const validity = std::vector<bool>{true, false, true, false};

  expect_output_uses_distinct_resources(
    [&]([[maybe_unused]] auto mr) { return cudf::test::strings_column_wrapper{}; });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::strings_column_wrapper(
      {"", "alpha", "beta", "gamma"}, stream, mr.get_output_mr());
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::strings_column_wrapper(
      strings.begin(), strings.end(), validity.begin(), stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::strings_column_wrapper(
      {"", "alpha", "beta", "gamma"}, validity.begin(), stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::strings_column_wrapper(
      {"", "alpha", "beta", "gamma"}, {true, false, true, false}, stream, mr);
  });

  expect_output_uses_distinct_resources([&](auto mr) {
    using pair_type = std::pair<std::string, bool>;
    return cudf::test::strings_column_wrapper(
      {pair_type{"", true}, pair_type{"alpha", false}, pair_type{"beta", true}}, stream, mr);
  });
}

TEST(DictionaryColumnWrapperMemoryResourceTest, FixedWidthDistinctOutputAndTemporaryResources)
{
  auto stream         = cudf::test::get_default_stream();
  auto const elements = std::vector<int32_t>{3, 1, 3, 2};
  auto const validity = std::vector<bool>{true, false, true, true};

  // Intermediate fixed-width column is allocated on temporary_mr before encode.
  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<int32_t>(
        elements.begin(), elements.end(), stream, mr);
    },
    uses_temporary);

  // Single-ref overload: temporaries go to the current resource, not the harness temporary.
  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>({3, 1, 3, 2}, stream, mr.get_output_mr());
  });

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<int32_t>(
        elements.begin(), elements.end(), validity.begin(), stream, mr);
    },
    uses_temporary);

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<int32_t>(
        {3, 1, 3, 2}, validity.begin(), stream, mr);
    },
    uses_temporary);

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<int32_t>(
        {3, 1, 3, 2}, {true, false, true, true}, stream, mr);
    },
    uses_temporary);

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<int32_t>(
        elements.begin(), elements.end(), {true, false, true, true}, stream, mr);
    },
    uses_temporary);
}

TEST(DictionaryColumnWrapperMemoryResourceTest, EmptyStringDictionaryPreservesChildTypes)
{
  expect_output_uses_distinct_resources([&]([[maybe_unused]] auto mr) {
    auto wrapper    = cudf::test::dictionary_column_wrapper<std::string>();
    auto dictionary = cudf::dictionary_column_view{static_cast<cudf::column_view>(wrapper)};

    EXPECT_EQ(0, static_cast<cudf::column_view>(wrapper).size());
    EXPECT_EQ(cudf::type_id::STRING, dictionary.keys().type().id());
    EXPECT_EQ(cudf::type_id::INT32, dictionary.indices().type().id());
    return wrapper;
  });
}

TEST(DictionaryColumnWrapperMemoryResourceTest, StringDistinctOutputAndTemporaryResources)
{
  auto stream         = cudf::test::get_default_stream();
  auto const strings  = std::vector<std::string>{"gamma", "alpha", "gamma", "beta"};
  auto const validity = std::vector<bool>{true, false, true, true};

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<std::string>(
        strings.begin(), strings.end(), stream, mr);
    },
    uses_temporary);

  // Single-ref overload: temporaries go to the current resource, not the harness temporary.
  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::dictionary_column_wrapper<std::string>(
      {"gamma", "alpha", "gamma", "beta"}, stream, mr.get_output_mr());
  });

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<std::string>(
        strings.begin(), strings.end(), validity.begin(), stream, mr);
    },
    uses_temporary);

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<std::string>(
        {"gamma", "alpha", "gamma", "beta"}, validity.begin(), stream, mr);
    },
    uses_temporary);

  expect_output_uses_distinct_resources(
    [&](auto mr) {
      return cudf::test::dictionary_column_wrapper<std::string>(
        {"gamma", "alpha", "gamma", "beta"}, {true, false, true, true}, stream, mr);
    },
    uses_temporary);
}

struct ColumnWrapperTestWithHarness : public cudf::test::BaseFixtureWithHarness {
  /**
   * @brief Validate that the harness owns the given result.
   *
   * Assert that the harness output resource holds bytes equal to `col->alloc_size()` and that no
   * temporary allocations remain live. `col` is destroyed on return, so `TearDown` can additionally
   * confirm that the output bytes were released.
   */
  void validate_with_harness(std::unique_ptr<cudf::column> col)
  {
    _harness.expect_resource_usage(col->alloc_size(), {}, this->stream());
  }
};

template <typename T>
struct FixedWidthColumnWrapperTest : public ColumnWrapperTestWithHarness,
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
    sequence, sequence, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
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

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableIteratorConstructor)
{
  auto sequence = cuda::counting_iterator{0};

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableListConstructor)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {1, 2, 3, 4, 5}, this->stream(), this->resources());

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllValid)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_valid = cudf::test::iterators::no_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_valid, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {1, 2, 3, 4, 5}, all_valid, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllNull)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_null = cudf::test::iterators::all_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_null, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), size);
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {1, 2, 3, 4, 5}, all_null, this->stream(), this->resources());
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {p{1, false}, p{2, false}, p{3, false}, p{4, false}, p{5, false}},
    this->stream(),
    this->resources());
  cudf::column_view view = col;

  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(col.release());
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> match_col(
    {1, 2, 3, 4, 5}, odd_valid, this->stream(), this->resources());
  cudf::column_view match_view = match_col;

  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({p{1, odd_valid[0]},
                                                                  p{2, odd_valid[1]},
                                                                  p{3, odd_valid[2]},
                                                                  p{4, odd_valid[3]},
                                                                  p{5, odd_valid[4]}},
                                                                 this->stream(),
                                                                 this->resources());
  cudf::column_view view = col;

  // TODO: Check the harness with a failing current resource once equality row preprocessing uses
  // the supplied memory resources.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view,
                                 match_view,
                                 cudf::test::debug_output_level::FIRST_ERROR,
                                 this->stream(),
                                 this->resources());
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {1, 2, 3, 4, 5}, all_valid, this->stream(), this->resources());
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(std::move(colPtr));
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {1, 2, 3, 4, 5}, all_null, this->stream(), this->resources());
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);

  this->validate_with_harness(std::move(colPtr));
}

template <typename T>
struct StringsColumnWrapperTest : public ColumnWrapperTestWithHarness {
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

  this->validate_with_harness(col.release());
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col(
    {p{"a", false}, p{"string", false}, p{"test", false}, p{"for", false}, p{"nulls", false}},
    this->stream(),
    this->resources());
  cudf::strings_column_view view = cudf::column_view(col);

  constexpr auto count = 5;
  EXPECT_EQ(view.size(), count);
  EXPECT_EQ(view.offsets().size(), count + 1);
  // all null entries results in no data allocated to chars
  EXPECT_EQ(nullptr, view.parent().head());
  EXPECT_NE(nullptr, view.offsets().head());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);

  this->validate_with_harness(col.release());
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::strings_column_wrapper match_col(
    {"a", "string", "", "test", "for", "nulls"}, odd_valid, this->stream(), this->resources());
  cudf::column_view match_view = match_col;

  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col({p{"a", odd_valid[0]},
                                          p{"string", odd_valid[1]},
                                          p{"", odd_valid[2]},
                                          p{"test", odd_valid[3]},
                                          p{"for", odd_valid[4]},
                                          p{"nulls", odd_valid[5]}},
                                         this->stream(),
                                         this->resources());
  cudf::column_view view = col;

  // TODO: Check the harness with a failing current resource once equality row preprocessing uses
  // the supplied memory resources.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view,
                                 match_view,
                                 cudf::test::debug_output_level::FIRST_ERROR,
                                 this->stream(),
                                 this->resources());
}
