/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <limits>
#include <type_traits>

struct DictionaryFactoriesTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryFactoriesTest, CreateFromColumnViews)
{
  cudf::test::strings_column_wrapper keys({"aaa", "ccc", "ddd", "www"});
  cudf::test::fixed_width_column_wrapper<int32_t> values{2, 0, 3, 1, 2, 2, 2, 3, 0};

  auto dictionary = cudf::make_dictionary_column(keys, values);
  cudf::dictionary_column_view view(dictionary->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values);
}

TEST_F(DictionaryFactoriesTest, ColumnViewsWithNulls)
{
  cudf::test::fixed_width_column_wrapper<float> keys{-11.75, 4.25, 7.125, 0.5, 12.0};
  std::vector<int32_t> h_values{1, 3, 2, 0, 1, 4, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> indices(
    h_values.begin(), h_values.end(), cuda::transform_iterator(h_values.begin(), [](auto v) {
      return v > 0;
    }));
  auto dictionary = cudf::make_dictionary_column(keys, indices);
  cudf::dictionary_column_view view(dictionary->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys);
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, CreateFromColumns)
{
  std::vector<std::string> h_keys{"pear", "apple", "fruit", "macintosh"};
  cudf::test::strings_column_wrapper keys(h_keys.begin(), h_keys.end());
  std::vector<int32_t> h_values{1, 2, 3, 1, 2, 3, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> values(h_values.begin(), h_values.end());

  auto dictionary = cudf::make_dictionary_column(
    keys.release(), values.release(), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::strings_column_wrapper keys_expected(h_keys.begin(), h_keys.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, ColumnsWithNulls)
{
  std::vector<int64_t> h_keys{-1234567890, -987654321, 0, 19283714};
  cudf::test::fixed_width_column_wrapper<int64_t> keys(h_keys.begin(), h_keys.end());
  std::vector<int32_t> h_values{1, 2, 3, 1, 2, 3, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> values(h_values.begin(), h_values.end());
  auto size = static_cast<cudf::size_type>(h_values.size());
  cuda::device_buffer<cudf::bitmask_type> null_mask =
    create_null_mask(size, cudf::mask_state::ALL_NULL);
  auto dictionary =
    cudf::make_dictionary_column(keys.release(), values.release(), std::move(null_mask), size);
  cudf::dictionary_column_view view(dictionary->view());
  EXPECT_EQ(size, view.size());
  EXPECT_EQ(size, view.null_count());

  cudf::test::fixed_width_column_wrapper<int64_t> keys_expected(h_keys.begin(), h_keys.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, DuplicateKeys)
{
  std::vector<std::string> h_keys{"pear", "apple", "fruit", "pear"};
  cudf::test::strings_column_wrapper keys(h_keys.begin(), h_keys.end());
  std::vector<int32_t> h_values{1, 2, 3, 1, 2, 3, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> values(h_values.begin(), h_values.end());

  auto dictionary = cudf::make_dictionary_column(
    keys.release(), values.release(), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
  cudf::dictionary_column_view view(dictionary->view());

  auto keys_expected = cudf::test::strings_column_wrapper(h_keys.begin(), h_keys.end());
  auto values_expected =
    cudf::test::fixed_width_column_wrapper<int32_t>(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, KeysWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{{0, 1, 2, 3, 4},
                                                       {true, true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{5, 4, 3, 2, 1, 0};
  EXPECT_THROW(cudf::make_dictionary_column(keys, indices), std::invalid_argument);
}

TEST_F(DictionaryFactoriesTest, IndicesWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{{5, 4, 3, 2, 1, 0},
                                                          {true, true, true, false, true, false}};
  EXPECT_THROW(
    cudf::make_dictionary_column(keys.release(),
                                 indices.release(),
                                 cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                 0),
    std::invalid_argument);
}

TEST_F(DictionaryFactoriesTest, InvalidIndices)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<uint16_t> indices{5, 4, 3, 2, 1, 0};
  EXPECT_THROW(cudf::make_dictionary_column(keys, indices), std::invalid_argument);
  EXPECT_THROW(
    cudf::make_dictionary_column(keys.release(),
                                 indices.release(),
                                 cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                 0),
    std::invalid_argument);
}

TEST_F(DictionaryFactoriesTest, EmptyKeys)
{
  cudf::test::strings_column_wrapper keys{};

  cudf::test::fixed_width_column_wrapper<int32_t> null_indices({0, 0, 0}, {0, 0, 0});
  auto dictionary = cudf::make_dictionary_column(keys, null_indices);
  EXPECT_EQ(dictionary->size(), 3);
  EXPECT_EQ(dictionary->null_count(), 3);
  ASSERT_EQ(dictionary->num_children(), 2);
  EXPECT_EQ(cudf::dictionary_column_view(dictionary->view()).keys().type().id(),
            cudf::type_id::STRING);

  cudf::test::fixed_width_column_wrapper<int32_t> empty_indices{};
  auto empty = cudf::make_dictionary_column(keys, empty_indices);
  EXPECT_EQ(empty->size(), 0);
  ASSERT_EQ(empty->num_children(), 2);
  EXPECT_EQ(cudf::dictionary_column_view(empty->view()).keys().type().id(), cudf::type_id::STRING);
}

template <typename T>
struct DictionaryFactoriesUnsignedIndicesTest : public cudf::test::BaseFixture {};

using NarrowUnsignedIndexTypes = cudf::test::Types<uint8_t, uint16_t>;
TYPED_TEST_SUITE(DictionaryFactoriesUnsignedIndicesTest, NarrowUnsignedIndexTypes);

TYPED_TEST(DictionaryFactoriesUnsignedIndicesTest, WidenAboveSignedLimit)
{
  using IndexType = TypeParam;
  using WiderType = std::conditional_t<std::is_same_v<IndexType, uint8_t>, int16_t, int32_t>;
  auto constexpr max_index =
    static_cast<WiderType>(std::numeric_limits<cuda::std::make_signed_t<IndexType>>::max() + 1);
  auto key_values = cuda::counting_iterator<int32_t>(0);
  auto keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(key_values, key_values + max_index + 1);
  auto indices = cudf::test::fixed_width_column_wrapper<IndexType>(
    {static_cast<IndexType>(max_index), IndexType{0}, IndexType{1}}, {1, 0, 1});

  auto dictionary = cudf::make_dictionary_column(keys.release(), indices.release());
  cudf::dictionary_column_view view(dictionary->view());

  auto expected_indices =
    cudf::test::fixed_width_column_wrapper<WiderType>({max_index, WiderType{0}, WiderType{1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), expected_indices);
  EXPECT_EQ(view.null_count(), 1);
}

TYPED_TEST(DictionaryFactoriesUnsignedIndicesTest, SameWidthAtSignedLimit)
{
  using IndexType          = TypeParam;
  using SignedType         = cuda::std::make_signed_t<IndexType>;
  auto constexpr max_index = std::numeric_limits<SignedType>::max();
  auto key_values          = cuda::counting_iterator<int32_t>(0);
  auto keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(key_values, key_values + max_index + 1);
  auto indices = cudf::test::fixed_width_column_wrapper<IndexType>(
    {static_cast<IndexType>(max_index), IndexType{0}, IndexType{1}}, {1, 0, 1});

  auto dictionary = cudf::make_dictionary_column(keys.release(), indices.release());
  cudf::dictionary_column_view view(dictionary->view());

  auto expected_indices =
    cudf::test::fixed_width_column_wrapper<SignedType>({max_index, SignedType{0}, SignedType{1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), expected_indices);
  EXPECT_EQ(view.null_count(), 1);
}
