/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "roaring_bitmap_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <cuda/stream>

#include <vector>

template <typename T>
struct RoaringBitmapTest : public cudf::test::BaseFixture {};

using RoaringTypes = cudf::test::Types<cuda::std::uint32_t, cuda::std::uint64_t>;

TYPED_TEST_SUITE(RoaringBitmapTest, RoaringTypes);

TYPED_TEST(RoaringBitmapTest, Basics)
{
  auto constexpr num_keys = 100'000;
  using Key               = TypeParam;

  auto insert_keys = std::vector<Key>(num_keys / 2);
  std::generate(insert_keys.begin(), insert_keys.end(), [k = Key{0}]() mutable {
    auto const result = k;
    k += 2;
    return result;
  });

  auto const [serialized_bitmap_data, bitmap_type, col_type] = [&]() {
    if constexpr (std::is_same_v<Key, cuda::std::uint64_t>) {
      return std::make_tuple(serialize_roaring_bitmap<Key>(insert_keys),
                             cudf::roaring_bitmap_type::BITS_64,
                             cudf::type_id::UINT64);
    } else {
      return std::make_tuple(serialize_roaring_bitmap<Key>(insert_keys),
                             cudf::roaring_bitmap_type::BITS_32,
                             cudf::type_id::UINT32);
    }
  }();

  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();

  auto bitmap = cudf::roaring_bitmap(bitmap_type, serialized_bitmap_data);

  auto key_iter = cuda::counting_iterator<Key>(0);
  auto keys_col = cudf::test::fixed_width_column_wrapper<Key>(key_iter, key_iter + num_keys);

  auto const is_even =
    cudf::detail::make_counting_transform_iterator(0, [](auto const i) { return i % 2 == 0; });

  {
    auto result_col = bitmap.contains_async(keys_col, stream, mr);
    auto results    = cudf::detail::make_host_vector_async(
      cudf::device_span<bool const>(result_col->view().template data<bool>(), num_keys), stream);
    stream.sync();
    EXPECT_TRUE(std::equal(results.begin(), results.end(), is_even));
  }

  {
    auto result_iter = cuda::constant_iterator<bool>(false);
    auto result_col =
      cudf::test::fixed_width_column_wrapper<bool>(result_iter, result_iter + num_keys).release();
    bitmap.contains_async(keys_col, result_col->mutable_view(), stream);
    auto results = cudf::detail::make_host_vector_async(
      cudf::device_span<bool const>(result_col->view().template data<bool>(), num_keys), stream);
    stream.sync();
    EXPECT_TRUE(std::equal(results.begin(), results.end(), is_even));
  }
}
struct RoaringBitmapErrorTest : public cudf::test::BaseFixture {};

TEST_F(RoaringBitmapErrorTest, EmptySerializedBitmapData)
{
  EXPECT_THROW(auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, {}),
               std::invalid_argument);
}

TEST_F(RoaringBitmapErrorTest, TypeMismatch)
{
  auto insert_keys            = std::vector<cuda::std::uint64_t>{1, 2, 3};
  auto serialized_bitmap_data = serialize_roaring_bitmap<cuda::std::uint64_t>(insert_keys);
  auto const stream           = cudf::get_default_stream();
  auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized_bitmap_data);
  bitmap.materialize(stream);
  auto probe_keys = cudf::test::fixed_width_column_wrapper<cuda::std::uint32_t>{1, 2, 3}.release();
  EXPECT_THROW(std::ignore = bitmap.contains_async(
                 probe_keys->view(), stream, cudf::get_current_device_resource_ref()),
               std::invalid_argument);
}

TEST_F(RoaringBitmapErrorTest, EmptyProbeKeys)
{
  auto insert_keys            = std::vector<cuda::std::uint64_t>{1, 2, 3};
  auto serialized_bitmap_data = serialize_roaring_bitmap<cuda::std::uint64_t>(insert_keys);
  auto const stream           = cudf::get_default_stream();
  auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized_bitmap_data);
  auto probe_keys = cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT64});
  EXPECT_NO_THROW(std::ignore = bitmap.contains_async(
                    probe_keys->view(), stream, cudf::get_current_device_resource_ref()));
}
