/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tests/utilities/roaring_bitmap_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/utilities/roaring_bitmap.hpp>

template <typename T>
struct RoaringBitmapStreamTest : public cudf::test::BaseFixture {};

using RoaringTypes = cudf::test::Types<cuda::std::uint32_t, cuda::std::uint64_t>;

TYPED_TEST_SUITE(RoaringBitmapStreamTest, RoaringTypes);

TYPED_TEST(RoaringBitmapStreamTest, ContainsAsync)
{
  using Key         = TypeParam;
  auto const stream = cudf::test::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  std::vector<Key> const insert_keys = {0, 2, 4};
  auto roaring_bitmap_data           = serialize_roaring_bitmap<Key>(insert_keys);

  auto constexpr bitmap_type = std::is_same_v<Key, cuda::std::uint32_t>
                                 ? cudf::roaring_bitmap_type::BITS_32
                                 : cudf::roaring_bitmap_type::BITS_64;
  auto const bitmap          = cudf::roaring_bitmap(bitmap_type, roaring_bitmap_data);
  bitmap.materialize(stream);

  cudf::size_type constexpr num_keys = 3;
  std::unique_ptr<cudf::column> result;

  auto const probe_keys = cudf::test::fixed_width_column_wrapper<Key>({0, 1, 2});
  result                = bitmap.contains_async(probe_keys, stream, mr);

  EXPECT_EQ(result->size(), num_keys);
}

CUDF_TEST_PROGRAM_MAIN()
