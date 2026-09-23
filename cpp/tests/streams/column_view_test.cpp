/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>

#include <numeric>
#include <random>
#include <vector>

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }

  TypedColumnTest(cuda::stream_ref stream = cudf::test::get_default_stream())
    : data{_num_elements * sizeof(T), stream},
      mask{cudf::bitmask_allocation_size_bytes(_num_elements), stream}
  {
    std::vector<char> h_data(std::max(data.size(), mask.size()));
    std::iota(h_data.begin(), h_data.end(), 0);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(data.data(), h_data.data(), data.size(), cudaMemcpyDefault, stream.get()));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(mask.data(), h_data.data(), mask.size(), cudaMemcpyDefault, stream.get()));
  }

  cudf::size_type num_elements() { return _num_elements; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  rmm::device_buffer all_valid_mask{create_null_mask(
    num_elements(), cudf::mask_state::ALL_VALID, cudf::test::get_default_stream())};
  rmm::device_buffer all_null_mask{
    create_null_mask(num_elements(), cudf::mask_state::ALL_NULL, cudf::test::get_default_stream())};
};

TYPED_TEST_SUITE(TypedColumnTest, cudf::test::Types<int32_t>);

/**
 * @brief Verifies equality of the properties and data of a `column`'s views.
 *
 * @param col The `column` to verify
 */
void verify_column_views(cudf::column& col)
{
  cudf::column_view view                 = col;
  cudf::mutable_column_view mutable_view = col;
  EXPECT_EQ(col.type(), view.type());
  EXPECT_EQ(col.type(), mutable_view.type());
  EXPECT_EQ(col.size(), view.size());
  EXPECT_EQ(col.size(), mutable_view.size());
  EXPECT_EQ(col.null_count(), view.null_count());
  EXPECT_EQ(col.null_count(), mutable_view.null_count());
  EXPECT_EQ(view.null_count(0, col.size(), cudf::test::get_default_stream()),
            mutable_view.null_count(0, col.size(), cudf::test::get_default_stream()));
  EXPECT_EQ(view.has_nulls(0, col.size(), cudf::test::get_default_stream()),
            mutable_view.has_nulls(0, col.size(), cudf::test::get_default_stream()));
  EXPECT_EQ(col.null_count(), mutable_view.null_count());
  EXPECT_EQ(col.nullable(), view.nullable());
  EXPECT_EQ(col.nullable(), mutable_view.nullable());
  EXPECT_EQ(col.num_children(), view.num_children());
  EXPECT_EQ(col.num_children(), mutable_view.num_children());
  EXPECT_EQ(view.head(), mutable_view.head());
  EXPECT_EQ(view.data<char>(), mutable_view.data<char>());
  EXPECT_EQ(view.offset(), mutable_view.offset());
}

TYPED_TEST(TypedColumnTest, CopyConstructorWithMask)
{
  cudf::column original{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask), 0};
  cudf::column copy{original, cudf::test::get_default_stream()};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view     = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}
