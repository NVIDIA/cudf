/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/column_buffer.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/types.hpp>

#include <cstring>
#include <vector>

struct ColumnBufferTest : public cudf::test::BaseFixture {};

TEST_F(ColumnBufferTest, MakeColumnFromRvalueFixedWidth)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  cudf::io::detail::inline_column_buffer buffer{cudf::data_type{cudf::type_id::INT32}, false};
  buffer.create(4, stream, mr);

  std::vector<int32_t> const host_values{1, 2, 3, 4};
  CUDF_CUDA_TRY(cudaMemcpyAsync(buffer.data(),
                                host_values.data(),
                                host_values.size() * sizeof(int32_t),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();

  auto column = std::move(buffer).make_column(nullptr, std::nullopt, stream);

  cudf::test::fixed_width_column_wrapper<int32_t> const expected{{1, 2, 3, 4}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column->view(), expected);
}

TEST_F(ColumnBufferTest, MakeColumnFromRvalueNullable)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  cudf::test::fixed_width_column_wrapper<int8_t> const expected{{1, 2, 0, 4}, {1, 1, 0, 1}};

  cudf::io::detail::inline_column_buffer buffer{cudf::data_type{cudf::type_id::INT8}, true};
  buffer.create_with_mask(4, cudf::mask_state::ALL_VALID, false, stream, mr);

  int8_t const host_values[]{1, 2, 3, 4};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    buffer.data(), host_values, sizeof(host_values), cudaMemcpyDefault, stream.value()));
  uint32_t const valid_bits{0b00001011u};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    buffer.null_mask(), &valid_bits, sizeof(valid_bits), cudaMemcpyDefault, stream.value()));
  buffer.null_count() = 1;
  stream.synchronize();

  auto column = std::move(buffer).make_column(nullptr, std::nullopt, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column->view(), expected);
}
