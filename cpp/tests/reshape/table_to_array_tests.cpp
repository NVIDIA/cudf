/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/functional>

template <typename T>
struct TableToDeviceArrayTypedTest : public cudf::test::BaseFixture {};

using SupportedTypes = cudf::test::Types<int8_t,
                                         int16_t,
                                         int32_t,
                                         int64_t,
                                         uint8_t,
                                         uint16_t,
                                         uint32_t,
                                         uint64_t,
                                         float,
                                         double,
                                         cudf::timestamp_D,
                                         cudf::timestamp_s,
                                         cudf::timestamp_ms,
                                         cudf::timestamp_us,
                                         cudf::timestamp_ns,
                                         cudf::duration_D,
                                         cudf::duration_s,
                                         cudf::duration_ms,
                                         cudf::duration_us,
                                         cudf::duration_ns>;

TYPED_TEST_SUITE(TableToDeviceArrayTypedTest, SupportedTypes);

TYPED_TEST(TableToDeviceArrayTypedTest, SupportedTypes)
{
  using T     = TypeParam;
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  int nrows = 3;
  int ncols = 4;

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::vector<T> expected;

  for (int col = 0; col < ncols; ++col) {
    std::vector<T> data(nrows);
    for (int row = 0; row < nrows; ++row) {
      auto val = col * nrows + row + 1;
      if constexpr (cudf::is_chrono<T>()) {
        data[row] = T(typename T::duration{val});
      } else {
        data[row] = static_cast<T>(val);
      }
      expected.push_back(data[row]);
    }
    cols.push_back(std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end())));
  }

  std::vector<cudf::column_view> views(cols.size());
  std::transform(
    cols.begin(), cols.end(), views.begin(), [](auto const& col) { return col->view(); });
  cudf::table_view input{views};

  auto output = cudf::detail::make_zeroed_device_uvector<T>(nrows * ncols, stream, mr);

  cudf::table_to_array(
    input,
    cudf::device_span<cuda::std::byte>(reinterpret_cast<cuda::std::byte*>(output.data()),
                                       output.size() * sizeof(T)),
    stream);

  auto host_result = cudf::detail::make_std_vector(output, stream);
  EXPECT_EQ(host_result, expected);
}

template <typename T>
struct FixedPointTableToDeviceArrayTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTableToDeviceArrayTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTableToDeviceArrayTest, SupportedFixedPointTypes)
{
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto scale  = numeric::scale_type{-2};

  fp_wrapper col0({123, 456, 789}, scale);
  fp_wrapper col1({321, 654, 987}, scale);

  cudf::table_view input({col0, col1});
  size_t num_elements = input.num_rows() * input.num_columns();

  auto output = cudf::detail::make_zeroed_device_uvector<RepType>(num_elements, stream, mr);

  cudf::table_to_array(
    input,
    cudf::device_span<cuda::std::byte>(reinterpret_cast<cuda::std::byte*>(output.data()),
                                       output.size() * sizeof(RepType)),
    stream);

  auto host_result = cudf::detail::make_std_vector(output, stream);

  std::vector<RepType> expected{123, 456, 789, 321, 654, 987};
  EXPECT_EQ(host_result, expected);
}

struct TableToDeviceArrayTest : public cudf::test::BaseFixture {};

TEST(TableToDeviceArrayTest, UnsupportedStringType)
{
  auto stream = cudf::get_default_stream();
  auto col    = cudf::test::strings_column_wrapper({"a", "b", "c"});
  cudf::table_view input_table({col});
  rmm::device_buffer output(3 * sizeof(int32_t), stream);

  EXPECT_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), output.size()),
                         stream),
    cudf::logic_error);
}

TEST(TableToDeviceArrayTest, FailsWithNullValues)
{
  auto stream = cudf::get_default_stream();

  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3}, {true, false, true});
  cudf::table_view input_table({col});
  rmm::device_buffer output(3 * sizeof(int32_t), stream);

  EXPECT_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), output.size()),
                         stream),
    std::invalid_argument);
}

TEST(TableToDeviceArrayTest, FailsWhenOutputSpanTooSmall)
{
  auto stream = cudf::get_default_stream();

  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::table_view input_table({col});

  rmm::device_buffer output(4, stream);

  EXPECT_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), output.size()),
                         stream),
    std::invalid_argument);
}

TEST(TableToDeviceArrayTest, NoRows)
{
  auto stream = cudf::get_default_stream();

  cudf::test::fixed_width_column_wrapper<int32_t> col({});
  cudf::table_view input_table({col});

  rmm::device_buffer output(0, stream);

  EXPECT_NO_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), output.size()),
                         stream));
}

TEST(TableToDeviceArrayTest, NoColumns)
{
  auto stream = cudf::get_default_stream();

  cudf::table_view input_table{std::vector<cudf::column_view>{}};

  rmm::device_buffer output(0, stream);

  EXPECT_NO_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), output.size()),
                         stream));
}

TEST(TableToDeviceArrayTest, FlatSizeExceedsSizeTypeLimit)
{
  auto stream      = cudf::get_default_stream();
  auto size_limit  = static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
  auto num_rows    = size_limit * 0.6;
  auto num_cols    = 2;
  auto flat_size   = num_rows * num_cols;
  auto total_bytes = flat_size * sizeof(int8_t);

  std::vector<int8_t> data(num_rows, 1);
  auto col = cudf::test::fixed_width_column_wrapper<int8_t>(data.begin(), data.end());

  cudf::table_view input_table({col, col});

  rmm::device_buffer output(total_bytes, stream);

  EXPECT_NO_THROW(
    cudf::table_to_array(input_table,
                         cudf::device_span<cuda::std::byte>(
                           reinterpret_cast<cuda::std::byte*>(output.data()), total_bytes),
                         stream));
}
