/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/fused_for.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t expected_wire_alignment      = 16;
constexpr std::size_t largest_supported_tile_bytes = 32 * 1024;
constexpr std::size_t too_small_tile_bytes         = 512;
constexpr std::size_t non_power_of_two_tile_bytes  = 1536;
constexpr std::size_t too_large_tile_bytes         = 64 * 1024;

}  // namespace

struct FusedForSplitTest : public cudf::test::BaseFixture {};

std::vector<cudf::detail::fused_for_segment> copy_descriptors_to_host(
  cudf::detail::fused_for_packed_columns const& input, cuda::stream_ref stream)
{
  std::vector<cudf::detail::fused_for_segment> result(input.segment_count);
  if (result.empty()) { return result; }

  CUDF_CUDA_TRY(cudaMemcpyAsync(result.data(),
                                input.wire_data->data(),
                                result.size() * sizeof(result.front()),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return result;
}

void expect_packed_equivalent(cudf::packed_columns const& expected,
                              cudf::packed_columns const& actual,
                              cuda::stream_ref stream)
{
  ASSERT_NE(expected.metadata, nullptr);
  ASSERT_NE(actual.metadata, nullptr);
  ASSERT_NE(expected.gpu_data, nullptr);
  ASSERT_NE(actual.gpu_data, nullptr);
  EXPECT_EQ(*expected.metadata, *actual.metadata);
  EXPECT_EQ(expected.gpu_data->size(), actual.gpu_data->size());
  auto const expected_table = cudf::unpack(expected);
  auto const actual_table   = cudf::unpack(actual);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, actual_table, stream);
}

void expect_fused_for_round_trip(cudf::table_view const& input,
                                 std::vector<cudf::size_type> const& splits)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto expected     = cudf::contiguous_split(input, splits, stream, mr);
  auto packed       = cudf::detail::contiguous_split_fused_for(input, splits, stream, mr);

  ASSERT_EQ(expected.size(), packed.size());
  for (std::size_t i = 0; i < packed.size(); ++i) {
    auto decoded = cudf::detail::decode_fused_for(std::move(packed[i]), stream, mr);
    expect_packed_equivalent(expected[i].data, decoded, stream);
  }
}

TEST_F(FusedForSplitTest, UsesAllByteAlignedWidthsAndPreservesRawColumns)
{
  constexpr cudf::size_type rows = 10'000;
  std::vector<std::int16_t> one_byte(rows);
  std::vector<std::int16_t> int16_raw(rows);
  std::vector<std::int32_t> two_bytes(rows);
  std::vector<std::int64_t> four_bytes(rows);
  std::vector<std::int64_t> eight_bytes(rows);
  std::vector<float> floating(rows);
  std::vector<std::string> strings(rows);
  std::vector<bool> valid(rows);

  for (cudf::size_type i = 0; i < rows; ++i) {
    one_byte[i]    = i % 2 == 0 ? -100 : 100;
    int16_raw[i]   = i % 2 == 0 ? -30'000 : 30'000;
    two_bytes[i]   = i % 2 == 0 ? -30'000 : 30'000;
    four_bytes[i]  = i % 2 == 0 ? -1'000'000'000LL : 1'000'000'000LL;
    eight_bytes[i] = i % 2 == 0 ? std::numeric_limits<std::int64_t>::lowest()
                                : std::numeric_limits<std::int64_t>::max();
    floating[i]    = static_cast<float>(i) * 0.25F;
    strings[i]     = i % 2 == 0 ? "alpha" : "beta-value";
    valid[i]       = i % 7 != 0;
  }

  cudf::test::fixed_width_column_wrapper<std::int16_t> byte_col(one_byte.begin(), one_byte.end());
  cudf::test::fixed_width_column_wrapper<std::int16_t> int16_raw_col(int16_raw.begin(),
                                                                     int16_raw.end());
  cudf::test::fixed_width_column_wrapper<std::int32_t> short_col(two_bytes.begin(),
                                                                 two_bytes.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> word_col(four_bytes.begin(),
                                                                four_bytes.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> raw_col(
    eight_bytes.begin(), eight_bytes.end(), valid.begin());
  cudf::test::fixed_width_column_wrapper<float> float_col(floating.begin(), floating.end());
  cudf::test::strings_column_wrapper string_col(strings.begin(), strings.end());
  cudf::table_view input{
    {byte_col, int16_raw_col, short_col, word_col, raw_col, float_col, string_col}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);

  auto has_width = [&descriptors](std::uint8_t logical, std::uint8_t wire) {
    return std::any_of(descriptors.begin(), descriptors.end(), [=](auto const& segment) {
      return segment.logical_width == logical && segment.wire_width == wire;
    });
  };
  EXPECT_TRUE(has_width(sizeof(std::int16_t), sizeof(std::uint8_t)));
  EXPECT_TRUE(has_width(sizeof(std::int32_t), sizeof(std::uint16_t)));
  EXPECT_TRUE(has_width(sizeof(std::int64_t), sizeof(std::uint32_t)));
  EXPECT_TRUE(has_width(sizeof(std::int16_t), sizeof(std::int16_t)));
  EXPECT_TRUE(has_width(sizeof(std::int64_t), sizeof(std::int64_t)));
  EXPECT_LT(packed.wire_data->size(), packed.logical_data_size);

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equivalent(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, BitpackedNonByteWidthsRoundTrip)
{
  constexpr cudf::size_type rows = 10'000;
  std::vector<std::int16_t> five_bit(rows);
  std::vector<std::int32_t> seventeen_bit(rows);
  std::vector<std::int64_t> forty_bit(rows);
  std::vector<std::int64_t> constants(rows, -8'000'000'000'000LL);
  std::vector<std::int64_t> raw(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    five_bit[i]      = i % 2 == 0 ? -10 : 21;
    seventeen_bit[i] = i % 2 == 0 ? -50'000 : 50'000;
    forty_bit[i]     = i % 2 == 0 ? -2'000'000'000'000LL : -900'488'372'225LL;
    raw[i]           = i % 2 == 0 ? std::numeric_limits<std::int64_t>::lowest()
                                  : std::numeric_limits<std::int64_t>::max();
  }

  cudf::test::fixed_width_column_wrapper<std::int16_t> five_bit_col(five_bit.begin(),
                                                                    five_bit.end());
  cudf::test::fixed_width_column_wrapper<std::int32_t> seventeen_bit_col(seventeen_bit.begin(),
                                                                         seventeen_bit.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> forty_bit_col(forty_bit.begin(),
                                                                     forty_bit.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> constant_col(constants.begin(),
                                                                    constants.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> raw_col(raw.begin(), raw.end());
  cudf::table_view input{{five_bit_col, seventeen_bit_col, forty_bit_col, constant_col, raw_col}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(
    input,
    stream,
    mr,
    cudf::detail::fused_for_options{largest_supported_tile_bytes,
                                    cudf::detail::fused_for_layout::bitpacked});
  auto descriptors = copy_descriptors_to_host(packed, stream);

  auto has_bits = [&descriptors](std::uint8_t logical_width, std::uint8_t bits) {
    return std::any_of(descriptors.begin(), descriptors.end(), [=](auto const& segment) {
      return segment.is_bitpacked() && segment.logical_width == logical_width &&
             segment.wire_width == bits;
    });
  };
  constexpr std::uint8_t five_bit_width      = 5;
  constexpr std::uint8_t seventeen_bit_width = 17;
  constexpr std::uint8_t forty_bit_width     = 40;
  constexpr std::uint8_t constant_bit_width  = 0;
  EXPECT_TRUE(has_bits(sizeof(std::int16_t), five_bit_width));
  EXPECT_TRUE(has_bits(sizeof(std::int32_t), seventeen_bit_width));
  EXPECT_TRUE(has_bits(sizeof(std::int64_t), forty_bit_width));
  EXPECT_TRUE(has_bits(sizeof(std::int64_t), constant_bit_width));
  EXPECT_TRUE(std::any_of(descriptors.begin(), descriptors.end(), [](auto const& segment) {
    return !segment.is_bitpacked() && segment.logical_width == sizeof(std::int64_t) &&
           segment.wire_width == sizeof(std::int64_t);
  }));

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equivalent(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, RawSegmentsReserveWireSpaceOncePerTile)
{
  constexpr cudf::size_type rows = 65'536;
  std::vector<float> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = static_cast<float>(i) * 0.25F;
  }

  cudf::test::fixed_width_column_wrapper<float> column(values.begin(), values.end());
  cudf::table_view input{{column}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);

  ASSERT_GT(descriptors.size(), 1);
  auto expected_wire_size = cudf::util::round_up_safe(
    descriptors.size() * sizeof(cudf::detail::fused_for_segment), expected_wire_alignment);
  for (auto const& segment : descriptors) {
    EXPECT_EQ(segment.logical_width, sizeof(std::uint8_t));
    EXPECT_EQ(segment.wire_width, sizeof(std::uint8_t));
    auto const segment_bytes = static_cast<std::size_t>(segment.element_count);
    ASSERT_LE(segment.wire_offset, packed.wire_data->size());
    EXPECT_LE(segment_bytes, packed.wire_data->size() - segment.wire_offset);
    expected_wire_size += cudf::util::round_up_safe(segment_bytes, expected_wire_alignment);
  }
  EXPECT_EQ(packed.wire_data->size(), expected_wire_size);

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equivalent(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, NonDefaultStreamPreservesAsyncDecodeLifetime)
{
  rmm::cuda_stream producer_stream;
  rmm::cuda_stream consumer_stream;
  cuda::stream_ref const producer_stream_view = producer_stream;
  cuda::stream_ref const consumer_stream_view = consumer_stream;
  auto const mr                               = cudf::get_current_device_resource_ref();

  constexpr cudf::size_type rows = 16'384;
  std::vector<std::int64_t> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = -500'000 + (i % 2'048);
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> column(values.begin(), values.end());
  cudf::table_view input{{column}};
  auto expected = cudf::pack(input, consumer_stream_view, mr);
  auto packed   = cudf::detail::pack_fused_for(input, producer_stream_view, mr);
  producer_stream.synchronize();

  // decode_fused_for launches asynchronously on a stream other than the one
  // that allocated the moved wire buffer. Its deallocation must follow the
  // decode on the consumer stream.
  auto decoded = cudf::detail::decode_fused_for(std::move(packed), consumer_stream_view, mr);
  expect_packed_equivalent(expected, decoded, consumer_stream_view);
}

TEST_F(FusedForSplitTest, RoundTripsNullableSliceAcrossPartitions)
{
  constexpr cudf::size_type rows = 25'000;
  std::vector<std::int64_t> values(rows);
  std::vector<bool> valid(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = -4'000'000 + (i % 4'000);
    valid[i]  = i % 11 != 0;
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> column(
    values.begin(), values.end(), valid.begin());
  cudf::table_view original{{column}};
  auto sliced = cudf::slice(original, {17, rows - 19});
  ASSERT_EQ(sliced.size(), 1);

  expect_fused_for_round_trip(sliced.front(), {4'000, 12'000, 20'000});
}

TEST_F(FusedForSplitTest, Decimal32And64PreserveScale)
{
  constexpr cudf::size_type rows = 16'384;
  std::vector<std::int32_t> decimal32_values(rows);
  std::vector<std::int64_t> decimal64_values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    decimal32_values[i] = -50'000 + (i % 4'000);
    decimal64_values[i] = -5'000'000'000'000LL + (i % 20'000);
  }

  cudf::test::fixed_point_column_wrapper<std::int32_t> decimal32(
    decimal32_values.begin(), decimal32_values.end(), numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<std::int64_t> decimal64(
    decimal64_values.begin(), decimal64_values.end(), numeric::scale_type{-4});
  cudf::table_view input{{decimal32, decimal64}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);
  EXPECT_TRUE(std::any_of(descriptors.begin(), descriptors.end(), [](auto const& segment) {
    return segment.logical_width == sizeof(std::int32_t) && segment.is_encoded();
  }));
  EXPECT_TRUE(std::any_of(descriptors.begin(), descriptors.end(), [](auto const& segment) {
    return segment.logical_width == sizeof(std::int64_t) && segment.is_encoded();
  }));

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equivalent(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, ConstantsAndSignedExtremaRoundTrip)
{
  constexpr cudf::size_type rows = 4'096;
  std::vector<std::int64_t> constants(rows, -9'223'372'036'854'000'000LL);
  std::vector<std::int64_t> extrema(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    extrema[i] = i % 2 == 0 ? std::numeric_limits<std::int64_t>::lowest()
                            : std::numeric_limits<std::int64_t>::max();
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> constant_col(constants.begin(),
                                                                    constants.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> extrema_col(extrema.begin(), extrema.end());
  cudf::table_view input{{constant_col, extrema_col}};
  expect_fused_for_round_trip(input, {});
}

TEST_F(FusedForSplitTest, UnsignedAndChronoTypesUseNumericPath)
{
  cudf::test::fixed_width_column_wrapper<std::uint16_t> uint16_col({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<std::uint32_t> uint32_col({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<std::uint64_t> uint64_col({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> timestamp_days(
    {-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>
    timestamp_seconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>
    timestamp_milliseconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>
    timestamp_microseconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>
    timestamp_nanoseconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::duration_D, cudf::duration_D::rep> duration_days(
    {-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::duration_s, cudf::duration_s::rep> duration_seconds(
    {-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms, cudf::duration_ms::rep>
    duration_milliseconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::duration_us, cudf::duration_us::rep>
    duration_microseconds({-2, -1, 0, 1});
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns, cudf::duration_ns::rep>
    duration_nanoseconds({-2, -1, 0, 1});

  cudf::table_view input{{uint16_col,
                          uint32_col,
                          uint64_col,
                          timestamp_days,
                          timestamp_seconds,
                          timestamp_milliseconds,
                          timestamp_microseconds,
                          timestamp_nanoseconds,
                          duration_days,
                          duration_seconds,
                          duration_milliseconds,
                          duration_microseconds,
                          duration_nanoseconds}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);

  ASSERT_EQ(descriptors.size(), input.num_columns());
  EXPECT_TRUE(std::all_of(descriptors.begin(), descriptors.end(), [](auto const& segment) {
    return segment.is_encoded();
  }));

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equivalent(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, PreservesMetadataOnlyZeroColumnTable)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto expected     = cudf::contiguous_split(input, {3}, stream, mr);
  auto packed       = cudf::detail::contiguous_split_fused_for(input, {3}, stream, mr);

  ASSERT_EQ(packed.size(), 2);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    EXPECT_EQ(packed[i].segment_count, 0);
    EXPECT_EQ(packed[i].logical_data_size, 0);
    auto decoded = cudf::detail::decode_fused_for(std::move(packed[i]), stream, mr);
    expect_packed_equivalent(expected[i].data, decoded, stream);
  }
}

TEST_F(FusedForSplitTest, RejectsInvalidOptions)
{
  cudf::test::fixed_width_column_wrapper<std::int64_t> column({1, 2, 3, 4, 5});
  cudf::table_view input{{column}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto options       = cudf::detail::fused_for_options{};
  options.tile_bytes = too_small_tile_bytes;
  EXPECT_THROW(cudf::detail::pack_fused_for(input, stream, mr, options), cudf::logic_error);

  options.tile_bytes = non_power_of_two_tile_bytes;
  EXPECT_THROW(cudf::detail::pack_fused_for(input, stream, mr, options), cudf::logic_error);

  options.tile_bytes = too_large_tile_bytes;
  EXPECT_THROW(cudf::detail::pack_fused_for(input, stream, mr, options), cudf::logic_error);

  options = cudf::detail::fused_for_options{};
  options.layout =
    static_cast<cudf::detail::fused_for_layout>(std::numeric_limits<std::uint8_t>::max());
  EXPECT_THROW(cudf::detail::pack_fused_for(input, stream, mr, options), cudf::logic_error);
}

TEST_F(FusedForSplitTest, RejectsMalformedEnvelope)
{
  cudf::test::fixed_width_column_wrapper<std::int64_t> column({1, 2, 3, 4, 5});
  cudf::table_view input{{column}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  {
    auto packed          = cudf::detail::pack_fused_for(input, stream, mr);
    packed.segment_count = packed.wire_data->size() / sizeof(cudf::detail::fused_for_segment) + 1;
    EXPECT_THROW(cudf::detail::decode_fused_for(std::move(packed), stream, mr), cudf::logic_error);
  }
  {
    auto packed = cudf::detail::pack_fused_for(input, stream, mr);
    ASSERT_NE(packed.logical_data_size, 0);
    packed.segment_count = 0;
    EXPECT_THROW(cudf::detail::decode_fused_for(std::move(packed), stream, mr), cudf::logic_error);
  }
}
