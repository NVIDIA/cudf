/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <src/partitioning/fixed_width.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using cudf::test::dictionary_column_wrapper;
using cudf::test::fixed_width_column_wrapper;
using cudf::test::lists_column_wrapper;
using cudf::test::strings_column_wrapper;

/** @brief Test fixture for fixed-width hash partitioning. */
class FixedWidthHashPartitionTest : public cudf::test::BaseFixture {};

TEST_F(FixedWidthHashPartitionTest, PartitionMetadataLayoutBoundaries)
{
  using layout = cudf::detail::partition_metadata::layout;

  EXPECT_EQ(cudf::detail::partition_metadata::pick_layout(1 << 20, 1 << 12), layout::PACKED32);
  EXPECT_EQ(cudf::detail::partition_metadata::pick_layout(1 << 20, (1 << 12) + 1), layout::DEFAULT);
  EXPECT_EQ(cudf::detail::partition_metadata::pick_layout(1, 1 << 12), layout::PACKED32);
}

TEST_F(FixedWidthHashPartitionTest, PartitionMetadataRoundTrips)
{
  std::uint32_t packed{};
  auto const packed_metadata =
    cudf::detail::partition_metadata::packed_view{cudf::device_span<std::uint32_t>{&packed, 1}, 20};
  packed_metadata.store(0, (1 << 20) - 1, (1 << 12) - 1);
  cudf::size_type packed_partition;
  cudf::size_type packed_offset;
  packed_metadata.load(0, packed_partition, packed_offset);
  EXPECT_EQ(packed_partition, (1 << 20) - 1);
  EXPECT_EQ(packed_offset, (1 << 12) - 1);
  EXPECT_EQ(packed_metadata.partition(0), packed_partition);

  std::uint32_t zero_partition_bits{};
  auto const zero_partition_bits_metadata = cudf::detail::partition_metadata::packed_view{
    cudf::device_span<std::uint32_t>{&zero_partition_bits, 1}, 0};
  zero_partition_bits_metadata.store(0, 0, 4095);
  cudf::size_type zero_partition;
  cudf::size_type zero_partition_offset;
  zero_partition_bits_metadata.load(0, zero_partition, zero_partition_offset);
  EXPECT_EQ(zero_partition, 0);
  EXPECT_EQ(zero_partition_offset, 4095);
}

/**
 * @brief Verifies hash partition output against the public Murmur3 hash implementation.
 *
 * @param input Table to partition
 * @param keys Indices of the key columns
 * @param num_partitions Number of partitions to produce
 * @param seed Initial Murmur3 seed
 */
void expect_murmur_partitioned(cudf::table_view const& input,
                               std::vector<cudf::size_type> const& keys,
                               cudf::size_type num_partitions,
                               uint32_t seed = cudf::DEFAULT_HASH_SEED)
{
  auto [output, offsets] =
    cudf::hash_partition(input, keys, num_partitions, cudf::hash_id::HASH_MURMUR3, seed);

  ASSERT_EQ(offsets.size(), static_cast<std::size_t>(num_partitions + 1));
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), input.num_rows());
  EXPECT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output->view());

  auto const sorted_input  = cudf::sort(input);
  auto const sorted_output = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_input->view(), sorted_output->view());

  auto hashes            = cudf::hashing::murmurhash3_x86_32(output->view().select(keys), seed);
  auto const host_hashes = cudf::test::to_host<uint32_t>(hashes->view()).first;
  for (cudf::size_type partition = 0; partition < num_partitions; ++partition) {
    for (auto row = offsets[partition]; row < offsets[partition + 1]; ++row) {
      EXPECT_EQ(host_hashes[row] % static_cast<uint32_t>(num_partitions),
                static_cast<uint32_t>(partition));
    }
  }
}

template <typename T>
class FixedPointHashPartitionTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointHashPartitionTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointHashPartitionTest, FixedPointKeys)
{
  using rep_type = cudf::device_storage_type_t<TypeParam>;

  cudf::test::fixed_point_column_wrapper<rep_type> keys{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                        numeric::scale_type{-2}};
  fixed_width_column_wrapper<int32_t> payload{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  expect_murmur_partitioned(cudf::table_view{{keys, payload}}, {0}, 5, 12345);
}

TEST_F(FixedWidthHashPartitionTest, MixedWidthsNullableCompositeKeys)
{
  constexpr cudf::size_type owner_rows = 2055;
  auto const values                    = cuda::counting_iterator<int32_t>{0};
  auto const valid = thrust::make_transform_iterator(values, [](auto row) { return row % 7 != 0; });

  fixed_width_column_wrapper<int8_t, int32_t> bytes(values, values + owner_rows);
  fixed_width_column_wrapper<int16_t, int32_t> shorts(values, values + owner_rows, valid);
  fixed_width_column_wrapper<int32_t> ints(values, values + owner_rows);
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    values, values + owner_rows, numeric::scale_type{-4});
  fixed_width_column_wrapper<int64_t, int32_t> payload(values, values + owner_rows);

  auto const owner  = cudf::table_view{{bytes, shorts, ints, decimal128, payload}};
  auto const sliced = cudf::slice(owner, {1, owner_rows - 1}).front();
  expect_murmur_partitioned(sliced, {0, 1, 2, 3}, 16, 12345);
  expect_murmur_partitioned(sliced, {0, 1, 2, 3}, 17, 12345);
}

TEST_F(FixedWidthHashPartitionTest, FloatingPointNormalizationAndSlice)
{
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  fixed_width_column_wrapper<double> doubles{
    0.0, -0.0, nan, -nan, 1.5, -2.25, 0.0, -0.0, nan, 9.0, 10.0};
  fixed_width_column_wrapper<int32_t> payload{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto const owner  = cudf::table_view{{doubles, payload}};
  auto const sliced = cudf::slice(owner, {1, 10}).front();

  expect_murmur_partitioned(sliced, {0}, 5, 9876);
}

TEST_F(FixedWidthHashPartitionTest, ExternalIdentityKeys)
{
  constexpr cudf::size_type num_rows       = 37;
  constexpr cudf::size_type num_partitions = 7;
  std::vector<std::string> values(num_rows);
  std::vector<int32_t> key_values(num_rows);
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    values[row]     = std::string{"row_"} + std::to_string(row);
    key_values[row] = (row / 2) % num_partitions + (row % 2) * num_partitions;
  }

  strings_column_wrapper strings(values.begin(), values.end());
  fixed_width_column_wrapper<int32_t> keys(key_values.begin(), key_values.end());
  auto const input = cudf::table_view{{strings}};

  auto [output, offsets] = cudf::hash_partition(
    input, cudf::table_view{{keys}}, num_partitions, cudf::hash_id::HASH_IDENTITY);
  auto const output_strings = cudf::test::to_host<std::string>(output->view().column(0)).first;

  for (cudf::size_type partition = 0; partition < num_partitions; ++partition) {
    for (auto row = offsets[partition]; row < offsets[partition + 1]; ++row) {
      auto const source_row = std::stoi(output_strings[row].substr(4));
      EXPECT_EQ(static_cast<uint32_t>(key_values[source_row]) % num_partitions, partition);
    }
  }
}

TEST_F(FixedWidthHashPartitionTest, FixedAndVariableWidthPayloads)
{
  constexpr cudf::size_type num_rows = 4099;
  auto const values                  = cuda::counting_iterator<int32_t>{0};
  auto const strings                 = thrust::make_transform_iterator(
    values, [](auto value) { return std::string{"value_"} + std::to_string(value); });

  fixed_width_column_wrapper<int32_t> keys(values, values + num_rows);
  fixed_width_column_wrapper<int8_t, int32_t> bytes(values, values + num_rows);
  fixed_width_column_wrapper<int64_t, int32_t> longs(values, values + num_rows);
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    values, values + num_rows, numeric::scale_type{-2});
  strings_column_wrapper string_payload(strings, strings + num_rows);

  expect_murmur_partitioned(
    cudf::table_view{{keys, bytes, decimal128, string_payload, longs}}, {0}, 2049, 2468);
}

TEST_F(FixedWidthHashPartitionTest, HybridPayloadColumnsStayAligned)
{
  fixed_width_column_wrapper<int32_t> row_id{0, 1, 2, 3, 4, 5, 6, 7};
  fixed_width_column_wrapper<int64_t> keys{17, 3, 91, 42, 8, 77, 11, 4};
  fixed_width_column_wrapper<int16_t> nullable_fixed(
    {100, 101, 102, 103, 104, 105, 106, 107}, {true, true, true, true, true, true, true, true});
  strings_column_wrapper first_strings(
    {"zero", "one", "two", "three", "four", "five", "six", "seven"},
    {true, false, true, true, false, true, true, true});
  lists_column_wrapper<int32_t> lists{{0}, {1, 1}, {2}, {3, 3}, {4}, {5, 5}, {6}, {7, 7}};
  dictionary_column_wrapper<int32_t> dictionary{10, 11, 12, 13, 14, 15, 16, 17};
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007}, numeric::scale_type{-2});
  strings_column_wrapper second_strings(
    {"eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"},
    {false, true, true, true, true, false, true, true});

  auto const input = cudf::table_view{
    {row_id, first_strings, nullable_fixed, lists, keys, dictionary, decimal128, second_strings}};
  auto [output, offsets] = cudf::hash_partition(input, {4}, 5);
  auto sorted            = cudf::sort_by_key(output->view(), output->view().select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted->view());
  EXPECT_TRUE(output->view().column(2).nullable());
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), input.num_rows());
}

TEST_F(FixedWidthHashPartitionTest, PartitionOffsetsAcrossBlocksAndEmptyPartitions)
{
  constexpr cudf::size_type num_rows = 10'000;
  auto const rows                    = cuda::counting_iterator<cudf::size_type>{0};
  auto const keys = thrust::make_transform_iterator(rows, [](auto row) { return row % 4; });

  fixed_width_column_wrapper<cudf::size_type> input(keys, keys + num_rows);
  for (auto const num_partitions : {cudf::size_type{17}, cudf::size_type{1025}}) {
    auto [output, offsets] = cudf::hash_partition(
      cudf::table_view{{input}}, {0}, num_partitions, cudf::hash_id::HASH_IDENTITY);

    std::vector<cudf::size_type> expected_offsets(num_partitions + 1, num_rows);
    expected_offsets[0] = 0;
    expected_offsets[1] = num_rows / 4;
    expected_offsets[2] = num_rows / 2;
    expected_offsets[3] = 3 * num_rows / 4;
    EXPECT_EQ(offsets, expected_offsets);

    auto const output_keys = cudf::test::to_host<cudf::size_type>(output->view().column(0)).first;
    for (cudf::size_type partition = 0; partition < 4; ++partition) {
      EXPECT_TRUE(std::all_of(output_keys.begin() + offsets[partition],
                              output_keys.begin() + offsets[partition + 1],
                              [partition](auto key) { return key == partition; }));
    }
  }
}

}  // namespace
