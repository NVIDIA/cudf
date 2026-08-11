/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda/devices>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <numeric>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

// Transform vector of column wrappers to vector of column views
template <typename T>
auto make_view_vector(std::vector<T> const& columns)
{
  std::vector<cudf::column_view> views(columns.size());
  std::transform(columns.begin(), columns.end(), views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });
  return views;
}

class HashPartition : public cudf::test::BaseFixture {};

TEST_F(HashPartition, InvalidColumnsToHash)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({-1});

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(cudf::hash_partition(input, columns_to_hash, num_partitions), std::out_of_range);
}

TEST_F(HashPartition, InvalidKeyRows)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = cudf::table_view({floats});
  auto keys  = cudf::table_view({integers});

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(cudf::hash_partition(input, keys, num_partitions), std::invalid_argument);
}

TEST_F(HashPartition, ZeroPartitions)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 0;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and zero partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions + 1}, offsets.size());
}

TEST_F(HashPartition, ZeroRows)
{
  fixed_width_column_wrapper<float> floats({});
  fixed_width_column_wrapper<int16_t> integers({});
  strings_column_wrapper strings;
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and same number of partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions + 1}, offsets.size());
}

TEST_F(HashPartition, ZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and same number of partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions + 1}, offsets.size());
}

TEST_F(HashPartition, ZeroColumnsNonEmptyTable)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and same number of partitions
  EXPECT_TRUE(cudf::have_same_types(input, output->view()));
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions + 1}, offsets.size());
}

// Test is very memory hungry (due to hash_partition algorithm and therefore doesn't fit in CI
// runners).
TEST_F(HashPartition, DISABLED_LargeRowCountNoOverflow)
{
  // We are also restricted by thrust 32bit offsets, so this test is quite delicate.
  cudf::size_type const num_rows       = 200'000'000;
  cudf::size_type const num_partitions = 5128;
  auto value                           = cudf::numeric_scalar<bool>(true, /* is_valid = */ true);
  auto bools                           = cudf::make_column_from_scalar(value, num_rows);
  auto input                           = cudf::table_view({bools->view()});

  auto [output, offsets] = cudf::hash_partition(input, {0}, num_partitions);

  EXPECT_EQ(1, output->num_columns());
  EXPECT_EQ(num_rows, output->num_rows());
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());
  EXPECT_EQ(num_rows, offsets.back());
}

// Returns a partition count that exceeds shared memory capacity, guaranteeing
// the global-memory code path is exercised regardless of GPU architecture.
cudf::size_type partitions_exceeding_shared_memory()
{
  int dev;
  CUDF_CUDA_TRY(cudaGetDevice(&dev));
  auto const smem_size =
    cuda::device_attributes::max_shared_memory_per_block(cuda::device_ref{dev});
  return smem_size / sizeof(cudf::size_type) + 10;
}

TEST_F(HashPartition, MorePartitionsThanSharedMemory)
{
  cudf::size_type const num_rows       = 48 * 1024;
  cudf::size_type const num_partitions = partitions_exceeding_shared_memory();
  auto const value                     = cudf::numeric_scalar<bool>(true, /* is_valid = */ true);
  auto const bools                     = cudf::make_column_from_scalar(value, num_rows);
  auto const input                     = cudf::table_view({bools->view()});

  auto [output, offsets] = cudf::hash_partition(input, {0}, num_partitions);

  EXPECT_EQ(1, output->num_columns());
  EXPECT_EQ(num_rows, output->num_rows());
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());
  EXPECT_EQ(num_rows, offsets.back());
  EXPECT_EQ(0, offsets.front());
}

TEST_F(HashPartition, LargePartitionCountCorrectness)
{
  cudf::size_type const num_rows       = 1000;
  cudf::size_type const num_partitions = partitions_exceeding_shared_memory();
  auto const iter                      = cuda::counting_iterator<int32_t>{0};
  fixed_width_column_wrapper<int32_t> col(iter, iter + num_rows);
  auto const input = cudf::table_view({col});

  auto [output, offsets] = cudf::hash_partition(input, {0}, num_partitions);

  // Offsets are monotonically non-decreasing
  EXPECT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));
  EXPECT_EQ(0, offsets.front());
  EXPECT_EQ(num_rows, offsets.back());
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());

  // Output is a permutation of input: sort both and compare
  auto const sorted_input  = cudf::sort(input);
  auto const sorted_output = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_input->view(), sorted_output->view());
}

TEST_F(HashPartition, LargePartitionCountMixedTypes)
{
  cudf::size_type const num_rows       = 100;
  cudf::size_type const num_partitions = partitions_exceeding_shared_memory();
  auto const iter                      = cuda::counting_iterator<int32_t>{0};

  fixed_width_column_wrapper<float> floats(iter, iter + num_rows);
  fixed_width_column_wrapper<int16_t> integers(iter, iter + num_rows);
  std::vector<std::string> str_data(num_rows);
  std::generate(
    str_data.begin(), str_data.end(), [n = 0]() mutable { return "str_" + std::to_string(n++); });
  strings_column_wrapper strings(str_data.begin(), str_data.end());

  auto const input = cudf::table_view({floats, integers, strings});

  auto [output, offsets] = cudf::hash_partition(input, {0, 1}, num_partitions);

  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(num_rows, output->num_rows());
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());
  EXPECT_EQ(0, offsets.front());
  EXPECT_EQ(num_rows, offsets.back());
  EXPECT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));
}

TEST_F(HashPartition, LargePartitionCountWithNulls)
{
  cudf::size_type const num_rows       = 200;
  cudf::size_type const num_partitions = partitions_exceeding_shared_memory();
  auto const iter                      = cuda::counting_iterator<int32_t>{0};
  auto const valids = thrust::make_transform_iterator(iter, [](auto i) { return i % 4 != 0; });

  fixed_width_column_wrapper<int32_t> ints(iter, iter + num_rows, valids);
  auto const str_iter =
    thrust::make_transform_iterator(iter, [](auto i) { return std::to_string(i); });
  strings_column_wrapper strings(str_iter, str_iter + num_rows, valids);
  auto const input = cudf::table_view({ints, strings});

  auto [output, offsets] = cudf::hash_partition(input, {0}, num_partitions);

  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(num_rows, output->num_rows());
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());
  EXPECT_EQ(0, offsets.front());
  EXPECT_EQ(num_rows, offsets.back());
  EXPECT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));

  // Verify null count is preserved
  EXPECT_EQ(output->get_column(0).null_count(), num_rows / 4);
  EXPECT_EQ(output->get_column(1).null_count(), num_rows / 4);
}

TEST_F(HashPartition, MixedColumnTypes)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  auto [output1, offsets1] = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto [output2, offsets2] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions + 1
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets1.size());
  EXPECT_EQ(offsets1.size(), offsets2.size());

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());

  // Expect deterministic result from hashing the same input
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

TEST_F(HashPartition, NullableStrings)
{
  strings_column_wrapper strings({"a", "bb", "ccc", "d"}, {true, true, true, true});
  cudf::table_view input({strings});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  auto const& col = result->get_column(0);
  EXPECT_EQ(0, col.null_count());
}

TEST_F(HashPartition, ColumnsToHash)
{
  fixed_width_column_wrapper<int32_t> to_hash({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto first_input  = cudf::table_view({to_hash, first_col});
  auto second_input = cudf::table_view({to_hash, second_col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be num_partitions + 1 in length
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

TEST_F(HashPartition, IdentityHashFailure)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(
    cudf::hash_partition(input, columns_to_hash, num_partitions, cudf::hash_id::HASH_IDENTITY),
    cudf::logic_error);
}

TEST_F(HashPartition, CustomSeedValue)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  auto [output1, offsets1]             = cudf::hash_partition(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);
  auto [output2, offsets2] = cudf::hash_partition(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);

  // Expect output to have size num_partitions + 1
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets1.size());
  EXPECT_EQ(offsets1.size(), offsets2.size());

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());

  // Expect deterministic result from hashing the same input
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

template <typename T>
class HashPartitionFixedWidth : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(HashPartitionFixedWidth, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(HashPartitionFixedWidth, NullableFixedWidth)
{
  fixed_width_column_wrapper<TypeParam, int32_t> fixed({1, 2, 3, 4}, {1, 1, 1, 1});
  cudf::table_view input({fixed});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  auto const& col = result->get_column(0);
  EXPECT_EQ(0, col.null_count());
}

template <typename T>
void run_fixed_width_test(size_t cols,
                          size_t rows,
                          cudf::size_type num_partitions,
                          bool has_nulls = false)
{
  std::vector<fixed_width_column_wrapper<T, int32_t>> columns;
  columns.reserve(cols);
  if (has_nulls) {
    std::generate_n(std::back_inserter(columns), cols, [rows]() {
      auto iter   = cuda::counting_iterator<int32_t>{0};
      auto valids = thrust::make_transform_iterator(iter, [](auto i) { return i % 4 != 0; });
      return fixed_width_column_wrapper<T, int32_t>(iter, iter + rows, valids);
    });
  } else {
    std::generate_n(std::back_inserter(columns), cols, [rows]() {
      auto iter = cuda::counting_iterator<int32_t>{0};
      return fixed_width_column_wrapper<T, int32_t>(iter, iter + rows);
    });
  }
  auto input = cudf::table_view(make_view_vector(columns));

  auto columns_to_hash = std::vector<cudf::size_type>(cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  // Either directly hash
  auto [output1, offsets1] =
    cudf::hash_partition(input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3);
  // Or hash externally and use this key column as the mapping
  auto key_column          = cudf::hashing::murmurhash3_x86_32(input.select(columns_to_hash));
  auto [output2, offsets2] = cudf::hash_partition(
    input, cudf::table_view{{key_column->view()}}, num_partitions, cudf::hash_id::HASH_IDENTITY);

  // Expect output to have size num_partitions + 1
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets1.size());
  EXPECT_TRUE(std::equal(offsets1.begin(), offsets1.end(), offsets2.begin()));

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(output1->view(), output2->view());

  // Compute number of rows in each partition
  EXPECT_EQ(0, offsets1[0]);
  EXPECT_EQ(rows, offsets1[num_partitions]);
  std::adjacent_difference(offsets1.begin() + 1, offsets1.end(), offsets1.begin() + 1);

  // Compute the partition number for each row
  cudf::size_type partition = 0;
  std::vector<cudf::size_type> partitions;
  std::for_each(offsets1.begin() + 1, offsets1.end(), [&](cudf::size_type const& count) {
    std::fill_n(std::back_inserter(partitions), count, partition++);
  });

  // Make a table view of the partition numbers
  constexpr cudf::data_type dtype{cudf::type_id::INT32};
  auto d_partitions = cudf::detail::make_device_uvector(
    partitions, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::column_view partitions_col(dtype, rows, d_partitions.data(), nullptr, 0);
  cudf::table_view partitions_table({partitions_col});

  // Sort partition numbers by the corresponding row hashes of each output
  auto hash1 = cudf::hashing::murmurhash3_x86_32(output1->view());
  cudf::table_view hash1_table({hash1->view()});
  auto sorted_partitions1 = cudf::sort_by_key(partitions_table, hash1_table);

  auto hash2 = cudf::hashing::murmurhash3_x86_32(output2->view());
  cudf::table_view hash2_table({hash2->view()});
  auto sorted_partitions2 = cudf::sort_by_key(partitions_table, hash2_table);

  // After sorting by row hashes, the corresponding partition numbers should be
  // equal
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_partitions1->view(), sorted_partitions2->view());
}

TYPED_TEST(HashPartitionFixedWidth, MorePartitionsThanRows)
{
  run_fixed_width_test<TypeParam>(5, 10, 50);
}

TYPED_TEST(HashPartitionFixedWidth, LargeInput) { run_fixed_width_test<TypeParam>(10, 1000, 10); }

TYPED_TEST(HashPartitionFixedWidth, HasNulls)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10, true);
}

TEST_F(HashPartition, FixedPointColumnsToHash)
{
  fixed_width_column_wrapper<int32_t> to_hash({1});
  cudf::test::fixed_point_column_wrapper<int64_t> first_col({7}, numeric::scale_type{-1});
  cudf::test::fixed_point_column_wrapper<__int128_t> second_col({77}, numeric::scale_type{0});

  auto input = cudf::table_view({to_hash, first_col, second_col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 1;
  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect offsets to be num_partitions + 1 in length
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), offsets.size());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0).view(), input.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(1).view(), input.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(2).view(), input.column(2));
}

TEST_F(HashPartition, ListWithNulls)
{
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  lcw to_hash{{{1}, {2, 2}, {0}, {1}, {2, 2}, {0}}, nulls_at({2, 5})};
  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto const first_input  = cudf::table_view({to_hash, first_col});
  auto const second_input = cudf::table_view({to_hash, second_col});

  auto const columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto const [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto const [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be num_partitions + 1 in length
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

TEST_F(HashPartition, StructofStructWithNulls)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  { {1, 1}, 5}  |
  // 1 |  { {1, 1}, 5}  |
  // 2 |  { {1, 2}, 4}  |
  // 3 |  Null          |
  // 4 |  { Null,   4}  |
  // 5 |  { Null,   4}  |

  auto const to_hash = [&] {
    auto a  = fixed_width_column_wrapper<int32_t>{1, 1, 1, 0, 0, 0};
    auto b  = fixed_width_column_wrapper<int32_t>{1, 1, 2, 0, 0, 0};
    auto s2 = structs_col{{a, b}, nulls_at({4, 5})};

    auto c = fixed_width_column_wrapper<int32_t>{5, 5, 4, 0, 4, 4};
    return structs_col({s2, c}, null_at(3));
  }();

  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto const first_input  = cudf::table_view({to_hash, first_col});
  auto const second_input = cudf::table_view({to_hash, second_col});

  auto const columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto const [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto const [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be num_partitions + 1 in length
  EXPECT_EQ(static_cast<size_t>(num_partitions + 1), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

CUDF_TEST_PROGRAM_MAIN()
