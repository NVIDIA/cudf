/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/parquet.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <array>
#include <vector>

////////////////////////////////
// delta encoding writer tests

// Test fixture for delta encoding tests
template <typename T>
struct ParquetWriterDeltaTest : public ParquetWriterTest {};

TYPED_TEST_SUITE(ParquetWriterDeltaTest, SupportedDeltaTestTypes);

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypes)
{
  using T   = TypeParam;
  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPacked.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypesSliced)
{
  using T                = TypeParam;
  constexpr int num_rows = 4'000;
  auto col0              = testdata::ascending<T>();
  auto col1              = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};
  auto expected_slice = cudf::slice(expected, {num_rows, 2 * num_rows});
  ASSERT_EQ(expected_slice[0].num_rows(), num_rows);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaListSliced)
{
  using T = TypeParam;

  constexpr int num_slice = 4'000;
  constexpr int num_rows  = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = cuda::counting_iterator<int64_t>{0};

  // list<T>
  constexpr int vals_per_row = 4;
  auto c1_offset_iter        = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * vals_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<T> c1_vals(
    values, values + (num_rows * vals_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_vals.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  auto const expected = table_view{{*c1}};
  auto expected_slice = cudf::slice(expected, {num_slice, 2 * num_slice});
  ASSERT_EQ(expected_slice[0].num_rows(), num_slice);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedListSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

////////////////////////
// sized tests

// Base test fixture for size-parameterized tests
class ParquetSizedTest : public ::cudf::test::BaseFixtureWithParam<int> {};

// Test the allowed bit widths: [1, 25) for dictionary encoding
// Note: Using a step of 3 and avoiding bit width of 24 to reduce the test suite execution time
INSTANTIATE_TEST_SUITE_P(ParquetDictionaryTest,
                         ParquetSizedTest,
                         testing::Range(2, 24, 3),
                         testing::PrintToStringParamName());

TEST_P(ParquetSizedTest, DictionaryTest)
{
  unsigned int const cardinality = (1 << (GetParam() - 1)) + 1;
  unsigned int const nrows       = std::max(cardinality * 3 / 2, 500'000U);

  auto const elements = cudf::detail::make_counting_transform_iterator(
    0, [cardinality](auto i) { return std::to_string(i % cardinality); });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto buffer = std::vector<char>{};
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&buffer), expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .row_group_size_rows(nrows)
      .row_group_size_bytes(512 * 1024 * 1024);
  cudf::io::write_parquet(out_opts);

  auto const buffer_span =
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(buffer.data()), buffer.size());
  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer_span));
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(cudf::host_span<std::byte const>{buffer_span});
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto const used_dict =
    std::all_of(fmd.row_groups.front().columns.front().meta_data.encodings.begin(),
                fmd.row_groups.front().columns.front().meta_data.encodings.end(),
                [](auto const& enc) {
                  return enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
                         enc == cudf::io::parquet::Encoding::RLE_DICTIONARY;
                });
  EXPECT_TRUE(used_dict);

  // and check that the correct number of bits was used
  auto const oi = read_offset_index(source, fmd.row_groups.front().columns.front());
  std::vector<int> page_dict_bits;
  page_dict_bits.reserve(oi.page_locations.size());
  std::transform(
    oi.page_locations.begin(),
    oi.page_locations.end(),
    std::back_inserter(page_dict_bits),
    [&source](auto const& page_location) { return read_dict_bits(source, page_location); });

  auto const max_bits = std::max_element(page_dict_bits.begin(), page_dict_bits.end());
  ASSERT_NE(max_bits, page_dict_bits.end());
  EXPECT_EQ(*max_bits, GetParam());
}

///////////////////////
// comparable tests

// Typed test fixture for comparable type tests
template <typename T>
struct ParquetWriterComparableTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(ParquetWriterComparableTypeTest, ComparableAndFixedTypes);

TYPED_TEST(ParquetWriterComparableTypeTest, ThreeColumnSorted)
{
  using T = TypeParam;

  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::descending<T>();
  auto col2 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1, col2}};

  auto const filepath = temp_env->get_temp_filepath("ThreeColumnSorted.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_size_for_ordered_tests)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);

  auto const& columns = fmd.row_groups[0].columns;
  ASSERT_EQ(columns.size(), static_cast<size_t>(expected.num_columns()));

  // now check that the boundary order for chunk 1 is ascending,
  // chunk 2 is descending, and chunk 3 is unordered
  std::array expected_orders{cudf::io::parquet::BoundaryOrder::ASCENDING,
                             cudf::io::parquet::BoundaryOrder::DESCENDING,
                             cudf::io::parquet::BoundaryOrder::UNORDERED};

  for (std::size_t i = 0; i < columns.size(); i++) {
    auto const ci = read_column_index(source, columns[i]);
    EXPECT_EQ(ci.boundary_order, expected_orders[i]);
  }
}
