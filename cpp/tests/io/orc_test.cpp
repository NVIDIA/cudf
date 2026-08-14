/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression_common.hpp"
#include "io_test_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>

#include <array>
#include <numeric>
#include <type_traits>

template <typename T, typename SourceElementT = T>
using column_wrapper =
  std::conditional_t<std::is_same_v<T, cudf::string_view>,
                     cudf::test::strings_column_wrapper,
                     cudf::test::fixed_width_column_wrapper<T, SourceElementT>>;

using str_col     = column_wrapper<cudf::string_view>;
using bool_col    = column_wrapper<bool>;
using int8_col    = column_wrapper<int8_t>;
using int16_col   = column_wrapper<int16_t>;
using int32_col   = column_wrapper<int32_t>;
using int64_col   = column_wrapper<int64_t>;
using float32_col = column_wrapper<float>;
using float64_col = column_wrapper<double>;
using dec32_col   = cudf::test::fixed_point_column_wrapper<numeric::decimal32::rep>;
using dec64_col   = cudf::test::fixed_point_column_wrapper<numeric::decimal64::rep>;
using dec128_col  = cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep>;
using struct_col  = cudf::test::structs_column_wrapper;
template <typename T>
using list_col = cudf::test::lists_column_wrapper<T>;

using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  std::vector<column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    auto rand_elements =
      cudf::detail::make_counting_transform_iterator(0, [](T i) { return rand(); });
    if (include_validity) {
      src_cols[idx] = column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] = column_wrapper<T>(rand_elements, rand_elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](column_wrapper<T>& in) {
    auto ret                    = in.release();
    [[maybe_unused]] auto nulls = ret->has_nulls();  // pre-cache the null count
    return ret;
  });
  return std::make_unique<cudf::table>(std::move(columns));
}

// Base test fixture for tests
struct OrcWriterTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct OrcWriterNumericTypeTest : public OrcWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for timestamp type tests
template <typename T>
struct OrcWriterTimestampTypeTest : public OrcWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5351
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;
TYPED_TEST_SUITE(OrcWriterNumericTypeTest, SupportedTypes);
using SupportedTimestampTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<cudf::timestamp_D>>,
                       cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(OrcWriterTimestampTypeTest, SupportedTimestampTypes);

// Base test fixture for chunked writer tests
struct OrcChunkedWriterTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct OrcChunkedWriterNumericTypeTest : public OrcChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
TYPED_TEST_SUITE(OrcChunkedWriterNumericTypeTest, SupportedTypes);

// Test fixture for reader tests
struct OrcReaderTest : public cudf::test::BaseFixture {};

// Test fixture for statistics tests
struct OrcStatisticsTest : public cudf::test::BaseFixture {};

// Test fixture for metadata tests
struct OrcMetadataReaderTest : public cudf::test::BaseFixture {};

using OrcCompressionTest = CompressionTest<OrcWriterTest>;

using OrcDecompressionTest = DecompressionTest<OrcReaderTest>;

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

struct SkipRowTest {
  int test_calls{0};
  SkipRowTest() {}

  std::unique_ptr<table> get_expected_result(std::string const& filepath,
                                             int skip_rows,
                                             int file_num_rows,
                                             int read_num_rows)
  {
    auto sequence = cuda::counting_iterator{0};
    column_wrapper<int32_t, typename decltype(sequence)::value_type> input_col(
      sequence, sequence + file_num_rows);
    table_view input_table({input_col});

    cudf::io::orc_writer_options out_opts =
      cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input_table);
    cudf::io::write_orc(out_opts);

    auto begin_sequence = sequence, end_sequence = sequence;
    if (skip_rows < file_num_rows) {
      begin_sequence += skip_rows;
      end_sequence += std::min(skip_rows + read_num_rows, file_num_rows);
    }
    column_wrapper<int32_t, typename decltype(sequence)::value_type> output_col(begin_sequence,
                                                                                end_sequence);
    std::vector<std::unique_ptr<column>> output_cols;
    output_cols.push_back(output_col.release());
    return std::make_unique<table>(std::move(output_cols));
  }

  void test(int skip_rows, int file_num_rows, int read_num_rows)
  {
    auto filepath =
      temp_env->get_temp_filepath("SkipRowTest" + std::to_string(test_calls++) + ".orc");
    auto expected_result = get_expected_result(filepath, skip_rows, file_num_rows, read_num_rows);
    cudf::io::orc_reader_options in_opts =
      cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
        .use_index(false)
        .skip_rows(skip_rows)
        .num_rows(read_num_rows);
    auto result = cudf::io::read_orc(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_result->view(), result.tbl->view());
  }

  void test(int skip_rows, int file_num_rows)
  {
    auto filepath =
      temp_env->get_temp_filepath("SkipRowTest" + std::to_string(test_calls++) + ".orc");
    auto expected_result =
      get_expected_result(filepath, skip_rows, file_num_rows, file_num_rows - skip_rows);
    cudf::io::orc_reader_options in_opts =
      cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
        .use_index(false)
        .skip_rows(skip_rows);
    auto result = cudf::io::read_orc(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_result->view(), result.tbl->view());
  }
};

void expect_selected_nested_empty_struct_table(cudf::table_view expected, cudf::table_view actual)
{
  ASSERT_EQ(1, actual.num_columns());
  ASSERT_EQ(1, actual.column(0).num_children());
  ASSERT_EQ(cudf::type_id::STRUCT, actual.column(0).child(0).type().id());
  ASSERT_EQ(0, actual.column(0).child(0).num_children());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual);
}

void expect_selected_nested_empty_struct_round_trip(std::unique_ptr<cudf::column> input_column,
                                                    std::string const& filename)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(std::move(input_column));

  auto expected = std::make_unique<cudf::table>(std::move(input_columns));
  cudf::io::table_input_metadata metadata(expected->view());
  metadata.column_metadata[0].set_name("name");
  metadata.column_metadata[0].child(0).set_name("empty");

  auto const filepath = temp_env->get_temp_filepath(filename);
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .metadata(std::move(metadata))
      .build();
  cudf::io::write_orc(write_opts);

  auto const read_opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
                           .columns({"name"})
                           .build();
  auto result = cudf::io::read_orc(read_opts);
  expect_selected_nested_empty_struct_table(expected->view(), result.tbl->view());
}

}  // namespace

TYPED_TEST(OrcWriterNumericTypeTest, SingleColumn)
{
  auto sequence = cuda::counting_iterator{0};

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(sequence,
                                                                         sequence + num_rows);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(OrcWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumnWithNulls.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(OrcWriterTimestampTypeTest, Timestamps)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (std::rand() / 10); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(sequence,
                                                                         sequence + num_rows);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcTimestamps.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      .timestamp_type(this->type());
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(OrcWriterTimestampTypeTest, TimestampsWithNulls)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (std::rand() / 10); });
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcTimestampsWithNulls.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      .timestamp_type(this->type());
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(OrcWriterTimestampTypeTest, TimestampOverflow)
{
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return max - i; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(sequence,
                                                                         sequence + num_rows);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcTimestampOverflow.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      .timestamp_type(this->type());
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcWriterTest, MultiColumn)
{
  constexpr auto num_rows = 10;

  auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int64_t>(num_rows);

  bool_col col0(col0_data.begin(), col0_data.end());
  int8_col col1(col1_data.begin(), col1_data.end());
  int16_col col2(col2_data.begin(), col2_data.end());
  int32_col col3(col3_data.begin(), col3_data.end());
  float32_col col4(col4_data.begin(), col4_data.end());
  float64_col col5(col5_data.begin(), col5_data.end());
  dec128_col col6{col6_vals.begin(), col6_vals.end(), numeric::scale_type{12}};
  dec128_col col7{col6_vals.begin(), col6_vals.end(), numeric::scale_type{-12}};

  list_col<int64_t> col8{
    {9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}, {}, {-1, -2}};

  int32_col child_col{48, 27, 25, 31, 351, 351, 29, 15, -1, -99};
  struct_col col9{child_col};

  table_view expected({col0, col1, col2, col3, col4, col5, col6, col7, col8, col9});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("bools");
  expected_metadata.column_metadata[1].set_name("int8s");
  expected_metadata.column_metadata[2].set_name("int16s");
  expected_metadata.column_metadata[3].set_name("int32s");
  expected_metadata.column_metadata[4].set_name("floats");
  expected_metadata.column_metadata[5].set_name("doubles");
  expected_metadata.column_metadata[6].set_name("decimal_pos_scale");
  expected_metadata.column_metadata[7].set_name("decimal_neg_scale");
  expected_metadata.column_metadata[8].set_name("lists");
  expected_metadata.column_metadata[9].set_name("structs");

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcWriterTest, MultiColumnWithNulls)
{
  constexpr auto num_rows = 10;

  auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col0_mask = cudf::test::iterators::nulls_at_multiples_of(2);
  auto col1_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i < 2); });
  auto col3_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i >= 4 && i <= 6); });
  auto col5_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 8); });
  auto col6_mask = cudf::test::iterators::nulls_at_multiples_of(3);

  bool_col col0{col0_data.begin(), col0_data.end(), col0_mask};
  int8_col col1{col1_data.begin(), col1_data.end(), col1_mask};
  int16_col col2(col2_data.begin(), col2_data.end());
  int32_col col3{col3_data.begin(), col3_data.end(), col3_mask};
  float32_col col4{col4_data.begin(), col4_data.end(), col4_mask};
  float64_col col5{col5_data.begin(), col5_data.end(), col5_mask};
  dec64_col col6{col6_vals.begin(), col6_vals.end(), col6_mask, numeric::scale_type{2}};
  list_col<int32_t> col7{
    {{9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}, {}, {-1, -2}},
    col0_mask};
  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351, 29, 15, -1, -99}, {1, 0, 1, 1, 0, 1, 1, 1, 0, 1}};
  struct_col col8{{ages_col}, {0, 1, 1, 0, 1, 1, 0, 1, 1, 0}};
  table_view expected({col0, col1, col2, col3, col4, col5, col6, col7, col8});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("bools");
  expected_metadata.column_metadata[1].set_name("int8s");
  expected_metadata.column_metadata[2].set_name("int16s");
  expected_metadata.column_metadata[3].set_name("int32s");
  expected_metadata.column_metadata[4].set_name("floats");
  expected_metadata.column_metadata[5].set_name("doubles");
  expected_metadata.column_metadata[6].set_name("decimal");
  expected_metadata.column_metadata[7].set_name("lists");
  expected_metadata.column_metadata[8].set_name("structs");

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumnWithNulls.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcWriterTest, ReadZeroRows)
{
  auto sequence = cuda::counting_iterator{0};

  constexpr auto num_rows = 10;
  column_wrapper<int64_t, typename decltype(sequence)::value_type> col(sequence,
                                                                       sequence + num_rows);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      .num_rows(0);
  auto result = cudf::io::read_orc(in_opts);

  EXPECT_EQ(0, result.tbl->num_rows());
  EXPECT_EQ(1, result.tbl->num_columns());
}

TEST_F(OrcWriterTest, Strings)
{
  std::vector<char const*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);

  int32_col col0(seq_col0.begin(), seq_col0.end());
  str_col col1(strings.begin(), strings.end());
  float32_col col2(seq_col2.begin(), seq_col2.end());

  table_view expected({col0, col1, col2});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");

  auto filepath = temp_env->get_temp_filepath("OrcStrings.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcWriterTest, SlicedTable)
{
  // This test checks for writing zero copy, offsetted views into existing cudf tables

  std::vector<char const*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0  = random_values<int32_t>(num_rows);
  auto seq_col2  = random_values<float>(num_rows);
  auto vals_col3 = random_values<int32_t>(num_rows);

  int32_col col0(seq_col0.begin(), seq_col0.end());
  str_col col1(strings.begin(), strings.end());
  float32_col col2(seq_col2.begin(), seq_col2.end());
  dec64_col col3{vals_col3.begin(), vals_col3.end(), numeric::scale_type{2}};

  list_col<int64_t> col4{
    {9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}};

  int16_col ages_col{{48, 27, 25, 31, 351, 351, 29, 15}, cudf::test::iterators::null_at(5)};
  struct_col col5{{ages_col}, cudf::test::iterators::null_at(4)};

  table_view expected({col0, col1, col2, col3, col4, col5});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");
  expected_metadata.column_metadata[3].set_name("col_decimal");
  expected_metadata.column_metadata[4].set_name("lists");
  expected_metadata.column_metadata[5].set_name("structs");

  auto expected_slice = cudf::slice(expected, {2, static_cast<cudf::size_type>(num_rows)});

  auto filepath = temp_env->get_temp_filepath("SlicedTable.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcWriterTest, HostBuffer)
{
  constexpr auto num_rows = 100 << 10;
  auto const seq_col      = random_values<int>(num_rows);
  int32_col col(seq_col.begin(), seq_col.end());

  table_view expected{{col}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info(&out_buffer), expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .use_index(false);
  auto const result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcWriterTest, negTimestampsNano)
{
  // This is a separate test because ORC format has a bug where writing a timestamp between -1 and 0
  // seconds from UNIX epoch is read as that timestamp + 1 second. We mimic that behavior and so
  // this test has to hardcode test values which are < -1 second.
  // Details: https://github.com/rapidsai/cudf/pull/5529#issuecomment-648768925
  auto timestamps_ns =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>{
      -131968727238000000,
      -1530705634500000000,
      -1674638741932929000,
    };
  cudf::table_view expected({timestamps_ns});

  auto filepath = temp_env->get_temp_filepath("OrcNegTimestamp.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);

  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected.column(0), result.tbl->view().column(0), cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcWriterTest, Slice)
{
  int32_col col{{1, 2, 3, 4, 5}, cudf::test::iterators::null_at(3)};
  std::vector<cudf::size_type> indices{2, 5};
  std::vector<cudf::column_view> result = cudf::slice(col, indices);
  cudf::table_view tbl{result};

  auto filepath = temp_env->get_temp_filepath("Slice.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto read_table = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(read_table.tbl->view(), tbl);
}

TEST_F(OrcWriterTest, BoolColumnLongRLE)
{
  // Test boolean column with extremely long RLE runs
  // Pattern: 130 trues followed by 130 falses, repeating for 10,000 rows
  constexpr auto num_rows   = 10000;
  constexpr auto run_length = 130;

  std::vector<bool> bool_data(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    // Alternating runs of 130 trues and 130 falses
    bool_data[i] = ((i / run_length) % 2) == 0;
  }

  bool_col col(bool_data.begin(), bool_data.end());
  table_view expected({col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("bools_long_rle");

  auto filepath = temp_env->get_temp_filepath("BoolColumnLongRLE.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(OrcChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1).write(*table2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, RootStatisticsAccumulateRows)
{
  auto table1 = create_random_fixed_table<int>(1, 5, false);
  auto table2 = create_random_fixed_table<int>(1, 1, false);
  auto table3 = create_random_fixed_table<int>(1, 0, false);

  auto filepath = temp_env->get_temp_filepath("ChunkedRootStatistics.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1).write(*table2).write(*table3);

  auto const stats = cudf::io::read_parsed_orc_statistics(cudf::io::source_info{filepath});
  ASSERT_FALSE(stats.file_stats.empty());
  ASSERT_TRUE(stats.file_stats.front().number_of_values.has_value());
  EXPECT_EQ(*stats.file_stats.front().number_of_values, 6);
}

TEST_F(OrcChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1).write(*table2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, ManyTables)
{
  srand(31337);
  std::vector<std::unique_ptr<table>> tables;
  std::vector<table_view> table_views;
  constexpr int num_tables = 96;
  for (int idx = 0; idx < num_tables; idx++) {
    auto tbl = create_random_fixed_table<int>(16, 64, true);
    table_views.push_back(*tbl);
    tables.push_back(std::move(tbl));
  }

  auto expected = cudf::concatenate(table_views);

  auto filepath = temp_env->get_temp_filepath("ChunkedManyTables.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer writer(opts);
  std::for_each(table_views.begin(), table_views.end(), [&writer](table_view const& tbl) {
    writer.write(tbl);
  });
  writer.close();

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(OrcChunkedWriterTest, Metadata)
{
  std::vector<char const*> strings{
    "Monday", "Tuesday", "THURSDAY", "Wednesday", "Friday", "Sunday", "Saturday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);

  int32_col col0(seq_col0.begin(), seq_col0.end());
  str_col col1{strings.begin(), strings.end()};
  float32_col col2(seq_col2.begin(), seq_col2.end());

  table_view expected({col0, col1, col2});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");

  auto filepath = temp_env->get_temp_filepath("ChunkedMetadata.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(expected_metadata);
  cudf::io::orc_chunked_writer(opts).write(expected).write(expected);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcChunkedWriterTest, Strings)
{
  std::array mask1{true, true, false, true, true, true, true};
  std::vector<char const*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  str_col strings1(h_strings1.begin(), h_strings1.end(), mask1.data());
  table_view tbl1({strings1});

  std::array mask2{false, true, true, true, true, true, true};
  std::vector<char const*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  str_col strings2(h_strings2.begin(), h_strings2.end(), mask2.data());
  table_view tbl2({strings2});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(tbl1).write(tbl2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(OrcChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer writer(opts);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
}

TEST_F(OrcChunkedWriterTest, ChunkedWritingAfterClosing)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedWritingAfterClosing.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer writer(opts);
  writer.write(*table1);
  writer.close();
  EXPECT_THROW(writer.write(*table1), cudf::logic_error);
}

TEST_F(OrcChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<int>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer writer(opts);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
}

TEST_F(OrcChunkedWriterTest, ReadStripes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table2, *table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedStripes.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1).write(*table2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).stripes({{1, 0, 1}});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, ReadStripesError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedStripesError.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(*table1);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).stripes({{0, 1}});
  EXPECT_THROW(cudf::io::read_orc(read_opts), cudf::logic_error);
  read_opts.set_stripes({{-1}});
  EXPECT_THROW(cudf::io::read_orc(read_opts), cudf::logic_error);
}

TYPED_TEST(OrcChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  constexpr int num_els{31};

  std::array<bool, num_els> mask{false, true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true};

  std::array<T, num_els> c1a;
  std::fill(c1a.begin(), c1a.end(), static_cast<T>(5));
  std::array<T, num_els> c1b;
  std::fill(c1b.begin(), c1b.end(), static_cast<T>(5));
  column_wrapper<T> c1a_w(c1a.begin(), c1a.end(), mask.begin());
  column_wrapper<T> c1b_w(c1b.begin(), c1b.end(), mask.begin());
  table_view tbl1({c1a_w, c1b_w});

  std::array<T, num_els> c2a;
  std::fill(c2a.begin(), c2a.end(), static_cast<T>(8));
  std::array<T, num_els> c2b;
  std::fill(c2b.begin(), c2b.end(), static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a.begin(), c2a.end(), mask.begin());
  column_wrapper<T> c2b_w(c2b.begin(), c2b.end(), mask.begin());
  table_view tbl2({c2a_w, c2b_w});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(tbl1).write(tbl2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(OrcChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  constexpr int num_els = 33;

  std::array<bool, num_els> mask{false, true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true};

  std::array<T, num_els> c1a;
  std::fill(c1a.begin(), c1a.end(), static_cast<T>(5));
  std::array<T, num_els> c1b;
  std::fill(c1b.begin(), c1b.end(), static_cast<T>(5));
  column_wrapper<T> c1a_w(c1a.begin(), c1a.end(), mask.begin());
  column_wrapper<T> c1b_w(c1b.begin(), c1b.end(), mask.begin());
  table_view tbl1({c1a_w, c1b_w});

  std::array<T, num_els> c2a;
  std::fill(c2a.begin(), c2a.end(), static_cast<T>(8));
  std::array<T, num_els> c2b;
  std::fill(c2b.begin(), c2b.end(), static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a.begin(), c2a.end(), mask.begin());
  column_wrapper<T> c2b_w(c2b.begin(), c2b.end(), mask.begin());
  table_view tbl2({c2a_w, c2b_w});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.orc");
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::orc_chunked_writer(opts).write(tbl1).write(tbl2);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(OrcReaderTest, CombinedSkipRowTest)
{
  SkipRowTest skip_row;
  skip_row.test(50, 75);
  skip_row.test(2, 100);
  skip_row.test(2, 100, 50);
  skip_row.test(2, 100, 98);
  skip_row.test(2, 100, 99);
  skip_row.test(2, 100, 100);
  skip_row.test(2, 100, 110);
}

TEST_F(OrcStatisticsTest, Basic)
{
  auto sequence = cuda::counting_iterator{0};
  auto ts_sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i - 4) * 1000002; });
  auto dec_sequence =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return i * 1001; });
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  std::vector<char const*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Wednesday", "Tuesday"};
  int num_rows = strings.size();

  column_wrapper<int32_t, typename decltype(sequence)::value_type> col1(
    sequence, sequence + num_rows, validity);
  column_wrapper<float, typename decltype(sequence)::value_type> col2(
    sequence, sequence + num_rows, validity);
  str_col col3{strings.begin(), strings.end()};
  column_wrapper<cudf::timestamp_ns, typename decltype(sequence)::value_type> col4(
    ts_sequence, ts_sequence + num_rows, validity);
  column_wrapper<cudf::timestamp_us, typename decltype(sequence)::value_type> col5(
    ts_sequence, ts_sequence + num_rows, validity);
  bool_col col6({true, true, true, true, true, false, false, false, false}, validity);

  cudf::test::fixed_point_column_wrapper<int64_t> col7(
    dec_sequence, dec_sequence + num_rows, numeric::scale_type{-1});

  table_view expected({col1, col2, col3, col4, col5, col6, col7});

  auto filepath = temp_env->get_temp_filepath("OrcStatsMerge.orc");

  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  auto const stats = cudf::io::read_parsed_orc_statistics(cudf::io::source_info{filepath});

  auto expected_column_names = std::vector<std::string>{""};
  std::generate_n(
    std::back_inserter(expected_column_names),
    expected.num_columns(),
    [starting_index = 0]() mutable { return "_col" + std::to_string(starting_index++); });
  EXPECT_EQ(stats.column_names, expected_column_names);

  auto validate_statistics = [&](std::vector<cudf::io::column_statistics> const& stats) {
    ASSERT_EQ(stats.size(), expected.num_columns() + 1);
    auto& s0 = stats[0];
    EXPECT_EQ(*s0.number_of_values, 9ul);
    EXPECT_TRUE(s0.has_null.has_value());
    EXPECT_FALSE(*s0.has_null);

    auto& s1 = stats[1];
    EXPECT_EQ(*s1.number_of_values, 4ul);
    EXPECT_TRUE(*s1.has_null);
    auto& ts1 = std::get<cudf::io::integer_statistics>(s1.type_specific_stats);
    EXPECT_EQ(*ts1.minimum, 1);
    EXPECT_EQ(*ts1.maximum, 7);
    EXPECT_EQ(*ts1.sum, 16);

    auto& s2 = stats[2];
    EXPECT_EQ(*s2.number_of_values, 4ul);
    EXPECT_TRUE(*s2.has_null);
    auto& ts2 = std::get<cudf::io::double_statistics>(s2.type_specific_stats);
    EXPECT_EQ(*ts2.minimum, 1.);
    EXPECT_EQ(*ts2.maximum, 7.);
    EXPECT_EQ(*ts2.sum, 16.);

    auto& s3 = stats[3];
    EXPECT_EQ(*s3.number_of_values, 9ul);
    EXPECT_FALSE(*s3.has_null);
    auto& ts3 = std::get<cudf::io::string_statistics>(s3.type_specific_stats);
    EXPECT_EQ(*ts3.minimum, "Friday");
    EXPECT_EQ(*ts3.maximum, "Wednesday");
    EXPECT_EQ(*ts3.sum, 58ul);

    auto& s4 = stats[4];
    EXPECT_EQ(*s4.number_of_values, 4ul);
    EXPECT_TRUE(*s4.has_null);
    auto& ts4 = std::get<cudf::io::timestamp_statistics>(s4.type_specific_stats);
    EXPECT_EQ(*ts4.minimum, -4);
    EXPECT_EQ(*ts4.maximum, 3);
    EXPECT_EQ(*ts4.minimum_utc, -4);
    EXPECT_EQ(*ts4.maximum_utc, 3);
    EXPECT_EQ(*ts4.minimum_nanos, 999994);
    EXPECT_EQ(*ts4.maximum_nanos, 6);

    auto& s5 = stats[5];
    EXPECT_EQ(*s5.number_of_values, 4ul);
    EXPECT_TRUE(*s5.has_null);
    auto& ts5 = std::get<cudf::io::timestamp_statistics>(s5.type_specific_stats);
    EXPECT_EQ(*ts5.minimum, -3001);
    EXPECT_EQ(*ts5.maximum, 3000);
    EXPECT_EQ(*ts5.minimum_utc, -3001);
    EXPECT_EQ(*ts5.maximum_utc, 3000);
    EXPECT_EQ(*ts5.minimum_nanos, 994000);
    EXPECT_EQ(*ts5.maximum_nanos, 6000);

    auto& s6 = stats[6];
    EXPECT_EQ(*s6.number_of_values, 4ul);
    EXPECT_TRUE(*s6.has_null);
    auto& ts6 = std::get<cudf::io::bucket_statistics>(s6.type_specific_stats);
    EXPECT_EQ(ts6.count[0], 2);

    auto& s7 = stats[7];
    EXPECT_EQ(*s7.number_of_values, 9ul);
    EXPECT_FALSE(*s7.has_null);
    auto& ts7 = std::get<cudf::io::decimal_statistics>(s7.type_specific_stats);
    EXPECT_EQ(*ts7.minimum, "0.0");
    EXPECT_EQ(*ts7.maximum, "800.8");
    EXPECT_EQ(*ts7.sum, "3603.6");
  };

  validate_statistics(stats.file_stats);
  // There's only one stripe, so column stats are the same as stripe stats
  validate_statistics(stats.stripes_stats[0]);
}

TEST_F(OrcWriterTest, SlicedValidMask)
{
  std::vector<char const*> strings;
  // Need more than 32 elements to reproduce the issue
  for (int i = 0; i < 34; ++i)
    strings.emplace_back("a long string to make sure overflow affects the output");
  // An element is null only to enforce the output column to be nullable
  str_col col{strings.begin(), strings.end(), cudf::test::iterators::null_at(32)};

  // Bug tested here is easiest to reproduce when column_offset % 32 is 31
  std::vector<cudf::size_type> indices{31, 34};
  auto sliced_col = cudf::slice(static_cast<cudf::column_view>(col), indices);
  cudf::table_view tbl{sliced_col};

  cudf::io::table_input_metadata expected_metadata(tbl);
  expected_metadata.column_metadata[0].set_name("col_string");

  auto filepath = temp_env->get_temp_filepath("OrcStrings.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcReaderTest, ZeroColumnsPreservesRowCount)
{
  GTEST_SKIP() << "Zero-column / N-row ORC reads are not yet supported. See "
                  "https://github.com/rapidsai/cudf/issues/22935).";

  constexpr cudf::size_type num_rows = 8;
  cudf::test::fixed_width_column_wrapper<int32_t> col{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::table_view input{{col}};

  auto filepath = temp_env->get_temp_filepath("OrcZeroColumns.orc");
  cudf::io::write_orc(cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input));

  // Project no columns: the result should be (num_rows, 0), not (0, 0).
  auto in_opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
                   .columns(std::vector<std::string>{})
                   .build();
  auto result = cudf::io::read_orc(in_opts);

  EXPECT_EQ(result.tbl->view().num_columns(), 0);
  EXPECT_EQ(result.tbl->view().num_rows(), num_rows);
}

TEST_F(OrcReaderTest, SingleInputs)
{
  srand(31533);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath1 = temp_env->get_temp_filepath("SimpleTable1.orc");
  cudf::io::orc_writer_options write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath1}, table1->view());
  cudf::io::write_orc(write_opts);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{{filepath1}});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(OrcReaderTest, zstdCompressionRegression)
{
  // Test with zstd compressed orc file with high compression ratio.
  constexpr std::array<uint8_t, 170> input_buffer{
    0x4f, 0x52, 0x43, 0x5a, 0x00, 0x00, 0x28, 0xb5, 0x2f, 0xfd, 0xa4, 0x34, 0xc7, 0x03, 0x00, 0x74,
    0x00, 0x00, 0x18, 0x41, 0xff, 0xaa, 0x02, 0x00, 0xbb, 0xff, 0x45, 0xc8, 0x01, 0x25, 0x30, 0x04,
    0x65, 0x00, 0x00, 0x10, 0xaa, 0x1f, 0x02, 0x00, 0x01, 0x29, 0x0b, 0xc7, 0x39, 0xb8, 0x02, 0xcb,
    0xaf, 0x38, 0xc0, 0x07, 0x00, 0x00, 0x40, 0x01, 0xc0, 0x05, 0x00, 0x00, 0x46, 0x4d, 0x45, 0x00,
    0x00, 0x0a, 0x06, 0x08, 0x01, 0x10, 0x01, 0x18, 0x30, 0x0a, 0x06, 0x08, 0x02, 0x10, 0x01, 0x18,
    0x06, 0x0a, 0x06, 0x08, 0x03, 0x10, 0x01, 0x18, 0x05, 0x12, 0x02, 0x08, 0x00, 0x12, 0x04, 0x08,
    0x03, 0x10, 0x02, 0x59, 0x00, 0x00, 0x08, 0x03, 0x10, 0x63, 0x1a, 0x0c, 0x08, 0x03, 0x10, 0x00,
    0x18, 0x3b, 0x20, 0x25, 0x28, 0xa0, 0x9e, 0x75, 0x22, 0x10, 0x08, 0x0c, 0x12, 0x01, 0x01, 0x1a,
    0x09, 0x63, 0x64, 0x5f, 0x67, 0x65, 0x6e, 0x64, 0x65, 0x72, 0x22, 0x02, 0x08, 0x07, 0x30, 0xa0,
    0x9e, 0x75, 0x08, 0x2f, 0x10, 0x05, 0x18, 0x80, 0x80, 0x10, 0x22, 0x02, 0x00, 0x0c, 0x28, 0x00,
    0x30, 0x09, 0x82, 0xf4, 0x03, 0x03, 0x4f, 0x52, 0x43, 0x17};

  auto source = cudf::io::source_info{cudf::host_span<std::byte const>{
    reinterpret_cast<std::byte const*>(input_buffer.data()), input_buffer.size()}};
  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(source).use_index(false);

  cudf::io::table_with_metadata result;
  CUDF_EXPECT_NO_THROW(result = cudf::io::read_orc(in_opts));
  EXPECT_EQ(1920800, result.tbl->num_rows());
}

TEST_F(OrcReaderTest, MultipleInputs)
{
  srand(31537);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto const filepath1 = temp_env->get_temp_filepath("SimpleTable1.orc");
  {
    cudf::io::orc_writer_options out_opts =
      cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath1}, table1->view());
    cudf::io::write_orc(out_opts);
  }

  auto const filepath2 = temp_env->get_temp_filepath("SimpleTable2.orc");
  {
    cudf::io::orc_writer_options out_opts =
      cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath2}, table2->view());
    cudf::io::write_orc(out_opts);
  }

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}});
  auto result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

struct OrcWriterTestDecimal
  : public OrcWriterTest,
    public ::testing::WithParamInterface<std::tuple<int, int, cudf::io::compression_type>> {};

TEST_P(OrcWriterTestDecimal, Decimal64)
{
  auto const [num_rows, scale, compression] = GetParam();

  // Using int16_t because scale causes values to overflow if they already require 32 bits
  auto const vals = random_values<int32_t>(num_rows);
  auto mask       = cudf::test::iterators::valids_at_multiples_of(7);
  dec64_col col{vals.begin(), vals.end(), mask, numeric::scale_type{scale}};
  cudf::table_view tbl({static_cast<cudf::column_view>(col)});

  auto filepath = temp_env->get_temp_filepath("Decimal64.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .compression(compression);

  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(tbl.column(0), result.tbl->view().column(0));
}

// The cases with more than 10000 rows and no compression test the writer's non-compaction path,
// where encoded streams are written straight from the encoder output, because decimal data stream
// sizes are known exactly up front and uncompressed streams get no alignment padding, which leaves
// the chunks of a multi-rowgroup stripe already contiguous.
INSTANTIATE_TEST_CASE_P(OrcWriterTest,
                        OrcWriterTestDecimal,
                        ::testing::Combine(::testing::Values(1, 10000, 10001, 34567),
                                           ::testing::Values(-2, 0, 2),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::NONE)));

TEST_F(OrcWriterTest, Decimal32)
{
  constexpr auto num_rows = 12000;

  // Using int16_t because scale causes values to overflow if they already require 32 bits
  auto const vals = random_values<int16_t>(num_rows);
  auto mask       = cudf::test::iterators::nulls_at_multiples_of(13);
  dec32_col col{vals.begin(), vals.end(), mask, numeric::scale_type{2}};
  cudf::table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("Decimal32.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);

  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, result.tbl->view().column(0));
}

TEST_F(OrcStatisticsTest, Overflow)
{
  int num_rows       = 10;
  auto too_large_seq = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i * (std::numeric_limits<int64_t>::max() / 20); });
  auto too_small_seq = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i * (std::numeric_limits<int64_t>::min() / 20); });
  auto not_too_large_seq = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i * (std::numeric_limits<int64_t>::max() / 200); });
  auto not_too_small_seq = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i * (std::numeric_limits<int64_t>::min() / 200); });
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  column_wrapper<int64_t, typename decltype(too_large_seq)::value_type> col1(
    too_large_seq, too_large_seq + num_rows, validity);
  column_wrapper<int64_t, typename decltype(too_small_seq)::value_type> col2(
    too_small_seq, too_small_seq + num_rows, validity);
  column_wrapper<int64_t, typename decltype(not_too_large_seq)::value_type> col3(
    not_too_large_seq, not_too_large_seq + num_rows, validity);
  column_wrapper<int64_t, typename decltype(not_too_small_seq)::value_type> col4(
    not_too_small_seq, not_too_small_seq + num_rows, validity);
  table_view tbl({col1, col2, col3, col4});

  auto filepath = temp_env->get_temp_filepath("OrcStatsOverflow.orc");

  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_orc(out_opts);

  auto const stats = cudf::io::read_parsed_orc_statistics(cudf::io::source_info{filepath});

  auto check_sum_exist = [&](int idx, bool expected) {
    auto const& s  = stats.file_stats[idx];
    auto const& ts = std::get<cudf::io::integer_statistics>(s.type_specific_stats);
    EXPECT_EQ(ts.sum.has_value(), expected);
  };
  check_sum_exist(1, false);
  check_sum_exist(2, false);
  check_sum_exist(3, true);
  check_sum_exist(4, true);
}

TEST_F(OrcStatisticsTest, HasNull)
{
  // This test can now be implemented with libcudf; keeping the pandas version to keep the test
  // inputs diversified
  // Method to create file:
  // >>> import pandas as pd
  // >>> df = pd.DataFrame({'a':pd.Series([1, 2, None], dtype="Int64"), 'b':[3, 4, 5]})
  // >>> df.to_orc("temp.orc")
  //
  // Contents of file:
  // >>> import pyarrow.orc as po
  // >>> po.ORCFile('temp.orc').read()
  // pyarrow.Table
  // a: int64
  // b: int64
  // ----
  // a: [[1,2,null]]
  // b: [[3,4,5]]
  auto nulls_orc = std::array<uint8_t, 308>{
    0x4F, 0x52, 0x43, 0x1D, 0x00, 0x00, 0x0A, 0x0C, 0x0A, 0x04, 0x00, 0x00, 0x00, 0x00, 0x12, 0x04,
    0x08, 0x03, 0x50, 0x00, 0x2C, 0x00, 0x00, 0xE3, 0x12, 0xE7, 0x62, 0x67, 0x80, 0x00, 0x21, 0x1E,
    0x0E, 0x26, 0x21, 0x36, 0x0E, 0x26, 0x01, 0x16, 0x09, 0xB6, 0x00, 0x46, 0x00, 0x2C, 0x00, 0x00,
    0xE3, 0x12, 0xE7, 0x62, 0x67, 0x80, 0x00, 0x21, 0x1E, 0x0E, 0x66, 0x21, 0x36, 0x0E, 0x36, 0x01,
    0x2E, 0x09, 0x89, 0x00, 0x06, 0x00, 0x05, 0x00, 0x00, 0xFF, 0xE0, 0x05, 0x00, 0x00, 0xFF, 0xC0,
    0x07, 0x00, 0x00, 0x46, 0x01, 0x24, 0x05, 0x00, 0x00, 0xFF, 0xE0, 0x09, 0x00, 0x00, 0x46, 0x02,
    0x68, 0xA0, 0x68, 0x00, 0x00, 0xE3, 0x62, 0xE3, 0x60, 0x13, 0x60, 0x90, 0x10, 0xE4, 0x02, 0xD1,
    0x8C, 0x12, 0x92, 0x60, 0x9A, 0x09, 0x4C, 0x33, 0x00, 0xC5, 0x59, 0xC1, 0x34, 0x23, 0x98, 0x66,
    0x04, 0xD2, 0x6C, 0x60, 0x3E, 0x13, 0x94, 0xCF, 0x24, 0xC1, 0x2E, 0xC4, 0x02, 0x52, 0x07, 0x24,
    0x99, 0x60, 0xA4, 0x14, 0x73, 0x68, 0x88, 0x33, 0x00, 0x46, 0x00, 0x00, 0xE3, 0x52, 0xE2, 0x62,
    0xE1, 0x60, 0x0E, 0x60, 0xE0, 0xE2, 0xE1, 0x60, 0x12, 0x62, 0xE3, 0x60, 0x12, 0x60, 0x91, 0x60,
    0x0B, 0x60, 0x04, 0xF2, 0x98, 0x81, 0x3C, 0x36, 0x01, 0x2E, 0x09, 0x89, 0x00, 0x06, 0x00, 0xB4,
    0x00, 0x00, 0xE3, 0x60, 0x16, 0x98, 0xC6, 0x28, 0xC5, 0xC5, 0xC1, 0x2C, 0xE0, 0x2C, 0x21, 0xA3,
    0x60, 0xAE, 0xC1, 0xAC, 0x24, 0xC4, 0xC1, 0x23, 0xC4, 0xC4, 0xC8, 0x24, 0xC5, 0x98, 0x28, 0xC5,
    0x98, 0xA4, 0xC0, 0xA0, 0xC1, 0x60, 0xC0, 0xA0, 0xC4, 0xC1, 0xC1, 0x82, 0xCE, 0x32, 0x60, 0xB6,
    0x62, 0xE1, 0x60, 0x0E, 0x60, 0xB0, 0xE2, 0xE1, 0x60, 0x12, 0x62, 0xE3, 0x60, 0x12, 0x60, 0x91,
    0x60, 0x0B, 0x60, 0x04, 0xF2, 0x98, 0x81, 0x3C, 0x36, 0x01, 0x2E, 0x09, 0x89, 0x00, 0x06, 0x87,
    0x09, 0x7E, 0x1E, 0x8C, 0x49, 0xAC, 0x86, 0x7A, 0xE6, 0x7A, 0xA6, 0x00, 0x08, 0x5D, 0x10, 0x01,
    0x18, 0x80, 0x80, 0x04, 0x22, 0x02, 0x00, 0x0C, 0x28, 0x26, 0x30, 0x06, 0x82, 0xF4, 0x03, 0x03,
    0x4F, 0x52, 0x43, 0x17,
  };

  auto const stats =
    cudf::io::read_parsed_orc_statistics(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(nulls_orc.data()), nulls_orc.size()}});

  EXPECT_EQ(stats.file_stats[1].has_null, true);
  EXPECT_EQ(stats.file_stats[2].has_null, false);

  EXPECT_EQ(stats.stripes_stats[0][1].has_null, true);
  EXPECT_EQ(stats.stripes_stats[0][2].has_null, false);
}

struct OrcWriterTestStripes
  : public OrcWriterTest,
    public ::testing::WithParamInterface<std::tuple<size_t, cudf::size_type>> {};

TEST_P(OrcWriterTestStripes, StripeSize)
{
  constexpr auto num_rows            = 1000000;
  auto const [size_bytes, size_rows] = GetParam();

  auto const seq_col  = random_values<int>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();
  column_wrapper<int64_t> col{seq_col.begin(), seq_col.end(), validity};

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto const expected = std::make_unique<table>(std::move(cols));

  auto validate =
    [&, &size_bytes = size_bytes, &size_rows = size_rows](std::vector<char> const& orc_buffer) {
      auto const expected_stripe_num =
        std::max<cudf::size_type>(num_rows / size_rows, (num_rows * sizeof(int64_t)) / size_bytes);
      auto const stats =
        cudf::io::read_parsed_orc_statistics(cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(orc_buffer.data()), orc_buffer.size()}});
      EXPECT_EQ(stats.stripes_stats.size(), expected_stripe_num);

      cudf::io::orc_reader_options in_opts =
        cudf::io::orc_reader_options::builder(
          cudf::io::source_info{cudf::host_span<std::byte const>{
            reinterpret_cast<std::byte const*>(orc_buffer.data()), orc_buffer.size()}})
          .use_index(false);
      auto result = cudf::io::read_orc(in_opts);

      CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
    };

  {
    std::vector<char> out_buffer_chunked;
    cudf::io::chunked_orc_writer_options opts =
      cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info(&out_buffer_chunked))
        .stripe_size_rows(size_rows)
        .stripe_size_bytes(size_bytes);
    cudf::io::orc_chunked_writer(opts).write(expected->view());
    validate(out_buffer_chunked);
  }
  {
    std::vector<char> out_buffer;
    cudf::io::orc_writer_options out_opts =
      cudf::io::orc_writer_options::builder(cudf::io::sink_info(&out_buffer), expected->view())
        .stripe_size_rows(size_rows)
        .stripe_size_bytes(size_bytes);
    cudf::io::write_orc(out_opts);
    validate(out_buffer);
  }
}

INSTANTIATE_TEST_CASE_P(OrcWriterTest,
                        OrcWriterTestStripes,
                        ::testing::Values(std::make_tuple(800000ul, 1000000),
                                          std::make_tuple(2000000ul, 1000000),
                                          std::make_tuple(4000000ul, 1000000),
                                          std::make_tuple(8000000ul, 1000000),
                                          std::make_tuple(8000000ul, 500000),
                                          std::make_tuple(8000000ul, 250000),
                                          std::make_tuple(8000000ul, 100000)));

TEST_F(OrcWriterTest, StripeSizeInvalid)
{
  auto const unused_table = std::make_unique<table>();
  std::vector<char> out_buffer;

  EXPECT_THROW(
    cudf::io::orc_writer_options::builder(cudf::io::sink_info(&out_buffer), unused_table->view())
      .stripe_size_rows(511),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::io::orc_writer_options::builder(cudf::io::sink_info(&out_buffer), unused_table->view())
      .stripe_size_bytes(63 << 10),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::io::orc_writer_options::builder(cudf::io::sink_info(&out_buffer), unused_table->view())
      .row_index_stride(511),
    cudf::logic_error);
}

TEST_F(OrcWriterTest, TestMap)
{
  auto const num_rows       = 1200000;
  auto const lists_per_row  = 4;
  auto const num_child_rows = (num_rows * lists_per_row) / 2;  // half due to validity

  auto keys      = random_values<int>(num_child_rows);
  auto vals      = random_values<float>(num_child_rows);
  auto vals_mask = cudf::test::iterators::nulls_at_multiples_of(3);
  int32_col keys_col(keys.begin(), keys.end());
  float32_col vals_col{vals.begin(), vals.end(), vals_mask};
  auto s_col = struct_col({keys_col, vals_col}).release();

  auto valids = cudf::test::iterators::nulls_at_multiples_of(2);

  std::vector<int> row_offsets(num_rows + 1);
  int offset = 0;
  for (int idx = 0; idx < (num_rows) + 1; ++idx) {
    row_offsets[idx] = offset;
    if (valids[idx]) { offset += lists_per_row; }
  }
  int32_col offsets(row_offsets.begin(), row_offsets.end());

  auto num_list_rows           = static_cast<cudf::column_view>(offsets).size() - 1;
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_list_rows);
  auto list_col                = cudf::make_lists_column(
    num_list_rows, offsets.release(), std::move(s_col), null_count, std::move(null_mask));

  table_view expected({*list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_list_column_as_map();

  auto filepath = temp_env->get_temp_filepath("MapColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(OrcReaderTest, NestedColumnSelection)
{
  auto const num_rows  = 1000;
  auto child_col1_data = random_values<int32_t>(num_rows);
  auto child_col2_data = random_values<int64_t>(num_rows);
  auto validity        = cudf::test::iterators::nulls_at_multiples_of(3);
  int32_col child_col1{child_col1_data.begin(), child_col1_data.end(), validity};
  int64_col child_col2{child_col2_data.begin(), child_col2_data.end(), validity};
  struct_col s_col{child_col1, child_col2};
  table_view expected({s_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("struct_s");
  expected_metadata.column_metadata[0].child(0).set_name("field_a");
  expected_metadata.column_metadata[0].child(1).set_name("field_b");

  auto filepath = temp_env->get_temp_filepath("OrcNestedSelection.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      .columns({"struct_s.field_b"});
  auto result = cudf::io::read_orc(in_opts);

  // Verify that only one child column is included in the output table
  ASSERT_EQ(1, result.tbl->view().column(0).num_children());
  // Verify that the first child column is `field_b`
  int64_col expected_col{child_col2_data.begin(), child_col2_data.end(), validity};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_col, result.tbl->view().column(0).child(0));
  ASSERT_EQ("field_b", result.metadata.schema_info[0].children[0].name);
}

TEST_F(OrcReaderTest, NestedEmptyStructColumnSelection)
{
  auto const validity = std::vector<bool>{true, false};
  auto const num_rows = static_cast<cudf::size_type>(validity.size());
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.emplace_back(cudf::make_structs_column(num_rows, {}, 0, {}));
  auto input_column = cudf::make_structs_column(
    num_rows, std::move(struct_children), null_count, std::move(null_mask));
  ASSERT_TRUE(input_column->nullable());
  ASSERT_EQ(null_count, input_column->null_count());
  // Struct parent nulls are superimposed on their children by make_structs_column.
  ASSERT_TRUE(input_column->child(0).nullable());
  ASSERT_EQ(null_count, input_column->child(0).null_count());

  expect_selected_nested_empty_struct_round_trip(std::move(input_column),
                                                 "reader_nested_empty_struct_outer_nullable.orc");
}

TEST_F(OrcReaderTest, NullableEmptyStructChildColumnSelection)
{
  auto const validity = std::vector<bool>{true, false};
  auto const num_rows = static_cast<cudf::size_type>(validity.size());
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.emplace_back(
    cudf::make_structs_column(num_rows, {}, null_count, std::move(null_mask)));
  auto input_column = cudf::make_structs_column(num_rows, std::move(struct_children), 0, {});
  ASSERT_FALSE(input_column->nullable());
  ASSERT_TRUE(input_column->child(0).nullable());
  ASSERT_EQ(null_count, input_column->child(0).null_count());

  expect_selected_nested_empty_struct_round_trip(std::move(input_column),
                                                 "reader_nested_empty_struct_inner_nullable.orc");
}

TEST_F(OrcReaderTest, DecimalOptions)
{
  constexpr auto num_rows = 10;
  auto col_vals           = random_values<int64_t>(num_rows);
  auto mask               = cudf::test::iterators::valids_at_multiples_of(3);

  dec128_col col{col_vals.begin(), col_vals.end(), mask, numeric::scale_type{2}};
  table_view expected({col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("dec");

  auto filepath = temp_env->get_temp_filepath("OrcDecimalOptions.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options valid_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .decimal128_columns({"dec", "fake_name"});
  // Should not throw, even with "fake name"
  EXPECT_NO_THROW(cudf::io::read_orc(valid_opts));
}

TEST_F(OrcWriterTest, DecimalOptionsNested)
{
  auto const num_rows = 100;

  auto dec_vals = random_values<int32_t>(num_rows);
  dec64_col dec1_col{dec_vals.begin(), dec_vals.end(), numeric::scale_type{2}};
  dec128_col dec2_col{dec_vals.begin(), dec_vals.end(), numeric::scale_type{2}};
  auto child_struct_col = cudf::test::structs_column_wrapper{dec1_col, dec2_col};

  auto int_vals = random_values<int32_t>(num_rows);
  int32_col int_col(int_vals.begin(), int_vals.end());
  auto map_struct_col = struct_col({child_struct_col, int_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  std::iota(row_offsets.begin(), row_offsets.end(), 0);
  int32_col offsets(row_offsets.begin(), row_offsets.end());

  auto map_list_col = cudf::make_lists_column(
    num_rows, offsets.release(), std::move(map_struct_col), 0, rmm::device_buffer{});

  table_view expected({*map_list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[0].child(1).child(0).child(0).set_name("dec64");
  expected_metadata.column_metadata[0].child(1).child(0).child(1).set_name("dec128");

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
      .use_index(false)
      // One less level of nesting because children of map columns are the child struct's children
      .decimal128_columns({"maps.0.dec64"});
  auto result = cudf::io::read_orc(in_opts);

  // Both columns should be read as decimal128
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0).child(1).child(0).child(0),
                                      result.tbl->view().column(0).child(1).child(0).child(1));
}

TEST_F(OrcReaderTest, EmptyColumnsParam)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(2, 4, false);

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options args =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, *expected);
  cudf::io::write_orc(args);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .columns({});
  auto const result = cudf::io::read_orc(read_opts);

  EXPECT_EQ(result.tbl->num_columns(), 0);
  EXPECT_EQ(result.tbl->num_rows(), 0);
}

TEST_F(OrcMetadataReaderTest, TestBasic)
{
  auto const num_rows = 1'200'000;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  int32_col int_col(ints.begin(), ints.end());
  float32_col float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("float_col");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_orc(out_opts);

  auto meta = read_orc_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  EXPECT_EQ(meta.schema().root().name(), "");
  EXPECT_EQ(meta.schema().root().type_kind(), cudf::io::orc::STRUCT);
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
  EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
}

TEST_F(OrcMetadataReaderTest, TestNested)
{
  auto const num_rows       = 1'200'000;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  int32_col keys_col(keys.begin(), keys.end());
  float32_col vals_col(vals.begin(), vals.end());
  auto s_col = struct_col({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  int32_col offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_orc(out_opts);

  auto meta = read_orc_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  EXPECT_EQ(meta.schema().root().name(), "");
  EXPECT_EQ(meta.schema().root().type_kind(), cudf::io::orc::STRUCT);
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type_kind(), cudf::io::orc::MAP);
  ASSERT_EQ(out_map_col.num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).name(), "");  // keys (no name in ORC)
  EXPECT_EQ(out_map_col.child(1).name(), "");  // values (no name in ORC)

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type_kind(), cudf::io::orc::LIST);
  ASSERT_EQ(out_list_col.num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0);
  EXPECT_EQ(out_list_struct_col.name(), "");  // elements (no name in ORC)
  EXPECT_EQ(out_list_struct_col.type_kind(), cudf::io::orc::STRUCT);
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type_kind(), cudf::io::orc::INT);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type_kind(), cudf::io::orc::FLOAT);
}

TEST_F(OrcReaderTest, ZstdMaxCompressionRate)
{
  // Encodes as 64KB of zeros, which compresses to 18 bytes with ZSTD
  std::vector<float> const h_data(8 * 1024);
  float32_col col(h_data.begin(), h_data.end());
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcHugeCompRatio.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::orc_writer_options opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&unused_buffer}, table->view())
      .compression_statistics(stats);
  cudf::io::write_orc(opts);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));
}

TEST_P(OrcCompressionTest, CompStats)
{
  auto const compression_type = std::get<1>(GetParam());

  auto table = create_random_fixed_table<int>(1, 55000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression(compression_type)
      .compression_statistics(stats);
  cudf::io::orc_chunked_writer(opts).write(*table);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));

  auto const single_table_comp_stats = *stats;
  cudf::io::orc_chunked_writer(opts).write(*table);

  EXPECT_EQ(stats->compression_ratio(), single_table_comp_stats.compression_ratio());
  EXPECT_EQ(stats->num_compressed_bytes(), 2 * single_table_comp_stats.num_compressed_bytes());

  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
}

void expect_compression_stats_empty(cudf::io::writer_compression_statistics const* stats)
{
  EXPECT_EQ(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_TRUE(std::isnan(stats->compression_ratio()));
}

TEST_P(OrcCompressionTest, CompStatsEmptyTable)
{
  auto const compression_type = std::get<1>(GetParam());

  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::orc_writer_options opts = cudf::io::orc_writer_options::builder(
                                        cudf::io::sink_info{&unused_buffer}, table_no_rows->view())
                                        .compression(compression_type)
                                        .compression_statistics(stats);
  cudf::io::write_orc(opts);

  expect_compression_stats_empty(stats.get());
}

TEST_P(OrcCompressionTest, ChunkedCompStatsEmptyTable)
{
  auto const compression_type = std::get<1>(GetParam());

  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_orc_writer_options opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression(compression_type)
      .compression_statistics(stats);
  cudf::io::orc_chunked_writer(opts).write(*table_no_rows);

  expect_compression_stats_empty(stats.get());
}

TEST_F(OrcWriterTest, EmptyRowGroup)
{
  std::vector<int> ints(10000 + 5, -1);
  auto mask = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i >= 10000; });
  int32_col col{ints.begin(), ints.end(), mask};
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcEmptyRowGroup.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcWriterTest, NoNullsAsNonNullable)
{
  auto valids = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> col{{1, 2, 3}, valids};
  table_view expected({col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  // Writer should be able to write a column without nulls as non-nullable
  EXPECT_NO_THROW(cudf::io::write_orc(out_opts));
}

TEST_F(OrcWriterTest, SlicedStringColumn)
{
  std::vector<char const*> strings{"a", "bc", "def", "longer", "strings", "at the end"};
  str_col col(strings.begin(), strings.end());
  table_view expected({col});

  // Slice the table to include the longer strings
  auto expected_slice = cudf::slice(expected, {2, 6});

  auto filepath = temp_env->get_temp_filepath("SlicedTable.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

TEST_F(OrcWriterTest, EmptyChildStringColumn)
{
  list_col<cudf::string_view> col{{}, {}};
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("OrcEmptyChildStringColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).use_index(false);
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

template <typename T>
void check_all_null_stats(cudf::io::column_statistics const& stats)
{
  EXPECT_EQ(stats.number_of_values, 0);
  EXPECT_TRUE(stats.has_null);

  auto const ts = std::get<T>(stats.type_specific_stats);
  EXPECT_FALSE(ts.minimum.has_value());
  EXPECT_FALSE(ts.maximum.has_value());
  EXPECT_TRUE(ts.sum.has_value());
  EXPECT_EQ(*ts.sum, 0);
}

TEST_F(OrcStatisticsTest, AllNulls)
{
  float64_col double_col({0., 0., 0.}, cudf::test::iterators::all_nulls());
  int32_col int_col({0, 0, 0}, cudf::test::iterators::all_nulls());
  str_col string_col({"", "", ""}, cudf::test::iterators::all_nulls());

  cudf::table_view expected({int_col, double_col, string_col});

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, expected);
  cudf::io::write_orc(out_opts);

  auto const stats =
    cudf::io::read_parsed_orc_statistics(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});

  check_all_null_stats<cudf::io::integer_statistics>(stats.file_stats[1]);
  check_all_null_stats<cudf::io::double_statistics>(stats.file_stats[2]);
  check_all_null_stats<cudf::io::string_statistics>(stats.file_stats[3]);
}

TEST_F(OrcWriterTest, UnorderedDictionary)
{
  std::vector<char const*> strings{
    "BBBB", "BBBB", "CCCC", "BBBB", "CCCC", "EEEE", "CCCC", "AAAA", "DDDD", "EEEE"};
  str_col col(strings.begin(), strings.end());

  table_view expected({col});

  std::vector<char> out_buffer_sorted;
  cudf::io::orc_writer_options out_opts_sorted =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer_sorted}, expected);
  cudf::io::write_orc(out_opts_sorted);

  cudf::io::orc_reader_options in_opts_sorted =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer_sorted.data()), out_buffer_sorted.size()}});
  auto const from_sorted = cudf::io::read_orc(in_opts_sorted).tbl;

  std::vector<char> out_buffer_unsorted;
  cudf::io::orc_writer_options out_opts_unsorted =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer_unsorted}, expected)
      .enable_dictionary_sort(false);
  cudf::io::write_orc(out_opts_unsorted);

  cudf::io::orc_reader_options in_opts_unsorted =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer_unsorted.data()), out_buffer_unsorted.size()}});
  auto const from_unsorted = cudf::io::read_orc(in_opts_unsorted).tbl;

  CUDF_TEST_EXPECT_TABLES_EQUAL(*from_sorted, *from_unsorted);
}

TEST_F(OrcStatisticsTest, Empty)
{
  int32_col col0{};
  float64_col col1{};
  str_col col2{};
  dec64_col col3{{}, numeric::scale_type{0}};
  column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> col4;
  bool_col col5{};
  table_view expected({col0, col1, col2, col3, col4, col5});

  std::vector<char> out_buffer;

  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, expected);
  cudf::io::write_orc(out_opts);

  auto const stats =
    cudf::io::read_parsed_orc_statistics(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});

  auto expected_column_names = std::vector<std::string>{""};
  std::generate_n(
    std::back_inserter(expected_column_names),
    expected.num_columns(),
    [starting_index = 0]() mutable { return "_col" + std::to_string(starting_index++); });
  EXPECT_EQ(stats.column_names, expected_column_names);

  EXPECT_EQ(stats.column_names.size(), 7);
  EXPECT_EQ(stats.stripes_stats.size(), 0);

  auto const& fstats = stats.file_stats;
  ASSERT_EQ(fstats.size(), 7);
  auto& s0 = fstats[0];
  EXPECT_TRUE(s0.number_of_values.has_value());
  EXPECT_EQ(*s0.number_of_values, 0ul);
  EXPECT_TRUE(s0.has_null.has_value());
  EXPECT_FALSE(*s0.has_null);

  auto& s1 = fstats[1];
  EXPECT_EQ(*s1.number_of_values, 0ul);
  EXPECT_FALSE(*s1.has_null);
  auto& ts1 = std::get<cudf::io::integer_statistics>(s1.type_specific_stats);
  EXPECT_FALSE(ts1.minimum.has_value());
  EXPECT_FALSE(ts1.maximum.has_value());
  EXPECT_TRUE(ts1.sum.has_value());
  EXPECT_EQ(*ts1.sum, 0);

  auto& s2 = fstats[2];
  EXPECT_EQ(*s2.number_of_values, 0ul);
  EXPECT_FALSE(*s2.has_null);
  auto& ts2 = std::get<cudf::io::double_statistics>(s2.type_specific_stats);
  EXPECT_FALSE(ts2.minimum.has_value());
  EXPECT_FALSE(ts2.maximum.has_value());
  EXPECT_TRUE(ts2.sum.has_value());
  EXPECT_EQ(*ts2.sum, 0);

  auto& s3 = fstats[3];
  EXPECT_EQ(*s3.number_of_values, 0ul);
  EXPECT_FALSE(*s3.has_null);
  auto& ts3 = std::get<cudf::io::string_statistics>(s3.type_specific_stats);
  EXPECT_FALSE(ts3.minimum.has_value());
  EXPECT_FALSE(ts3.maximum.has_value());
  EXPECT_TRUE(ts3.sum.has_value());
  EXPECT_EQ(*ts3.sum, 0);

  auto& s4 = fstats[4];
  EXPECT_EQ(*s4.number_of_values, 0ul);
  EXPECT_FALSE(*s4.has_null);
  auto& ts4 = std::get<cudf::io::decimal_statistics>(s4.type_specific_stats);
  EXPECT_FALSE(ts4.minimum.has_value());
  EXPECT_FALSE(ts4.maximum.has_value());
  EXPECT_TRUE(ts4.sum.has_value());
  EXPECT_EQ(*ts4.sum, "0");

  auto& s5 = fstats[5];
  EXPECT_EQ(*s5.number_of_values, 0ul);
  EXPECT_FALSE(*s5.has_null);
  auto& ts5 = std::get<cudf::io::timestamp_statistics>(s5.type_specific_stats);
  EXPECT_FALSE(ts5.minimum.has_value());
  EXPECT_FALSE(ts5.maximum.has_value());
  EXPECT_FALSE(ts5.minimum_utc.has_value());
  EXPECT_FALSE(ts5.maximum_utc.has_value());
  EXPECT_FALSE(ts5.minimum_nanos.has_value());
  EXPECT_FALSE(ts5.maximum_nanos.has_value());

  auto& s6 = fstats[6];
  EXPECT_EQ(*s6.number_of_values, 0ul);
  EXPECT_FALSE(*s6.has_null);
  auto& ts6 = std::get<cudf::io::bucket_statistics>(s6.type_specific_stats);
  EXPECT_EQ(ts6.count[0], 0);
}

void round_trip_basic(cudf::io::compression_type compression_type)
{
  constexpr auto num_rows = 12345;

  // Generate compressible data
  auto int_sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 100; });
  auto float_sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 32; });

  int32_col int_col(int_sequence, int_sequence + num_rows);
  float32_col float_col(float_sequence, float_sequence + num_rows);

  table_view expected({int_col, float_col});

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, expected)
      .compression(compression_type)
      .compression_statistics(stats);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  auto result = cudf::io::read_orc(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));
}

TEST_P(OrcCompressionTest, RoundTripBasic) { round_trip_basic(std::get<1>(GetParam())); }

TEST_P(OrcDecompressionTest, RoundTripBasic) { round_trip_basic(std::get<1>(GetParam())); }

TEST_F(OrcWriterTest, BounceBufferBug)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 100; });

  constexpr auto num_rows = 150000;
  column_wrapper<int8_t> col(sequence, sequence + num_rows);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("BounceBufferBug.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_orc(out_opts);
}

TEST_F(OrcReaderTest, SizeTypeRowsOverflow)
{
  // this test runs over 1.5 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  using cudf::test::iterators::no_nulls;
  constexpr auto num_rows   = 500'000'000l;
  constexpr auto num_reps   = 5;
  constexpr auto total_rows = num_rows * num_reps;
  static_assert(total_rows > std::numeric_limits<cudf::size_type>::max());

  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 127; });
  column_wrapper<int8_t> col(sequence, sequence + num_rows);
  table_view chunk_table({col});

  std::vector<char> out_buffer;
  {
    cudf::io::chunked_orc_writer_options write_opts =
      cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&out_buffer});

    auto writer = cudf::io::orc_chunked_writer(write_opts);
    for (int i = 0; i < num_reps; i++) {
      writer.write(chunk_table);
    }
  }

  // Test reading the metadata
  auto metadata = read_orc_metadata(cudf::io::source_info{cudf::host_span<std::byte const>{
    reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  EXPECT_EQ(metadata.num_rows(), total_rows);
  EXPECT_EQ(metadata.num_stripes(), total_rows / 1'000'000);

  auto const stats =
    cudf::io::read_parsed_orc_statistics(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  ASSERT_FALSE(stats.file_stats.empty());
  ASSERT_TRUE(stats.file_stats.front().number_of_values.has_value());
  EXPECT_EQ(*stats.file_stats.front().number_of_values, static_cast<uint64_t>(total_rows));

  constexpr auto num_rows_to_read = 1'000'000;
  auto const num_rows_to_skip     = metadata.num_rows() - num_rows_to_read;

  // Read the last million rows
  cudf::io::orc_reader_options skip_opts =
    cudf::io::orc_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .use_index(false)
      .skip_rows(num_rows_to_skip);
  auto const got_with_skip = cudf::io::read_orc(skip_opts).tbl;

  auto const sequence_start = num_rows_to_skip % num_rows;
  column_wrapper<int8_t, typename decltype(sequence)::value_type> skipped_col(
    sequence + sequence_start, sequence + sequence_start + num_rows_to_read, no_nulls());
  table_view expected({skipped_col});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_with_skip->view());

  // Read the last stripe (still the last million rows)
  cudf::io::orc_reader_options stripe_opts =
    cudf::io::orc_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .use_index(false)
      .stripes({{metadata.num_stripes() - 1}});
  auto const got_with_stripe_selection = cudf::io::read_orc(stripe_opts).tbl;

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got_with_stripe_selection->view());
}

TEST_F(OrcChunkedWriterTest, NoWriteCloseNotThrow)
{
  std::vector<char> out_buffer;

  cudf::io::chunked_orc_writer_options write_opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&out_buffer});
  auto writer = cudf::io::orc_chunked_writer(write_opts);

  EXPECT_NO_THROW(writer.close());
}

TEST_F(OrcChunkedWriterTest, FailedWriteCloseNotThrow)
{
  // A sink that throws on write()
  class throw_sink : public cudf::io::data_sink {
   public:
    void host_write(void const* data, size_t size) override { throw std::runtime_error("write"); }
    void flush() override {}
    size_t bytes_written() override { return 0; }
  };

  auto sequence = cuda::counting_iterator<int32_t>{0};
  column_wrapper<int8_t, int32_t> col(sequence, sequence + 10);
  table_view table({col});

  throw_sink sink;
  cudf::io::chunked_orc_writer_options write_opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&sink});
  auto writer = cudf::io::orc_chunked_writer(write_opts);

  try {
    writer.write(table);
  } catch (...) {
    // ignore the exception; we're testing that close() doesn't throw when the only write() fails
  }

  EXPECT_NO_THROW(writer.close());
}

TEST_F(OrcChunkedWriterTest, NoDataInSinkWhenNoWrite)
{
  std::vector<char> out_buffer;

  cudf::io::chunked_orc_writer_options write_opts =
    cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&out_buffer});
  auto writer = cudf::io::orc_chunked_writer(write_opts);

  EXPECT_NO_THROW(writer.close());
  EXPECT_EQ(out_buffer.size(), 0);
}

// Tests whether Y dimension of grid sizes depends on the number of row groups
// Disabled because of the high execution time (especially compared to the likelihood of regression)
TEST_F(OrcReaderTest, DISABLED_Over65kRowGroups)
{
  auto constexpr row_group_size = 512;
  constexpr auto num_rows       = (1 << 16) * row_group_size + 1;

  auto vals_col = random_values<int32_t>(num_rows);
  dec64_col col{vals_col.begin(), vals_col.end(), numeric::scale_type{2}};
  table_view chunk_table({col});

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, chunk_table)
      .row_index_stride(row_group_size);

  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  auto result = cudf::io::read_orc(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(chunk_table, result.tbl->view());
}

// Tests whether Y dimension of grid sizes depends on the number of stripes
// Disabled because of the high execution time (especially compared to the likelihood of regression)
TEST_F(OrcReaderTest, DISABLED_Over65kStripes)
{
  auto constexpr stripe_size = 512;
  constexpr auto num_rows    = (1 << 16) * stripe_size + 1;

  auto vals_col = random_values<int32_t>(num_rows);
  dec64_col col{vals_col.begin(), vals_col.end(), numeric::scale_type{2}};
  table_view chunk_table({col});

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, chunk_table)
      .stripe_size_rows(stripe_size)
      .compression(cudf::io::compression_type::NONE);

  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  auto result = cudf::io::read_orc(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(chunk_table, result.tbl->view());
}

// Tests whether Y dimension of grid sizes depends on the number of columns
// Disabled because of the high execution time (especially compared to the likelihood of regression)
TEST_F(OrcWriterTest, DISABLED_Over65kColumns)
{
  auto vals_col = random_values<int32_t>(8);
  dec64_col col{vals_col.begin(), vals_col.end(), numeric::scale_type{2}};
  cudf::column_view col_view = col;
  table_view expected(std::vector<cudf::column_view>{(1 << 16) + 1, col_view});

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, expected)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  auto result = cudf::io::read_orc(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcWriterTest, MultipleBlocksInStripeFooter)
{
  std::vector<std::string> vals_col(8, "a");
  str_col col{vals_col.begin(), vals_col.end()};
  cudf::column_view col_view = col;
  table_view expected(std::vector<cudf::column_view>{6400, col_view});

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, expected);
  // Write with compression on (default)
  cudf::io::write_orc(out_opts);

  cudf::io::orc_reader_options in_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
  auto result = cudf::io::read_orc(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(OrcReaderTest, DeviceReadAsyncThrows)
{
  // Create a simple ORC file in memory
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  std::vector<char> out_buffer;
  cudf::io::orc_writer_options write_args =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{&out_buffer}, table_to_write);
  cudf::io::write_orc(write_args);

  // Create our throwing datasource
  auto throwing_source = std::make_unique<cudf::test::ThrowingDeviceReadDatasource>(out_buffer);
  cudf::io::source_info source_info(throwing_source.get());

  // Try to read the ORC file - this should either succeed or propagate AsyncException
  // from device_read_async.
  cudf::io::orc_reader_options read_args = cudf::io::orc_reader_options::builder(source_info);
  try {
    cudf::io::read_orc(read_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

TEST_F(OrcReaderTest, DeviceWriteAsyncThrows)
{
  // Create a simple table to write
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  auto throwing_sink = std::make_unique<cudf::test::ThrowingDeviceWriteDataSink>();

  cudf::io::orc_writer_options write_args =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{throwing_sink.get()}, table_to_write);

  // The write_orc call should either succeed or throw AsyncException.
  // Should only fail if a different exception is thrown.
  try {
    cudf::io::write_orc(write_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        OrcCompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::LZ4,
                                                             cudf::io::compression_type::ZLIB,
                                                             cudf::io::compression_type::ZSTD)));

INSTANTIATE_TEST_CASE_P(DeviceInternal,
                        OrcCompressionTest,
                        ::testing::Combine(::testing::Values("DEVICE_INTERNAL"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY)));

INSTANTIATE_TEST_CASE_P(Host,
                        OrcCompressionTest,
                        ::testing::Combine(::testing::Values("HOST"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::ZSTD)));

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        OrcDecompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::LZ4,
                                                             cudf::io::compression_type::ZLIB,
                                                             cudf::io::compression_type::ZSTD)));

INSTANTIATE_TEST_CASE_P(DeviceInternal,
                        OrcDecompressionTest,
                        ::testing::Combine(::testing::Values("DEVICE_INTERNAL"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY)));

INSTANTIATE_TEST_CASE_P(Host,
                        OrcDecompressionTest,
                        ::testing::Combine(::testing::Values("HOST", "HYBRID", "AUTO"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::ZSTD)));

TEST_F(OrcWriterTest, DISABLED_SizeTypeOverflow)
{
  constexpr auto num_rows = std::numeric_limits<cudf::size_type>::max();

  auto const val = cudf::numeric_scalar<bool>(true, true);
  auto const col = cudf::make_column_from_scalar(val, num_rows);
  ASSERT_EQ(col->size(), num_rows);

  std::vector<char> buffer;
  buffer.reserve(num_rows);
  cudf::io::orc_writer_options const options = cudf::io::orc_writer_options::builder(
    cudf::io::sink_info{&buffer}, cudf::table_view({col->view()}));
  EXPECT_NO_THROW(cudf::io::write_orc(options));

  cudf::io::orc_reader_options const read_opts = cudf::io::orc_reader_options::builder(
    cudf::io::source_info{cudf::host_span<char const>(buffer.data(), buffer.size())});
  auto const result = cudf::io::read_orc(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({col->view()}), result.tbl->view());
}

CUDF_TEST_PROGRAM_MAIN()
