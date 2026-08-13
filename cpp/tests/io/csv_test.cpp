/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using cudf::data_type;
using cudf::type_id;
using cudf::type_to_id;

template <typename T>
auto dtype()
{
  return data_type{type_to_id<T>()};
}

template <typename T, typename SourceElementT = T>
using column_wrapper =
  std::conditional_t<std::is_same_v<T, cudf::string_view>,
                     cudf::test::strings_column_wrapper,
                     cudf::test::fixed_width_column_wrapper<T, SourceElementT>>;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Base test fixture for tests
struct CsvWriterTest : public cudf::test::BaseFixture {};

template <typename T>
struct CsvFixedPointWriterTest : public CsvWriterTest {};

TYPED_TEST_SUITE(CsvFixedPointWriterTest, cudf::test::FixedPointTypes);

// Base test fixture for tests
struct CsvReaderTest : public cudf::test::BaseFixture {};

// Typed test fixture for timestamp type tests
template <typename T>
struct CsvReaderNumericTypeTest : public CsvReaderTest {};

// Declare typed test cases
using SupportedNumericTypes = cudf::test::Types<int64_t, double>;
TYPED_TEST_SUITE(CsvReaderNumericTypeTest, SupportedNumericTypes);

template <typename DecimalType>
struct CsvFixedPointReaderTest : public CsvReaderTest {
  void run_tests(std::vector<std::string> const& reference_strings, numeric::scale_type scale)
  {
    cudf::test::strings_column_wrapper const strings(reference_strings.begin(),
                                                     reference_strings.end());
    auto const expected = cudf::strings::to_fixed_point(
      cudf::strings_column_view(strings), data_type{type_to_id<DecimalType>(), scale});

    auto const buffer = std::accumulate(reference_strings.begin(),
                                        reference_strings.end(),
                                        std::string{},
                                        [](std::string const& acc, std::string const& rhs) {
                                          return acc.empty() ? rhs : (acc + "\n" + rhs);
                                        });

    cudf::io::csv_reader_options const in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
        .compression(cudf::io::compression_type::NONE)
        .dtypes({data_type{type_to_id<DecimalType>(), scale}})
        .header(-1);

    auto const result      = cudf::io::read_csv(in_opts);
    auto const result_view = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, result_view.column(0));
    EXPECT_EQ(result_view.num_columns(), 1);
  }
};

TYPED_TEST_SUITE(CsvFixedPointReaderTest, cudf::test::FixedPointTypes);

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(std::size_t size)
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

MATCHER_P(FloatNearPointwise, tolerance, "Out-of-range")
{
  return (std::get<0>(arg) > std::get<1>(arg) - tolerance &&
          std::get<0>(arg) < std::get<1>(arg) + tolerance);
}

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

// temporary method to verify the float columns until
// CUDF_TEST_EXPECT_COLUMNS_EQUAL supports floating point
template <typename T, typename valid_t>
void check_float_column(cudf::column_view const& col_lhs,
                        cudf::column_view const& col_rhs,
                        T tol,
                        valid_t const& validity)
{
  auto h_data = cudf::test::to_host<T>(col_rhs).first;

  std::vector<T> data(h_data.size());
  std::copy(h_data.begin(), h_data.end(), data.begin());

  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(col_lhs,
                                                (wrapper<T>{data.begin(), data.end(), validity}));
  EXPECT_TRUE(col_lhs.null_count() == 0 and col_rhs.null_count() == 0);
  EXPECT_THAT(cudf::test::to_host<T>(col_lhs).first,
              ::testing::Pointwise(FloatNearPointwise(tol), data));
}

// timestamp column checker within tolerance
// given by `tol_ms` (milliseconds)
void check_timestamp_column(cudf::column_view const& col_lhs,
                            cudf::column_view const& col_rhs,
                            long tol_ms = 1000l)
{
  using T = cudf::timestamp_ms;
  using namespace cuda::std::chrono;

  auto h_lhs = cudf::test::to_host<T>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<T>(col_rhs).first;

  cudf::size_type nrows = h_lhs.size();
  EXPECT_TRUE(nrows == static_cast<cudf::size_type>(h_rhs.size()));

  auto begin_count = cuda::counting_iterator<cudf::size_type>{0};
  auto end_count   = cuda::counting_iterator<cudf::size_type>{nrows};

  auto* ptr_lhs = h_lhs.data();  // cannot capture host_vector in thrust,
                                 // not even in host lambda
  auto* ptr_rhs = h_rhs.data();

  auto found = thrust::find_if(
    thrust::host, begin_count, end_count, [ptr_lhs, ptr_rhs, tol_ms](auto row_index) {
      auto delta_ms = cuda::std::chrono::duration_cast<cuda::std::chrono::milliseconds>(
        ptr_lhs[row_index] - ptr_rhs[row_index]);
      return delta_ms.count() >= tol_ms;
    });

  EXPECT_TRUE(found == end_count);  // not found...
}

// helper to replace in `str`  _all_ occurrences of `from` with `to`
std::string replace_all_helper(std::string str, std::string const& from, std::string const& to)
{
  std::size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

// compare string columns accounting for special character
// treatment: double double quotes ('\"')
// and surround whole string by double quotes if it contains:
// newline '\n', <delimiter>, and double quotes;
void check_string_column(cudf::column_view const& col_lhs,
                         cudf::column_view const& col_rhs,
                         std::string const& delimiter = ",")
{
  auto h_lhs = cudf::test::to_host<std::string>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<std::string>(col_rhs).first;

  std::string newline("\n");
  std::string quotes("\"");
  std::string quotes_repl("\"\"");

  std::vector<std::string> v_lhs;
  std::transform(h_lhs.begin(),
                 h_lhs.end(),
                 std::back_inserter(v_lhs),
                 [delimiter, newline, quotes, quotes_repl](std::string const& str_row) {
                   auto found_quote = str_row.find(quotes);
                   auto found_newl  = str_row.find(newline);
                   auto found_delim = str_row.find(delimiter);

                   bool flag_found_quotes = (found_quote != std::string::npos);
                   bool need_surround = flag_found_quotes || (found_newl != std::string::npos) ||
                                        (found_delim != std::string::npos);

                   std::string str_repl;
                   if (flag_found_quotes) {
                     str_repl = replace_all_helper(str_row, quotes, quotes_repl);
                   } else {
                     str_repl = str_row;
                   }
                   return need_surround ? quotes + str_repl + quotes : str_row;
                 });
  EXPECT_TRUE(std::equal(v_lhs.begin(), v_lhs.end(), h_rhs.begin()));
}

// Helper function to compare two floating-point column contents
template <typename T>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
  requires(std::is_floating_point_v<T>)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), lhs));
}

// Helper function to compare two column contents
template <typename T>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
  requires(!std::is_floating_point_v<T>)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first, ::testing::ElementsAreArray(lhs));
}

void write_csv_helper(std::string const& filename,
                      cudf::table_view const& table,
                      std::vector<std::string> const& names = {})
{
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filename), table)
      .include_header(not names.empty())
      .names(names);

  cudf::io::write_csv(writer_options);
}

template <typename T>
std::string assign(T input)
{
  return std::to_string(input);
}

std::string assign(std::string input) { return input; }

template <typename T>
std::vector<std::string> prepend_zeros(std::vector<T> const& input,
                                       int zero_count         = 0,
                                       bool add_positive_sign = false)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](T const& num) {
    auto str         = assign(num);
    bool is_negative = (str[0] == '-');
    if (is_negative) {
      str.insert(1, zero_count, '0');
      return str;
    } else if (add_positive_sign) {
      return "+" + std::string(zero_count, '0') + str;
    } else {
      str.insert(0, zero_count, '0');
      return str;
    }
  });
  return output;
}

}  // namespace

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumn)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });

  auto filepath = temp_env->get_temp_filepath("SingleColumn.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<TypeParam> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(std::vector<TypeParam>(sequence, sequence + num_rows), view.column(0));
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnNegativeScale)
{
  this->run_tests({"1.23", "876e-2", "5.43e1", "-0.12", "0.25", "-0.23", "-0.27", "0.00", "0.00"},
                  numeric::scale_type{-2});
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnNoScale)
{
  this->run_tests({"123", "-87600e-2", "54.3e1", "-12", "25", "-23", "-27", "0", "0"},
                  numeric::scale_type{0});
}

TYPED_TEST(CsvFixedPointReaderTest, SingleColumnPositiveScale)
{
  this->run_tests(
    {"123000", "-87600000e-2", "54300e1", "-12000", "25000", "-23000", "-27000", "0000", "0000"},
    numeric::scale_type{3});
}

TYPED_TEST(CsvFixedPointWriterTest, SingleColumnNegativeScale)
{
  std::vector<std::string> reference_strings = {
    "1.23", "-8.76", "5.43", "-0.12", "0.25", "-0.23", "-0.27", "0.00", "0.00"};

  auto validity = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper strings(
    reference_strings.begin(), reference_strings.end(), validity);

  std::vector<std::string> valid_reference_strings;
  thrust::copy_if(thrust::host,
                  reference_strings.begin(),
                  reference_strings.end(),
                  cuda::counting_iterator<std::size_t>{0},
                  std::back_inserter(valid_reference_strings),
                  validity.functor());
  reference_strings = valid_reference_strings;

  using DecimalType = TypeParam;
  auto input_column =
    cudf::strings::to_fixed_point(cudf::strings_column_view(strings),
                                  data_type{type_to_id<DecimalType>(), numeric::scale_type{-2}});

  auto input_table = cudf::table_view{std::vector<cudf::column_view>{*input_column}};

  auto filepath = temp_env->get_temp_dir() + "FixedPointSingleColumnNegativeScale.csv";

  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filepath), input_table)
      .include_header(false);

  cudf::io::write_csv(writer_options);

  std::vector<std::string> result_strings;
  result_strings.reserve(reference_strings.size());

  std::ifstream read_result_file(filepath);
  ASSERT_TRUE(read_result_file.is_open());

  std::copy(std::istream_iterator<std::string>(read_result_file),
            std::istream_iterator<std::string>(),
            std::back_inserter(result_strings));

  EXPECT_EQ(result_strings, reference_strings);
}

TYPED_TEST(CsvFixedPointWriterTest, SingleColumnPositiveScale)
{
  std::vector<std::string> reference_strings = {
    "123000", "-876000", "543000", "-12000", "25000", "-23000", "-27000", "0000", "0000"};

  auto validity = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper strings(
    reference_strings.begin(), reference_strings.end(), validity);

  std::vector<std::string> valid_reference_strings;
  thrust::copy_if(thrust::host,
                  reference_strings.begin(),
                  reference_strings.end(),
                  cuda::counting_iterator<std::size_t>{0},
                  std::back_inserter(valid_reference_strings),
                  validity.functor());
  reference_strings = valid_reference_strings;

  using DecimalType = TypeParam;
  auto input_column =
    cudf::strings::to_fixed_point(cudf::strings_column_view(strings),
                                  data_type{type_to_id<DecimalType>(), numeric::scale_type{3}});

  auto input_table = cudf::table_view{std::vector<cudf::column_view>{*input_column}};

  auto filepath = temp_env->get_temp_dir() + "FixedPointSingleColumnPositiveScale.csv";

  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(filepath), input_table)
      .include_header(false);

  cudf::io::write_csv(writer_options);

  std::vector<std::string> result_strings;
  result_strings.reserve(reference_strings.size());

  std::ifstream read_result_file(filepath);
  ASSERT_TRUE(read_result_file.is_open());

  std::copy(std::istream_iterator<std::string>(read_result_file),
            std::istream_iterator<std::string>(),
            std::back_inserter(result_strings));

  EXPECT_EQ(result_strings, reference_strings);
}

void test_quoting_disabled_with_delimiter(char delimiter_char)
{
  auto const delimiter     = std::string{delimiter_char};
  auto const input_strings = cudf::test::strings_column_wrapper{
    std::string{"All"} + delimiter + "the" + delimiter + "leaves",
    "are\"brown",
    "and\nthe\nsky\nis\ngrey"};
  auto const input_table = table_view{{input_strings}};

  auto const filepath = temp_env->get_temp_dir() + "unquoted.csv";
  auto w_options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, input_table)
                     .include_header(false)
                     .inter_column_delimiter(delimiter_char)
                     .quoting(cudf::io::quote_style::NONE);
  cudf::io::write_csv(w_options.build());

  auto r_options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
                     .header(-1)
                     .delimiter(delimiter_char)
                     .quoting(cudf::io::quote_style::NONE);
  auto r_table = cudf::io::read_csv(r_options.build());

  auto const expected =
    cudf::test::strings_column_wrapper{"All", "are\"brown", "and", "the", "sky", "is", "grey"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(r_table.tbl->view().column(0), expected);
}

TEST_F(CsvWriterTest, QuotingDisabled)
{
  test_quoting_disabled_with_delimiter(',');
  test_quoting_disabled_with_delimiter('\u0001');
}

TEST_F(CsvReaderTest, MultiColumn)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint8_values       = random_values<uint8_t>(num_rows);
  auto uint16_values      = random_values<uint16_t>(num_rows);
  auto uint32_values      = random_values<uint32_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
           << "," << int64_values[i] << "," << std::to_string(uint8_values[i]) << ","
           << uint16_values[i] << "," << uint32_values[i] << "," << uint64_values[i] << ","
           << float32_values[i] << "," << float64_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .header(-1)
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<uint8_t>(),
               dtype<uint16_t>(),
               dtype<uint32_t>(),
               dtype<uint64_t>(),
               dtype<float>(),
               dtype<double>()});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int32_values, view.column(2));
  expect_column_data_equal(int64_values, view.column(3));
  expect_column_data_equal(uint8_values, view.column(4));
  expect_column_data_equal(uint16_values, view.column(5));
  expect_column_data_equal(uint32_values, view.column(6));
  expect_column_data_equal(uint64_values, view.column(7));
  expect_column_data_equal(float32_values, view.column(8));
  expect_column_data_equal(float64_values, view.column(9));
}

TEST_F(CsvReaderTest, RepeatColumn)
{
  constexpr auto num_rows = 10;
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "RepeatColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << int16_values[i] << "," << int64_values[i] << "," << uint64_values[i] << ","
           << float32_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  // repeats column in indexes and names, misses 1 column.
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes({dtype<int16_t>(), dtype<int64_t>(), dtype<uint64_t>(), dtype<float>()})
      .names({"A", "B", "C", "D"})
      .use_cols_indexes({1, 0, 0})
      .use_cols_names({"D", "B", "B"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(3, view.num_columns());
  expect_column_data_equal(int16_values, view.column(0));
  expect_column_data_equal(int64_values, view.column(1));
  expect_column_data_equal(float32_values, view.column(2));
}

TEST_F(CsvReaderTest, Booleans)
{
  auto filepath = temp_env->get_temp_dir() + "Booleans.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
               "true\nYes,5,foo,false\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A", "B", "C", "D"})
      .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int16_t>(), dtype<bool>()})
      .true_values({"yes", "Yes", "YES", "foo", "FOO"})
      .false_values({"no", "No", "NO", "Bar", "bar"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  auto const view = result.tbl->view();
  EXPECT_EQ(4, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::INT32, view.column(1).type().id());
  ASSERT_EQ(type_id::INT16, view.column(2).type().id());
  ASSERT_EQ(type_id::BOOL8, view.column(3).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 0, 0, 0, 1}, view.column(0));
  expect_column_data_equal(std::vector<int16_t>{0, 1, 1, 0, 1}, view.column(2));
  expect_column_data_equal(std::vector<bool>{true, true, false, true, false}, view.column(3));
}

TEST_F(CsvReaderTest, Dates)
{
  auto filepath = temp_env->get_temp_dir() + "Dates.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampS.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_SECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_s>{cudf::timestamp_s{983750400s},
                                                          cudf::timestamp_s{1288483200s},
                                                          cudf::timestamp_s{782611200s},
                                                          cudf::timestamp_s{656208000s},
                                                          cudf::timestamp_s{0s},
                                                          cudf::timestamp_s{798163200s},
                                                          cudf::timestamp_s{774144000s},
                                                          cudf::timestamp_s{1149679230s},
                                                          cudf::timestamp_s{1126875750s},
                                                          cudf::timestamp_s{2764800s}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMilliSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampMs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMicroSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampUs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MICROSECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_us>{cudf::timestamp_us{983750400000000us},
                                                           cudf::timestamp_us{1288483200000000us},
                                                           cudf::timestamp_us{782611200000000us},
                                                           cudf::timestamp_us{656208000000000us},
                                                           cudf::timestamp_us{0us},
                                                           cudf::timestamp_us{798163200000000us},
                                                           cudf::timestamp_us{774144000000000us},
                                                           cudf::timestamp_us{1149679230400000us},
                                                           cudf::timestamp_us{1126875750400000us},
                                                           cudf::timestamp_us{2764800000000us}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampNanoSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampNs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_NANOSECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(
    std::vector<cudf::timestamp_ns>{cudf::timestamp_ns{983750400000000000ns},
                                    cudf::timestamp_ns{1288483200000000000ns},
                                    cudf::timestamp_ns{782611200000000000ns},
                                    cudf::timestamp_ns{656208000000000000ns},
                                    cudf::timestamp_ns{0ns},
                                    cudf::timestamp_ns{798163200000000000ns},
                                    cudf::timestamp_ns{774144000000000000ns},
                                    cudf::timestamp_ns{1149679230400000000ns},
                                    cudf::timestamp_ns{1126875750400000000ns},
                                    cudf::timestamp_ns{2764800000000000ns}},
    view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampS.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column =
    column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>(input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_SECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampMilliSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampMs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampMicroSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampUs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MICROSECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, IntegersCastToTimestampNanoSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "IntegersCastToTimestampNs.csv";
  std::vector<int64_t> input_vals{1, 10, 111, 2, 11, 112, 3, 12, 113, 43432423, 13342, 13243214};
  auto expected_column = column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>(
    input_vals.begin(), input_vals.end());
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (auto v : input_vals) {
      outfile << v << "\n";
    }
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_NANOSECONDS}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_column, view.column(0));
}

TEST_F(CsvReaderTest, FloatingPoint)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73."
               "98007199999998;";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<float>()})
      .lineterminator(';')
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::FLOAT32, view.column(0).type().id());

  auto const ref_vals =
    std::vector<float>{5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  expect_column_data_equal(ref_vals, view.column(0));

  auto const bitmask = cudf::test::bitmask_to_host(view.column(0));
  ASSERT_EQ((1u << ref_vals.size()) - 1, bitmask[0]);
}

TEST_F(CsvReaderTest, Strings)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "Strings.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << R"(30,stu ""vwx"" yz)" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc def ghi", "\"jkl mno pqr\"", R"(stu ""vwx"" yz)"},
    view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotes)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotes.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quotechar('`');
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc,\ndef, ghi", "jkl, `mno`, pqr", "stu `vwx` yz"}, view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesIgnored)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << R"(20,"jkl ""mno"" pqr")" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE)
      .doublequote(false);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"\"abcdef ghi\"", R"("jkl ""mno"" pqr")", "stu \"vwx\" yz"},
    view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesWhitespace)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << '\n';
    outfile << "A,a" << '\n';              // unquoted no whitespace
    outfile << "    B,b" << '\n';          // unquoted leading whitespace
    outfile << "C    ,c" << '\n';          // unquoted trailing whitespace
    outfile << "    D    ,d" << '\n';      // unquoted leading and trailing whitespace
    outfile << "\"E\",e" << '\n';          // quoted no whitespace
    outfile << "\"F\"    ,f" << '\n';      // quoted trailing whitespace
    outfile << "    \"G\",g" << '\n';      // quoted leading whitespace
    outfile << "    \"H\"    ,h" << '\n';  // quoted leading and trailing whitespace
    outfile << "    \"    I    \"    ,i"
            << '\n';  // quoted leading and trailing whitespace with spaces inside quotes
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes(std::vector<data_type>{dtype<cudf::string_view>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::ALL)
      .doublequote(false)
      .detect_whitespace_around_quotes(true);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  ASSERT_EQ(2, view.num_columns());
  ASSERT_EQ(type_id::STRING, view.column(0).type().id());
  ASSERT_EQ(type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"A", "    B", "C    ", "    D    ", "E", "F", "G", "H", "    I    "},
    view.column(0));
  expect_column_data_equal(std::vector<std::string>{"a", "b", "c", "d", "e", "f", "g", "h", "i"},
                           view.column(1));
}

TEST_F(CsvReaderTest, SkiprowsNrows)
{
  auto filepath = temp_env->get_temp_dir() + "SkiprowsNrows.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(1)
      .skiprows(2)
      .nrows(2);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{5, 6}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRange)
{
  auto filepath = temp_env->get_temp_dir() + "ByteRange.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(-1)
      .byte_range_offset(11)
      .byte_range_size(15);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{4000, 5000, 6000}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRangeStrings)
{
  std::string input = "\"a\"\n\"b\"\n\"c\"";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(input.c_str()), input.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A"})
      .dtypes({dtype<cudf::string_view>()})
      .header(-1)
      .byte_range_offset(4);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::STRING, view.column(0).type().id());

  expect_column_data_equal(std::vector<std::string>{"c"}, view.column(0));
}

TEST_F(CsvReaderTest, BlanksAndComments)
{
  auto filepath = temp_env->get_temp_dir() + "BlanksAndComments.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<int32_t>()})
      .header(-1)
      .comment('#');
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 3, 4, 5, 8, 9}, view.column(0));
}

TEST_F(CsvReaderTest, EmptyFile)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, NoDataFile)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\n\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, HeaderOnlyFile)
{
  auto filepath = temp_env->get_temp_dir() + "HeaderOnlyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\"a\",\"b\",\"c\"\n\n";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_rows());
  EXPECT_EQ(3, view.num_columns());
}

TEST_F(CsvReaderTest, InvalidFloatingPoint)
{
  auto const filepath = temp_env->get_temp_dir() + "InvalidFloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1.2e1+\n3.4e2-\n5.6e3e\n7.8e3A\n9.0Be1\n1C.2";
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<float>()})
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(type_id::FLOAT32, view.column(0).type().id());

  // ignore all data because it is all nulls.
  ASSERT_EQ(6u, result.tbl->view().column(0).null_count());
}

TEST_F(CsvReaderTest, StringInference)
{
  std::string buffer = "\"-1\"\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), type_id::STRING);
}

TEST_F(CsvReaderTest, DelimWhitespaceNoHeaderLeadingTrailingDelimiter)
{
  std::string buffer = "  1   2  \n  3   4  \n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delim_whitespace(true)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  ASSERT_EQ(result_view.num_columns(), 2);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  expect_column_data_equal(std::vector<int64_t>{1, 3}, result_view.column(0));
  expect_column_data_equal(std::vector<int64_t>{2, 4}, result_view.column(1));
}

// Exercises whitespace-shape variations (no padding, leading-only, trailing-only,
// internal-runs-only, leading+trailing+internal, and quoted header names with and
// without surrounding whitespace) that should all yield the same `(col_a, col_b)`
// schema and `[(1,2),(3,4)]` data under `delim_whitespace=true` (pandas parity).
class CsvDelimWhitespaceShapeTest : public CsvReaderTest,
                                    public ::testing::WithParamInterface<std::string> {};

TEST_P(CsvDelimWhitespaceShapeTest, ProducesTwoColumns)
{
  auto const buffer = GetParam();
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delim_whitespace(true)
      .header(0);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "col_a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col_b");
  ASSERT_EQ(result_view.num_columns(), 2);
  expect_column_data_equal(std::vector<int64_t>{1, 3}, result_view.column(0));
  expect_column_data_equal(std::vector<int64_t>{2, 4}, result_view.column(1));
}

INSTANTIATE_TEST_SUITE_P(CsvReaderTest,
                         CsvDelimWhitespaceShapeTest,
                         ::testing::Values(std::string{"col_a col_b\n1 2\n3 4\n"},
                                           std::string{"  col_a col_b\n  1 2\n  3 4\n"},
                                           std::string{"col_a col_b  \n1 2  \n3 4  \n"},
                                           std::string{"col_a   col_b\n1   2\n3   4\n"},
                                           std::string{"  col_a   col_b  \n  1   2  \n  3   4  \n"},
                                           std::string{"\"col_a\" \"col_b\"\n1 2\n3 4\n"},
                                           std::string{"  \"col_a\"   \"col_b\"  \n1 2\n3 4\n"}));

TEST_F(CsvReaderTest, TypeInferenceEmptyDelimitedFields)
{
  std::string const buffer = "1,,3\n4,,6\n";
  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .na_filter(false)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::STRING);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::INT64);

  expect_column_data_equal(std::vector<int64_t>{1, 4}, result_view.column(0));
  expect_column_data_equal(std::vector<std::string>{"", ""}, result_view.column(1));
  expect_column_data_equal(std::vector<int64_t>{3, 6}, result_view.column(2));
}

TEST_F(CsvReaderTest, MultiChunkRowCount)
{
  // TODO: add reader option to set chunk size and use it here
  constexpr size_t chunk_threshold = 64ull * 1024 * 1024;
  std::string const row            = "123,456,789\n";
  size_t const num_rows            = (chunk_threshold / row.size()) + 1024;

  std::string buffer;
  buffer.reserve(num_rows * row.size());
  for (size_t i = 0; i < num_rows; ++i) {
    buffer.append(row);
  }

  cudf::io::csv_reader_options const in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(static_cast<size_t>(result_view.num_rows()), num_rows);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::INT64);

  // All rows are identical, so verifying min == max == expected
  auto const i64       = cudf::data_type{cudf::type_id::INT64};
  auto const min_agg   = cudf::make_min_aggregation<cudf::reduce_aggregation>();
  auto const max_agg   = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto const all_equal = [&](cudf::column_view const& col, int64_t expected) {
    using scalar_t = cudf::numeric_scalar<int64_t>;
    auto const min = cudf::reduce(col, *min_agg, i64);
    auto const max = cudf::reduce(col, *max_agg, i64);
    return static_cast<scalar_t const&>(*min).value() == expected &&
           static_cast<scalar_t const&>(*max).value() == expected;
  };
  EXPECT_TRUE(all_equal(result_view.column(0), 123));
  EXPECT_TRUE(all_equal(result_view.column(1), 456));
  EXPECT_TRUE(all_equal(result_view.column(2), 789));
}

TEST_F(CsvReaderTest, TypeInferenceThousands)
{
  std::string buffer = "1`400,123,1`234.56\n123`456,123456,12.34";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .thousands('`');
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::FLOAT64);

  auto tsnd_sep_col = std::vector<int64_t>{1400L, 123456L};
  auto int_col      = std::vector<int64_t>{123L, 123456L};
  auto dbl_col      = std::vector<double>{1234.56, 12.34};
  expect_column_data_equal(tsnd_sep_col, result_view.column(0));
  expect_column_data_equal(int_col, result_view.column(1));
  expect_column_data_equal(dbl_col, result_view.column(2));
}

TEST_F(CsvReaderTest, TypeInferenceWithDecimal)
{
  // Given that thousands:'`' and decimal(';'), we expect:
  // col#0 => INT64 (column contains only digits & thousands sep)
  // col#1 => STRING (contains digits and period character, which is NOT the decimal point here)
  // col#2 => FLOAT64 (column contains digits and decimal point (i.e., ';'))
  std::string buffer = "1`400,1.23,1`234;56\n123`456,123.456,12;34";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .thousands('`')
      .decimal(';');
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_columns(), 3);
  EXPECT_EQ(result_view.column(0).type().id(), type_id::INT64);
  EXPECT_EQ(result_view.column(1).type().id(), type_id::STRING);
  EXPECT_EQ(result_view.column(2).type().id(), type_id::FLOAT64);

  auto int_col = std::vector<int64_t>{1400L, 123456L};
  auto str_col = std::vector<std::string>{"1.23", "123.456"};
  auto dbl_col = std::vector<double>{1234.56, 12.34};
  expect_column_data_equal(int_col, result_view.column(0));
  expect_column_data_equal(str_col, result_view.column(1));
  expect_column_data_equal(dbl_col, result_view.column(2));
}

TEST_F(CsvReaderTest, SkipRowsXorSkipFooter)
{
  std::string buffer = "1,2,3";

  cudf::io::csv_reader_options skiprows_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .skiprows(1);
  EXPECT_NO_THROW(cudf::io::read_csv(skiprows_options));

  cudf::io::csv_reader_options skipfooter_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .skipfooter(1);
  EXPECT_NO_THROW(cudf::io::read_csv(skipfooter_options));
}

TEST_F(CsvReaderTest, nullHandling)
{
  auto const filepath = temp_env->get_temp_dir() + "NullValues.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "NULL\n\nnull\nn/a\nNull\nNA\nnan";
  }

  // Test disabling na_filter
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .na_filter(false)
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Test enabling na_filter
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, true, false, false});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Setting na_values with default values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .na_values({"Null"})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, false, false, false});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Setting na_values without default values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .keep_default_na(false)
        .na_values({"Null"})
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {true, true, true, true, false, true, true, true});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }

  // Filter enabled, but no NA values
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .keep_default_na(false)
        .dtypes({dtype<cudf::string_view>()})
        .header(-1)
        .skip_blank_lines(false);
    auto const result = cudf::io::read_csv(in_opts);
    auto const view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, view.column(0));
  }
}

TEST_F(CsvReaderTest, FailCases)
{
  std::string buffer = "1,2,3";
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .skiprows(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .skipfooter(1),
                 std::invalid_argument);
  }

  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_offset(4)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .skiprows(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .skipfooter(1),
                 std::invalid_argument);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .byte_range_size(4)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skiprows(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .byte_range_offset(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skiprows(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .byte_range_size(4),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .nrows(1)
                   .skipfooter(1),
                 std::invalid_argument);
    ;
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .skipfooter(1)
                   .nrows(1),
                 cudf::logic_error);
  }
  {
    EXPECT_THROW(cudf::io::csv_reader_options::builder(
                   cudf::io::source_info{cudf::host_span<std::byte const>{
                     reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
                   .na_filter(false)
                   .na_values({"Null"}),
                 cudf::logic_error);
  }
}

TEST_F(CsvReaderTest, HexTest)
{
  auto filepath = temp_env->get_temp_filepath("Hexadecimal.csv");
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "0x0\n-0x1000\n0xfedcba\n0xABCDEF\n0xaBcDeF\n9512c20b\n";
  }
  // specify hex columns by name
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .dtypes({dtype<int64_t>()})
        .header(-1)
        .parse_hex({"A"});
    auto result = cudf::io::read_csv(in_opts);

    expect_column_data_equal(
      std::vector<int64_t>{0, -4096, 16702650, 11259375, 11259375, 2501034507},
      result.tbl->view().column(0));
  }

  // specify hex columns by index
  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .dtypes({dtype<int64_t>()})
        .header(-1)
        .parse_hex(std::vector<int>{0});
    auto result = cudf::io::read_csv(in_opts);

    expect_column_data_equal(
      std::vector<int64_t>{0, -4096, 16702650, 11259375, 11259375, 2501034507},
      result.tbl->view().column(0));
  }
}

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });
  auto input_column = column_wrapper<TypeParam>(sequence, sequence + num_rows);
  auto input_table  = cudf::table_view{std::vector<cudf::column_view>{input_column}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithWriter.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, MultiColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto int8_column        = []() {
    auto values = random_values<int8_t>(num_rows);
    return column_wrapper<int8_t>(values.begin(), values.end());
  }();
  auto int16_column = []() {
    auto values = random_values<int16_t>(num_rows);
    return column_wrapper<int16_t>(values.begin(), values.end());
  }();
  auto int32_column = []() {
    auto values = random_values<int32_t>(num_rows);
    return column_wrapper<int32_t>(values.begin(), values.end());
  }();
  auto int64_column = []() {
    auto values = random_values<int64_t>(num_rows);
    return column_wrapper<int64_t>(values.begin(), values.end());
  }();
  auto uint8_column = []() {
    auto values = random_values<uint8_t>(num_rows);
    return column_wrapper<uint8_t>(values.begin(), values.end());
  }();
  auto uint16_column = []() {
    auto values = random_values<uint16_t>(num_rows);
    return column_wrapper<uint16_t>(values.begin(), values.end());
  }();
  auto uint32_column = []() {
    auto values = random_values<uint32_t>(num_rows);
    return column_wrapper<uint32_t>(values.begin(), values.end());
  }();
  auto uint64_column = []() {
    auto values = random_values<uint64_t>(num_rows);
    return column_wrapper<uint64_t>(values.begin(), values.end());
  }();
  auto float32_column = []() {
    auto values = random_values<float>(num_rows);
    return column_wrapper<float>(values.begin(), values.end());
  }();
  auto float64_column = []() {
    auto values = random_values<double>(num_rows);
    return column_wrapper<double>(values.begin(), values.end());
  }();

  std::vector<cudf::column_view> input_columns{int8_column,
                                               int16_column,
                                               int32_column,
                                               int64_column,
                                               uint8_column,
                                               uint16_column,
                                               uint32_column,
                                               uint64_column,
                                               float32_column,
                                               float64_column};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_dir() + "MultiColumnWithWriter.csv";

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .header(-1)
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<uint8_t>(),
               dtype<uint16_t>(),
               dtype<uint32_t>(),
               dtype<uint64_t>(),
               dtype<float>(),
               dtype<double>()});
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();

  std::vector<cudf::size_type> non_float64s{0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto const input_sliced_view  = input_table.select(non_float64s);
  auto const result_sliced_view = result_table.select(non_float64s);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_sliced_view, result_sliced_view);

  auto validity = cudf::test::iterators::no_nulls();
  double tol{1.0e-6};
  auto float64_col_idx = non_float64s.size();
  check_float_column(
    input_table.column(float64_col_idx), result_table.column(float64_col_idx), tol, validity);
}

TEST_F(CsvReaderTest, DatesWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DatesWithWriter.csv";

  auto input_column = column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{983750400000,
                                                                                  1288483200000,
                                                                                  782611200000,
                                                                                  656208000000,
                                                                                  0L,
                                                                                  798163200000,
                                                                                  774144000000,
                                                                                  1149679230400,
                                                                                  1126875750400,
                                                                                  2764800000};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO need to add a dayfirst flag?
  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .dayfirst(true)
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();

  check_timestamp_column(input_table.column(0), result_table.column(0));
}

TEST_F(CsvReaderTest, DatesStringWithWriter)
{
  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_D.csv";

    auto input_column = column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{-106751, 106751};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-22", "2262-04-11"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_s.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{-9223372036, 9223372036};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:44Z", "2262-04-11T23:47:16Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ms.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{-9223372036854, 9223372036854};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.146Z", "2262-04-11T23:47:16.854Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_us.csv";

    auto input_column = column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>{
      -9223372036854775, 9223372036854775};
    auto cast_column     = cudf::strings::from_timestamps(input_column, "%Y-%m-%dT%H:%M:%S.%fZ");
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145225Z",
                                                             "2262-04-11T23:47:16.854775Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ns.csv";

    auto input_column = column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>{
      -9223372036854775807, 9223372036854775807};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145224193Z",
                                                             "2262-04-11T23:47:16.854775807Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table);

    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
        .names({"A"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }
}

TEST_F(CsvReaderTest, FloatingPointWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPointWithWriter.csv";

  auto input_column =
    column_wrapper<double>{5.6, 56.79, 12000000000., 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO add lineterminator=";"
  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names({"A"})
      .dtypes({dtype<double>()})
      .header(-1);
  // in_opts.lineterminator = ';';
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, StringsWithWriter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriter.csv";

  auto int_column = column_wrapper<int32_t>{10, 20, 30};
  auto string_column =
    column_wrapper<cudf::string_view>{"abc def ghi", "\"jkl mno pqr\"", R"(stu ""vwx"" yz)"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, StringsWithWriterSimple)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl mno pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()})
      .quoting(cudf::io::quote_style::NONE);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, StringsEmbeddedDelimiter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl,mno,pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, HeaderEmbeddedDelimiter)
{
  std::vector<std::string> names{
    "header1", "header,2", "quote\"embedded", "new\nline", "\"quoted\""};

  auto filepath = temp_env->get_temp_dir() + "HeaderEmbeddedDelimiter.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc", "jkl,mno", "xyz"};
  cudf::table_view input_table(
    std::vector<cudf::column_view>{int_column, string_column, int_column, int_column, int_column});

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes({dtype<int32_t>(),
               dtype<cudf::string_view>(),
               dtype<int32_t>(),
               dtype<int32_t>(),
               dtype<int32_t>()});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, EmptyFileWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFileWithWriter.csv";

  cudf::table_view empty_table;
  write_csv_helper(filepath, empty_table);
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty_table, result.tbl->view());
}

class TestSource : public cudf::io::datasource {
 public:
  std::string const str;

  TestSource(std::string s) : str(std::move(s)) {}
  std::unique_ptr<buffer> host_read(std::size_t offset, std::size_t size) override
  {
    size = std::min(size, str.size() - offset);
    return std::make_unique<non_owning_buffer>((uint8_t*)str.data() + offset, size);
  }

  std::size_t host_read(std::size_t offset, std::size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, str.size() - offset);
    memcpy(dst, str.data() + offset, size);
    return read_size;
  }

  [[nodiscard]] std::size_t size() const override { return str.size(); }
};

TEST_F(CsvReaderTest, UserImplementedSource)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);

  std::ostringstream csv_data;
  for (int i = 0; i < num_rows; ++i) {
    csv_data << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
             << "\n";
  }
  TestSource source{csv_data.str()};
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{&source})
      .compression(cudf::io::compression_type::NONE)
      .dtypes({dtype<int8_t>(), dtype<int16_t>(), dtype<int32_t>()})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int32_values, view.column(2));
}

TEST_F(CsvReaderTest, DurationsWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DurationsWithWriter.csv";

  constexpr long max_value_d  = std::numeric_limits<cudf::duration_D::rep>::max();
  constexpr long min_value_d  = std::numeric_limits<cudf::duration_D::rep>::min();
  constexpr long max_value_ns = std::numeric_limits<cudf::duration_s::rep>::max();
  constexpr long min_value_ns = std::numeric_limits<cudf::duration_s::rep>::min();
  column_wrapper<cudf::duration_D, cudf::duration_D::rep> durations_D{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_d, max_value_d}};
  column_wrapper<cudf::duration_s, int64_t> durations_s{{-86400L,
                                                         -3600L,
                                                         -2L,
                                                         -1L,
                                                         0L,
                                                         1L,
                                                         2L,
                                                         min_value_ns / 1000000000 + 1,
                                                         max_value_ns / 1000000000}};
  column_wrapper<cudf::duration_ms, int64_t> durations_ms{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000000 + 1, max_value_ns / 1000000}};
  column_wrapper<cudf::duration_us, int64_t> durations_us{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000 + 1, max_value_ns / 1000}};
  column_wrapper<cudf::duration_ns, int64_t> durations_ns{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns, max_value_ns}};

  cudf::table_view input_table(std::vector<cudf::column_view>{
    durations_D, durations_s, durations_ms, durations_us, durations_ns});
  std::vector<std::string> names{"D", "s", "ms", "us", "ns"};

  write_csv_helper(filepath, input_table, names);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .names(names)
      .dtypes({data_type{type_id::DURATION_DAYS},
               data_type{type_id::DURATION_SECONDS},
               data_type{type_id::DURATION_MILLISECONDS},
               data_type{type_id::DURATION_MICROSECONDS},
               data_type{type_id::DURATION_NANOSECONDS}});
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
  ASSERT_EQ(result.metadata.schema_info.size(), names.size());
  for (auto i = 0ul; i < names.size(); ++i)
    EXPECT_EQ(names[i], result.metadata.schema_info[i].name);
}

TEST_F(CsvReaderTest, ParseInRangeIntegers)
{
  std::vector<int64_t> small_int               = {0, -10, 20, -30};
  std::vector<int64_t> less_equal_int64_max    = {std::numeric_limits<int64_t>::max() - 3,
                                                  std::numeric_limits<int64_t>::max() - 2,
                                                  std::numeric_limits<int64_t>::max() - 1,
                                                  std::numeric_limits<int64_t>::max()};
  std::vector<int64_t> greater_equal_int64_min = {std::numeric_limits<int64_t>::min() + 3,
                                                  std::numeric_limits<int64_t>::min() + 2,
                                                  std::numeric_limits<int64_t>::min() + 1,
                                                  std::numeric_limits<int64_t>::min()};
  std::vector<uint64_t> greater_int64_max      = {uint64_t{std::numeric_limits<int64_t>::max()} - 1,
                                                  uint64_t{std::numeric_limits<int64_t>::max()},
                                                  uint64_t{std::numeric_limits<int64_t>::max()} + 1,
                                                  uint64_t{std::numeric_limits<int64_t>::max()} + 2};
  std::vector<uint64_t> less_equal_uint64_max  = {std::numeric_limits<uint64_t>::max() - 3,
                                                  std::numeric_limits<uint64_t>::max() - 2,
                                                  std::numeric_limits<uint64_t>::max() - 1,
                                                  std::numeric_limits<uint64_t>::max()};
  auto input_small_int = column_wrapper<int64_t>(small_int.begin(), small_int.end());
  auto input_less_equal_int64_max =
    column_wrapper<int64_t>(less_equal_int64_max.begin(), less_equal_int64_max.end());
  auto input_greater_equal_int64_min =
    column_wrapper<int64_t>(greater_equal_int64_min.begin(), greater_equal_int64_min.end());
  auto input_greater_int64_max =
    column_wrapper<uint64_t>(greater_int64_max.begin(), greater_int64_max.end());
  auto input_less_equal_uint64_max =
    column_wrapper<uint64_t>(less_equal_uint64_max.begin(), less_equal_uint64_max.end());

  auto small_int_append_zeros               = prepend_zeros(small_int, 32, true);
  auto less_equal_int64_max_append_zeros    = prepend_zeros(less_equal_int64_max, 32, true);
  auto greater_equal_int64_min_append_zeros = prepend_zeros(greater_equal_int64_min, 17);
  auto greater_int64_max_append_zeros       = prepend_zeros(greater_int64_max, 5);
  auto less_equal_uint64_max_append_zeros   = prepend_zeros(less_equal_uint64_max, 8, true);

  auto input_small_int_append =
    column_wrapper<cudf::string_view>(small_int_append_zeros.begin(), small_int_append_zeros.end());
  auto input_less_equal_int64_max_append = column_wrapper<cudf::string_view>(
    less_equal_int64_max_append_zeros.begin(), less_equal_int64_max_append_zeros.end());
  auto input_greater_equal_int64_min_append = column_wrapper<cudf::string_view>(
    greater_equal_int64_min_append_zeros.begin(), greater_equal_int64_min_append_zeros.end());
  auto input_greater_int64_max_append = column_wrapper<cudf::string_view>(
    greater_int64_max_append_zeros.begin(), greater_int64_max_append_zeros.end());
  auto input_less_equal_uint64_max_append = column_wrapper<cudf::string_view>(
    less_equal_uint64_max_append_zeros.begin(), less_equal_uint64_max_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_small_int,
                                               input_less_equal_int64_max,
                                               input_greater_equal_int64_min,
                                               input_greater_int64_max,
                                               input_less_equal_uint64_max,
                                               input_small_int_append,
                                               input_less_equal_int64_max_append,
                                               input_greater_equal_int64_min_append,
                                               input_greater_int64_max_append,
                                               input_less_equal_uint64_max_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseInRangeIntegers.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(4));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(9));
}

TEST_F(CsvReaderTest, ParseOutOfRangeIntegers)
{
  std::vector<std::string> out_of_range_positive = {"111111111111111111111",
                                                    "2222222222222222222222",
                                                    "33333333333333333333333",
                                                    "444444444444444444444444"};
  std::vector<std::string> out_of_range_negative = {"-111111111111111111111",
                                                    "-2222222222222222222222",
                                                    "-33333333333333333333333",
                                                    "-444444444444444444444444"};
  std::vector<std::string> greater_uint64_max    = {
    "18446744073709551615", "18446744073709551616", "18446744073709551617", "18446744073709551618"};
  std::vector<std::string> less_int64_min = {
    "-9223372036854775807", "-9223372036854775808", "-9223372036854775809", "-9223372036854775810"};
  std::vector<std::string> mixed_range = {
    "18446744073709551613", "18446744073709551614", "18446744073709551615", "-5"};
  auto input_out_of_range_positive =
    column_wrapper<cudf::string_view>(out_of_range_positive.begin(), out_of_range_positive.end());
  auto input_out_of_range_negative =
    column_wrapper<cudf::string_view>(out_of_range_negative.begin(), out_of_range_negative.end());
  auto input_greater_uint64_max =
    column_wrapper<cudf::string_view>(greater_uint64_max.begin(), greater_uint64_max.end());
  auto input_less_int64_min =
    column_wrapper<cudf::string_view>(less_int64_min.begin(), less_int64_min.end());
  auto input_mixed_range =
    column_wrapper<cudf::string_view>(mixed_range.begin(), mixed_range.end());

  auto out_of_range_positive_append_zeros = prepend_zeros(out_of_range_positive, 32, true);
  auto out_of_range_negative_append_zeros = prepend_zeros(out_of_range_negative, 5);
  auto greater_uint64_max_append_zeros    = prepend_zeros(greater_uint64_max, 8, true);
  auto less_int64_min_append_zeros        = prepend_zeros(less_int64_min, 17);
  auto mixed_range_append_zeros           = prepend_zeros(mixed_range, 2, true);

  auto input_out_of_range_positive_append = column_wrapper<cudf::string_view>(
    out_of_range_positive_append_zeros.begin(), out_of_range_positive_append_zeros.end());
  auto input_out_of_range_negative_append = column_wrapper<cudf::string_view>(
    out_of_range_negative_append_zeros.begin(), out_of_range_negative_append_zeros.end());
  auto input_greater_uint64_max_append = column_wrapper<cudf::string_view>(
    greater_uint64_max_append_zeros.begin(), greater_uint64_max_append_zeros.end());
  auto input_less_int64_min_append = column_wrapper<cudf::string_view>(
    less_int64_min_append_zeros.begin(), less_int64_min_append_zeros.end());
  auto input_mixed_range_append = column_wrapper<cudf::string_view>(
    mixed_range_append_zeros.begin(), mixed_range_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_out_of_range_positive,
                                               input_out_of_range_negative,
                                               input_greater_uint64_max,
                                               input_less_int64_min,
                                               input_mixed_range,
                                               input_out_of_range_positive_append,
                                               input_out_of_range_negative_append,
                                               input_greater_uint64_max_append,
                                               input_less_int64_min_append,
                                               input_mixed_range_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseOutOfRangeIntegers.csv");

  write_csv_helper(filepath, input_table);

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range, view.column(4));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive_append, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative_append, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max_append, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min_append, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range_append, view.column(9));
}

TEST_F(CsvReaderTest, ReadMaxNumericValue)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::numeric_limits<uint64_t>::max() - i; });

  auto filepath = temp_env->get_temp_filepath("ReadMaxNumericValue.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<uint64_t> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(std::vector<uint64_t>(sequence, sequence + num_rows), view.column(0));
}

TEST_F(CsvReaderTest, DefaultWriteChunkSize)
{
  for (auto num_rows : {1, 20, 100, 1000}) {
    auto sequence = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<int32_t>(i + 1000.50f); });
    auto input_column = column_wrapper<int32_t>(sequence, sequence + num_rows);
    auto input_table  = cudf::table_view{std::vector<cudf::column_view>{input_column}};

    cudf::io::csv_writer_options opts =
      cudf::io::csv_writer_options::builder(cudf::io::sink_info{"unused.path"}, input_table);
    ASSERT_EQ(num_rows, opts.get_rows_per_chunk());
  }
}

TEST_F(CsvReaderTest, DtypesMap)
{
  std::string csv_in{"12,9\n34,8\n56,7"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B"})
      .dtypes({{"B", dtype<int16_t>()}, {"A", dtype<int32_t>()}})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 2);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::INT32});
  ASSERT_EQ(result_table.column(1).type(), data_type{type_id::INT16});
  expect_column_data_equal(std::vector<int32_t>{12, 34, 56}, result_table.column(0));
  expect_column_data_equal(std::vector<int16_t>{9, 8, 7}, result_table.column(1));
}

TEST_F(CsvReaderTest, DtypesMapPartial)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{nullptr, 0}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B"})
      .dtypes({{"A", dtype<int16_t>()}});
  {
    auto result = cudf::io::read_csv(in_opts);

    auto const view = result.tbl->view();
    ASSERT_EQ(type_id::INT16, view.column(0).type().id());
    // Default to String if there's no data
    ASSERT_EQ(type_id::STRING, view.column(1).type().id());
  }

  in_opts.set_dtypes({{"B", dtype<uint32_t>()}});
  {
    auto result = cudf::io::read_csv(in_opts);

    auto const view = result.tbl->view();
    ASSERT_EQ(type_id::STRING, view.column(0).type().id());
    ASSERT_EQ(type_id::UINT32, view.column(1).type().id());
  }
}

TEST_F(CsvReaderTest, DtypesArrayInvalid)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{nullptr, 0}})
      .compression(cudf::io::compression_type::NONE)
      .names({"A", "B", "C"})
      .dtypes(std::vector<cudf::data_type>{dtype<int16_t>(), dtype<int8_t>()});

  EXPECT_THROW(cudf::io::read_csv(in_opts), cudf::logic_error);
}

TEST_F(CsvReaderTest, CsvDefaultOptionsWriteReadMatch)
{
  auto const filepath = temp_env->get_temp_dir() + "issue.csv";

  // make up some kind of dataframe
  auto int_column = column_wrapper<int32_t>{10, 20, 30};
  auto str_column = column_wrapper<cudf::string_view>{"abc", "mno", "xyz"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, str_column});

  // write that dataframe to a csv using default options to some temporary file
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, input_table);
  cudf::io::write_csv(writer_options);

  // read the temp csv file using default options
  cudf::io::csv_reader_options read_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});

  cudf::io::table_with_metadata new_table_and_metadata = cudf::io::read_csv(read_options);

  // verify that the tables are identical, or as identical as expected.
  auto const new_table_view = new_table_and_metadata.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, new_table_view);
  EXPECT_EQ(new_table_and_metadata.metadata.schema_info[0].name, "0");
  EXPECT_EQ(new_table_and_metadata.metadata.schema_info[1].name, "1");
}

TEST_F(CsvReaderTest, UseColsValidation)
{
  std::string const buffer = "1,2,3";

  cudf::io::csv_reader_options const idx_cnt_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b"})
      .use_cols_indexes({0});
  EXPECT_THROW(cudf::io::read_csv(idx_cnt_options), cudf::logic_error);

  cudf::io::csv_reader_options unique_idx_cnt_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b"})
      .use_cols_indexes({0, 0});
  EXPECT_THROW(cudf::io::read_csv(unique_idx_cnt_options), cudf::logic_error);

  cudf::io::csv_reader_options bad_name_options =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b", "c"})
      .use_cols_names({"nonexistent_name"});
  EXPECT_THROW(cudf::io::read_csv(bad_name_options), cudf::logic_error);
}

TEST_F(CsvReaderTest, CropColumns)
{
  std::string const csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<float>()})
      .names({"a", "b"})
      .header(-1);
  auto const result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 2);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::INT32});
  ASSERT_EQ(result_table.column(1).type(), data_type{type_id::FLOAT32});
  expect_column_data_equal(std::vector<int32_t>{12, 34, 56}, result_table.column(0));
  expect_column_data_equal(std::vector<float>{9., 8., 7.}, result_table.column(1));
}

TEST_F(CsvReaderTest, CropColumnsUseColsNames)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<float>()})
      .names({"a", "b"})
      .use_cols_names({"b"})
      .header(-1);
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  ASSERT_EQ(result_table.num_columns(), 1);
  ASSERT_EQ(result_table.column(0).type(), data_type{type_id::FLOAT32});
  expect_column_data_equal(std::vector<float>{9., 8., 7.}, result_table.column(0));
}

TEST_F(CsvReaderTest, ExtraColumns)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};
  {
    cudf::io::csv_reader_options opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .header(-1);
    auto result = cudf::io::read_csv(opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 4);
    ASSERT_EQ(result_table.column(3).type(), data_type{type_id::INT8});
    ASSERT_EQ(result_table.column(3).null_count(), 3);
  }
  {
    cudf::io::csv_reader_options with_dtypes_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>(), dtype<float>()})
        .header(-1);
    auto result = cudf::io::read_csv(with_dtypes_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 4);
    ASSERT_EQ(result_table.column(3).type(), data_type{type_id::FLOAT32});
    ASSERT_EQ(result_table.column(3).null_count(), 3);
  }
}

TEST_F(CsvReaderTest, ExtraColumnsUseCols)
{
  std::string csv_in{"12,9., 10\n34,8., 20\n56,7., 30"};

  {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .use_cols_names({"b", "d"})
        .header(-1);
    auto result = cudf::io::read_csv(in_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 2);
    ASSERT_EQ(result_table.column(1).type(), data_type{type_id::INT8});
    ASSERT_EQ(result_table.column(1).null_count(), 3);
  }
  {
    cudf::io::csv_reader_options with_dtypes_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .names({"a", "b", "c", "d"})
        .use_cols_names({"b", "d"})
        .dtypes({dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>(), dtype<cudf::string_view>()})
        .header(-1);
    auto result = cudf::io::read_csv(with_dtypes_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 2);
    ASSERT_EQ(result_table.column(1).type(), data_type{type_id::STRING});
    ASSERT_EQ(result_table.column(1).null_count(), 3);
  }
}

TEST_F(CsvReaderTest, EmptyColumns)
{
  // First column only has empty fields. second column contains only "null" literals
  std::string csv_in{",null\n,null"};

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
      .compression(cudf::io::compression_type::NONE)
      .names({"a", "b", "c", "d"})
      .header(-1);
  // More elements in `names` than in the file; additional columns are filled with nulls
  auto result = cudf::io::read_csv(in_opts);

  auto const result_table = result.tbl->view();
  EXPECT_EQ(result_table.num_columns(), 4);
  // All columns should contain only nulls; expect INT8 type to use as little memory as possible
  for (auto& column : result_table) {
    EXPECT_EQ(column.type(), data_type{type_id::INT8});
    EXPECT_EQ(column.null_count(), 2);
  }
}

TEST_F(CsvReaderTest, BlankLineAfterFirstRow)
{
  std::string csv_in{"12,9., 10\n\n"};

  {
    cudf::io::csv_reader_options no_header_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE)
        .header(-1);
    // No header, getting column names/count from first row
    auto result = cudf::io::read_csv(no_header_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 3);
  }
  {
    cudf::io::csv_reader_options header_opts =
      cudf::io::csv_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(csv_in.c_str()), csv_in.size()}})
        .compression(cudf::io::compression_type::NONE);
    // Getting column names/count from header
    auto result = cudf::io::read_csv(header_opts);

    auto const result_table = result.tbl->view();
    ASSERT_EQ(result_table.num_columns(), 3);
  }
}

TEST_F(CsvReaderTest, NullCount)
{
  std::string buffer = "0,,\n1,1.,\n2,,\n3,,\n4,4.,\n5,5.,\n6,6.,\n7,7.,\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();

  EXPECT_EQ(result_view.num_rows(), 8);
  EXPECT_EQ(result_view.column(0).null_count(), 0);
  EXPECT_EQ(result_view.column(1).null_count(), 3);
  EXPECT_EQ(result_view.column(2).null_count(), 8);
}

TEST_F(CsvReaderTest, UTF8BOM)
{
  std::string buffer = "\xEF\xBB\xBFMonth,Day,Year\nJune,6,2023\nAugust,25,1990\nMay,1,2000\n";
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE);
  auto const result      = cudf::io::read_csv(in_opts);
  auto const result_view = result.tbl->view();
  EXPECT_EQ(result_view.num_rows(), 3);
  EXPECT_EQ(result.metadata.schema_info.front().name, "Month");

  auto col1     = cudf::test::strings_column_wrapper({"June", "August", "May"});
  auto col2     = cudf::test::fixed_width_column_wrapper<int64_t>({6, 25, 1});
  auto col3     = cudf::test::fixed_width_column_wrapper<int64_t>({2023, 1990, 2000});
  auto expected = cudf::table_view({col1, col2, col3});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result_view, expected);
}

void expect_buffers_equal(cudf::io::datasource::buffer* lhs, cudf::io::datasource::buffer* rhs)
{
  ASSERT_EQ(lhs->size(), rhs->size());
  EXPECT_EQ(0, std::memcmp(lhs->data(), rhs->data(), lhs->size()));
}

TEST_F(CsvReaderTest, OutOfMapBoundsReads)
{
  // write a lot of data into a file
  auto filepath        = temp_env->get_temp_dir() + "OutOfMapBoundsReads.csv";
  auto const num_rows  = 1 << 20;
  auto const row       = std::string{"0,1,2,3,4,5,6,7,8,9\n"};
  auto const file_size = num_rows * row.size();
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    for (std::size_t i = 0; i < num_rows; ++i) {
      outfile << row;
    }
  }

  // Only memory map the middle of the file
  auto source         = cudf::io::datasource::create(filepath, file_size / 2, file_size / 4);
  auto full_source    = cudf::io::datasource::create(filepath);
  auto const all_data = source->host_read(0, file_size);
  auto ref_data       = full_source->host_read(0, file_size);
  expect_buffers_equal(ref_data.get(), all_data.get());

  auto const start_data = source->host_read(file_size / 2, file_size / 2);
  expect_buffers_equal(full_source->host_read(file_size / 2, file_size / 2).get(),
                       start_data.get());

  auto const end_data = source->host_read(0, file_size / 2 + 512);
  expect_buffers_equal(full_source->host_read(0, file_size / 2 + 512).get(), end_data.get());
}

struct CsvWriterTypeSupportTest : public cudf::test::BaseFixture {};

TEST(CsvWriterTypeSupportTest, SupportedTypes)
{
  using cudf::io::is_supported_write_csv;

  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::INT32}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::FLOAT64}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::STRING}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::DECIMAL64}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}));
  EXPECT_TRUE(is_supported_write_csv(cudf::data_type{cudf::type_id::DURATION_SECONDS}));
}

TEST(CsvWriterTypeSupportTest, UnsupportedTypes)
{
  using cudf::io::is_supported_write_csv;

  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::LIST}));
  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::STRUCT}));
  EXPECT_FALSE(is_supported_write_csv(cudf::data_type{cudf::type_id::DICTIONARY32}));
}

TEST_F(CsvReaderTest, DoubleQuotesOddCount)
{
  std::string const content{R"(Monday" "Tuesday" Wednesday)"};
  std::string const buffer = content + "\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .dtypes({dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);
  EXPECT_EQ(result.tbl->view().num_columns(), 1);

  auto const expected = cudf::test::strings_column_wrapper({content});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(CsvReaderTest, UnquotedStringDoubledQuotes)
{
  // Tab delimiter makes the entire line a single unquoted field containing commas and quotes
  std::string const buffer = R"(hello,"",world
foo
bar
end
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .header(-1)
      .dtypes({dtype<cudf::string_view>()})
      .delimiter('\t');
  auto const result = cudf::io::read_csv(in_opts);
  EXPECT_EQ(result.tbl->view().num_columns(), 1);
  EXPECT_EQ(result.tbl->view().num_rows(), 4);

  auto const expected =
    cudf::test::strings_column_wrapper({R"(hello,"",world)", R"(foo)", R"(bar)", R"(end)"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(CsvReaderTest, QuotedFieldWithTrailingDelimiter)
{
  std::string const buffer = R"(a,b
1,"trailing,"
2,"normal"
3,end
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"trailing,", "normal", "end"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

TEST_F(CsvReaderTest, EscapedQuotesWithSemicolonDelimiter)
{
  std::string const buffer = R"(a;b
1;"hello""world"
2;"test"
3;"end"
)";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delimiter(';')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"hello\"world", "test", "end"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

TEST_F(CsvReaderTest, EscapedQuoteBeforeDelimiterInQuotedField)
{
  std::string const buffer =
    "a;b\n"
    "1;\"hello\"\";\"\n"  // hello"; escaped as "hello"";"
    "2;\"test\"\n"
    "3;\"end\"\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .delimiter(';')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 3);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);
}

TEST_F(CsvReaderTest, CommentLines)
{
  std::string const buffer =
    "# This is a comment\n"
    "a,b,c\n"
    "1,2,3\n"
    "# Another comment\n"
    "4,5,6\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .comment('#')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<int32_t>(), dtype<int32_t>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 2);
  EXPECT_EQ(result.tbl->view().num_columns(), 3);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 4});
  auto const expected_col1 = cudf::test::fixed_width_column_wrapper<int32_t>({2, 5});
  auto const expected_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({3, 6});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(2), expected_col2);
}

TEST_F(CsvReaderTest, CommentLinesWithQuotedStrings)
{
  std::string const buffer =
    "# comment\n"
    "a,b\n"
    "1,\"hello\"\"world\"\n"
    "# another comment\n"
    "2,\"test\"\n";

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
      .compression(cudf::io::compression_type::NONE)
      .comment('#')
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<cudf::string_view>()});
  auto const result = cudf::io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->view().num_rows(), 2);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  auto const expected_col0 = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  auto const expected_col1 = cudf::test::strings_column_wrapper({"hello\"world", "test"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(0), expected_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.tbl->view().column(1), expected_col1);
}

CUDF_TEST_PROGRAM_MAIN()
