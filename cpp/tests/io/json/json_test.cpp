/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../io_test_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <cuda/iterator>

#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper        = wrapper<float>;
using float64_wrapper      = wrapper<double>;
using int_wrapper          = wrapper<int>;
using int8_wrapper         = wrapper<int8_t>;
using int16_wrapper        = wrapper<int16_t>;
using int64_wrapper        = wrapper<int64_t>;
using timestamp_ms_wrapper = wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>;
using bool_wrapper         = wrapper<bool>;
using size_type_wrapper    = wrapper<cudf::size_type>;
using strings_wrapper      = cudf::test::strings_column_wrapper;

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

cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T>
std::vector<std::string> prepend_zeros(std::vector<T> const& input,
                                       int zero_count         = 0,
                                       bool add_positive_sign = false)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](T const& num) {
    auto str         = std::to_string(num);
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

template <>
std::vector<std::string> prepend_zeros<std::string>(std::vector<std::string> const& input,
                                                    int zero_count,
                                                    bool add_positive_sign)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](std::string const& num) {
    auto str         = num;
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

MATCHER_P(FloatNearPointwise, tolerance, "Out-of-range")
{
  return (std::get<0>(arg) > std::get<1>(arg) - tolerance &&
          std::get<0>(arg) < std::get<1>(arg) + tolerance);
}

// temporary method to verify the float columns until
// CUDF_TEST_EXPECT_COLUMNS_EQUAL supports floating point
template <typename T>
void check_float_column(cudf::column_view const& col, std::vector<T> const& data)
{
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(col, (wrapper<T>(data.begin(), data.end())));
  EXPECT_EQ(col.null_count(), 0);
  EXPECT_THAT(cudf::test::to_host<T>(col).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), data));
}

/**
 * @brief Base test fixture for JSON reader tests
 */
struct JsonReaderTest : public cudf::test::BaseFixture {};

/**
 * @brief Enum class to be used to specify the test case of parametrized tests
 */
enum class json_test_t {
  // Run test with the nested JSON lines reader using record-orient input data
  json_record_orient,
  // Run test with the nested JSON lines reader using row-orient input data
  json_row_orient
};

constexpr bool is_row_orient_test(json_test_t test_opt)
{
  return test_opt == json_test_t::json_row_orient;
}

/**
 * @brief Test fixture for parametrized JSON reader tests
 */
struct JsonReaderParamTest : public cudf::test::BaseFixture,
                             public testing::WithParamInterface<json_test_t> {};

/**
 * @brief Test fixture for parametrized JSON reader tests with both orients
 */
struct JsonReaderRecordTest : public cudf::test::BaseFixture,
                              public testing::WithParamInterface<json_test_t> {};

/**
 * @brief Generates a JSON lines string that uses the record orient
 *
 * @param records An array of a map of key-value pairs
 * @param record_delimiter The delimiter to be used to delimit a record
 * @param prefix The prefix prepended to the whole string
 * @param suffix The suffix to be appended after the whole string
 * @return The JSON lines string that uses the record orient
 */
std::string to_records_orient(std::vector<std::map<std::string, std::string>> const& records,
                              std::string record_delimiter,
                              std::string prefix = "",
                              std::string suffix = "")
{
  std::string result = prefix;
  for (auto record_it = std::cbegin(records); record_it != std::cend(records); record_it++) {
    result += "{";
    for (auto kv_pair_it = std::cbegin(*record_it); kv_pair_it != std::cend(*record_it);
         kv_pair_it++) {
      auto const& [key, value] = *kv_pair_it;
      result += "\"" + key + "\":" + value;
      result += (kv_pair_it != std::prev(std::end(*record_it))) ? ", " : "";
    }
    result += "}";
    if (record_it != std::prev(std::end(records))) { result += record_delimiter; }
  }
  return (result + suffix);
}

template <typename DecimalType>
struct JsonFixedPointReaderTest : public JsonReaderTest {};

template <typename DecimalType>
struct JsonValidFixedPointReaderTest : public JsonFixedPointReaderTest<DecimalType> {
  void run_test(std::vector<std::string> const& reference_strings, numeric::scale_type scale)
  {
    cudf::test::strings_column_wrapper const strings(reference_strings.begin(),
                                                     reference_strings.end());
    auto const expected = cudf::strings::to_fixed_point(
      cudf::strings_column_view(strings), data_type{type_to_id<DecimalType>(), scale});

    auto const buffer =
      std::accumulate(reference_strings.begin(),
                      reference_strings.end(),
                      std::string{},
                      [](std::string const& acc, std::string const& rhs) {
                        return acc + (acc.empty() ? "" : "\n") + "{\"col0\":" + rhs + "}";
                      });
    cudf::io::json_reader_options const in_opts =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
        .dtypes(std::vector{data_type{type_to_id<DecimalType>(), scale}})
        .lines(true);

    auto const result      = cudf::io::read_json(in_opts);
    auto const result_view = result.tbl->view();

    ASSERT_EQ(result_view.num_columns(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].name, "col0");
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, result_view.column(0));
  }

  void run_tests(std::vector<std::string> const& reference_strings, numeric::scale_type scale)
  {
    // Test both parsers
    run_test(reference_strings, scale);
    run_test(reference_strings, scale);
  }
};

TYPED_TEST_SUITE(JsonFixedPointReaderTest, cudf::test::FixedPointTypes);
TYPED_TEST_SUITE(JsonValidFixedPointReaderTest, cudf::test::FixedPointTypes);

// Parametrize qualifying JSON tests for supported orients
INSTANTIATE_TEST_CASE_P(JsonReaderParamTest,
                        JsonReaderParamTest,
                        ::testing::Values(json_test_t::json_record_orient,
                                          json_test_t::json_row_orient));

// Parametrize qualifying JSON tests for supported orients
INSTANTIATE_TEST_CASE_P(JsonReaderRecordTest,
                        JsonReaderRecordTest,
                        ::testing::Values(json_test_t::json_record_orient));

TEST_P(JsonReaderParamTest, BasicJsonLines)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  std::string record_orient = to_records_orient(
    {{{"0", "1"}, {"1", "1.1"}}, {{"0", "2"}, {"1", "2.2"}}, {{"0", "3"}, {"1", "3.3"}}}, "\n");
  std::string data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<double>()})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2, 3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3}});
}

TEST_P(JsonReaderParamTest, FloatingPoint)
{
  auto const test_opt = GetParam();
  std::string row_orient =
    "[5.6]\n[0.5679e2]\n[1.2e10]\n[0.07e1]\n[3000e-3]\n[12.34e0]\n[3.1e-001]\n[-73."
    "98007199999998]\n";
  std::string record_orient = to_records_orient({{{"0", "5.6"}},
                                                 {{"0", "0.5679e2"}},
                                                 {{"0", "1.2e10"}},
                                                 {{"0", "0.07e1"}},
                                                 {{"0", "3000e-3"}},
                                                 {{"0", "12.34e0"}},
                                                 {{"0", "3.1e-001"}},
                                                 {{"0", "-73.98007199999998"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector{dtype<float>()})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT32);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    float_wrapper{{5.6, 56.79, 12000000000., 0.7, 3.000, 12.34, 0.31, -73.98007199999998}});
}

TEST_P(JsonReaderParamTest, JsonLinesStrings)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";
  std::string record_orient = to_records_orient({{{"0", "1"}, {"1", "1.1"}, {"2", R"("aa ")"}},
                                                 {{"0", "2"}, {"1", "2.2"}, {"2", R"("  bbb")"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(std::map<std::string, data_type>{
        {"2", dtype<cudf::string_view>()}, {"0", dtype<int32_t>()}, {"1", dtype<double>()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  EXPECT_EQ(result.metadata.schema_info[2].name, "2");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_P(JsonReaderParamTest, MultiColumn)
{
  auto const test_opt   = GetParam();
  bool const row_orient = is_row_orient_test(test_opt);

  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.json";
  {
    std::ostringstream line;
    if (row_orient) {
      for (int i = 0; i < num_rows; ++i) {
        line << "[" << std::to_string(int8_values[i]) << "," << int16_values[i] << ","
             << int32_values[i] << "," << int64_values[i] << "," << float32_values[i] << ","
             << float64_values[i] << "]\n";
      }
    } else {
      std::vector<std::map<std::string, std::string>> records;
      for (int i = 0; i < num_rows; ++i) {
        records.push_back({
          {"0", std::to_string(int8_values[i])},     //
          {"1", std::to_string(int16_values[i])},    //
          {"2", std::to_string(int32_values[i])},    //
          {"3", std::to_string(int64_values[i])},    //
          {"4", std::to_string(float32_values[i])},  //
          {"5", std::to_string(float64_values[i])},  //
        });
      }
      line << to_records_orient(records, "\n");
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<float>(),
               dtype<double>()})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();

  EXPECT_EQ(view.num_columns(), 6);
  EXPECT_EQ(view.column(0).type().id(), cudf::type_id::INT8);
  EXPECT_EQ(view.column(1).type().id(), cudf::type_id::INT16);
  EXPECT_EQ(view.column(2).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(view.column(3).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(view.column(4).type().id(), cudf::type_id::FLOAT32);
  EXPECT_EQ(view.column(5).type().id(), cudf::type_id::FLOAT64);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0),
                                 int8_wrapper(int8_values.begin(), int8_values.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1),
                                 int16_wrapper(int16_values.begin(), int16_values.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(2),
                                 int_wrapper(int32_values.begin(), int32_values.end()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(3),
                                 int64_wrapper(int64_values.begin(), int64_values.end()));
  check_float_column(view.column(4), float32_values);
  check_float_column(view.column(5), float64_values);
}

TEST_P(JsonReaderParamTest, Booleans)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[true]\n[true]\n[false]\n[false]\n[true]";
  std::string record_orient = to_records_orient(
    {
      {{"0", "true"}},
      {{"0", "true"}},
      {{"0", "false"}},
      {{"0", "false"}},
      {{"0", "true"}},
    },
    "\n");
  std::string data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  auto filepath = temp_env->get_temp_dir() + "Booleans.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector{dtype<bool>()})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  auto const view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::BOOL8);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 bool_wrapper{{true, true, false, false, true}});
}

TEST_P(JsonReaderParamTest, Dates)
{
  auto const test_opt = GetParam();
  std::string row_orient =
    "[05/03/2001]\n[31/10/2010]\n[20/10/1994]\n[18/10/1990]\n[1/1/1970]\n"
    "[18/04/1995]\n[14/07/1994]\n[\"07/06/2006 11:20:30.400\"]\n"
    "[\"16/09/2005T1:2:30.400PM\"]\n[2/2/1970]\n[null]";
  std::string record_orient = to_records_orient({{{"0", R"("05/03/2001")"}},
                                                 {{"0", R"("31/10/2010")"}},
                                                 {{"0", R"("20/10/1994")"}},
                                                 {{"0", R"("18/10/1990")"}},
                                                 {{"0", R"("1/1/1970")"}},
                                                 {{"0", R"("18/04/1995")"}},
                                                 {{"0", R"("14/07/1994")"}},
                                                 {{"0", R"("07/06/2006 11:20:30.400")"}},
                                                 {{"0", R"("16/09/2005T1:2:30.400PM")"}},
                                                 {{"0", R"("2/2/1970")"}},
                                                 {{"0", R"(null)"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  auto filepath = temp_env->get_temp_dir() + "Dates.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector{data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .lines(true)
      .dayfirst(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::TIMESTAMP_MILLISECONDS);

  auto validity = cudf::test::iterators::nulls_at({10});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 timestamp_ms_wrapper{{983750400000,
                                                       1288483200000,
                                                       782611200000,
                                                       656208000000,
                                                       0L,
                                                       798163200000,
                                                       774144000000,
                                                       1149679230400,
                                                       1126875750400,
                                                       2764800000,
                                                       0L},
                                                      validity});
}

TEST_P(JsonReaderParamTest, Durations)
{
  auto const test_opt = GetParam();
  std::string row_orient =
    "[-2]\n[-1]\n[0]\n"
    "[\"1 days\"]\n[\"0 days 23:01:00\"]\n[\"0 days 00:00:00.000000123\"]\n"
    "[\"0:0:0.000123\"]\n[\"0:0:0.000123000\"]\n[\"00:00:00.100000001\"]\n"
    "[-2147483648]\n[2147483647]\n[null]";
  std::string record_orient = to_records_orient({{{"0", "-2"}},
                                                 {{"0", "-1"}},
                                                 {{"0", "0"}},
                                                 {{"0", R"("1 days")"}},
                                                 {{"0", R"("0 days 23:01:00")"}},
                                                 {{"0", R"("0 days 00:00:00.000000123")"}},
                                                 {{"0", R"("0:0:0.000123")"}},
                                                 {{"0", R"("0:0:0.000123000")"}},
                                                 {{"0", R"("00:00:00.100000001")"}},
                                                 {{"0", R"(-2147483648)"}},
                                                 {{"0", R"(2147483647)"}},
                                                 {{"0", R"(null)"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;
  auto filepath             = temp_env->get_temp_dir() + "Durations.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector{data_type{type_id::DURATION_NANOSECONDS}})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::DURATION_NANOSECONDS);

  auto validity = cudf::test::iterators::nulls_at({11});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    wrapper<cudf::duration_ns, cudf::duration_ns::rep>{{-2L,
                                                        -1L,
                                                        0L,
                                                        1L * 60 * 60 * 24 * 1000000000L,
                                                        (23 * 60 + 1) * 60 * 1000000000L,
                                                        123L,
                                                        123000L,
                                                        123000L,
                                                        100000001L,
                                                        -2147483648L,
                                                        2147483647L,
                                                        0L},
                                                       validity});
}

TEST_P(JsonReaderParamTest, JsonLinesDtypeInference)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[100, 1.1, \"aa \"]\n[200, 2.2, \"  bbb\"]";
  std::string record_orient = to_records_orient({{{"0", "100"}, {"1", "1.1"}, {"2", R"("aa ")"}},
                                                 {{"0", "200"}, {"1", "2.2"}, {"2", R"("  bbb")"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  EXPECT_EQ(result.metadata.schema_info[2].name, "2");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_P(JsonReaderParamTest, JsonLinesFileInput)
{
  auto const test_opt    = GetParam();
  std::string row_orient = "[11, 1.1]\n[22, 2.2]";
  std::string record_orient =
    to_records_orient({{{"0", "11"}, {"1", "1.1"}}, {{"0", "22"}, {"1", "2.2"}}}, "\n");
  std::string data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  std::string const fname = temp_env->get_temp_dir() + "JsonLinesFileTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << data;
  outfile.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{fname}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{11, 22}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
}

TEST_F(JsonReaderTest, JsonLinesByteRangeCompleteRecord)
{
  std::string const fname = temp_env->get_temp_dir() + "JsonLinesByteRangeTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]\n";
  outfile.close();

  // Requesting 0]\n[3000]\n[4000]\n[5000]\n but reading 0]\n[3000]\n[4000]\n[5000]\n[6000]\n
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{fname})
      .lines(true)
      .byte_range_offset(11)
      .byte_range_size(24);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 4);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{3000, 4000, 5000, 6000}});
}

TEST_F(JsonReaderTest, JsonLinesByteRangeIncompleteRecord)
{
  std::string const fname = temp_env->get_temp_dir() + "JsonLinesByteRangeTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]\n";
  outfile.close();

  // Reading 0]\n[3000]\n[4000]\n[50
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{fname})
      .lines(true)
      .byte_range_offset(11)
      .byte_range_size(20);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{3000, 4000, 5000}});
}

TEST_F(JsonReaderTest, JsonLinesByteRangeWithRealloc)
{
  std::string long_string     = "haha";
  std::size_t log_repetitions = 12;
  long_string.reserve(long_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    long_string += long_string;
  }

  auto json_string = [&long_string]() {
    std::string json_string   = R"(
      { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
      { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
      { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
      { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
    std::string replace_chars = "c";
    std::size_t pos           = json_string.find(replace_chars);
    while (pos != std::string::npos) {
      // Replace the substring with the specified string
      json_string.replace(pos, replace_chars.size(), long_string);

      // Find the next occurrence of the substring
      pos = json_string.find(replace_chars, pos + long_string.size());
    }
    return json_string;
  }();

  // Initialize parsing options (reading json lines). Set byte range offset and size so as to read
  // the second row of input
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte>(
        reinterpret_cast<std::byte*>(json_string.data()), json_string.size())})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL)
      .byte_range_offset(16430)
      .byte_range_size(30);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata result = cudf::io::read_json(json_lines_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 1);
  EXPECT_EQ(result.metadata.schema_info[2].name, long_string);
}

TEST_F(JsonReaderTest, JsonLinesMultipleFilesByteRange_AcrossFiles)
{
  std::string const file1 = temp_env->get_temp_dir() + "JsonLinesMultipleFilesByteRangeTest1.json";
  std::ofstream outfile1(file1, std::ofstream::out);
  outfile1 << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]";
  outfile1.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{{file1, file1}})
      .lines(true)
      .byte_range_offset(11)
      .byte_range_size(70);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 10);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    int64_wrapper{{3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000, 2000, 3000}});
}

TEST_F(JsonReaderTest, JsonLinesMultipleFilesByteRange_ExcessRangeSize)
{
  std::string const file1 = temp_env->get_temp_dir() + "JsonLinesMultipleFilesByteRangeTest1.json";
  std::ofstream outfile1(file1, std::ofstream::out);
  outfile1 << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]";
  outfile1.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{{file1, file1}})
      .lines(true)
      .byte_range_offset(11)
      .byte_range_size(1000);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 16);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{3000,
                                                4000,
                                                5000,
                                                6000,
                                                7000,
                                                8000,
                                                9000,
                                                1000,
                                                2000,
                                                3000,
                                                4000,
                                                5000,
                                                6000,
                                                7000,
                                                8000,
                                                9000}});
}

TEST_F(JsonReaderTest, JsonLinesMultipleFilesByteRange_LoadAllFiles)
{
  std::string const file1 = temp_env->get_temp_dir() + "JsonLinesMultipleFilesByteRangeTest1.json";
  std::ofstream outfile1(file1, std::ofstream::out);
  outfile1 << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]";
  outfile1.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{{file1, file1}}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 18);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{1000,
                                                2000,
                                                3000,
                                                4000,
                                                5000,
                                                6000,
                                                7000,
                                                8000,
                                                9000,
                                                1000,
                                                2000,
                                                3000,
                                                4000,
                                                5000,
                                                6000,
                                                7000,
                                                8000,
                                                9000}});
}

TEST_P(JsonReaderRecordTest, JsonLinesObjects)
{
  std::string const fname = temp_env->get_temp_dir() + "JsonLinesObjectsTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << " {\"co\\\"l1\" : 1, \"col2\" : 2.0} \n";
  outfile.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{fname}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 1);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "co\"l1");
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.metadata.schema_info[1].name, "col2");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{2.0}});
}

TEST_P(JsonReaderRecordTest, JsonLinesObjectsStrings)
{
  auto test_json_objects = [](std::string const& data) {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data.data()), data.size()}})
        .lines(true);

    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 3);
    EXPECT_EQ(result.tbl->num_rows(), 2);

    EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
    EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
    EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

    EXPECT_EQ(result.metadata.schema_info[0].name, "col1");
    EXPECT_EQ(result.metadata.schema_info[1].name, "col2");
    EXPECT_EQ(result.metadata.schema_info[2].name, "col3");

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                   cudf::test::strings_column_wrapper({"aaa", "bbb"}));
  };
  // simple case
  test_json_objects(
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col1\":200, \"col2\":2.2, \"col3\":\"bbb\"}\n");
  // out of order fields
  test_json_objects(
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col3\":\"bbb\", \"col1\":200, \"col2\":2.2}\n");
}

TEST_P(JsonReaderRecordTest, JsonLinesObjectsMissingData)
{
  //  Note: columns will be ordered based on which fields appear first
  std::string const data =
    "{              \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col1\":200,               \"col3\":\"bbb\"}\n";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::INT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col2");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col3");
  EXPECT_EQ(result.metadata.schema_info[2].name, "col1");

  auto col1_validity = cudf::test::iterators::null_at(0);
  auto col2_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 0; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2), int64_wrapper{{0, 200}, col1_validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 float64_wrapper{{1.1, 0.}, col2_validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 cudf::test::strings_column_wrapper({"aaa", "bbb"}));
}

TEST_P(JsonReaderRecordTest, JsonLinesObjectsOutOfOrder)
{
  std::string const data =
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col3\":\"bbb\", \"col1\":200, \"col2\":2.2}\n";

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col1");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col2");
  EXPECT_EQ(result.metadata.schema_info[2].name, "col3");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aaa", "bbb"}));
}

TEST_F(JsonReaderTest, EmptyFile)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.json";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath}).lines(true);
  auto result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(JsonReaderTest, NoDataFile)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.json";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "{}\n";
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath}).lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

// empty input in values orient
TEST_F(JsonReaderTest, NoDataFileValues)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFileValues.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "[]\n";
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath}).lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  auto const view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_P(JsonReaderParamTest, InvalidFloatingPoint)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[1.2e1+]\n[3.4e2-]\n[5.6e3e]\n[7.8e3A]\n[9.0Be1]\n[1C.2]";
  std::string record_orient = to_records_orient({{{"0", "1.2e1+"}},
                                                 {{"0", "3.4e2-"}},
                                                 {{"0", "5.6e3e"}},
                                                 {{"0", "7.8e3A"}},
                                                 {{"0", "9.0Be1"}},
                                                 {{"0", "1C.2"}}},
                                                "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  auto const filepath = temp_env->get_temp_dir() + "InvalidFloatingPoint.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .dtypes(std::vector{dtype<float>()})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT32);

  // ignore all data because it is all nulls.
  ASSERT_EQ(6u, result.tbl->view().column(0).null_count());
}

TEST_P(JsonReaderParamTest, StringInference)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[\"-1\"]";
  std::string record_orient = to_records_orient({{{"0", R"("-1")"}}}, "\n");
  std::string data          = is_row_orient_test(test_opt) ? row_orient : record_orient;

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
}

TEST_P(JsonReaderParamTest, ParseInRangeIntegers)
{
  auto const test_opt   = GetParam();
  bool const row_orient = is_row_orient_test(test_opt);

  constexpr auto num_rows                      = 4;
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

  auto filepath = temp_env->get_temp_dir() + "ParseInRangeIntegers.json";
  {
    std::ostringstream line;
    if (row_orient) {
      for (int i = 0; i < num_rows; ++i) {
        line << "[" << small_int[i] << "," << less_equal_int64_max[i] << ","
             << greater_equal_int64_min[i] << "," << greater_int64_max[i] << ","
             << less_equal_uint64_max[i] << "," << small_int_append_zeros[i] << ","
             << less_equal_int64_max_append_zeros[i] << ","
             << greater_equal_int64_min_append_zeros[i] << "," << greater_int64_max_append_zeros[i]
             << "," << less_equal_uint64_max_append_zeros[i] << "]\n";
      }
    } else {
      std::vector<std::map<std::string, std::string>> records;
      for (int i = 0; i < num_rows; ++i) {
        records.push_back({
          {"0", std::to_string(small_int[i])},                //
          {"1", std::to_string(less_equal_int64_max[i])},     //
          {"2", std::to_string(greater_equal_int64_min[i])},  //
          {"3", std::to_string(greater_int64_max[i])},        //
          {"4", std::to_string(less_equal_uint64_max[i])},    //
          {"5", small_int_append_zeros[i]},                   //
          {"6", less_equal_int64_max_append_zeros[i]},        //
          {"7", greater_equal_int64_min_append_zeros[i]},     //
          {"8", greater_int64_max_append_zeros[i]},           //
          {"9", less_equal_uint64_max_append_zeros[i]},       //
        });
      }
      line << to_records_orient(records, "\n");
    }

    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

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

TEST_P(JsonReaderParamTest, ParseOutOfRangeIntegers)
{
  auto const test_opt   = GetParam();
  bool const row_orient = is_row_orient_test(test_opt);

  constexpr auto num_rows                        = 4;
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

  auto filepath = temp_env->get_temp_dir() + "ParseOutOfRangeIntegers.json";
  {
    std::ostringstream line;
    if (row_orient) {
      for (int i = 0; i < num_rows; ++i) {
        line << "[" << out_of_range_positive[i] << "," << out_of_range_negative[i] << ","
             << greater_uint64_max[i] << "," << less_int64_min[i] << "," << mixed_range[i] << ","
             << out_of_range_positive_append_zeros[i] << ","
             << out_of_range_negative_append_zeros[i] << "," << greater_uint64_max_append_zeros[i]
             << "," << less_int64_min_append_zeros[i] << "," << mixed_range_append_zeros[i]
             << "]\n";
      }
    } else {
      std::vector<std::map<std::string, std::string>> records;
      for (int i = 0; i < num_rows; ++i) {
        records.push_back({
          {"0", out_of_range_positive[i]},               //
          {"1", out_of_range_negative[i]},               //
          {"2", greater_uint64_max[i]},                  //
          {"3", less_int64_min[i]},                      //
          {"4", mixed_range[i]},                         //
          {"5", out_of_range_positive_append_zeros[i]},  //
          {"6", out_of_range_negative_append_zeros[i]},  //
          {"7", greater_uint64_max_append_zeros[i]},     //
          {"8", less_int64_min_append_zeros[i]},         //
          {"9", mixed_range_append_zeros[i]},            //
        });
      }
      line << to_records_orient(records, "\n");
    }

    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

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

TEST_P(JsonReaderParamTest, JsonLinesMultipleFileInputs)
{
  auto const test_opt = GetParam();
  std::vector<std::string> row_orient{"[11, 1.1]\n[22, 2.2]\n", "[33, 3.3]\n[44, 4.4]"};
  std::vector<std::string> record_orient{
    to_records_orient({{{"0", "11"}, {"1", "1.1"}}, {{"0", "22"}, {"1", "2.2"}}}, "\n") + "\n",
    to_records_orient({{{"0", "33"}, {"1", "3.3"}}, {{"0", "44"}, {"1", "4.4"}}}, "\n") + "\n"};
  auto const& data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  std::string const file1 = temp_env->get_temp_dir() + "JsonLinesFileTest1.json";
  std::ofstream outfile(file1, std::ofstream::out);
  outfile << data[0];
  outfile.close();

  std::string const file2 = temp_env->get_temp_dir() + "JsonLinesFileTest2.json";
  std::ofstream outfile2(file2, std::ofstream::out);
  outfile2 << data[1];
  outfile2.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{{file1, file2}}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 4);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{11, 22, 33, 44}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3, 4.4}});
}

TEST_P(JsonReaderParamTest, JsonLinesMultipleFileInputsNoNL)
{
  auto const test_opt = GetParam();
  // Strings for the two separate input files in row-orient that do not end with a newline
  std::vector<std::string> row_orient{"[11, 1.1]\n[22, 2.2]", "[33, 3.3]\n[44, 4.4]"};
  // Strings for the two separate input files in record-orient that do not end with a newline
  std::vector<std::string> record_orient{
    to_records_orient({{{"0", "11"}, {"1", "1.1"}}, {{"0", "22"}, {"1", "2.2"}}}, "\n"),
    to_records_orient({{{"0", "33"}, {"1", "3.3"}}, {{"0", "44"}, {"1", "4.4"}}}, "\n")};
  auto const& data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  std::string const file1 = temp_env->get_temp_dir() + "JsonLinesFileTest1.json";
  std::ofstream outfile(file1, std::ofstream::out);
  outfile << data[0];
  outfile.close();

  std::string const file2 = temp_env->get_temp_dir() + "JsonLinesFileTest2.json";
  std::ofstream outfile2(file2, std::ofstream::out);
  outfile2 << data[1];
  outfile2.close();

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{{file1, file2}}).lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 4);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{11, 22, 33, 44}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3, 4.4}});
}

TEST_F(JsonReaderTest, JsonBasic)
{
  std::string const fname = temp_env->get_temp_dir() + "JsonBasic.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << R"([{"a":"11", "b":"1.1"},{"a":"22", "b":"2.2"}])";
  outfile.close();

  cudf::io::json_reader_options options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{fname});
  auto result = cudf::io::read_json(options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "b");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 cudf::test::strings_column_wrapper({"11", "22"}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 cudf::test::strings_column_wrapper({"1.1", "2.2"}));
}

TEST_F(JsonReaderTest, JsonLines)
{
  std::string const json_string =
    R"({"a":"a0"}
    {"a":"a1"}
    {"a":"a2", "b":"b2"}
    {"a":"a3", "c":"c3"}
    {"a":"a4"})";

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
      .lines(true);

  // Read test data via nested JSON reader
  auto const table = cudf::io::read_json(json_lines_options);

  // TODO: Rewrite this test to check against a fixed value
  CUDF_TEST_EXPECT_TABLES_EQUAL(table.tbl->view(), table.tbl->view());
}

TEST_F(JsonReaderTest, JsonLongString)
{
  // Unicode
  // 0000-FFFF     Basic Multilingual Plane
  // 10000-10FFFF  Supplementary Plane
  cudf::test::strings_column_wrapper col1{
    {
      "\"\\/\b\f\n\r\t",
      "\"",
      "\\",
      "/",
      "\b",
      "\f\n",
      "\r\t",
      "$€",
      "ராபிட்ஸ்",
      "C𝞵𝓓𝒻",
      "",  // null
      "",  // null
      "கார்த்தி",
      "CႮ≪ㇳ䍏凹沦王辿龸ꁗ믜스폶ﴠ",  //  0000-FFFF
      "𐀀𑿪𒐦𓃰𔙆 𖦆𗿿𘳕𚿾[↳] 𜽆𝓚𞤁🄰",                            // 10000-1FFFF
      "𠘨𡥌𢗉𣇊𤊩𥅽𦉱𧴱𨁲𩁹𪐢𫇭𬬭𭺷𮊦屮",                // 20000-2FFFF
      "𰾑𱔈𲍉",                                          // 30000-3FFFF
      R"("$€ \u0024\u20ac \\u0024\\u20ac  \\\u0024\\\u20ac \\\\u0024\\\\u20ac)",
      R"(        \\\\\\\\\\\\\\\\)",
      R"(\\\\\\\\\\\\\\\\)",
      R"(\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\)",
      R"( \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\)",
      R"(                      \\abcd)",
      R"(                 \\\\\\\\\\\\\\\\                 \\\\\\\\\\\\\\\\)",
      R"(                \\\\\\\\\\\\\\\\                 \\\\\\\\\\\\\\\\)",
    },
    cudf::test::iterators::nulls_at({10, 11})};

  cudf::test::fixed_width_column_wrapper<int16_t> repeat_times{
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 19, 37, 81, 161, 323, 631, 1279, 10, 1, 2, 1, 100, 1000, 1, 3},
    cudf::test::iterators::no_nulls()};
  auto d_col2 = cudf::strings::repeat_strings(cudf::strings_column_view{col1}, repeat_times);
  auto col2   = d_col2->view();
  cudf::table_view const tbl_view{{col1, col2, repeat_times}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination     = cudf::io::sink_info(&out_buffer);
  auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                           .include_nulls(true)
                           .metadata(mt)
                           .lines(true)
                           .na_rep("null");

  cudf::io::write_json(options_builder.build(), cudf::test::get_default_stream());

  cudf::column_view int16_with_mask(repeat_times);
  cudf::column_view int16(
    int16_with_mask.type(), int16_with_mask.size(), int16_with_mask.head(), nullptr, 0);
  cudf::table_view const expected = cudf::table_view{{col1, col2, int16}};
  std::map<std::string, data_type> types;
  types["col1"]  = data_type{type_id::STRING};
  types["col2"]  = data_type{type_id::STRING};
  types["int16"] = data_type{type_id::INT16};

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
      .lines(true)
      .dtypes(types);

  // Read test data via nested JSON reader
  auto const table = cudf::io::read_json(json_lines_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, table.tbl->view());
}

TEST_F(JsonReaderTest, ErrorStrings)
{
  // cases of invalid escape characters, invalid unicode encodings.
  // Error strings will decode to nulls
  auto const buffer = std::string{R"(
    {"col0": "\"\a"}
    {"col0": "\u"}
    {"col0": "\u0"}
    {"col0": "\u0b"}
    {"col0": "\u00b"}
    {"col0": "\u00bz"}
    {"col0": "\t34567890123456\t9012345678901\ug0bc"}
    {"col0": "\t34567890123456\t90123456789012\u0hbc"}
    {"col0": "\t34567890123456\t90123456789012\u00ic"}
    {"col0": "\u0b95\u0bbe\u0bb0\u0bcd\u0ba4\u0bcd\u0ba4\u0bbfகார்த்தி"}
)"};
  // Last one is not an error case, but shows that unicode in json is copied string column output.

  cudf::io::json_reader_options const in_opts =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .dtypes(std::vector{data_type{cudf::type_id::STRING}})
      .lines(true);

  auto const result      = cudf::io::read_json(in_opts);
  auto const result_view = result.tbl->view().column(0);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col0");
  EXPECT_EQ(result_view.null_count(), 9);
  cudf::test::strings_column_wrapper expected{
    {"",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "கார்த்தி\xe0\xae\x95\xe0\xae\xbe\xe0\xae\xb0\xe0\xaf\x8d\xe0\xae\xa4\xe0\xaf\x8d\xe0\xae\xa4"
     "\xe0\xae\xbf"},
    // unicode hex 0xe0 0xae 0x95 0xe0 0xae 0xbe 0xe0 0xae 0xb0 0xe0 0xaf 0x8d
    //             0xe0 0xae 0xa4 0xe0 0xaf 0x8d 0xe0 0xae 0xa4 0xe0 0xae 0xbf
    cudf::test::iterators::nulls_at({0, 1, 2, 3, 4, 5, 6, 7, 8})};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_view, expected);
}

TEST_F(JsonReaderTest, TokenAllocation)
{
  std::array<std::string const, 3> const json_inputs{
    R"({"":1})",
    "{}\n{}\n{}",
    R"({"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":{"":1}}}}}}}}}}}})",
  };

  for (auto const& json_string : json_inputs) {
    // Initialize parsing options (reading json lines)
    cudf::io::json_reader_options json_lines_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
        .lines(true);

    EXPECT_NO_THROW(cudf::io::read_json(json_lines_options));
  }
}

TEST_F(JsonReaderTest, LinesNoOmissions)
{
  std::array<std::string const, 4> const json_inputs
    // single column
    {R"({"a":"a0"}
    {"a":"a1"}
    {"a":"a2"}
    {"a":"a3"}
    {"a":"a4"})",
     // single column, single row
     R"({"a":"a0"})",
     // single row
     R"({"a":"a0", "b":"b0"})",
     // two column, two rows
     R"({"a":"a0", "b":"b0"}
    {"a":"a1", "b":"b1"})"};

  for (auto const& json_string : json_inputs) {
    // Initialize parsing options (reading json lines)
    cudf::io::json_reader_options json_lines_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
        .lines(true);

    // Read test data via nested JSON reader
    auto const table = cudf::io::read_json(json_lines_options);

    // TODO: Rewrite this test to check against a fixed value
    CUDF_TEST_EXPECT_TABLES_EQUAL(table.tbl->view(), table.tbl->view());
  }
}

TEST_F(JsonReaderTest, TestColumnOrder)
{
  std::string const json_string =
    // Expected order:
    // root: b, c, a, d
    // a: 2, 0, 1
    {R"({"b":"b0"}
    {"c":"c1","a":{"2":null}}
    {"d":"d2","a":{"0":"a2.0", "2":"a2.2"}}
    {"b":"b3","a":{"1":null, "2":"a3.2"}})"};

  std::vector<std::string> const root_col_names{"b", "c", "a", "d"};
  std::vector<std::string> const a_child_col_names{"2", "0", "1"};

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
      .lines(true);

  // Read in data using nested JSON reader
  cudf::io::table_with_metadata new_reader_table = cudf::io::read_json(json_lines_options);

  // Verify root column order before accessing schema entries.
  ASSERT_EQ(new_reader_table.metadata.schema_info.size(), root_col_names.size());

  for (std::size_t i = 0; i < root_col_names.size(); i++) {
    auto const& root_col_name = root_col_names[i];
    EXPECT_EQ(new_reader_table.metadata.schema_info[i].name, root_col_name);
  }

  // Verify nested child column order before accessing schema entries.
  ASSERT_EQ(new_reader_table.metadata.schema_info[2].children.size(), a_child_col_names.size());
  for (std::size_t i = 0; i < a_child_col_names.size(); i++) {
    auto const& a_child_col_name = a_child_col_names[i];
    EXPECT_EQ(new_reader_table.metadata.schema_info[2].children[i].name, a_child_col_name);
  }

  // Verify data of root columns
  ASSERT_EQ(root_col_names.size(), new_reader_table.tbl->num_columns());
  column_wrapper<cudf::string_view> root_col_data_b{{"b0", "", "", "b3"},
                                                    {true, false, false, true}};
  column_wrapper<cudf::string_view> root_col_data_c{{"", "c1", "", ""},
                                                    {false, true, false, false}};
  column_wrapper<cudf::string_view> root_col_data_d{{"", "", "d2", ""},
                                                    {false, false, true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_b, new_reader_table.tbl->get_column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_c, new_reader_table.tbl->get_column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_d, new_reader_table.tbl->get_column(3));

  // Verify data of child columns of column 'a'
  auto const col_a = new_reader_table.tbl->get_column(2);
  ASSERT_EQ(a_child_col_names.size(), col_a.num_children());
  column_wrapper<cudf::string_view> col_a2{{"", "", "a2.2", "a3.2"}, {false, false, true, true}};
  column_wrapper<cudf::string_view> col_a0{{"", "", "a2.0", ""}, {false, false, true, false}};
  // col a.1 is inferred as all-null
  int8_wrapper col_a1{{0, 0, 0, 0}, {false, false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a2, col_a.child(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a0, col_a.child(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a1, col_a.child(2));
}

TEST_P(JsonReaderParamTest, JsonDtypeSchema)
{
  auto const test_opt       = GetParam();
  std::string row_orient    = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";
  std::string record_orient = to_records_orient({{{"0", "1"}, {"1", "1.1"}, {"2", R"("aa ")"}},
                                                 {{"0", "2"}, {"1", "2.2"}, {"2", R"("  bbb")"}}},
                                                "\n");

  std::string data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  std::map<std::string, cudf::io::schema_element> dtype_schema{
    {"2", {dtype<cudf::string_view>()}}, {"0", {dtype<int32_t>()}}, {"1", {dtype<double>()}}};
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(dtype_schema)
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  EXPECT_EQ(result.metadata.schema_info[2].name, "2");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_F(JsonReaderTest, JsonNestedDtypeSchema)
{
  std::string json_string = R"( [{"a":[123, {"0": 123}], "b":1.0}, {"b":1.1}, {"b":2.1}])";

  std::map<std::string, cudf::io::schema_element> dtype_schema{
    {"a",
     {
       data_type{cudf::type_id::LIST},
       {{"element", {data_type{cudf::type_id::STRUCT}, {{"0", {dtype<float>()}}}}}},
     }},
    {"b", {dtype<int32_t>()}},
  };

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
      .dtypes(dtype_schema)
      .lines(false);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  // Make sure we have columns "a" and "b"
  ASSERT_EQ(result.tbl->num_columns(), 2);
  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "b");
  // Make sure column "a" is a list column (offsets and elements)
  ASSERT_EQ(result.tbl->get_column(0).num_children(), 2);
  ASSERT_EQ(result.metadata.schema_info[0].children.size(), 2);
  // Make sure column "b" is a leaf column
  ASSERT_EQ(result.tbl->get_column(1).num_children(), 0);
  ASSERT_EQ(result.metadata.schema_info[1].children.size(), 0);
  // Offsets child with no other child columns
  ASSERT_EQ(result.tbl->get_column(0).child(0).num_children(), 0);
  ASSERT_EQ(result.metadata.schema_info[0].children[0].children.size(), 0);
  EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "offsets");
  // Elements is the struct column with a single child column "0"
  ASSERT_EQ(result.tbl->get_column(0).child(1).num_children(), 1);
  ASSERT_EQ(result.metadata.schema_info[0].children[1].children.size(), 1);
  EXPECT_EQ(result.metadata.schema_info[0].children[1].name, "element");

  // Verify column "a" being a list column
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::LIST);
  // Verify column "a->element->0" is a float column
  EXPECT_EQ(result.tbl->get_column(0).child(1).child(0).type().id(), cudf::type_id::FLOAT32);
  // Verify column "b" is an int column
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::INT32);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0).child(0), int_wrapper{{0, 2, 2, 2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0).child(1).child(0),
                                 float_wrapper{{0.0, 123.0}, {false, true}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), int_wrapper{{1, 1, 2}});
  // List column expected
  auto leaf_child     = float_wrapper{{0.0, 123.0}, {false, true}};
  auto const validity = {1, 0, 0};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expected = cudf::make_lists_column(
    3,
    int_wrapper{{0, 2, 2, 2}}.release(),
    cudf::test::structs_column_wrapper{{leaf_child}, {false, true}}.release(),
    null_count,
    std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), *expected);
}

TEST_P(JsonReaderParamTest, JsonDtypeParsing)
{
  auto const test_opt = GetParam();
  // All corner cases of dtype parsing
  //  0, "0", " 0", 1, "1", " 1", "a", "z", null, true, false,  "null", "true", "false", nan, "nan"
  // Test for dtypes: bool, int, float, str, duration, timestamp
  std::string row_orient =
    "[0]\n[\"0\"]\n[\" 0\"]\n[1]\n[\"1\"]\n[\" 1\"]\n[\"a\"]\n[\"z\"]\n"
    "[null]\n[true]\n[false]\n[\"null\"]\n[\"true\"]\n[\"false\"]\n[nan]\n[\"nan\"]\n";
  std::string record_orient = to_records_orient({{{"0", "0"}},
                                                 {{"0", "\"0\""}},
                                                 {{"0", "\" 0\""}},
                                                 {{"0", "1"}},
                                                 {{"0", "\"1\""}},
                                                 {{"0", "\" 1\""}},
                                                 {{"0", "\"a\""}},
                                                 {{"0", "\"z\""}},
                                                 {{"0", "null"}},
                                                 {{"0", "true"}},
                                                 {{"0", "false"}},
                                                 {{"0", "\"null\""}},
                                                 {{"0", "\"true\""}},
                                                 {{"0", "\"false\""}},
                                                 {{"0", "nan"}},
                                                 {{"0", "\"nan\""}}},
                                                "\n");

  std::string data = is_row_orient_test(test_opt) ? row_orient : record_orient;

  auto make_validity = [](std::vector<int> const& validity) {
    return cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) -> bool { return static_cast<bool>(validity[i]); });
  };

  constexpr int int_ignore{};
  constexpr double double_ignore{};
  constexpr bool bool_ignore{};

  std::vector<int> const validity = {1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0};

  auto int_col   = int_wrapper{{0,
                                0,
                                int_ignore,
                                1,
                                1,
                                int_ignore,
                                int_ignore,
                                int_ignore,
                                int_ignore,
                                1,
                                0,
                                int_ignore,
                                1,
                                0,
                                int_ignore,
                                int_ignore},
                             make_validity(validity)};
  auto float_col = float_wrapper{{0.0,
                                  0.0,
                                  double_ignore,
                                  1.0,
                                  1.0,
                                  double_ignore,
                                  double_ignore,
                                  double_ignore,
                                  double_ignore,
                                  1.0,
                                  0.0,
                                  double_ignore,
                                  1.0,
                                  0.0,
                                  double_ignore,
                                  double_ignore},
                                 make_validity(validity)};
  auto str_col   = cudf::test::strings_column_wrapper{
    // clang-format off
    {"0", "0", " 0", "1", "1", " 1", "a", "z", "", "true", "false", "null", "true", "false", "nan", "nan"},
     cudf::test::iterators::nulls_at(std::vector<int>{8})};
  // clang-format on
  auto bool_col = bool_wrapper{{false,
                                false,
                                bool_ignore,
                                true,
                                true,
                                bool_ignore,
                                bool_ignore,
                                bool_ignore,
                                bool_ignore,
                                true,
                                false,
                                bool_ignore,
                                true,
                                false,
                                bool_ignore,
                                bool_ignore},
                               make_validity(validity)};

  // Types to test
  std::vector<data_type> const dtypes = {
    dtype<int32_t>(), dtype<float>(), dtype<cudf::string_view>(), dtype<bool>()};
  std::vector<cudf::column_view> const cols{cudf::column_view(int_col),
                                            cudf::column_view(float_col),
                                            cudf::column_view(str_col),
                                            cudf::column_view(bool_col)};
  for (size_t col_type = 0; col_type < cols.size(); col_type++) {
    std::map<std::string, cudf::io::schema_element> dtype_schema{{"0", {dtypes[col_type]}}};
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data.data()), data.size()}})
        .dtypes(dtype_schema)
        .lines(true);

    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 1);
    EXPECT_EQ(result.tbl->num_rows(), 16);
    EXPECT_EQ(result.metadata.schema_info[0].name, "0");

    EXPECT_EQ(result.tbl->get_column(0).type().id(), dtypes[col_type].id());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), cols[col_type]);
  }
}

TYPED_TEST(JsonValidFixedPointReaderTest, SingleColumnNegativeScale)
{
  this->run_tests({"1.23", "876e-2", "5.43e1", "-0.12", "0.25", "-0.23", "-0.27", "0.00", "0.00"},
                  numeric::scale_type{-2});
}

TYPED_TEST(JsonValidFixedPointReaderTest, SingleColumnNoScale)
{
  this->run_tests({"123", "-87600e-2", "54.3e1", "-12", "25", "-23", "-27", "0", "0"},
                  numeric::scale_type{0});
}

TYPED_TEST(JsonValidFixedPointReaderTest, SingleColumnPositiveScale)
{
  this->run_tests(
    {"123000", "-87600000e-2", "54300e1", "-12000", "25000", "-23000", "-27000", "0000", "0000"},
    numeric::scale_type{3});
}

TYPED_TEST(JsonFixedPointReaderTest, EmptyValues)
{
  auto const buffer = std::string{R"({"col0":""})"};

  cudf::io::json_reader_options const in_opts =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(buffer.c_str()), buffer.size()}})
      .dtypes(std::vector{data_type{type_to_id<TypeParam>(), 0}})
      .lines(true);

  auto const result      = cudf::io::read_json(in_opts);
  auto const result_view = result.tbl->view();

  ASSERT_EQ(result_view.num_columns(), 1);
  EXPECT_EQ(result_view.num_rows(), 1);
  EXPECT_EQ(result.metadata.schema_info[0].name, "col0");
  EXPECT_EQ(result_view.column(0).null_count(), 0);
}

TEST_F(JsonReaderTest, UnsupportedMultipleFileInputs)
{
  std::string const data = "{\"col\":0}";
  auto const span =
    cudf::host_span<std::byte const>(reinterpret_cast<std::byte const*>(data.data()), data.size());
  std::vector<cudf::host_span<std::byte const>> spans{span, span};
  auto const src = cudf::io::source_info(cudf::host_span<cudf::host_span<std::byte const>>(spans));

  cudf::io::json_reader_options const not_lines_opts = cudf::io::json_reader_options::builder(src);
  EXPECT_THROW(cudf::io::read_json(not_lines_opts), cudf::logic_error);

  cudf::io::json_reader_options const comp_exp_opts =
    cudf::io::json_reader_options::builder(src).compression(cudf::io::compression_type::GZIP);
  EXPECT_THROW(cudf::io::read_json(comp_exp_opts), cudf::logic_error);

  cudf::io::json_reader_options const comp_opts =
    cudf::io::json_reader_options::builder(src).compression(cudf::io::compression_type::GZIP);
  EXPECT_THROW(cudf::io::read_json(comp_opts), cudf::logic_error);
}

TEST_F(JsonReaderTest, TrailingCommas)
{
  std::vector<std::string> const json_lines_valid{
    R"({"a":"a0",}
    {"a":"a2", "b":"b2",}
    {"a":"a4",})",
    R"({"a":"a0"}
    {"a":"a2", "b": [1, 2,]})",
    R"({"a":"a0",}
    {"a":"a2", "b": [1, 2,],})",
  };
  for (size_t i = 0; i < json_lines_valid.size(); i++) {
    auto const& json_string = json_lines_valid[i];
    // Initialize parsing options (reading json lines)
    cudf::io::json_reader_options json_parser_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}})
        .lines(true);
    EXPECT_NO_THROW(cudf::io::read_json(json_parser_options)) << "Failed on test case " << i;
  }

  std::vector<std::string> const json_valid{
    R"([{"a":"a0",},  {"a":"a2", "b":"b2",}, {"a":"a4"},])",
    R"([{"a":"a0"},  {"a":"a2", "b": [1, 2,]}])",
    R"([{"a":"a0",}, {"a":"a2", "b": [1, 2,],}])",
    R"([{"a": 1,}, {"a": null, "b": [null,],}])",
  };
  for (size_t i = 0; i < json_valid.size(); i++) {
    auto const& json_string = json_valid[i];
    cudf::io::json_reader_options json_parser_options =
      cudf::io::json_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}});
    EXPECT_NO_THROW(cudf::io::read_json(json_parser_options)) << "Failed on test case " << i;
  }

  std::vector<std::string> const json_invalid{
    R"([{"a":"a0",,}])",
    R"([{"a":"a0"},,])",
    R"([,{"a":"a0"}])",
    R"([{,"a":"a0"}])",
    R"([{,}])",
    R"([,])",
    R"([,,])",
    R"([{,,}])",
  };
  for (size_t i = 0; i < json_invalid.size(); i++) {
    auto const& json_string = json_invalid[i];
    cudf::io::json_reader_options json_parser_options =
      cudf::io::json_reader_options::builder(cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.c_str()), json_string.size()}});
    EXPECT_THROW(cudf::io::read_json(json_parser_options), cudf::logic_error)
      << "Failed on test case " << i;
  }
}

TEST_F(JsonReaderTest, JSONLinesRecovering)
{
  std::string data =
    // 0 -> a: -2 (valid)
    R"({"a":-2})"
    "\n"
    // 1 -> (invalid)
    R"({"a":])"
    "\n"
    // 2 -> (invalid)
    R"({"b":{"a":[321})"
    "\n"
    // 3 -> c: 1.2 (valid)
    R"({"c":1.2})"
    "\n"
    "\n"
    // 4 -> a: 4 (valid)
    R"({"a":4})"
    "\n"
    // 5 -> (invalid)
    R"({"a":5)"
    "\n"
    // 6 -> (invalid)
    R"({"a":6 )"
    "\n"
    // 7 -> (invalid)
    R"({"b":[7 )"
    "\n"
    // 8 -> a: 8 (valid)
    R"({"a":8})"
    "\n"
    // 9 -> (invalid)
    R"({"d":{"unterminated_field_name)"
    "\n"
    // 10 -> (invalid)
    R"({"d":{)"
    "\n"
    // 11 -> (invalid)
    R"({"d":{"123",)"
    "\n"
    // 12 -> a: 12 (valid)
    R"({"a":12})";

  auto filepath = temp_env->get_temp_dir() + "RecoveringLines.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 13);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  std::vector<bool> a_validity{
    true, false, false, false, true, false, false, false, true, false, false, false, true};
  std::vector<bool> c_validity{
    false, false, false, true, false, false, false, false, false, false, false, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    int64_wrapper{{-2, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 12}, a_validity.cbegin()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(1),
    float64_wrapper{{0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                    c_validity.cbegin()});
}

TEST_F(JsonReaderTest, JSONLinesRecoveringMalformedOpenBraces)
{
  // Test recovery mode with various malformed JSON patterns
  for (int num_lines : {2, 4, 8, 16}) {
    // Test "{\n" pattern
    {
      std::string data;
      for (int i = 0; i < num_lines - 1; ++i) {
        data += "{\n";
      }
      data += "{";

      cudf::io::json_reader_options in_options =
        cudf::io::json_reader_options::builder(
          cudf::io::source_info{cudf::host_span<std::byte const>{
            reinterpret_cast<std::byte const*>(data.data()), data.size()}})
          .lines(true)
          .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      EXPECT_EQ(result.tbl->num_rows(), 0) << "Failed {\\n pattern with " << num_lines << " lines";
    }

    // Test {"\n pattern
    {
      std::string data;
      for (int i = 0; i < num_lines - 1; ++i) {
        data += "{\"\n";
      }
      data += "{\"";

      cudf::io::json_reader_options in_options =
        cudf::io::json_reader_options::builder(
          cudf::io::source_info{cudf::host_span<std::byte const>{
            reinterpret_cast<std::byte const*>(data.data()), data.size()}})
          .lines(true)
          .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      EXPECT_EQ(result.tbl->num_rows(), 0)
        << "Failed {\"\\n pattern with " << num_lines << " lines";
    }
  }
}

TEST_F(JsonReaderTest, JSONLinesRecoveringIgnoreExcessChars)
{
  /**
   * @brief Spark has the specific need to ignore extra characters that come after the first record
   * on a JSON line
   */
  std::string data =
    // 0 -> a: -2 (valid)
    R"({"a":-2}{})"
    "\n"
    // 1 -> (invalid)
    R"({"b":{}should_be_invalid})"
    "\n"
    // 2 -> b (valid)
    R"({"b":{"a":3} })"
    "\n"
    // 3 -> c: (valid)
    R"({"c":1.2 } )"
    "\n"
    "\n"
    // 4 -> (valid)
    R"({"a":4} 123)"
    "\n"
    // 5 -> (valid)
    R"({"a":5}//Comment after record)"
    "\n"
    // 6 -> (valid)
    R"({"a":6} //Comment after whitespace)"
    "\n"
    // 7 -> (invalid)
    R"({"a":5 //Invalid Comment within record})";

  auto filepath = temp_env->get_temp_dir() + "RecoveringLinesExcessChars.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 8);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::FLOAT64);

  std::vector<bool> a_validity{true, false, false, false, true, true, true, false};
  std::vector<bool> b_validity{false, false, true, false, false, false, false, false};
  std::vector<bool> c_validity{false, false, false, true, false, false, false, false};

  // Child column b->a
  auto b_a_col = int64_wrapper({0, 0, 3, 0, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{-2, 0, 0, 0, 4, 5, 6, 0}, a_validity.cbegin()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(1), cudf::test::structs_column_wrapper({b_a_col}, b_validity.cbegin()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(2),
    float64_wrapper{{0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0}, c_validity.cbegin()});
}

// Sanity test that checks whether there's a race on the FST destructor
TEST_F(JsonReaderTest, JSONLinesRecoveringSync)
{
  // Set up host pinned memory pool to avoid implicit synchronizations to test for any potential
  // races due to missing host-device synchronizations
  cudf::test::pinned_pool mr{size_t{128} * 1024 * 1024};

  // Set new resource
  auto last_mr = cudf::set_pinned_memory_resource(mr);

  /**
   * @brief Spark has the specific need to ignore extra characters that come after the first record
   * on a JSON line
   */
  std::string data =
    // 0 -> a: -2 (valid)
    R"({"a":-2}{})"
    "\n"
    // 1 -> (invalid)
    R"({"b":{}should_be_invalid})"
    "\n"
    // 2 -> b (valid)
    R"({"b":{"a":3} })"
    "\n"
    // 3 -> c: (valid)
    R"({"c":1.2 } )"
    "\n"
    "\n"
    // 4 -> (valid)
    R"({"a":4} 123)"
    "\n"
    // 5 -> (valid)
    R"({"a":5}//Comment after record)"
    "\n"
    // 6 -> (valid)
    R"({"a":6} //Comment after whitespace)"
    "\n"
    // 7 -> (invalid)
    R"({"a":5 //Invalid Comment within record})";

  // Create input of a certain size to potentially reveal a missing host/device sync
  std::size_t const target_size = 40000000;
  auto const repetitions_log2 =
    static_cast<std::size_t>(std::ceil(std::log2(target_size / data.size())));
  auto const repetitions = 1ULL << repetitions_log2;

  for (std::size_t i = 0; i < repetitions_log2; ++i) {
    data = data + "\n" + data;
  }

  auto filepath = temp_env->get_temp_dir() + "RecoveringLinesExcessChars.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << data;
  }

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 8 * repetitions);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::FLOAT64);

  std::vector<bool> a_validity{true, false, false, false, true, true, true, false};
  std::vector<bool> b_validity{false, false, true, false, false, false, false, false};
  std::vector<bool> c_validity{false, false, false, true, false, false, false, false};

  std::vector<std::int32_t> a_data{-2, 0, 0, 0, 4, 5, 6, 0};
  std::vector<std::int32_t> b_a_data{0, 0, 3, 0, 0, 0, 0, 0};
  std::vector<double> c_data{0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0};

  for (std::size_t i = 0; i < repetitions_log2; ++i) {
    a_validity.insert(a_validity.end(), a_validity.cbegin(), a_validity.cend());
    b_validity.insert(b_validity.end(), b_validity.cbegin(), b_validity.cend());
    c_validity.insert(c_validity.end(), c_validity.cbegin(), c_validity.cend());
    a_data.insert(a_data.end(), a_data.cbegin(), a_data.cend());
    b_a_data.insert(b_a_data.end(), b_a_data.cbegin(), b_a_data.cend());
    c_data.insert(c_data.end(), c_data.cbegin(), c_data.cend());
  }

  // Child column b->a
  auto b_a_col = int64_wrapper(b_a_data.cbegin(), b_a_data.cend());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0), int64_wrapper{a_data.cbegin(), a_data.cend(), a_validity.cbegin()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(1), cudf::test::structs_column_wrapper({b_a_col}, b_validity.cbegin()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(2),
    float64_wrapper{c_data.cbegin(), c_data.cend(), c_validity.cbegin()});

  // Restore original memory source
  cudf::set_pinned_memory_resource(last_mr);
}

// Validation
TEST_F(JsonReaderTest, ValueValidation)
{
  // parsing error as null rows
  std::string data =
    // 0 -> a: -2 (valid)
    R"({"a":-2 }{})"
    "\n"
    // 1 -> (invalid)
    R"({"b":{}should_be_invalid})"
    "\n"
    // 2 -> b (valid)
    R"({"b":{"a":3} })"
    "\n"
    // 3 -> c: (valid/null based on option)
    R"({"a": 1, "c":nan, "d": "null" } )"
    "\n"
    "\n"
    // 4 -> (valid/null based on option)
    R"({"a":04, "c": 1.23, "d": "abc"} 123)"
    "\n"
    // 5 -> (valid)
    R"({"a":5}//Comment after record)"
    "\n"
    // 6 -> ((valid/null based on option)
    R"({"a":06} //Comment after whitespace)"
    "\n"
    // 7 -> (invalid)
    R"({"a":5 //Invalid Comment within record})";

  // leadingZeros allowed
  // na_values,
  {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data.data()), data.size()}})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
        .strict_validation(true);
    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 4);
    EXPECT_EQ(result.tbl->num_rows(), 8);
    auto b_a_col  = int64_wrapper({0, 0, 3, 0, 0, 0, 0, 0});
    auto a_column = int64_wrapper{{-2, 0, 0, 0, 4, 5, 6, 0},
                                  {true, false, false, false, true, true, true, false}};
    auto b_column = cudf::test::structs_column_wrapper(
      {b_a_col}, {false, false, true, false, false, false, false, false});
    auto c_column = float64_wrapper({0.0, 0.0, 0.0, 0.0, 1.23, 0.0, 0.0, 0.0},
                                    {false, false, false, false, true, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), a_column);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), b_column);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2), c_column);
  }
  // leadingZeros not allowed, NaN allowed
  {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(data.data()), data.size()}})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
        .strict_validation(true)
        .numeric_leading_zeros(false)
        .na_values({"nan"});
    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 4);
    EXPECT_EQ(result.tbl->num_rows(), 8);
    EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::INT8);  // empty column
    auto b_a_col  = int64_wrapper({0, 0, 3, 0, 0, 0, 0, 0});
    auto a_column = int64_wrapper{{-2, 0, 0, 1, 4, 5, 6, 0},
                                  {true, false, false, true, false, true, false, false}};
    auto b_column = cudf::test::structs_column_wrapper(
      {b_a_col}, {false, false, true, false, false, false, false, false});
    auto c_column = int8_wrapper({0, 0, 0, 0, 0, 0, 0, 0},
                                 {false, false, false, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), a_column);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), b_column);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2), c_column);
  }
}

TEST_F(JsonReaderTest, MixedTypes)
{
  using LCWS    = cudf::test::lists_column_wrapper<cudf::string_view>;
  using LCWI    = cudf::test::lists_column_wrapper<int64_t>;
  using valid_t = std::vector<cudf::valid_type>;
  {
    // Simple test for mixed types
    std::string json_string = R"({ "foo": [1,2,3], "bar": 123 }
                               { "foo": { "a": 1 }, "bar": 456 })";

    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
        .mixed_types_as_string(true)
        .lines(true);

    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 2);
    EXPECT_EQ(result.tbl->num_rows(), 2);
    EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
    EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::INT64);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                   cudf::test::strings_column_wrapper({"[1,2,3]", "{ \"a\": 1 }"}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                   cudf::test::fixed_width_column_wrapper<int64_t>({123, 456}));
  }

  // Testing function for mixed types in JSON (for spark json reader)
  auto test_fn = [](std::string_view json_string, cudf::column_view expected) {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
        .mixed_types_as_string(true)
        .lines(true);

    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected);
  };
  // value + string (not mixed type case)
  test_fn(R"(
{ "a": "123" }
{ "a": 123 }
)",
          cudf::test::strings_column_wrapper({"123", "123"}));

  // test cases.
  // STR + STRUCT, STR + LIST, STR + null
  // STRUCT + STR, STRUCT + LIST, STRUCT + null
  // LIST + STR, LIST + STRUCT, LIST + null
  // LIST + STRUCT + STR, STRUCT + LIST + STR, STR + STRUCT + LIST, STRUCT + LIST + null
  // STR + STRUCT + LIST + null

  // STRING mixed:
  // STR + STRUCT, STR + LIST, STR + null
  test_fn(R"(
{ "a": "123" }
{ "a": { "b": 1 } }
)",
          cudf::test::strings_column_wrapper({"123", "{ \"b\": 1 }"}));
  test_fn(R"(
{ "a": "123" }
{ "a": [1,2,3] }
)",
          cudf::test::strings_column_wrapper({"123", "[1,2,3]"}));
  test_fn(R"(
{ "a": "123" }
{ "a": null }
)",
          cudf::test::strings_column_wrapper({"123", ""}, std::vector<bool>{1, 0}.begin()));

  // STRUCT mixed:
  // STRUCT + STR, STRUCT + LIST, STRUCT + null
  test_fn(R"(
{ "a": { "b": 1 } }
{ "a": "fox" }
)",
          cudf::test::strings_column_wrapper({"{ \"b\": 1 }", "fox"}));
  test_fn(R"(
{ "a": { "b": 1 } }
{ "a": [1,2,3] }
)",
          cudf::test::strings_column_wrapper({"{ \"b\": 1 }", "[1,2,3]"}));
  cudf::test::fixed_width_column_wrapper<int64_t> child_int_col_wrapper{1, 2};
  test_fn(R"(
{ "a": { "b": 1 } }
{ "a": null }
)",
          cudf::test::structs_column_wrapper{
            {child_int_col_wrapper}, {1, 0} /*Validity*/
          });

  // LIST mixed:
  // LIST + STR, LIST + STRUCT, LIST + null
  test_fn(R"(
{ "a": [1,2,3] }
{ "a": "123" }
)",
          cudf::test::strings_column_wrapper({"[1,2,3]", "123"}));
  test_fn(R"(
{ "a": [1,2,3] }
{ "a": { "b": 1 } }
)",
          cudf::test::strings_column_wrapper({"[1,2,3]", "{ \"b\": 1 }"}));
  test_fn(
    R"(
{ "a": [1,2,3] }
{ "a": null }
)",
    cudf::test::lists_column_wrapper{{LCWI{1L, 2L, 3L}, LCWI{4L, 5L}}, valid_t{1, 0}.begin()});

  // All mixed:
  // LIST + STRUCT + STR, STRUCT + LIST + STR, STR + STRUCT + LIST, STRUCT + LIST + null
  test_fn(R"(
{ "a": [1,2,3]  }
{ "a": { "b": 1 } }
{ "a": "fox"}
)",
          cudf::test::strings_column_wrapper({"[1,2,3]", "{ \"b\": 1 }", "fox"}));
  test_fn(R"(
{ "a": { "b": 1 } }
{ "a": [1,2,3]  }
{ "a": "fox"}
)",
          cudf::test::strings_column_wrapper({"{ \"b\": 1 }", "[1,2,3]", "fox"}));
  test_fn(R"(
{ "a": "fox"}
{ "a": { "b": 1 } }
{ "a": [1,2,3]  }
)",
          cudf::test::strings_column_wrapper({"fox", "{ \"b\": 1 }", "[1,2,3]"}));
  test_fn(R"(
{ "a": [1,2,3]  }
{ "a": { "b": 1 } }
{ "a": null}
)",
          cudf::test::strings_column_wrapper({"[1,2,3]", "{ \"b\": 1 }", "NA"},
                                             valid_t{1, 1, 0}.begin()));  // RIGHT

  // value + string inside list
  test_fn(R"(
{ "a": [1,2,3] }
{ "a": [true,false,true] }
{ "a": ["a", "b", "c"] }
)",
          cudf::test::lists_column_wrapper<cudf::string_view>{
            {"1", "2", "3"}, {"true", "false", "true"}, {"a", "b", "c"}});

  // null + list of mixed types and null
  test_fn(R"(
{ "var1": null }
{ "var1": [{ "var0": true, "var1": "hello", "var2": null }, null, [true, null, null]] }
  )",
          cudf::test::lists_column_wrapper<cudf::string_view>(
            {{"NA", "NA"},
             {{R"({ "var0": true, "var1": "hello", "var2": null })", "null", "[true, null, null]"},
              valid_t{1, 0, 1}.begin()}},
            valid_t{0, 1}.begin()));

  // test to confirm if reinitialize a non-string column as string affects max_rowoffsets.
  // max_rowoffsets is generated based on parent col id,
  // so, even if mixed types are present, their row offset will be correct.

  cudf::test::lists_column_wrapper expected_list{
    {
      cudf::test::lists_column_wrapper({LCWS({"1", "2", "3"}), LCWS({"4", "5", "6"})}),
      cudf::test::lists_column_wrapper({LCWS()}),
      cudf::test::lists_column_wrapper({LCWS()}),  // null
      cudf::test::lists_column_wrapper({LCWS()}),  // null
      cudf::test::lists_column_wrapper({LCWS({"{\"c\": -1}"}), LCWS({"5"})}),
      cudf::test::lists_column_wrapper({LCWS({"7"}), LCWS({"8", "9"})}),
      cudf::test::lists_column_wrapper({LCWS()}),  // null
    },
    valid_t{1, 1, 0, 0, 1, 1, 0}.begin()};
  test_fn(R"(
{"b": [ [1, 2, 3], [ 4, 5, 6] ]}
{"b": [[]]}
{}
{}
{"b": [ [ {"c": -1} ], [ 5 ] ]}
{"b": [ [7], [8, 9]]}
{}
)",
          expected_list);
}

TEST_F(JsonReaderTest, MapTypes)
{
  using cudf::type_id;
  // Testing function for mixed types in JSON (for spark json reader)
  auto test_fn = [](std::string_view json_string, bool lines, std::vector<type_id> types) {
    std::map<std::string, cudf::io::schema_element> dtype_schema{
      {"foo1", {data_type{type_id::STRING}}},  // list forced as a string
      {"foo2", {data_type{type_id::STRING}}},  // struct forced as a string
      {"1", {data_type{type_id::STRING}}},
      {"2", {data_type{type_id::STRING}}},
      {"bar", {dtype<int32_t>()}},
    };

    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
        .dtypes(dtype_schema)
        .mixed_types_as_string(true)
        .lines(lines);

    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
    EXPECT_EQ(result.tbl->num_columns(), types.size());
    int i = 0;
    for (auto& col : result.tbl->view()) {
      EXPECT_EQ(col.type().id(), types[i]) << "column[" << i << "].type";
      i++;
    }
  };

  // json
  test_fn(R"([{ "foo1": [1,2,3], "bar": 123 },
              { "foo2": { "a": 1 }, "bar": 456 }])",
          false,
          {type_id::STRING, type_id::INT32, type_id::STRING});
  // jsonl
  test_fn(R"( { "foo1": [1,2,3], "bar": 123 }
              { "foo2": { "a": 1 }, "bar": 456 })",
          true,
          {type_id::STRING, type_id::INT32, type_id::STRING});
  // jsonl-array
  test_fn(R"([123, [1,2,3]]
              [456, null,  { "a": 1 }])",
          true,
          {type_id::INT64, type_id::STRING, type_id::STRING});
  // json-array
  test_fn(R"([[[1,2,3], null, 123],
              [null, { "a": 1 }, 456 ]])",
          false,
          {type_id::LIST, type_id::STRING, type_id::STRING});
}

/**
 * @brief Test fixture for parametrized JSON reader tests
 */
struct JsonDelimiterParamTest : public cudf::test::BaseFixture,
                                public testing::WithParamInterface<char> {};

// Parametrize qualifying JSON tests for multiple delimiters
INSTANTIATE_TEST_SUITE_P(JsonDelimiterParamTest,
                         JsonDelimiterParamTest,
                         ::testing::Values('\n', '\b', '\v', '\f', 'h'));

TEST_P(JsonDelimiterParamTest, JsonLinesDelimiter)
{
  using SymbolT = char;

  SymbolT const random_delimiter = GetParam();

  // Test input
  std::string input             = R"({"col1":100, "col2":1.1, "col3":"aaa"})";
  std::size_t const string_size = 400;
  /*
   * We are constructing a JSON lines string where each row is {"col1":100, "col2":1.1,
   * "col3":"aaa"} and rows are separated by random_delimiter. Instead of concatenating lines
   * linearly in O(n), we can do it in O(log n) by doubling the input in each iteration. The total
   * number of such iterations is log_repetitions.
   */
  auto const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(string_size / input.size())));
  std::size_t const repetitions = 1UL << log_repetitions;
  for (std::size_t i = 0; i < log_repetitions; i++) {
    input = input + random_delimiter + input;
  }

  cudf::io::json_reader_options json_parser_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(input.c_str()), input.size()}})
      .lines(true)
      .delimiter(random_delimiter);

  cudf::io::table_with_metadata result = cudf::io::read_json(json_parser_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), repetitions);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col1");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col2");
  EXPECT_EQ(result.metadata.schema_info[2].name, "col3");

  auto col1_iterator = cuda::constant_iterator<int64_t>(100);
  auto col2_iterator = cuda::constant_iterator<double>(1.1);
  auto col3_iterator = cuda::constant_iterator<std::string>("aaa");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper(col1_iterator, col1_iterator + repetitions));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 float64_wrapper(col2_iterator, col2_iterator + repetitions));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(2),
    cudf::test::strings_column_wrapper(col3_iterator, col3_iterator + repetitions));
}

TEST_F(JsonReaderTest, ViableDelimiter)
{
  // Test input
  std::string input = R"({"col1":100, "col2":1.1, "col3":"aaa"})";

  cudf::io::json_reader_options json_parser_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(input.c_str()), input.size()}})
      .lines(true);

  json_parser_options.set_delimiter('\f');
  CUDF_EXPECT_NO_THROW(cudf::io::read_json(json_parser_options));

  EXPECT_THROW(json_parser_options.set_delimiter('\t'), std::invalid_argument);
}

TEST_F(JsonReaderTest, ViableDelimiterNewlineWS)
{
  // Test input
  std::string input = R"({"a":
  100})";

  cudf::io::json_reader_options json_parser_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(input.c_str()), input.size()}})
      .lines(true)
      .delimiter('\0');

  auto result = cudf::io::read_json(json_parser_options);
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 1);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "a");

  auto col1_iterator = cuda::constant_iterator<int64_t>(100);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper(col1_iterator, col1_iterator + 1));
}

// Test case for dtype prune:
// all paths, only one.
// one present, another not present, nothing present
// nested, flat, not-jsonlines
TEST_F(JsonReaderTest, JsonNestedDtypeFilter)
{
  std::string json_stringl = R"(
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true}
    {"a": 1, "b": {"0": "abc"          }, "c": false}
    {"a": 1, "b": {}}
    {"a": 1,                              "c": null}
    )";
  std::string json_string  = R"([
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true},
    {"a": 1, "b": {"0": "abc"          }, "c": false},
    {"a": 1, "b": {}},
    {"a": 1,                              "c": null}
    ])";
  for (auto& [json_string, lines] : {std::pair{json_stringl, true}, {json_string, false}}) {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
        .prune_columns(true)
        .lines(lines);

    // include all columns
    //// schema
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{
        {"b",
         {data_type{cudf::type_id::STRUCT},
          {{"0", {data_type{cudf::type_id::STRING}}},
           {"1", {data_type{cudf::type_id::LIST}, {{"element", {dtype<float>()}}}}}}}},
        {"a", {dtype<int32_t>()}},
        {"c", {dtype<bool>()}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", "b" and "c"
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "b");
      EXPECT_EQ(result.metadata.schema_info[2].name, "c");
      // "b" children checks
      ASSERT_EQ(result.metadata.schema_info[1].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[1].children[0].name, "0");
      EXPECT_EQ(result.metadata.schema_info[1].children[1].name, "1");
      ASSERT_EQ(result.metadata.schema_info[1].children[1].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[1].children[1].children[0].name, "offsets");
      EXPECT_EQ(result.metadata.schema_info[1].children[1].children[1].name, "element");
      // types
      EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRUCT);
      EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::BOOL8);
      EXPECT_EQ(result.tbl->get_column(1).child(0).type().id(), cudf::type_id::STRING);
      EXPECT_EQ(result.tbl->get_column(1).child(1).type().id(), cudf::type_id::LIST);
      EXPECT_EQ(result.tbl->get_column(1).child(1).child(0).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(1).child(1).child(1).type().id(), cudf::type_id::FLOAT32);
    }
    //// vector
    {
      std::vector<data_type> types{
        {dtype<int32_t>()}, data_type{cudf::type_id::STRUCT}, {dtype<bool>()}};
      in_options.set_dtypes(types);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", "b" and "c"
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "b");
      EXPECT_EQ(result.metadata.schema_info[2].name, "c");
    }
    //// map
    {
      std::map<std::string, data_type> dtype_map{
        {"b",
         {
           data_type{cudf::type_id::STRUCT},
         }},
        {"a", {dtype<int32_t>()}},
        {"c", {dtype<bool>()}},
      };
      in_options.set_dtypes(dtype_map);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", "b" and "c"
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "b");
      EXPECT_EQ(result.metadata.schema_info[2].name, "c");
    }

    // include only one column
    //// schema
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{
        {"a", {dtype<int32_t>()}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "a"
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
    }
    //// vector
    {
      std::vector<data_type> types{{dtype<int32_t>()}};
      in_options.set_dtypes(types);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "a"
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
    }
    //// map
    {
      std::map<std::string, data_type> dtype_map{
        {"a", {dtype<int32_t>()}},
      };
      in_options.set_dtypes(dtype_map);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "a"
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
    }

    // include only one column (nested)
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{
        {"b",
         {data_type{cudf::type_id::STRUCT},
          {{"1", {data_type{cudf::type_id::LIST}, {{"element", {dtype<float>()}}}}}}}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "b":"1":[float]
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "b");
      ASSERT_EQ(result.metadata.schema_info[0].children.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "1");
      ASSERT_EQ(result.metadata.schema_info[0].children[0].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].children[0].children[0].name, "offsets");
      EXPECT_EQ(result.metadata.schema_info[0].children[0].children[1].name, "element");
      EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
      EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::LIST);
      EXPECT_EQ(result.tbl->get_column(0).child(0).child(0).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(0).child(0).child(1).type().id(), cudf::type_id::FLOAT32);
    }
    // multiple - all present
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{
        {"a", {dtype<int32_t>()}},
        {"c", {dtype<bool>()}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", and "c"
      ASSERT_EQ(result.tbl->num_columns(), 2);
      ASSERT_EQ(result.metadata.schema_info.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "c");
    }
    // multiple - not all present
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{
        {"a", {dtype<int32_t>()}},
        {"d", {dtype<bool>()}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "a"
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
    }
    // multiple - not all present nested
    {
      std::map<std::string, cudf::io::schema_element> dtype_schema{

        {"b",
         {data_type{cudf::type_id::STRUCT},
          {
            {"2", {data_type{cudf::type_id::STRING}}},
          }}},
        {"c", {dtype<bool>()}},
      };
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "b" (empty struct) and "c"
      ASSERT_EQ(result.tbl->num_columns(), 2);
      ASSERT_EQ(result.metadata.schema_info.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].name, "b");
      ASSERT_EQ(result.metadata.schema_info[0].children.size(), 0);
      EXPECT_EQ(result.metadata.schema_info[1].name, "c");
    }
  }
}

TEST_F(JsonReaderTest, JSONMixedTypeChildren)
{
  // struct mixed.
  {
    std::string const json_str = R"(
  { "Root": { "Key": [ { "EE": "A" } ] } }
  { "Root": { "Key": {  } } }
  { "Root": { "Key": [{ "YY": 1}] } }
  )";
    // Column "EE" is created and destroyed
    // Column "YY" should not be created

    cudf::io::json_reader_options options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_str.c_str()), json_str.size()}})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
        .normalize_single_quotes(true)
        .normalize_whitespace(false)
        .mixed_types_as_string(true)
        .keep_quotes(true);

    auto result = cudf::io::read_json(options);

    ASSERT_EQ(result.tbl->num_columns(), 1);
    ASSERT_EQ(result.metadata.schema_info.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].name, "Root");
    ASSERT_EQ(result.metadata.schema_info[0].children.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "Key");
    ASSERT_EQ(result.metadata.schema_info[0].children[0].children.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].children[0].children[0].name, "offsets");
    // types
    EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
    EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::STRING);
    cudf::test::strings_column_wrapper expected(
      {R"([ { "EE": "A" } ])", "{  }", R"([{ "YY": 1}])"});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result.tbl->get_column(0).child(0));
  }

  // list mixed.
  {
    std::string const json_str = R"(
  { "Root": { "Key": [ { "EE": "A" } ] } }
  { "Root": { "Key": "abc" } }
  { "Root": { "Key": [{ "YY": 1}] } }
  )";
    // Column "EE" is created and destroyed
    // Column "YY" should not be created

    cudf::io::json_reader_options options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_str.c_str()), json_str.size()}})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
        .normalize_single_quotes(true)
        .normalize_whitespace(false)
        .mixed_types_as_string(true)
        .keep_quotes(true);

    auto result = cudf::io::read_json(options);

    ASSERT_EQ(result.tbl->num_columns(), 1);
    ASSERT_EQ(result.metadata.schema_info.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].name, "Root");
    ASSERT_EQ(result.metadata.schema_info[0].children.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "Key");
    ASSERT_EQ(result.metadata.schema_info[0].children[0].children.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[0].children[0].children[0].name, "offsets");
    // types
    EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
    EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::STRING);
    cudf::test::strings_column_wrapper expected(
      {R"([ { "EE": "A" } ])", "\"abc\"", R"([{ "YY": 1}])"});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result.tbl->get_column(0).child(0));
  }
}

TEST_F(JsonReaderTest, MixedTypesWithSchema)
{
  std::string data = "{\"data\": {\"A\": 0, \"B\": 1}}\n{\"data\": [1,0]}\n";

  std::map<std::string, cudf::io::schema_element> data_types;
  std::map<std::string, cudf::io::schema_element> child_types;
  child_types.insert(
    std::pair{"element", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}, {}}});
  data_types.insert(
    std::pair{"data", cudf::io::schema_element{cudf::data_type{cudf::type_id::LIST}, child_types}});

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(data_types)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .normalize_single_quotes(true)
      .normalize_whitespace(true)
      .mixed_types_as_string(true)
      .experimental(true)
      .keep_quotes(true)
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 2);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result.tbl->get_column(0).child(1).type().id(), cudf::type_id::STRING);
}

TEST_F(JsonReaderTest, UnicodeFieldname)
{
  // unicode at nested and leaf levels
  std::string data = R"({"data": {"a": 0, "b	c": 1}}
  {"data": {"\u0061": 2, "\u0062\tc": 3}}
  {"d\u0061ta": {"a": 4}})";

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .experimental(true)
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 3);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(result.tbl->get_column(0).num_children(), 2);
  EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(0).child(1).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info.at(0).name, "data");
  EXPECT_EQ(result.metadata.schema_info.at(0).children.at(0).name, "a");
  EXPECT_EQ(result.metadata.schema_info.at(0).children.at(1).name, "b\tc");
  EXPECT_EQ(result.metadata.schema_info.at(0).children.size(), 2);
}

TEST_F(JsonReaderTest, JsonDtypeSchema)
{
  std::string data = R"(
    {"a": 1, "b": {"0": "abc", "1": ["a", "b"]}, "c": true}
    {"a": 1, "b": {"0": "abc"          }, "c": false}
    {"a": 1, "b": {"0": "lolol  "}, "c": true}
    )";

  std::map<std::string, cudf::io::schema_element> dtype_schema{{"c", {data_type{type_id::STRING}}},
                                                               {"b", {data_type{type_id::STRING}}},
                                                               {"a", {dtype<double>()}}};
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(dtype_schema)
      .prune_columns(true)
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "b");
  EXPECT_EQ(result.metadata.schema_info[2].name, "c");

  // cudf::column::contents contents = result.tbl->get_column(1).release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), float64_wrapper{{1, 1, 1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(1),
    cudf::test::strings_column_wrapper({"{\"0\": \"abc\", \"1\": [\"a\", \"b\"]}",
                                        "{\"0\": \"abc\"          }",
                                        "{\"0\": \"lolol  \"}"}),
    cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"true", "false", "true"}),
                                 cudf::test::debug_output_level::ALL_ERRORS);
}

TEST_F(JsonReaderTest, LastRecordInvalid)
{
  std::string data = R"({"key": "1"}
    {"key": "})";
  std::map<std::string, cudf::io::schema_element> schema{{"key", {dtype<cudf::string_view>()}}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(data.data()), data.size()}})
                .dtypes(schema)
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .build();
  auto const result = cudf::io::read_json(opts);

  EXPECT_EQ(result.metadata.schema_info[0].name, "key");
  cudf::test::strings_column_wrapper expected{{"1", ""}, cudf::test::iterators::nulls_at({1})};
  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), cudf::table_view{{expected}});
}

// Test case for dtype pruning with column order
TEST_F(JsonReaderTest, JsonNestedDtypeFilterWithOrder)
{
  std::string json_stringl = R"(
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true}
    {"a": 1, "b": {"0": "abc"          }, "c": false}
    {"a": 1, "b": {}}
    {"a": 1,                              "c": null}
    )";
  std::string json_string  = R"([
    {"a": 1, "b": {"0": "abc", "1": [-1.]}, "c": true},
    {"a": 1, "b": {"0": "abc"          }, "c": false},
    {"a": 1, "b": {}},
    {"a": 1,                              "c": null}
    ])";
  for (auto& [json_string, lines] : {std::pair{json_stringl, true}, {json_string, false}}) {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
        .prune_columns(true)
        .lines(lines);

    // include all columns
    //// schema with partial ordering
    {
      cudf::io::schema_element dtype_schema{
        data_type{cudf::type_id::STRUCT},
        {
          {"b",
           {data_type{cudf::type_id::STRUCT},
            {{"0", {data_type{cudf::type_id::STRING}}},
             {"1", {data_type{cudf::type_id::LIST}, {{"element", {dtype<float>()}}}}}},
            {{"0", "1"}}}},
          {"a", {dtype<int32_t>()}},
          {"c", {dtype<bool>()}},
        },
        {{"b", "a", "c"}}};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", "b" and "c"
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "b");
      EXPECT_EQ(result.metadata.schema_info[1].name, "a");
      EXPECT_EQ(result.metadata.schema_info[2].name, "c");
      // "b" children checks
      ASSERT_EQ(result.metadata.schema_info[0].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "0");
      EXPECT_EQ(result.metadata.schema_info[0].children[1].name, "1");
      ASSERT_EQ(result.metadata.schema_info[0].children[1].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].children[1].children[0].name, "offsets");
      EXPECT_EQ(result.metadata.schema_info[0].children[1].children[1].name, "element");
      // types
      EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
      EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::BOOL8);
      EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::STRING);
      EXPECT_EQ(result.tbl->get_column(0).child(1).type().id(), cudf::type_id::LIST);
      EXPECT_EQ(result.tbl->get_column(0).child(1).child(0).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(0).child(1).child(1).type().id(), cudf::type_id::FLOAT32);
    }
    //// schema with pruned columns and different order.
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"c", {dtype<bool>()}},
                                              {"b",
                                               {
                                                 data_type{cudf::type_id::STRUCT},
                                               }},
                                              {"a", {dtype<int32_t>()}},
                                            },
                                            {{"c", "b", "a"}}};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // "c", "b" and "a" order
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "c");
      EXPECT_EQ(result.metadata.schema_info[1].name, "b");
      EXPECT_EQ(result.metadata.schema_info[2].name, "a");
      // pruned
      EXPECT_EQ(result.metadata.schema_info[1].children.size(), 0);
    }
    //// schema with pruned columns and different sub-order.
    {
      cudf::io::schema_element dtype_schema{
        data_type{cudf::type_id::STRUCT},
        {
          {"c", {dtype<bool>()}},
          {"b",
           {data_type{cudf::type_id::STRUCT},
            //  {},
            {{"0", {data_type{cudf::type_id::STRING}}},
             {"1", {data_type{cudf::type_id::LIST}, {{"element", {dtype<float>()}}}}}},
            {{"1", "0"}}}},
          {"a", {dtype<int32_t>()}},
        }};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Order of occurance in json
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "b");
      EXPECT_EQ(result.metadata.schema_info[2].name, "c");
      // Sub-order of "b"
      EXPECT_EQ(result.metadata.schema_info[1].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[1].children[0].name, "1");
      EXPECT_EQ(result.metadata.schema_info[1].children[1].name, "0");
    }
    //// schema with 1 dtype, but 2 column order
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"a", {dtype<int32_t>()}},
                                            },
                                            {{"a", "b"}}};
      EXPECT_THROW(in_options.set_dtypes(dtype_schema), std::invalid_argument);
      // Input schema column order size mismatch with input schema child types
    }
    //// repetition, Error
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"a", {dtype<int32_t>()}},
                                            },
                                            {{"a", "a"}}};
      EXPECT_THROW(in_options.set_dtypes(dtype_schema), std::invalid_argument);
      // Input schema column order size mismatch with input schema child types
    }
    //// different column name in order, Error
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"a", {dtype<int32_t>()}},
                                            },
                                            {{"b"}}};
      EXPECT_THROW(in_options.set_dtypes(dtype_schema), std::invalid_argument);
      // Column name not found in input schema map, but present in column order and
      // prune_columns is enabled
    }
    // include only one column (nested)
    {
      cudf::io::schema_element dtype_schema{
        data_type{cudf::type_id::STRUCT},
        {
          {"b",
           {data_type{cudf::type_id::STRUCT},
            {{"1", {data_type{cudf::type_id::LIST}, {{"element", {dtype<float>()}}}}}},
            {{"1"}}}},
        }};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "b":"1":[float]
      ASSERT_EQ(result.tbl->num_columns(), 1);
      ASSERT_EQ(result.metadata.schema_info.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].name, "b");
      ASSERT_EQ(result.metadata.schema_info[0].children.size(), 1);
      EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "1");
      ASSERT_EQ(result.metadata.schema_info[0].children[0].children.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].children[0].children[0].name, "offsets");
      EXPECT_EQ(result.metadata.schema_info[0].children[0].children[1].name, "element");
      EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRUCT);
      EXPECT_EQ(result.tbl->get_column(0).child(0).type().id(), cudf::type_id::LIST);
      EXPECT_EQ(result.tbl->get_column(0).child(0).child(0).type().id(), cudf::type_id::INT32);
      EXPECT_EQ(result.tbl->get_column(0).child(0).child(1).type().id(), cudf::type_id::FLOAT32);
    }
    // multiple - all present
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"a", {dtype<int32_t>()}},
                                              {"c", {dtype<bool>()}},
                                            },
                                            {{"a", "c"}}};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "a", and "c"
      ASSERT_EQ(result.tbl->num_columns(), 2);
      ASSERT_EQ(result.metadata.schema_info.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "c");
    }
    // multiple - not all present
    {
      cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                            {
                                              {"a", {dtype<int32_t>()}},
                                              {"d", {dtype<bool>()}},
                                            },
                                            {{"a", "d"}}};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have column "a"
      ASSERT_EQ(result.tbl->num_columns(), 2);
      ASSERT_EQ(result.metadata.schema_info.size(), 2);
      EXPECT_EQ(result.metadata.schema_info[0].name, "a");
      EXPECT_EQ(result.metadata.schema_info[1].name, "d");
      auto all_null_bools =
        cudf::test::fixed_width_column_wrapper<bool>{{true, true, true, true}, {0, 0, 0, 0}};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), all_null_bools);
    }
    // test struct, list of string, list of struct.
    //  multiple - not all present nested
    {
      cudf::io::schema_element dtype_schema{
        data_type{cudf::type_id::STRUCT},
        {
          {"b",
           {data_type{cudf::type_id::STRUCT},
            {
              {"2", {data_type{cudf::type_id::STRING}}},
            },
            {{"2"}}}},
          {"d", {data_type{cudf::type_id::LIST}, {{"element", {dtype<int32_t>()}}}}},
          {"e",
           {data_type{cudf::type_id::LIST},
            {{"element",
              {
                data_type{cudf::type_id::STRUCT},
                {
                  {"3", {data_type{cudf::type_id::STRING}}},
                },  //{{"3"}} missing column_order, but output should not have it.
              }}}}},
        },
        {{"b", "d", "e"}}};
      in_options.set_dtypes(dtype_schema);
      cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
      // Make sure we have columns "b" (empty struct) and "c"
      ASSERT_EQ(result.tbl->num_columns(), 3);
      ASSERT_EQ(result.metadata.schema_info.size(), 3);
      EXPECT_EQ(result.metadata.schema_info[0].name, "b");
      ASSERT_EQ(result.metadata.schema_info[0].children.size(), 1);
      ASSERT_EQ(result.metadata.schema_info[0].children[0].name, "2");
      EXPECT_EQ(result.metadata.schema_info[1].name, "d");
      auto all_null_strings = cudf::test::strings_column_wrapper{{"", "", "", ""}, {0, 0, 0, 0}};
      EXPECT_EQ(result.tbl->get_column(0).num_children(), 1);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0).child(0), all_null_strings);
      auto const all_null_list = cudf::test::lists_column_wrapper<int32_t>{
        {{0, 0}, {1, 1}, {2, 2}, {3, 3}}, cudf::test::iterators::all_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), all_null_list);
      EXPECT_EQ(result.metadata.schema_info[2].name, "e");
      ASSERT_EQ(result.metadata.schema_info[2].children.size(), 2);
      ASSERT_EQ(result.metadata.schema_info[2].children[1].children.size(), 0);
      // ASSERT_EQ(result.metadata.schema_info[2].children[1].children[0].name, "3");
      auto empty_string_col = cudf::test::strings_column_wrapper{};
      cudf::test::structs_column_wrapper expected_structs{{}, cudf::test::iterators::all_nulls()};
      // make all null column of list of struct of string
      auto wrapped = make_lists_column(
        4,
        cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0}.release(),
        expected_structs.release(),
        4,
        cudf::create_null_mask(4, cudf::mask_state::ALL_NULL));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2), *wrapped);
    }
  }

  // test list (all-null) of struct (empty) of string (empty)
  {
    std::string json_stringl = R"(
    {"a" : [1], "c2": [1, 2]}
    {}
    )";
    auto lines               = true;
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(json_stringl.data()), json_stringl.size()}})
        .prune_columns(true)
        .experimental(true)
        .lines(lines);

    cudf::io::schema_element dtype_schema{
      data_type{cudf::type_id::STRUCT},
      {
        {"a", {data_type{cudf::type_id::LIST}, {{"element", {dtype<int64_t>()}}}}},
        {"c2",
         {data_type{cudf::type_id::LIST},
          {{"element",
            {data_type{cudf::type_id::STRUCT},
             {
               {"d", {data_type{cudf::type_id::STRING}}},
             },
             {{"d"}}}}}}},
      },
      {{"a", "c2"}}};
    in_options.set_dtypes(dtype_schema);
    cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
    // Make sure we have column "a":[int64_t]
    ASSERT_EQ(result.tbl->num_columns(), 2);
    ASSERT_EQ(result.metadata.schema_info.size(), 2);
    EXPECT_EQ(result.metadata.schema_info[0].name, "a");
    ASSERT_EQ(result.metadata.schema_info[0].children.size(), 2);
    EXPECT_EQ(result.metadata.schema_info[0].children[0].name, "offsets");
    EXPECT_EQ(result.metadata.schema_info[0].children[1].name, "element");
    // Make sure we have all null list "c2": [{"d": ""}]
    EXPECT_EQ(result.metadata.schema_info[1].name, "c2");
    ASSERT_EQ(result.metadata.schema_info[1].children.size(), 2);
    EXPECT_EQ(result.metadata.schema_info[1].children[0].name, "offsets");
    EXPECT_EQ(result.metadata.schema_info[1].children[1].name, "element");
    ASSERT_EQ(result.metadata.schema_info[1].children[1].children.size(), 1);
    EXPECT_EQ(result.metadata.schema_info[1].children[1].children[0].name, "d");

    auto const expected0 = [&] {
      auto const valids = std::vector<bool>{1, 0};
      auto [null_mask, null_count] =
        cudf::test::detail::make_null_mask(valids.begin(), valids.end());
      return cudf::make_lists_column(2,
                                     size_type_wrapper{0, 1, 1}.release(),
                                     int64_wrapper{1}.release(),
                                     null_count,
                                     std::move(null_mask));
    }();

    auto const expected1 = [&] {
      auto const get_structs = [] {
        auto child = cudf::test::strings_column_wrapper{};
        return cudf::test::structs_column_wrapper{{child}};
      };
      auto const valids = std::vector<bool>{0, 0};
      auto [null_mask, null_count] =
        cudf::test::detail::make_null_mask(valids.begin(), valids.end());
      return cudf::make_lists_column(2,
                                     size_type_wrapper{0, 0, 0}.release(),
                                     get_structs().release(),
                                     null_count,
                                     std::move(null_mask));
    }();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected0, result.tbl->get_column(0).view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected1, result.tbl->get_column(1).view());
  }
}

TEST_F(JsonReaderTest, NullifyMixedList)
{
  using namespace cudf::test::iterators;
  // test list
  std::string json_stringl = R"(
      {"c2": []}
      {"c2": [{}]}
      {"c2": [[]]}
      {"c2": [{}, [], {}]}
      {"c2": [[123], {"b": "1"}]}
      {"c2": [{"x": "y"}, {"b": "1"}]}
      {}
    )";
  // [], [{null, null}], null, null, null, [{null, null}, {1, null}], null
  // valid     1  1  0  0  0  1  0
  // ofset  0, 0, 1, 1, 1, 1, 3, 3
  // child  {null, null}, {null, null}, {1, null}
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_stringl.data()), json_stringl.size()}})
      .prune_columns(true)
      .experimental(true)
      .lines(true);

  // struct<c2: array<struct<b: string, c: string>>> eg. {"c2": [{"b": "1", "c": "2"}]}
  cudf::io::schema_element dtype_schema{data_type{cudf::type_id::STRUCT},
                                        {
                                          {"c2",
                                           {data_type{cudf::type_id::LIST},
                                            {{"element",
                                              {data_type{cudf::type_id::STRUCT},
                                               {
                                                 {"b", {data_type{cudf::type_id::STRING}}},
                                                 {"c", {data_type{cudf::type_id::STRING}}},
                                               },
                                               {{"b", "c"}}}}}}},
                                        },
                                        {{"c2"}}};
  in_options.set_dtypes(dtype_schema);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  ASSERT_EQ(result.tbl->num_columns(), 1);
  ASSERT_EQ(result.metadata.schema_info.size(), 1);

  // Expected: A list of struct of 2-string columns
  // [], [{null, null}], null, null, null, [{null, null}, {1, null}], null
  auto get_structs = [] {
    strings_wrapper child0{{"", "", "1"}, nulls_at({0, 0, 1})};
    strings_wrapper child1{{"", "", ""}, all_nulls()};
    // purge non-empty nulls in list seems to retain nullmask in struct child column
    return cudf::test::structs_column_wrapper{{child0, child1}, no_nulls()}.release();
  };
  std::vector<bool> const list_nulls{1, 1, 0, 0, 0, 1, 0};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nulls.cbegin(), list_nulls.cend());
  auto const expected = cudf::make_lists_column(
    7,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 1, 1, 1, 1, 3, 3}.release(),
    get_structs(),
    null_count,
    std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, result.tbl->get_column(0).view());
}

struct JsonCompressedIOTest : public cudf::test::BaseFixture,
                              public testing::WithParamInterface<cudf::io::compression_type> {};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonCompressedIOTest,
                         JsonCompressedIOTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::SNAPPY,
                                           cudf::io::compression_type::ZSTD,
                                           cudf::io::compression_type::NONE));

TEST_P(JsonCompressedIOTest, BasicJsonLines)
{
  cudf::io::compression_type const comptype = GetParam();
  std::string data                          = to_records_orient(
    {{{"0", "1"}, {"1", "1.1"}}, {{"0", "2"}, {"1", "2.2"}}, {{"0", "3"}, {"1", "3.3"}}}, "\n");

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(data.data()), data.size()));
    auto decomp_out_buffer = cudf::io::detail::decompress(
      comptype, cudf::host_span<uint8_t const>(cdata.data(), cdata.size()));
    std::string const expected = R"({"0":1, "1":1.1}
{"0":2, "1":2.2}
{"0":3, "1":3.3})";
    EXPECT_EQ(
      expected,
      std::string(reinterpret_cast<char*>(decomp_out_buffer.data()), decomp_out_buffer.size()));
  } else
    cdata = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(data.data()),
                                 reinterpret_cast<uint8_t*>(data.data()) + data.size());

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<uint8_t>(cdata.data(), cdata.size())})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<double>()})
      .compression(comptype)
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2, 3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3}});
}

TEST_F(JsonReaderTest, MismatchedBeginEndTokens)
{
  std::string data = R"({"not_valid": "json)";
  auto opts        = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(data.data()), data.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::FAIL)
                .build();
  EXPECT_THROW(cudf::io::read_json(opts), cudf::logic_error);
}

/**
 * @brief Base test fixture for JSON batched reader tests
 */
struct JsonBatchedReaderTest : public cudf::test::BaseFixture {
 public:
  void set_batch_size(size_t batch_size_upper_bound)
  {
    setenv("LIBCUDF_JSON_BATCH_SIZE", std::to_string(batch_size_upper_bound).c_str(), 1);
  }

  ~JsonBatchedReaderTest() { unsetenv("LIBCUDF_JSON_BATCH_SIZE"); }
};

TEST_F(JsonBatchedReaderTest, EmptyLastBatch)
{
  std::string data          = R"(
  {"a": "b"}
  {"a": "b"}
  {"a": "b"}
  {"a": "b"}
  )";
  size_t size_of_last_batch = 5;
  // This test constructs two batches by setting the batch size such that the last batch is an
  // incomplete line. The JSON string corresponding to the first batch is
  // '\n{"a": "b"}\n{"a": "b"}\n{"a": "b"}\n{"a": '
  // The JSON string corresponding to the second batch is
  // '"b"}\n'
  this->set_batch_size(data.size() - size_of_last_batch);
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(data.data()), data.size()}})
                .lines(true)
                .build();
  auto result = cudf::io::read_json(opts);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 4);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 cudf::test::strings_column_wrapper{{"b", "b", "b", "b"}});
}

TEST_F(JsonReaderTest, DeviceReadAsyncThrows)
{
  // Create simple JSON data
  std::string json_string = R"({"a": 1}
{"a": 2}
{"a": 3}
{"a": 4}
{"a": 5})";

  // Convert to char vector
  std::vector<char> json_data(json_string.begin(), json_string.end());

  // Create our throwing datasource
  auto throwing_source = std::make_unique<cudf::test::ThrowingDeviceReadDatasource>(json_data);
  cudf::io::source_info source_info(throwing_source.get());

  // Try to read the JSON data - this should either succeed or propagate AsyncException
  // from device_read_async.
  cudf::io::json_reader_options read_args =
    cudf::io::json_reader_options::builder(source_info).lines(true);
  try {
    cudf::io::read_json(read_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

TEST_F(JsonReaderTest, DeviceWriteAsyncThrows)
{
  // Create a simple table to write
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = cudf::table_view{{col0}};

  auto throwing_sink = std::make_unique<cudf::test::ThrowingDeviceWriteDataSink>();

  cudf::io::json_writer_options write_args = cudf::io::json_writer_options::builder(
    cudf::io::sink_info{throwing_sink.get()}, table_to_write);

  // The write_json call should either succeed or throw AsyncException.
  try {
    cudf::io::write_json(write_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

// Tests for the per-top-level-column schema-mismatch diagnostic surfaced by
// `read_json_with_row_diagnostics`. Downstream callers (e.g. spark-rapids-jni) use this signal to
// implement their own policy when a JSON value's node category does not match the requested
// schema. cuDF itself does NOT change the column contents on mismatch (existing
// CHILD_NULL_ONLY behavior); the diagnostic is reported on a separate side channel so the
// public `column_name_info`/`table_metadata` ABI is unaffected.
// See https://github.com/rapidsai/cudf/issues/22423.

// Build schema for: struct<c1: int64, c2: list<struct<c3: int64, c4: string>>>.
// Pin column order so cuDF emits columns in (c1, c2) regardless of JSON-encounter order.
cudf::io::schema_element make_nested_mismatch_schema_st()
{
  cudf::io::schema_element st_c2_elem{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"c3", cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}}},
     {"c4", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}}}},
    std::vector<std::string>{"c3", "c4"}};
  cudf::io::schema_element st_c2{
    cudf::data_type{cudf::type_id::LIST}, {{"element", st_c2_elem}}, std::vector<std::string>{}};
  return cudf::io::schema_element{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"c1", cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}}}, {"c2", st_c2}},
    std::vector<std::string>{"c1", "c2"}};
}

cudf::io::schema_element make_nested_mismatch_schema_st_strings()
{
  cudf::io::schema_element st_c2_elem{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"c3", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}}},
     {"c4", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}}}},
    std::vector<std::string>{"c3", "c4"}};
  cudf::io::schema_element st_c2{
    cudf::data_type{cudf::type_id::LIST}, {{"element", st_c2_elem}}, std::vector<std::string>{}};
  return cudf::io::schema_element{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"c1", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}}}, {"c2", st_c2}},
    std::vector<std::string>{"c1", "c2"}};
}

TEST_F(JsonReaderTest, SchemaMismatchDiagNoMismatchEmpty)
{
  // Well-formed nested JSON: schema matches everywhere, so the diagnostic list stays empty.
  std::string const json = "{\"data\": {\"c2\": [{\"c3\": 19, \"c4\": \"x\"}], \"c1\": 123456}}\n";
  cudf::io::schema_element root{cudf::data_type{cudf::type_id::STRUCT},
                                {{"data", make_nested_mismatch_schema_st()}}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 1u);
  EXPECT_TRUE(result.diagnostics.top_level_columns_with_schema_mismatch.empty());
  EXPECT_TRUE(result.diagnostics.top_level_columns_with_schema_mismatch_rows.empty());
}

TEST_F(JsonReaderTest, SchemaMismatchDiagNestedMismatchListed)
{
  // df2 case from SPARK-33134: {"data": {"c2":[19],"c1":123456}}
  // c2 expects list<struct> but JSON has list<int>; nested mismatch at depth 3.
  // Diagnostic should list top-level "data" column. The cudf result table is unchanged
  // (data row stays populated, partial — existing CHILD_NULL_ONLY behavior).
  std::string const json = "{\"data\": {\"c2\": [19], \"c1\": 123456}}\n";
  cudf::io::schema_element root{cudf::data_type{cudf::type_id::STRUCT},
                                {{"data", make_nested_mismatch_schema_st()}}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>{"data"});
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "data");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>{0});
  // Default cudf behavior: column itself is not nulled (caller applies policy).
  EXPECT_EQ(result.data.tbl->get_column(0).null_count(), 0);
}

TEST_F(JsonReaderTest, SchemaMismatchDiagRootLevelMismatchListed)
{
  // c2 is a scalar while the schema requests list<struct>. This fully prunes c2 from the parsed
  // device column tree, so the reader synthesizes it from the schema as an all-null top-level
  // column. The diagnostic must still list "c2", while "c1" must not appear.
  std::string const json = "{\"c2\": 19, \"c1\": 123456}\n";
  auto root_schema       = make_nested_mismatch_schema_st();
  auto opts              = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root_schema)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 2u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>{"c2"});
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "c2");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>{0});
  EXPECT_EQ(result.data.tbl->get_column(1).type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result.data.tbl->get_column(1).null_count(), 1);
}

TEST_F(JsonReaderTest, SchemaMismatchDiagMixedRowsListed)
{
  std::string const json =
    "{\"data\": {\"c2\": [{\"c3\": 19, \"c4\": \"x\"}], \"c1\": 1}}\n"
    "{\"data\": {\"c2\": [19], \"c1\": 2}}\n";
  cudf::io::schema_element root{cudf::data_type{cudf::type_id::STRUCT},
                                {{"data", make_nested_mismatch_schema_st()}}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>{"data"});
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "data");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>{1});
  EXPECT_EQ(result.data.tbl->get_column(0).null_count(), 0);
}

TEST_F(JsonReaderTest, SchemaMismatchDiagRowsFollowOutputColumnOrder)
{
  std::string const json = "{\"a\": {\"c2\": [19], \"c1\": 1}, \"b\": {\"c2\": [29], \"c1\": 2}}\n";
  cudf::io::schema_element root{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"a", make_nested_mismatch_schema_st()}, {"b", make_nested_mismatch_schema_st()}},
    std::vector<std::string>{"b", "a"}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 2u);
  EXPECT_EQ(result.data.metadata.schema_info[0].name, "b");
  EXPECT_EQ(result.data.metadata.schema_info[1].name, "a");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>({"b", "a"}));
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 2u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "b");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>{0});
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[1].column_name, "a");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[1].row_indices,
            std::vector<cudf::size_type>{0});
}

TEST_F(JsonReaderTest, SchemaMismatchDiagMixedRowsListedWithMixedTypesAsString)
{
  std::string const json =
    "{\"data\": {\"c2\": [{\"c3\": 19, \"c4\": \"x\"}], \"c1\": 1}, \"id\": 10}\n"
    "{\"data\": {\"c2\": [19], \"c1\": 2}, \"id\": 20}\n"
    "{\"data\": {\"c2\": [{\"c3\": 39, \"c4\": \"z\"}], \"c1\": 3}, \"id\": 30}\n";
  cudf::io::schema_element root{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"data", make_nested_mismatch_schema_st_strings()},
     {"id", cudf::io::schema_element{cudf::data_type{cudf::type_id::STRING}}}},
    std::vector<std::string>{"data", "id"}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .normalize_whitespace(true)
                .mixed_types_as_string(true)
                .keep_quotes(true)
                .experimental(true)
                .strict_validation(true)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  ASSERT_EQ(result.data.metadata.schema_info.size(), 2u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>{"data"});
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "data");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>{1});
  EXPECT_EQ(result.data.tbl->get_column(0).null_count(), 0);
  EXPECT_EQ(result.data.tbl->get_column(1).null_count(), 0);
}

TEST_F(JsonBatchedReaderTest, SchemaMismatchDiagBatchedRowsUseGlobalOffsets)
{
  std::string const row0 = "{\"data\": 1}\n";
  std::string const row1 = "{\"data\": {\"c1\": 2}}\n";
  std::string const row2 = "{\"data\": 3}\n";
  std::string const json = row0 + row1 + row2;
  this->set_batch_size(row0.size() + row1.size() + 1);

  cudf::io::schema_element data_schema{
    cudf::data_type{cudf::type_id::STRUCT},
    {{"c1", cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}}}},
    std::vector<std::string>{"c1"}};
  cudf::io::schema_element root{cudf::data_type{cudf::type_id::STRUCT},
                                {{"data", data_schema}},
                                std::vector<std::string>{"data"}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();

  auto result = cudf::io::read_json_with_row_diagnostics(opts);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch,
            std::vector<std::string>{"data"});
  ASSERT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows.size(), 1u);
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].column_name, "data");
  EXPECT_EQ(result.diagnostics.top_level_columns_with_schema_mismatch_rows[0].row_indices,
            std::vector<cudf::size_type>({0, 2}));
}

TEST_F(JsonReaderTest, SchemaMismatchDiagPlainReadJsonUnaffected)
{
  // The plain `read_json` path does not pay the cost of collecting diagnostics: its result
  // matches the pre-PR behavior bit-for-bit (column count, null counts, schema_info contents).
  // The row diagnostic side-channel is exposed only through `read_json_with_row_diagnostics`, so
  // existing downstream consumers — including those that bake `column_name_info`'s destructor
  // into their own binaries — see no observable change.
  std::string const json = "{\"data\": {\"c2\": [19], \"c1\": 123456}}\n";
  cudf::io::schema_element root{cudf::data_type{cudf::type_id::STRUCT},
                                {{"data", make_nested_mismatch_schema_st()}}};
  auto opts = cudf::io::json_reader_options::builder(
                cudf::io::source_info{cudf::host_span<std::byte const>{
                  reinterpret_cast<std::byte const*>(json.data()), json.size()}})
                .lines(true)
                .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                .dtypes(root)
                .prune_columns(true)
                .build();
  auto result = cudf::io::read_json(opts);
  ASSERT_EQ(result.metadata.schema_info.size(), 1u);
  EXPECT_EQ(result.tbl->get_column(0).null_count(), 0);
}

TEST_F(JsonReaderTest, MalformedFieldNameWithBrace)
{
  // Garbled field name containing '{' creates structural ambiguity in the
  // token tree.  Combined with an invalid byte in a sibling row, the parser
  // may produce column-tree nodes that are never materialised as columns.
  // Recovery mode must handle this without accessing uninitialised memory.
  std::string json_string =
    R"({"name":"Alice","address":{"city":"NYC","zip":"1"},"score{":[95,2]})"
    "\n"
    R"({"name":"Bob","address":{"city":"LA","zip":"1"},"scores":[)"
    "\xbf"
    R"(8,82]})"
    "\n"
    R"({"name":"C","address":{"city":"Chi","zip":"60601"},"scores":[1,97]})";

  cudf::io::json_reader_options options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);
  CUDF_EXPECT_NO_THROW(cudf::io::read_json(options));
}

TEST_F(JsonReaderTest, MalformedStructuralCharsInValues)
{
  // A malformed middle line must not affect surrounding well-formed records.
  std::string json_string = R"({"a":1,"b":[10,20]})"
                            "\n"
                            R"({"phantom":{"nested":[30,{"c":40)"
                            "\n"
                            R"({"a":3,"b":[50,60]})";

  cudf::io::json_reader_options options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(json_string.data()), json_string.size()}})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);
  cudf::io::table_with_metadata tbl;
  ASSERT_NO_THROW(tbl = cudf::io::read_json(options));
  ASSERT_EQ(tbl.tbl->num_rows(), 3);
  ASSERT_EQ(tbl.tbl->num_columns(), 2);
  cudf::test::fixed_width_column_wrapper<int64_t> expected_a{{1, 0, 3}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tbl.tbl->view().column(0), expected_a);
  EXPECT_EQ(tbl.tbl->get_column(0).null_count(), 1);
  // Column "b": list<int64> [[10,20], null, [50,60]].
  using LCWI = cudf::test::lists_column_wrapper<int64_t>;
  LCWI expected_b{{LCWI{10, 20}, LCWI{}, LCWI{50, 60}}, std::vector<bool>{1, 0, 1}.begin()};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(tbl.tbl->view().column(1), expected_b);
}

CUDF_TEST_PROGRAM_MAIN()
