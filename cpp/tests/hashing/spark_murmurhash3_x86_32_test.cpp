/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
class SparkMurmurHashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SparkMurmurHashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(SparkMurmurHashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const col{0, 127, 1, 2, 8};
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const spark_output1 = cudf::hashing::spark_murmurhash3_x86_32(input, 0);
  auto const spark_output2 = cudf::hashing::spark_murmurhash3_x86_32(input);

  EXPECT_EQ(input.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

TYPED_TEST(SparkMurmurHashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const spark_output1 = cudf::hashing::spark_murmurhash3_x86_32(input1, 0);
  auto const spark_output2 = cudf::hashing::spark_murmurhash3_x86_32(input2);

  EXPECT_EQ(input1.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

template <typename T>
class SparkMurmurHashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SparkMurmurHashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SparkMurmurHashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_zero(
    {T(-0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_nan(
    {T(0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const table_col          = cudf::table_view({col});
  auto const table_col_neg_zero = cudf::table_view({col_neg_zero});
  auto const table_col_neg_nan  = cudf::table_view({col_neg_nan});

  // Spark 3.2+ normalizes signed zero before invoking the JNI hasher.
  // https://github.com/NVIDIA/spark-rapids/blob/aef688e232b9e2c710696d380b2539230897450a/sql-plugin/src/main/scala/com/nvidia/spark/rapids/shims/HashUtils.scala#L21-L44
  // https://github.com/NVIDIA/spark-rapids-jni/blob/09bae9c7dccf050b5db53a70282c3001cba5a015/src/main/cpp/src/hash/murmur_hash.cuh#L163-L172
  auto const spark_col          = cudf::hashing::spark_murmurhash3_x86_32(table_col, 0);
  auto const spark_col_neg_zero = cudf::hashing::spark_murmurhash3_x86_32(table_col_neg_zero, 0);
  auto const spark_col_neg_nan  = cudf::hashing::spark_murmurhash3_x86_32(table_col_neg_nan, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*spark_col, *spark_col_neg_zero);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*spark_col, *spark_col_neg_nan);
}

class SparkMurmurHashTest : public cudf::test::BaseFixture {};

TEST_F(SparkMurmurHashTest, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const empty{};
  auto const expected = cudf::test::fixed_width_column_wrapper<int32_t>{};

  auto const empty_table_output = cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view{}, 42);
  auto const empty_column_output =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view{{empty}}, 42);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, empty_table_output->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, empty_column_output->view());
}

TEST_F(SparkMurmurHashTest, SlicedInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const first({999, 0, 1, -1, 42, 123456789, -999});
  cudf::test::fixed_width_column_wrapper<int32_t> const second(
    {999, 10, 20, 30, -40, -987654321, -999});

  auto const first_slice  = cudf::slice(first, {1, 6}).front();
  auto const second_slice = cudf::slice(second, {1, 6}).front();
  auto const output =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view{{first_slice, second_slice}}, 42);

  cudf::test::fixed_width_column_wrapper<int32_t> const expected(
    {-1721723333, 1151116018, 1549484878, -1287750896, -1980733329});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TEST_F(SparkMurmurHashTest, IterativeSeeding)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const first({0, 1, -1, 42, 123456789});
  cudf::test::fixed_width_column_wrapper<int32_t> const second({10, 20, 30, -40, -987654321});
  auto const input = cudf::table_view({first, second});

  auto const output = cudf::hashing::spark_murmurhash3_x86_32(input, 42);

  // Values produced by Apache Spark for the same input and seed.
  cudf::test::fixed_width_column_wrapper<int32_t> const expected(
    {-1721723333, 1151116018, 1549484878, -1287750896, -1980733329});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(), expected);
}

TEST_F(SparkMurmurHashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});
  cudf::test::strings_column_wrapper const strings_col2(
    {"different but null",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col1(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col2(
    {ts::duration::zero(),
     static_cast<ts::duration>(-200),
     static_cast<ts::duration>(200),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});

  auto const input1        = cudf::table_view({strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2        = cudf::table_view({strings_col2, ints_col2, bools_col2, secs_col2});
  auto const spark_output1 = cudf::hashing::spark_murmurhash3_x86_32(input1, 0);
  auto const spark_output2 = cudf::hashing::spark_murmurhash3_x86_32(input2);

  EXPECT_EQ(input1.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

TEST_F(SparkMurmurHashTest, MultiValueWithSeeds)
{
  // The hash values were determined by running the following Scala code in Apache Spark.
  // Note that Spark >= 3.2 normalizes the float/double value of -0. to +0. so that both values
  // hash to the same result, and libcudf normalizes signed zero to match. The reference values
  // for -0. below are therefore the Spark >= 3.2 values, which are the ones for 0., and the
  // combined hash values follow from them. The Spark RAPIDS plugin applies the same
  // normalization in its calling code, so normalizing here is idempotent for that caller.

  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.catalyst.util.DateTimeUtils

  val schema = new StructType()
      .add("structs", new StructType()
          .add("a", IntegerType)
          .add("b", StringType)
          .add("c", new StructType()
              .add("x", FloatType)
              .add("y", LongType)))
      .add("strings", StringType)
      .add("doubles", DoubleType)
      .add("timestamps", TimestampType)
      .add("decimal64", DecimalType(18, 7))
      .add("longs", LongType)
      .add("floats", FloatType)
      .add("dates", DateType)
      .add("decimal32", DecimalType(9, 3))
      .add("ints", IntegerType)
      .add("shorts", ShortType)
      .add("bytes", ByteType)
      .add("bools", BooleanType)
      .add("decimal128", DecimalType(38, 11))

  val data = Seq(
      Row(Row(0, "a", Row(0f, 0L)), "", 0.toDouble,
          DateTimeUtils.toJavaTimestamp(0), BigDecimal(0), 0.toLong, 0.toFloat,
          DateTimeUtils.toJavaDate(0), BigDecimal(0), 0, 0.toShort, 0.toByte,
          false, BigDecimal(0)),
      Row(Row(100, "bc", Row(100f, 100L)), "The quick brown fox", -(0.toDouble),
          DateTimeUtils.toJavaTimestamp(100), BigDecimal("0.00001"), 100.toLong, -(0.toFloat),
          DateTimeUtils.toJavaDate(100), BigDecimal("0.1"), 100, 100.toShort, 100.toByte,
          true, BigDecimal("0.000000001")),
      Row(Row(-100, "def", Row(-100f, -100L)), "jumps over the lazy dog.", -Double.NaN,
          DateTimeUtils.toJavaTimestamp(-100), BigDecimal("-0.00001"), -100.toLong, -Float.NaN,
          DateTimeUtils.toJavaDate(-100), BigDecimal("-0.1"), -100, -100.toShort, -100.toByte,
          true, BigDecimal("-0.00000000001")),
      Row(Row(0x12345678, "ghij", Row(Float.PositiveInfinity, 0x123456789abcdefL)),
          "All work and no play makes Jack a dull boy", Double.MinValue,
          DateTimeUtils.toJavaTimestamp(Long.MinValue/1000000), BigDecimal("-99999999999.9999999"),
          Long.MinValue, Float.MinValue, DateTimeUtils.toJavaDate(Int.MinValue/100),
          BigDecimal("-999999.999"), Int.MinValue, Short.MinValue, Byte.MinValue, true,
          BigDecimal("-9999999999999999.99999999999")),
      Row(Row(-0x76543210, "klmno", Row(Float.NegativeInfinity, -0x123456789abcdefL)),
          "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721", Double.MaxValue,
          DateTimeUtils.toJavaTimestamp(Long.MaxValue/1000000), BigDecimal("99999999999.9999999"),
          Long.MaxValue, Float.MaxValue, DateTimeUtils.toJavaDate(Int.MaxValue/100),
          BigDecimal("999999.999"), Int.MaxValue, Short.MaxValue, Byte.MaxValue, false,
          BigDecimal("99999999999999999999999999.99999999999")))

  val df = spark.createDataFrame(sc.parallelize(data), schema)
  df.columns.foreach(c => println(s"$c => ${df.select(hash(col(c))).collect.mkString(",")}"))
  println(s"combined => ${df.select(hash(col("*"))).collect.mkString(",")}")
  */

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_structs_expected(
    {-105406170, 90479889, -678041645, 1667387937, 301478567});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected(
    {142593372, 1217302703, -715697185, -2061143941, -111635966});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_doubles_expected(
    {-1670924195, -1670924195, -1281358385, 1897734433, -508695674});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_timestamps_expected(
    {-1670924195, 1114849490, 904948192, -1832979433, 1752430209});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal64_expected(
    {-1670924195, 1114849490, 904948192, 1962370902, -1795328666});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_longs_expected(
    {-1670924195, 1114849490, 904948192, -853646085, -1604625029});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_floats_expected(
    {933211791, 933211791, -349261430, -1225560532, -338752985});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_dates_expected(
    {933211791, 751823303, -1080202046, -1906567553, -1503850410});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal32_expected(
    {-1670924195, 1114849490, 904948192, -1454351396, -193774131});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_ints_expected(
    {933211791, 751823303, -1080202046, 723455942, 133916647});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_shorts_expected(
    {933211791, 751823303, -1080202046, -1871935946, 1249274084});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bytes_expected(
    {933211791, 751823303, -1080202046, 1110053733, 1135925485});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bools_expected(
    {933211791, -559580957, -559580957, -559580957, 933211791});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal128_expected(
    {-783713497, -295670906, 1398487324, -52622807, -1359749815});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_combined_expected(
    {401603227, 1728350528, 552160517, 1132537411, -326043017});

  using double_limits = std::numeric_limits<double>;
  using long_limits   = std::numeric_limits<int64_t>;
  using float_limits  = std::numeric_limits<float>;
  using int_limits    = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> a_col{0, 100, -100, 0x1234'5678, -0x7654'3210};
  cudf::test::strings_column_wrapper b_col{"a", "bc", "def", "ghij", "klmno"};
  cudf::test::fixed_width_column_wrapper<float> x_col{
    0.f, 100.f, -100.f, float_limits::infinity(), -float_limits::infinity()};
  cudf::test::fixed_width_column_wrapper<int64_t> y_col{
    0L, 100L, -100L, 0x0123'4567'89ab'cdefL, -0x0123'4567'89ab'cdefL};
  cudf::test::structs_column_wrapper c_col{{x_col, y_col}};
  cudf::test::structs_column_wrapper const structs_col{{a_col, b_col, c_col}};

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});
  cudf::test::fixed_width_column_wrapper<double> const doubles_col(
    {0., -0., -double_limits::quiet_NaN(), double_limits::lowest(), double_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> const
    timestamps_col({0L, 100L, -100L, long_limits::min() / 1000000, long_limits::max() / 1000000});
  cudf::test::fixed_point_column_wrapper<int64_t> const decimal64_col(
    {0L, 100L, -100L, -999999999999999999L, 999999999999999999L}, numeric::scale_type{-7});
  cudf::test::fixed_width_column_wrapper<int64_t> const longs_col(
    {0L, 100L, -100L, long_limits::min(), long_limits::max()});
  cudf::test::fixed_width_column_wrapper<float> const floats_col(
    {0.f, -0.f, -float_limits::quiet_NaN(), float_limits::lowest(), float_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> dates_col(
    {0, 100, -100, int_limits::min() / 100, int_limits::max() / 100});
  cudf::test::fixed_point_column_wrapper<int32_t> const decimal32_col(
    {0, 100, -100, -999999999, 999999999}, numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, int_limits::min(), int_limits::max()});
  cudf::test::fixed_width_column_wrapper<int16_t> const shorts_col({0, 100, -100, -32768, 32767});
  cudf::test::fixed_width_column_wrapper<int8_t> const bytes_col({0, 100, -100, -128, 127});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});
  cudf::test::fixed_point_column_wrapper<__int128_t> const decimal128_col(
    {static_cast<__int128>(0),
     static_cast<__int128>(100),
     static_cast<__int128>(-1),
     (static_cast<__int128>(0xFFFF'FFFF'FCC4'D1C3u) << 64 | 0x602F'7FC3'1800'0001u),
     (static_cast<__int128>(0x0785'EE10'D5DA'46D9u) << 64 | 0x00F4'369F'FFFF'FFFFu)},
    numeric::scale_type{-11});

  auto const hash_structs =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({structs_col}), 42);
  auto const hash_strings =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({strings_col}), 42);
  auto const hash_doubles =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({doubles_col}), 42);
  auto const hash_timestamps =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({timestamps_col}), 42);
  auto const hash_decimal64 =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({decimal64_col}), 42);
  auto const hash_longs =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({longs_col}), 42);
  auto const hash_floats =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({floats_col}), 42);
  auto const hash_dates =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({dates_col}), 42);
  auto const hash_decimal32 =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({decimal32_col}), 42);
  auto const hash_ints = cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({ints_col}), 42);
  auto const hash_shorts =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({shorts_col}), 42);
  auto const hash_bytes =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({bytes_col}), 42);
  auto const hash_bools1 =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({bools_col1}), 42);
  auto const hash_bools2 =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({bools_col2}), 42);
  auto const hash_decimal128 =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({decimal128_col}), 42);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_structs, hash_structs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_doubles, hash_doubles_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_timestamps, hash_timestamps_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal64, hash_decimal64_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_longs, hash_longs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_floats, hash_floats_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_dates, hash_dates_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal32, hash_decimal32_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_ints, hash_ints_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_shorts, hash_shorts_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bytes, hash_bytes_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools1, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools2, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal128, hash_decimal128_expected, verbosity);

  auto const combined_table = cudf::table_view({structs_col,
                                                strings_col,
                                                doubles_col,
                                                timestamps_col,
                                                decimal64_col,
                                                longs_col,
                                                floats_col,
                                                dates_col,
                                                decimal32_col,
                                                ints_col,
                                                shorts_col,
                                                bytes_col,
                                                bools_col2,
                                                decimal128_col});
  auto const hash_combined  = cudf::hashing::spark_murmurhash3_x86_32(combined_table, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_combined, hash_combined_expected, verbosity);
}

TEST_F(SparkMurmurHashTest, StringsWithSeed)
{
  // The hash values were determined by running the following Scala code in Apache Spark:
  // val strs = Seq("", "The quick brown fox",
  //              "jumps over the lazy dog.",
  //              "All work and no play makes Jack a dull boy",
  //              "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721")
  // println(strs.map(org.apache.spark.unsafe.types.UTF8String.fromString)
  //   .map(org.apache.spark.sql.catalyst.expressions.Murmur3HashFunction.hash(
  //     _, org.apache.spark.sql.types.StringType, 314)))

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected_seed_314(
    {1467149710, 723257560, -1620282500, -2001858707, 1588473657});

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});

  auto const hash_strings =
    cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({strings_col}), 314);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected_seed_314, verbosity);
}

TEST_F(SparkMurmurHashTest, ListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists",ArrayType(ArrayType(IntegerType)))

  val data = Seq(
    Row(null),
    Row(List(null)),
    Row(List(List())),
    Row(List(List(1))),
    Row(List(List(1, 2))),
    Row(List(List(1, 2, 3))),
    Row(List(List(1, 2), List(3))),
    Row(List(List(1), List(2, 3))),
    Row(List(List(1), List(null, 2, 3))),
    Row(List(List(1, 2), List(3), List(null))),
    Row(List(List(1, 2), null, List(3))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto nested_list =
    cudf::test::lists_column_wrapper<int>({{},
                                           {1},
                                           {1, 2},
                                           {1, 2, 3},
                                           {1, 2},
                                           {3},
                                           {1},
                                           {2, 3},
                                           {1},
                                           {{null, 2, 3}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {3},
                                           {{null}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {},
                                           {3}},
                                          cudf::test::iterators::nulls_at({0, 14}));
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 16};
  auto list_validity = cudf::test::iterators::nulls_at({0});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_validity, list_validity + 11);
  auto list_column = cudf::make_lists_column(
    11, offsets.release(), nested_list.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{42,
                                                                42,
                                                                42,
                                                                -559580957,
                                                                -222940379,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097};

  auto output = cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({*list_column}), 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHashTest, StructOfListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("structs", new StructType()
        .add("a", ArrayType(IntegerType))
        .add("b", ArrayType(IntegerType)))

  val data = Seq(
    Row(Row(List(), List())),
    Row(Row(List(0), List(0))),
    Row(Row(List(1, null), null)),
    Row(Row(List(1, null), List())),
    Row(Row(List(), List(null, 1))),
    Row(Row(null, List(1))),
    Row(Row(List(2, 3), List(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("structs", "hash(structs) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1 =
    cudf::test::lists_column_wrapper<int>({{},
                                           {0},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {},
                                           {} /*NULL*/,
                                           {2, 3}},
                                          cudf::test::iterators::nulls_at({5}));
  auto col2 = cudf::test::lists_column_wrapper<int>(
    {{}, {0}, {} /*NULL*/, {}, {{null, 1}, cudf::test::iterators::nulls_at({0})}, {1}, {4, 5}},
    cudf::test::iterators::nulls_at({2}));
  auto struct_column = cudf::test::structs_column_wrapper{{col1, col2}};

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    42, 59727262, -559580957, -559580957, -559580957, -559580957, 170038658};

  auto output = cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({struct_column}), 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHashTest, ListOfStructValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists", ArrayType(new StructType()
      .add("a", IntegerType)
      .add("b", IntegerType)))

  val data = Seq(
    Row(List(Row(0, 0))),
    Row(List(null)),
    Row(List(Row(null, null))),
    Row(List(Row(1, null))),
    Row(List(Row(null, 1))),
    Row(List(Row(null, 1), Row(2, 3))),
    Row(List(Row(2, 3), null)),
    Row(List(Row(2, 3), Row(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1       = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, 1, null, null, 2, 2, null, 2, 4},
    cudf::test::iterators::nulls_at({1, 2, 4, 5, 8}));
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, null, 1, 1, 3, 3, null, 3, 5}, cudf::test::iterators::nulls_at({1, 2, 3, 8}));
  auto struct_column =
    cudf::test::structs_column_wrapper{{col1, col2}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1}};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3, 4, 5, 7, 9, 11};
  auto list_nullmask = std::vector<bool>(8, true);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    8, offsets.release(), struct_column.release(), null_count, std::move(null_mask));

  // Lists of structs are rejected, matching the Spark RAPIDS JNI implementation, which raises the
  // same error from check_hash_compatibility.
  // https://github.com/NVIDIA/spark-rapids-jni/blob/09bae9c7dccf050b5db53a70282c3001cba5a015/src/main/cpp/src/hash/murmur_hash.cu#L167-L182
  EXPECT_THROW(cudf::hashing::spark_murmurhash3_x86_32(cudf::table_view({*list_column}), 42),
               cudf::logic_error);
}
