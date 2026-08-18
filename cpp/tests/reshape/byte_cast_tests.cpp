/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/reshape.hpp>

class ByteCastTest : public cudf::test::BaseFixture {};

TEST_F(ByteCastTest, int16ValuesWithSplit)
{
  using limits = std::numeric_limits<int16_t>;
  cudf::test::fixed_width_column_wrapper<int16_t> const int16_col(
    {short(0), short(100), short(-100), limits::min(), limits::max()});
  cudf::test::lists_column_wrapper<uint8_t> const int16_expected(
    {{0x00, 0x00}, {0x64, 0x00}, {0x9c, 0xff}, {0x00, 0x80}, {0xff, 0x7f}});
  cudf::test::lists_column_wrapper<uint8_t> const int16_expected_slice1(
    {{0x00, 0x00}, {0x00, 0x64}, {0xff, 0x9c}});
  cudf::test::lists_column_wrapper<uint8_t> const int16_expected_slice2(
    {{0x80, 0x00}, {0x7f, 0xff}});

  std::vector<cudf::size_type> splits({3});
  std::vector<cudf::column_view> split_column = cudf::split(int16_col, splits);

  auto const output_int16        = cudf::byte_cast(int16_col, cudf::flip_endianness::NO);
  auto const output_int16_slice1 = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::YES);
  auto const output_int16_slice2 = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16->view(), int16_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16_slice1->view(), int16_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16_slice2->view(), int16_expected_slice2);
}

TEST_F(ByteCastTest, int16ValuesWithNulls)
{
  using limits      = std::numeric_limits<int16_t>;
  auto odd_validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int16_t> const int16_col(
    {short(0), short(100), short(-100), limits::min(), limits::max()},
    {false, true, false, true, false});

  auto int16_data = cudf::test::fixed_width_column_wrapper<uint8_t>{0x00, 0x64, 0x80, 0x00};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(odd_validity, odd_validity + 5);
  auto int16_expected          = cudf::make_lists_column(
    5,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 2, 2, 4, 4}.release(),
    int16_data.release(),
    null_count,
    std::move(null_mask));

  auto const output_int16 = cudf::byte_cast(int16_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int16->view(), int16_expected->view());
}

TEST_F(ByteCastTest, int32Values)
{
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const int32_col(
    {0, 100, -100, limits::min(), limits::max()});
  cudf::test::lists_column_wrapper<uint8_t> const int32_expected_flipped(
    {{0x00, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0x00, 0x64},
     {0xff, 0xff, 0xff, 0x9c},
     {0x80, 0x00, 0x00, 0x00},
     {0x7f, 0xff, 0xff, 0xff}});
  cudf::test::lists_column_wrapper<uint8_t> const int32_expected({{0x00, 0x00, 0x00, 0x00},
                                                                  {0x64, 0x00, 0x00, 0x00},
                                                                  {0x9c, 0xff, 0xff, 0xff},
                                                                  {0x00, 0x00, 0x00, 0x80},
                                                                  {0xff, 0xff, 0xff, 0x7f}});

  auto const output_int32_flipped = cudf::byte_cast(int32_col, cudf::flip_endianness::YES);
  auto const output_int32         = cudf::byte_cast(int32_col, cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32_flipped->view(), int32_expected_flipped);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32->view(), int32_expected);
}

TEST_F(ByteCastTest, int32ValuesWithNulls)
{
  using limits       = std::numeric_limits<int32_t>;
  auto even_validity = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int32_t> const int32_col(
    {0, 100, -100, limits::min(), limits::max()}, {true, false, true, false, true});

  auto int32_data = cudf::test::fixed_width_column_wrapper<uint8_t>{
    0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x9c, 0x7f, 0xff, 0xff, 0xff};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(even_validity, even_validity + 5);

  auto int32_expected = cudf::make_lists_column(
    5,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 4, 4, 8, 8, 12}.release(),
    int32_data.release(),
    null_count,
    std::move(null_mask));

  auto const output_int32 = cudf::byte_cast(int32_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int32->view(), int32_expected->view());
}

TEST_F(ByteCastTest, int64ValuesWithSplit)
{
  using limits = std::numeric_limits<int64_t>;
  cudf::test::fixed_width_column_wrapper<int64_t> const int64_col(
    {long(0), long(100), long(-100), limits::min(), limits::max()});
  cudf::test::lists_column_wrapper<uint8_t> const int64_expected_flipped(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9c},
     {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});
  cudf::test::lists_column_wrapper<uint8_t> const int64_expected_slice1(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x9c, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});
  cudf::test::lists_column_wrapper<uint8_t> const int64_expected_slice2(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f}});

  std::vector<cudf::size_type> splits({3});
  std::vector<cudf::column_view> split_column = cudf::split(int64_col, splits);

  auto const output_int64_flipped = cudf::byte_cast(int64_col, cudf::flip_endianness::YES);
  auto const output_int64_slice1  = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::NO);
  auto const output_int64_slice2  = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_flipped->view(), int64_expected_flipped);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_slice1->view(), int64_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_slice2->view(), int64_expected_slice2);
}

TEST_F(ByteCastTest, int64ValuesWithNulls)
{
  using limits      = std::numeric_limits<int64_t>;
  auto odd_validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int64_t> const int64_col(
    {long(0), long(100), long(-100), limits::min(), limits::max()},
    {false, true, false, true, false});

  auto int64_data = cudf::test::fixed_width_column_wrapper<uint8_t>{
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(odd_validity, odd_validity + 5);
  auto int64_expected          = cudf::make_lists_column(
    5,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 8, 8, 16, 16}.release(),
    int64_data.release(),
    null_count,
    std::move(null_mask));

  auto const output_int64 = cudf::byte_cast(int64_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int64->view(), int64_expected->view());
}

TEST_F(ByteCastTest, fp32ValuesWithSplit)
{
  using limits = std::numeric_limits<float>;
  float nan    = limits::quiet_NaN();
  float inf    = limits::infinity();
  cudf::test::fixed_width_column_wrapper<float> const fp32_col(
    {float(0.0), float(100.0), float(-100.0), limits::min(), limits::max(), nan, -nan, inf, -inf});
  cudf::test::lists_column_wrapper<uint8_t> const fp32_expected({{0x00, 0x00, 0x00, 0x00},
                                                                 {0x00, 0x00, 0xc8, 0x42},
                                                                 {0x00, 0x00, 0xc8, 0xc2},
                                                                 {0x00, 0x00, 0x80, 0x00},
                                                                 {0xff, 0xff, 0x7f, 0x7f},
                                                                 {0x00, 0x00, 0xc0, 0x7f},
                                                                 {0x00, 0x00, 0xc0, 0xff},
                                                                 {0x00, 0x00, 0x80, 0x7f},
                                                                 {0x00, 0x00, 0x80, 0xff}});
  cudf::test::lists_column_wrapper<uint8_t> const fp32_expected_slice1({{0x00, 0x00, 0x00, 0x00},
                                                                        {0x42, 0xc8, 0x00, 0x00},
                                                                        {0xc2, 0xc8, 0x00, 0x00},
                                                                        {0x00, 0x80, 0x00, 0x00},
                                                                        {0x7f, 0x7f, 0xff, 0xff}});
  cudf::test::lists_column_wrapper<uint8_t> const fp32_expected_slice2({{0x7f, 0xc0, 0x00, 0x00},
                                                                        {0xff, 0xc0, 0x00, 0x00},
                                                                        {0x7f, 0x80, 0x00, 0x00},
                                                                        {0xff, 0x80, 0x00, 0x00}});

  std::vector<cudf::size_type> splits({5});
  std::vector<cudf::column_view> split_column = cudf::split(fp32_col, splits);

  auto const output_fp32        = cudf::byte_cast(fp32_col, cudf::flip_endianness::NO);
  auto const output_fp32_slice1 = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::YES);
  auto const output_fp32_slice2 = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32->view(), fp32_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32_slice1->view(), fp32_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32_slice2->view(), fp32_expected_slice2);
}

TEST_F(ByteCastTest, fp32ValuesWithNulls)
{
  using limits       = std::numeric_limits<float>;
  auto even_validity = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<float> const fp32_col(
    {float(0.0), float(100.0), float(-100.0), limits::min(), limits::max()},
    {true, false, true, false, true});

  auto fp32_data = cudf::test::fixed_width_column_wrapper<uint8_t>{
    0x00, 0x00, 0x00, 0x00, 0xc2, 0xc8, 0x00, 0x00, 0x7f, 0x7f, 0xff, 0xff};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(even_validity, even_validity + 5);
  auto fp32_expected = cudf::make_lists_column(
    5,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 4, 4, 8, 8, 12}.release(),
    fp32_data.release(),
    null_count,
    std::move(null_mask));

  auto const output_fp32 = cudf::byte_cast(fp32_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_fp32->view(), fp32_expected->view());
}

TEST_F(ByteCastTest, fp64ValuesWithSplit)
{
  using limits = std::numeric_limits<double>;
  double nan   = limits::quiet_NaN();
  double inf   = limits::infinity();
  cudf::test::fixed_width_column_wrapper<double> const fp64_col({double(0.0),
                                                                 double(100.0),
                                                                 double(-100.0),
                                                                 limits::min(),
                                                                 limits::max(),
                                                                 nan,
                                                                 -nan,
                                                                 inf,
                                                                 -inf});
  cudf::test::lists_column_wrapper<uint8_t> const fp64_flipped_expected(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x40, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xc0, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
     {0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xff, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}});
  cudf::test::lists_column_wrapper<uint8_t> const fp64_expected_slice1(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0xc0},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7f}});
  cudf::test::lists_column_wrapper<uint8_t> const fp64_expected_slice2(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0xff},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff}});

  std::vector<cudf::size_type> splits({5});
  std::vector<cudf::column_view> split_column = cudf::split(fp64_col, splits);

  auto const output_fp64_flipped = cudf::byte_cast(fp64_col, cudf::flip_endianness::YES);
  auto const output_fp64_slice1  = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::NO);
  auto const output_fp64_slice2  = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_flipped->view(), fp64_flipped_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_slice1->view(), fp64_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_slice2->view(), fp64_expected_slice2);
}

TEST_F(ByteCastTest, fp64ValuesWithNulls)
{
  using limits      = std::numeric_limits<double>;
  auto odd_validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<double> const fp64_col(
    {double(0.0), double(100.0), double(-100.0), limits::min(), limits::max()},
    {false, true, false, true, false});

  auto fp64_data = cudf::test::fixed_width_column_wrapper<uint8_t>{
    0x40, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(odd_validity, odd_validity + 5);
  auto fp64_expected           = cudf::make_lists_column(
    5,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 8, 8, 16, 16}.release(),
    fp64_data.release(),
    null_count,
    std::move(null_mask));

  auto const output_fp64 = cudf::byte_cast(fp64_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_fp64->view(), fp64_expected->view());
}

TEST_F(ByteCastTest, StringValuesNoNulls)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"", "The quick", " brown fox...", "!\"#$%&\'()*+,-./", "0123456789:;<=>?@", "[\\]^_`{|}~"});
  cudf::test::lists_column_wrapper<uint8_t> const strings_expected(
    {{},
     {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b},
     {0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f, 0x78, 0x2e, 0x2e, 0x2e},
     {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f},
     {0x30,
      0x31,
      0x32,
      0x33,
      0x34,
      0x35,
      0x36,
      0x37,
      0x38,
      0x39,
      0x3a,
      0x3b,
      0x3c,
      0x3d,
      0x3e,
      0x3f,
      0x40},
     {0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x7b, 0x7c, 0x7d, 0x7e}});

  auto const output_strings = cudf::byte_cast(strings_col, cudf::flip_endianness::YES);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_strings->view(), strings_expected);
}

TEST_F(ByteCastTest, StringValuesWithNulls)
{
  auto const strings_col = [] {
    auto output =
      cudf::test::strings_column_wrapper(
        {"", "The quick", " brown fox...", "!\"#$%&\'()*+,-./", "0123456789:;<=>?@", "[\\]^_`{|}~"})
        .release();

    // Set nulls by `set_null_mask` so the output column will have non-empty nulls.
    // This is intentional.
    auto const null_iter = cudf::test::iterators::nulls_at({2, 4});
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(null_iter, null_iter + output->size());
    output->set_null_mask(std::move(null_mask), null_count);
    return output;
  }();

  auto const strings_expected = cudf::test::lists_column_wrapper<uint8_t>{
    {{},
     {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b},
     {} /*NULL*/,
     {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f},
     {} /*NULL*/,
     {0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x7b, 0x7c, 0x7d, 0x7e}},
    cudf::test::iterators::nulls_at({2, 4})};

  auto const output_strings = cudf::byte_cast(*strings_col, cudf::flip_endianness::YES);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_strings->view(), strings_expected);
}

TEST_F(ByteCastTest, int32Empty)
{
  auto const input    = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto const expected = cudf::test::lists_column_wrapper<uint8_t>{};
  auto const output   = cudf::byte_cast(input, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);
}

TEST_F(ByteCastTest, int32sAllNulls)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0}, cudf::test::iterators::all_nulls()};
  auto const output     = cudf::byte_cast(input, cudf::flip_endianness::YES);
  auto const& out_child = output->child(cudf::lists_column_view::child_column_index);
  EXPECT_EQ(output->size(), 3);
  EXPECT_EQ(output->null_count(), 3);
  EXPECT_EQ(out_child.size(), 0);
  EXPECT_EQ(out_child.type().id(), cudf::type_id::UINT8);
}

TEST_F(ByteCastTest, StringEmpty)
{
  auto const input    = cudf::test::strings_column_wrapper{};
  auto const expected = cudf::test::lists_column_wrapper<uint8_t>{};
  auto const output   = cudf::byte_cast(input, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);
}

TEST_F(ByteCastTest, StringsAllNulls)
{
  auto const input =
    cudf::test::strings_column_wrapper{{"", "", ""}, cudf::test::iterators::all_nulls()};
  auto const output     = cudf::byte_cast(input, cudf::flip_endianness::YES);
  auto const& out_child = output->child(cudf::lists_column_view::child_column_index);
  EXPECT_EQ(output->size(), 3);
  EXPECT_EQ(output->null_count(), 3);
  EXPECT_EQ(out_child.size(), 0);
  EXPECT_EQ(out_child.type().id(), cudf::type_id::UINT8);
}
