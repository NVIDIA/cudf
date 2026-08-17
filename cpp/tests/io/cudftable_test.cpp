/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_test_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct CudftableTest : public cudf::test::BaseFixture {
  void run_test_file(cudf::table_view const& expected)
  {
    auto const filepath = temp_env->get_temp_filepath("test.cudftbl");

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{filepath},
                                                                expected)
        .build());

    auto result = cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }

  void run_test_buffer(cudf::table_view const& expected)
  {
    std::vector<char> buffer;

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer},
                                                                expected)
        .build());

    auto host_buffer = cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
    auto result = cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
        .build());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }

  void run_test(cudf::table_view const& t)
  {
    run_test_file(t);
    run_test_buffer(t);
  }

  void run_roundtrip(cudf::table_view const& expected,
                     cudf::io::compression_type compression,
                     uint32_t block_size = 256 * 1024)
  {
    std::vector<char> buffer;

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer},
                                                                expected)
        .compression(compression)
        .block_size(block_size)
        .build());

    auto host_buffer = cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
    auto result = cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
        .build());

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }

  static cudf::table_view make_sample_table()
  {
    static cudf::test::fixed_width_column_wrapper<int32_t> col1(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {true, true, true, false, true, false, true, true, true, true});
    static cudf::test::fixed_width_column_wrapper<double> col2(
      {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10});
    static cudf::test::strings_column_wrapper col3(
      {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"},
      {true, true, false, true, true, true, true, false, true, true});
    return cudf::table_view{{col1, col2, col3}};
  }

  // Returns the packed (uncompressed) data length by writing the table once with
  // no compression and reading the `uncompressed_data_length` header field. This
  // lets tests target block sizes exactly around the packed-data boundary.
  static uint64_t packed_data_length(cudf::table_view const& t)
  {
    std::vector<char> buffer;
    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer}, t)
        .compression(cudf::io::compression_type::NONE)
        .build());
    // Header: magic(4)+version(4)+compression(4)+block_size(4)+metadata_version(4)
    //   +reserved(4)+metadata_length(8) -> uncompressed_data_length at offset 32.
    uint64_t len{};
    std::memcpy(&len, buffer.data() + 32, sizeof(uint64_t));
    return len;
  }
};

TEST_F(CudftableTest, SingleColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7},
                                                      {true, true, true, false, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CudftableTest, SingleColumnFixedWidthNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CudftableTest, MultiColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> col2({7, 8, 6, 5, 4, 3, 2},
                                                     {true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<double> col3({8, 4, 2, 0, 7, 1, 3},
                                                      {false, true, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<bool> col4({true, false, true, false, true, false, true},
                                                    {true, true, false, true, true, true, false});

  auto const expected = cudf::table_view{{col1, col2, col3, col4}};
  run_test(expected);
}

TEST_F(CudftableTest, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col({});

  auto const expected = cudf::table_view{{empty_col}};
  run_test(expected);
}

TEST_F(CudftableTest, EmptyTable)
{
  auto const expected = cudf::table_view{std::vector<cudf::column_view>{}};
  run_test(expected);
}

TEST_F(CudftableTest, MultiColumnCompound)
{
  cudf::test::strings_column_wrapper string_col({"Lorem", "ipsum", "dolor", "sit"},
                                                {true, false, true, true});

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::lists_column_wrapper<int32_t> list_col{
    {{1, 2, 3}, valids}, {4, 5}, {}, {{6, 7, 8, 9}, valids}};

  cudf::test::fixed_width_column_wrapper<int32_t> struct_col1{{1, 2, 3, 4},
                                                              {true, false, true, true}};
  cudf::test::strings_column_wrapper struct_col2{{"a", "b", "", "d"}, {true, true, false, true}};
  cudf::test::structs_column_wrapper struct_col{{struct_col1, struct_col2},
                                                {true, true, true, false}};

  auto const expected = cudf::table_view{{string_col, list_col, struct_col}};
  run_test(expected);
}

TEST_F(CudftableTest, LargeTable)
{
  constexpr int num_rows = 23'456'789;

  auto col1 = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
  auto col2 = cudf::sequence(num_rows, cudf::numeric_scalar<double>(0.0));

  auto const expected = cudf::table_view{{col1->view(), col2->view()}};
  run_test(expected);
}

TEST_F(CudftableTest, AllNullColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5},
                                                      cudf::test::iterators::all_nulls());

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CudftableTest, MixedNullability)
{
  cudf::test::fixed_width_column_wrapper<int32_t> nullable_col({1, 2, 3, 4, 5},
                                                               {true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> non_nullable_col({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  cudf::test::strings_column_wrapper string_col({"a", "b", "c", "d", "e"},
                                                {true, true, false, true, true});

  auto const expected = cudf::table_view{{nullable_col, non_nullable_col, string_col}};
  run_test(expected);
}

TEST_F(CudftableTest, InvalidHeaderMagic)
{
  auto const filepath = temp_env->get_temp_filepath("invalid.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .build());

  // Corrupt the magic number
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  uint32_t bad_magic = 0xDEADBEEF;
  file.write(reinterpret_cast<char*>(&bad_magic), sizeof(uint32_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, InvalidHeaderVersion)
{
  auto const filepath = temp_env->get_temp_filepath("invalid_version.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .build());

  // Corrupt the version
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(uint32_t));  // Skip magic
  uint32_t bad_version = 999;
  file.write(reinterpret_cast<char*>(&bad_version), sizeof(uint32_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, TruncatedFile)
{
  auto const filepath = temp_env->get_temp_filepath("truncated.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .build());

  // Truncate the file
  std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
  file.write("TRUNCATED", 9);
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, IntegralTypes)
{
  // Test all integral types in a single table
  cudf::test::fixed_width_column_wrapper<int8_t> int8_col({-128, -64, 0, 64, 127},
                                                          {true, true, false, true, true});
  cudf::test::fixed_width_column_wrapper<uint8_t> uint8_col({0, 1, 255, 128, 64},
                                                            {true, true, true, false, true});
  cudf::test::fixed_width_column_wrapper<uint16_t> uint16_col({0, 1, 65535, 32768, 16384},
                                                              {true, true, true, false, true});
  std::vector<uint32_t> uint32_data{0, 1, 4294967295U, 2147483648U, 1073741824U};
  std::vector<bool> uint32_validity{true, true, true, false, true};
  cudf::test::fixed_width_column_wrapper<uint32_t> uint32_col(
    uint32_data.begin(), uint32_data.end(), uint32_validity.begin());
  cudf::test::fixed_width_column_wrapper<uint64_t> uint64_col(
    {0ULL, 1ULL, 18446744073709551615ULL, 9223372036854775808ULL, 4611686018427387904ULL},
    {true, true, true, false, true});

  auto const expected = cudf::table_view{{int8_col, uint8_col, uint16_col, uint32_col, uint64_col}};
  run_test(expected);
}

TEST_F(CudftableTest, TimestampTypes)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_D> timestamp_days_col{
    cudf::timestamp_D{cudf::duration_D{0}},
    cudf::timestamp_D{cudf::duration_D{100}},
    cudf::timestamp_D{cudf::duration_D{200}},
    cudf::timestamp_D{cudf::duration_D{300}}};

  fixed_width_column_wrapper<cudf::timestamp_s> timestamp_seconds_col{cudf::timestamp_s{0s},
                                                                      cudf::timestamp_s{100s},
                                                                      cudf::timestamp_s{200s},
                                                                      cudf::timestamp_s{300s}};

  fixed_width_column_wrapper<cudf::timestamp_ms> timestamp_milliseconds_col{
    cudf::timestamp_ms{0ms},
    cudf::timestamp_ms{100ms},
    cudf::timestamp_ms{200ms},
    cudf::timestamp_ms{300ms}};

  fixed_width_column_wrapper<cudf::timestamp_us> timestamp_microseconds_col{
    cudf::timestamp_us{0us},
    cudf::timestamp_us{100us},
    cudf::timestamp_us{200us},
    cudf::timestamp_us{300us}};

  fixed_width_column_wrapper<cudf::timestamp_ns> timestamp_nanoseconds_col{
    cudf::timestamp_ns{0ns},
    cudf::timestamp_ns{100ns},
    cudf::timestamp_ns{200ns},
    cudf::timestamp_ns{300ns}};

  auto const expected = cudf::table_view{{timestamp_days_col,
                                          timestamp_seconds_col,
                                          timestamp_milliseconds_col,
                                          timestamp_microseconds_col,
                                          timestamp_nanoseconds_col}};
  run_test(expected);
}

TEST_F(CudftableTest, DurationTypes)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_D> duration_days_col{
    cudf::duration_D{0}, cudf::duration_D{100}, cudf::duration_D{200}, cudf::duration_D{300}};

  fixed_width_column_wrapper<cudf::duration_s> duration_seconds_col{
    cudf::duration_s{0s}, cudf::duration_s{100s}, cudf::duration_s{200s}, cudf::duration_s{300s}};

  fixed_width_column_wrapper<cudf::duration_ms> duration_milliseconds_col{cudf::duration_ms{0ms},
                                                                          cudf::duration_ms{100ms},
                                                                          cudf::duration_ms{200ms},
                                                                          cudf::duration_ms{300ms}};

  fixed_width_column_wrapper<cudf::duration_us> duration_microseconds_col{
    cudf::duration_us{0us},
    cudf::duration_us{1000us},
    cudf::duration_us{2000us},
    cudf::duration_us{3000us}};

  fixed_width_column_wrapper<cudf::duration_ns> duration_nanoseconds_col{cudf::duration_ns{0ns},
                                                                         cudf::duration_ns{1000ns},
                                                                         cudf::duration_ns{2000ns},
                                                                         cudf::duration_ns{3000ns}};

  auto const expected = cudf::table_view{{duration_days_col,
                                          duration_seconds_col,
                                          duration_milliseconds_col,
                                          duration_microseconds_col,
                                          duration_nanoseconds_col}};
  run_test(expected);
}

TEST_F(CudftableTest, DecimalTypes)
{
  using namespace numeric;

  cudf::test::fixed_point_column_wrapper<int32_t> decimal32_col{
    {12345, -67890, 0, 99999}, {true, true, false, true}, scale_type{2}};

  cudf::test::fixed_point_column_wrapper<int64_t> decimal64_col{
    {123456789012345LL, -987654321098765LL, 0LL, 111LL}, {true, true, false, true}, scale_type{-5}};

  __int128_t val1 = 1234567890123456789LL;
  auto val2       = static_cast<__int128_t>(9876543210987654321ULL);
  val2            = -val2;
  __int128_t val3 = val1 * 1000LL;
  __int128_t val4 = 0;
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128_col1{
    {val1, val2, val3, val4}, {true, true, true, false}, scale_type{5}};

  __int128_t val5 = 12345;
  __int128_t val6 = -67890;
  __int128_t val7 = 999999999999999999LL;
  __int128_t val8 = 0;
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128_col2{
    {val5, val6, val7, val8}, {true, true, true, true}, scale_type{0}};

  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128_col3{
    {val5, val6, val7, val8}, {true, true, true, true}, scale_type{-10}};

  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128_col4{
    {val5, val6, val7, val8}, {true, true, true, true}, scale_type{15}};

  auto const expected = cudf::table_view{{decimal32_col,
                                          decimal64_col,
                                          decimal128_col1,
                                          decimal128_col2,
                                          decimal128_col3,
                                          decimal128_col4}};
  run_test(expected);
}

TEST_F(CudftableTest, DictionaryColumn)
{
  // Dictionary columns: pack/unpack does not preserve dictionary structure correctly
  // This test is expected to fail until pack/unpack supports dictionary columns
  GTEST_SKIP() << "Dictionary columns not supported by pack/unpack - expected failure";

  cudf::test::fixed_width_column_wrapper<int64_t> keys{10, 20, 30, 40};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{{0, 1, 2, 0, 1, 3},
                                                          {true, true, false, true, true, true}};

  auto dictionary     = cudf::make_dictionary_column(keys, indices);
  auto const expected = cudf::table_view{{dictionary->view()}};
  run_test(expected);
}

TEST_F(CudftableTest, Lists)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;
  using namespace numeric;

  cudf::test::lists_column_wrapper<int32_t> child_list{{1, 2}, {3, 4}, {5, 6, 7}, {8}, {}, {9, 10}};
  auto lists_of_lists_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 6, 6}.release();
  auto lists_of_lists_col = cudf::make_lists_column(
    4, std::move(lists_of_lists_offsets), child_list.release(), 0, rmm::device_buffer{});

  cudf::test::strings_column_wrapper strings_child{"hello", "world", "foo", "bar", "baz", "test"};
  auto strings_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 5, 5, 6}.release();
  auto lists_of_strings_col = cudf::make_lists_column(
    4, std::move(strings_offsets), strings_child.release(), 0, rmm::device_buffer{});

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms> timestamps_child{
    cudf::timestamp_ms{100ms},
    cudf::timestamp_ms{200ms},
    cudf::timestamp_ms{300ms},
    cudf::timestamp_ms{400ms},
    cudf::timestamp_ms{500ms},
    cudf::timestamp_ms{600ms}};
  auto timestamps_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 3, 6}.release();
  auto lists_of_timestamps_col = cudf::make_lists_column(
    4, std::move(timestamps_offsets), timestamps_child.release(), 0, rmm::device_buffer{});

  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_child{
    cudf::duration_ns{1000ns},
    cudf::duration_ns{2000ns},
    cudf::duration_ns{3000ns},
    cudf::duration_ns{4000ns},
    cudf::duration_ns{5000ns}};
  auto durations_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 3, 5}.release();
  auto lists_of_durations_col = cudf::make_lists_column(
    4, std::move(durations_offsets), durations_child.release(), 0, rmm::device_buffer{});

  cudf::test::fixed_point_column_wrapper<int32_t> decimal32_child{
    {12345, -67890, 99999, 0}, {true, true, true, true}, scale_type{2}};
  auto decimal32_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 3, 4}.release();
  auto lists_of_decimals_col = cudf::make_lists_column(
    4, std::move(decimal32_offsets), decimal32_child.release(), 0, rmm::device_buffer{});

  cudf::test::fixed_width_column_wrapper<int32_t> struct_col1{1, 2, 3, 4, 5, 6};
  cudf::test::strings_column_wrapper struct_col2{"a", "b", "c", "d", "e", "f"};
  cudf::test::structs_column_wrapper struct_col{{struct_col1, struct_col2}};
  auto lists_of_structs_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 3, 4, 6}.release();
  auto lists_of_structs_col = cudf::make_lists_column(
    4, std::move(lists_of_structs_offsets), struct_col.release(), 0, rmm::device_buffer{});

  auto const expected = cudf::table_view{{lists_of_lists_col->view(),
                                          lists_of_strings_col->view(),
                                          lists_of_timestamps_col->view(),
                                          lists_of_durations_col->view(),
                                          lists_of_decimals_col->view(),
                                          lists_of_structs_col->view()}};
  run_test(expected);
}

TEST_F(CudftableTest, StructsContainingLists)
{
  cudf::test::lists_column_wrapper<int32_t> list_col{{1, 2, 3}, {4, 5}, {}, {6}};
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{10, 20, 30, 40};
  cudf::test::structs_column_wrapper struct_col{{list_col, int_col}};

  auto const expected = cudf::table_view{{struct_col}};
  run_test(expected);
}

TEST_F(CudftableTest, DeepNestingStructs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> level4_col{10, 20};
  cudf::test::strings_column_wrapper level4_str{"x", "y"};
  cudf::test::structs_column_wrapper level4_struct{{level4_col, level4_str}};

  cudf::test::fixed_width_column_wrapper<float> level3_col{1.1f, 2.2f};
  cudf::test::structs_column_wrapper level3_struct{{level4_struct, level3_col}};

  cudf::test::fixed_width_column_wrapper<double> level2_col{100.5, 200.5};
  cudf::test::structs_column_wrapper level2_struct{{level3_struct, level2_col}};

  cudf::test::fixed_width_column_wrapper<int64_t> level1_col{1000LL, 2000LL};
  cudf::test::structs_column_wrapper level1_struct{{level2_struct, level1_col}};

  auto const expected = cudf::table_view{{level1_struct}};
  run_test(expected);
}

TEST_F(CudftableTest, DeepNestingLists)
{
  cudf::test::lists_column_wrapper<int32_t> level3_list{{1, 2}, {3, 4}, {5}, {6, 7, 8}};

  auto level2_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4}.release();
  auto level2_list    = cudf::make_lists_column(
    2, std::move(level2_offsets), level3_list.release(), 0, rmm::device_buffer{});

  auto level1_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2}.release();
  auto level1_list    = cudf::make_lists_column(
    1, std::move(level1_offsets), std::move(level2_list), 0, rmm::device_buffer{});

  auto const expected = cudf::table_view{{level1_list->view()}};
  run_test(expected);
}

TEST_F(CudftableTest, DeviceBufferSource)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});

  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&buffer}, expected)
                                            .build());

  rmm::device_buffer device_buffer(buffer.size(), cudf::get_default_stream());
  auto const stream = cudf::get_default_stream();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    device_buffer.data(), buffer.data(), buffer.size(), cudaMemcpyHostToDevice, stream.value()));
  // Ensure the data is copied to the device before the host read, because the host read does not
  // take the stream
  stream.synchronize();

  auto device_span = cudf::device_span<std::byte const>(
    static_cast<std::byte const*>(device_buffer.data()), device_buffer.size());
  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{device_span})
      .build());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, UnicodeStrings)
{
  cudf::test::strings_column_wrapper col({"Hello", "世界", "🌍", "café", "résumé"},
                                         {true, true, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CudftableTest, LongStringColumns)
{
  // Set threshold to enable int64 offsets for strings exceeding this size
  constexpr int64_t threshold = 10000;
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

  std::string long_str1(threshold + 1000, 'a');
  std::string long_str2(threshold + 5000, 'b');
  std::string short_str = "short";
  cudf::test::strings_column_wrapper col({long_str1, short_str, long_str2, long_str1 + "suffix"});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);

  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}

TEST_F(CudftableTest, ManyColumns)
{
  constexpr int num_cols = 12'345;
  std::vector<cudf::column_view> columns;
  for (int i = 0; i < num_cols; ++i) {
    cudf::test::fixed_width_column_wrapper<int32_t> col({i % 10, (i + 1) % 10, (i + 2) % 10});
    columns.push_back(col);
  }

  cudf::table_view expected(columns);
  run_test(expected);
}

TEST_F(CudftableTest, FileTooSmallForHeader)
{
  auto const filepath = temp_env->get_temp_filepath("too_small.cudftbl");

  // Create a file that's too small to contain a header
  std::ofstream file(filepath, std::ios::binary);
  file.write("SMALL", 5);
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, CorruptedMetadataLength)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_meta.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .build());

  // Get the actual file size
  std::ifstream size_check(filepath, std::ios::binary | std::ios::ate);
  auto const file_size = size_check.tellg();
  size_check.close();

  // Corrupt the metadata length (set to a value larger than the file).
  // metadata_length lives at offset 24 in the header (after magic, version,
  // compression, block_size, metadata_version, reserved).
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(24);
  uint64_t bad_meta_length = static_cast<uint64_t>(file_size) + 1000;
  file.write(reinterpret_cast<char*>(&bad_meta_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, CorruptedDataLength)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_data.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .build());

  // Header layout (56 bytes):
  // magic(4) + version(4) + compression(4) + block_size(4)
  //   + metadata_version(4) + reserved(4)
  //   + metadata_length(8) + uncompressed_data_length(8)
  //   + num_blocks(8) + compressed_data_length(8)
  // Corrupt compressed_data_length at offset 48 so that the file size check fails.
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(48);
  uint64_t bad_compressed_length = 999999999ULL;
  file.write(reinterpret_cast<char*>(&bad_compressed_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, MultipleSourcesError)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};

  auto const filepath1 = temp_env->get_temp_filepath("source1.cudftbl");
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath1}, expected)
                                            .build());

  auto const filepath2 = temp_env->get_temp_filepath("source2.cudftbl");
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath2}, expected)
                                            .build());

  // Try to read with multiple sources - should fail
  std::vector<std::string> filepaths{filepath1, filepath2};
  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepaths})
        .build()),
    cudf::logic_error);
}

// =============================================================================
// Block Compression Tests
// =============================================================================

TEST_F(CudftableTest, SnappyRoundtrip)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY);
}

TEST_F(CudftableTest, ZstdRoundtrip)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::ZSTD);
}

TEST_F(CudftableTest, GzipRoundtrip)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::GZIP);
}

TEST_F(CudftableTest, NoneRoundtrip)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::NONE);
}

TEST_F(CudftableTest, SmallBlockSize)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, 64);
}

TEST_F(CudftableTest, LargeBlockSize)
{
  auto const expected = make_sample_table();
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, 1024 * 1024);
}

TEST_F(CudftableTest, EmptyTableCompressed)
{
  auto const expected = cudf::table_view{std::vector<cudf::column_view>{}};
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY);
}

TEST_F(CudftableTest, EmptyColumnCompressed)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col({});
  auto const expected = cudf::table_view{{empty_col}};
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY);
}

TEST_F(CudftableTest, LargeTableCompressed)
{
  constexpr int num_rows = 100'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col1(sequence, sequence + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col2(sequence, sequence + num_rows);

  auto const expected = cudf::table_view{{col1, col2}};
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, 32 * 1024);
}

TEST_F(CudftableTest, NestedTypesCompressed)
{
  cudf::test::strings_column_wrapper string_col({"Lorem", "ipsum", "dolor", "sit"},
                                                {true, false, true, true});
  cudf::test::lists_column_wrapper<int32_t> list_col{{1, 2, 3}, {4, 5}, {}, {6, 7, 8, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> struct_child1{{1, 2, 3, 4}};
  cudf::test::strings_column_wrapper struct_child2{{"a", "b", "c", "d"}};
  cudf::test::structs_column_wrapper struct_col{{struct_child1, struct_child2}};

  auto const expected = cudf::table_view{{string_col, list_col, struct_col}};
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY);
  run_roundtrip(expected, cudf::io::compression_type::ZSTD);
}

TEST_F(CudftableTest, DefaultCompressionIsNone)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&buffer}, expected)
                                            .build());

  // Header is a single version-1 layout; verify magic, version, and that the
  // compression field reports NONE by default.
  ASSERT_GE(buffer.size(), 12u);
  uint32_t magic{};
  uint32_t version{};
  uint32_t compression{};
  std::memcpy(&magic, buffer.data(), sizeof(uint32_t));
  std::memcpy(&version, buffer.data() + sizeof(uint32_t), sizeof(uint32_t));
  std::memcpy(&compression, buffer.data() + 2 * sizeof(uint32_t), sizeof(uint32_t));
  EXPECT_EQ(magic, 0x4C425443u);
  EXPECT_EQ(version, 1u);
  EXPECT_EQ(compression, static_cast<uint32_t>(cudf::io::compression_type::NONE));
}

TEST_F(CudftableTest, NoneWithCustomBlockSizeWarns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;
  EXPECT_CUDF_LOG_WARN(cudf::io::experimental::write_cudftable(
    cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer},
                                                              expected)
      .compression(cudf::io::compression_type::NONE)
      .block_size(64)
      .build()));
}

TEST_F(CudftableTest, NoneNormalizesBlockSize)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&buffer}, expected)
                                            .compression(cudf::io::compression_type::NONE)
                                            .block_size(64)
                                            .build());

  // Header: magic(4) + version(4) + compression(4) + block_size(4) ...
  // For NONE, block_size must be normalized to 0 on disk regardless of the
  // caller-supplied value.
  ASSERT_GE(buffer.size(), 16u);
  uint32_t block_size{};
  std::memcpy(&block_size, buffer.data() + 3 * sizeof(uint32_t), sizeof(uint32_t));
  EXPECT_EQ(block_size, 0u);

  // Roundtrip still works.
  auto host_buffer = cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, ExplicitSnappyHeader)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&buffer}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  ASSERT_GE(buffer.size(), 12u);
  uint32_t magic{};
  uint32_t version{};
  uint32_t compression{};
  std::memcpy(&magic, buffer.data(), sizeof(uint32_t));
  std::memcpy(&version, buffer.data() + sizeof(uint32_t), sizeof(uint32_t));
  std::memcpy(&compression, buffer.data() + 2 * sizeof(uint32_t), sizeof(uint32_t));
  EXPECT_EQ(magic, 0x4C425443u);
  EXPECT_EQ(version, 1u);
  EXPECT_EQ(compression, static_cast<uint32_t>(cudf::io::compression_type::SNAPPY));

  auto host_buffer = cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, CompressedFileIsSmaller)
{
  constexpr int num_rows = 50'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 100; });
  cudf::test::fixed_width_column_wrapper<int32_t> col(sequence, sequence + num_rows);
  auto const table = cudf::table_view{{col}};

  std::vector<char> uncompressed_buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&uncompressed_buffer}, table)
                                            .compression(cudf::io::compression_type::NONE)
                                            .build());

  std::vector<char> compressed_buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&compressed_buffer}, table)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  EXPECT_LT(compressed_buffer.size(), uncompressed_buffer.size());
}

TEST_F(CudftableTest, CorruptedCompressedDataLength)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_compressed.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  // Header layout (56 bytes):
  // magic(4) + version(4) + compression(4) + block_size(4)
  //   + metadata_version(4) + reserved(4)
  //   + metadata_length(8) + uncompressed_data_length(8)
  //   + num_blocks(8) + compressed_data_length(8)
  // Corrupt compressed_data_length at offset 48.
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(48);
  uint64_t bad_length = 999999999ULL;
  file.write(reinterpret_cast<char*>(&bad_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, CompressedFileRoundtrip)
{
  auto const filepath = temp_env->get_temp_filepath("compressed_roundtrip.cudftbl");
  cudf::test::fixed_width_column_wrapper<int64_t> col({100, 200, 300, 400, 500},
                                                      {true, false, true, true, true});
  auto const expected = cudf::table_view{{col}};

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
      .build());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, LargeTableCompressedZstd)
{
  constexpr int num_rows = 100'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col1(sequence, sequence + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col2(sequence, sequence + num_rows);

  auto const expected = cudf::table_view{{col1, col2}};
  run_roundtrip(expected, cudf::io::compression_type::ZSTD, 32 * 1024);
}

TEST_F(CudftableTest, LargeTableCompressedGzip)
{
  constexpr int num_rows = 100'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col1(sequence, sequence + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col2(sequence, sequence + num_rows);

  auto const expected = cudf::table_view{{col1, col2}};
  run_roundtrip(expected, cudf::io::compression_type::GZIP, 32 * 1024);
}

TEST_F(CudftableTest, CompressedWithZeroBlockSize)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};

  std::vector<char> buffer;
  EXPECT_THROW(cudf::io::experimental::write_cudftable(
                 cudf::io::experimental::cudftable_writer_options::builder(
                   cudf::io::sink_info{&buffer}, expected)
                   .compression(cudf::io::compression_type::SNAPPY)
                   .block_size(0)
                   .build()),
               cudf::logic_error);
}

TEST_F(CudftableTest, CorruptedBlockIndex)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_block_index.cudftbl");

  // Create a table large enough to produce multiple blocks at the requested block size.
  constexpr int num_rows = 10'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col(sequence, sequence + num_rows);
  auto const expected = cudf::table_view{{col}};

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .block_size(4 * 1024)
                                            .build());

  // Read the header to find the block index offset, then stamp out a clearly
  // inconsistent per-block compressed_size (which must sum to
  // compressed_data_length in a valid file).
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  uint64_t metadata_length{};
  file.seekg(24);
  file.read(reinterpret_cast<char*>(&metadata_length), sizeof(uint64_t));

  constexpr size_t header_size       = 56;
  auto const block_index_offset      = header_size + metadata_length;
  uint64_t const bad_compressed_size = 1;  // too small to match the header
  file.seekp(static_cast<std::streamoff>(block_index_offset));
  file.write(reinterpret_cast<char const*>(&bad_compressed_size), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

namespace {
/**
 * @brief Host-only data sink that does not support device writes.
 *
 * Used to exercise the compressed/uncompressed fallback path where the writer
 * must stage bytes through a host bounce buffer.
 */
class host_only_sink : public cudf::io::data_sink {
 public:
  void host_write(void const* data, size_t size) override
  {
    auto const* bytes = static_cast<char const*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
  }

  [[nodiscard]] bool supports_device_write() const override { return false; }

  void flush() override {}

  [[nodiscard]] size_t bytes_written() override { return buffer_.size(); }

  [[nodiscard]] std::vector<char> const& buffer() const { return buffer_; }

 private:
  std::vector<char> buffer_;
};
}  // namespace

TEST_F(CudftableTest, HostOnlySinkCompressedRoundtrip)
{
  constexpr int num_rows = 10'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 64; });
  cudf::test::fixed_width_column_wrapper<int32_t> col(sequence, sequence + num_rows);
  auto const expected = cudf::table_view{{col}};

  host_only_sink sink;
  cudf::io::experimental::write_cudftable(
    cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&sink}, expected)
      .compression(cudf::io::compression_type::SNAPPY)
      .block_size(4 * 1024)
      .build());

  ASSERT_FALSE(sink.buffer().empty());
  auto const& data = sink.buffer();
  auto host_buffer =
    cudf::host_span<std::byte const>(reinterpret_cast<std::byte const*>(data.data()), data.size());
  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, HostOnlySinkUncompressedRoundtrip)
{
  auto const expected = make_sample_table();

  host_only_sink sink;
  cudf::io::experimental::write_cudftable(
    cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&sink}, expected)
      .build());

  ASSERT_FALSE(sink.buffer().empty());
  auto const& data = sink.buffer();
  auto host_buffer =
    cudf::host_span<std::byte const>(reinterpret_cast<std::byte const*>(data.data()), data.size());
  auto result = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
}

TEST_F(CudftableTest, BlockSizeBoundaries)
{
  constexpr int num_rows = 20'000;
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col(sequence, sequence + num_rows);
  auto const expected = cudf::table_view{{col}};

  auto const len = packed_data_length(expected);
  ASSERT_GT(len, 2u);
  ASSERT_LE(len, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

  // Exercise block sizes exactly one-less-than (last block is a 1-byte partial),
  // equal-to, and one-more-than the packed data size (both single-block).
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, static_cast<uint32_t>(len - 1));
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, static_cast<uint32_t>(len));
  run_roundtrip(expected, cudf::io::compression_type::SNAPPY, static_cast<uint32_t>(len + 1));

  // Exact multiple: two full blocks with no partial final block.
  if (len % 2 == 0) {
    run_roundtrip(expected, cudf::io::compression_type::SNAPPY, static_cast<uint32_t>(len / 2));
  }
}

TEST_F(CudftableTest, CallerProvidedMemoryResource)
{
  auto const expected = make_sample_table();

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{&buffer}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  auto host_buffer = cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());

  auto stats_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto result   = cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build(),
    cudf::get_default_stream(),
    stats_mr);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  // The returned table's device data must have been allocated from the caller
  // supplied resource.
  EXPECT_GT(stats_mr.get_bytes_counter().peak, 0);
}

TEST_F(CudftableTest, CorruptedNumBlocksOverflow)
{
  auto const filepath = temp_env->get_temp_filepath("overflow_blocks.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  // num_blocks lives at offset 40. A value near UINT64_MAX overflows the
  // num_blocks * sizeof(block_index_entry) product used to compute offsets.
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(40);
  uint64_t bad_num_blocks = std::numeric_limits<uint64_t>::max();
  file.write(reinterpret_cast<char*>(&bad_num_blocks), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

TEST_F(CudftableTest, CorruptedMetadataLengthOverflow)
{
  auto const filepath = temp_env->get_temp_filepath("overflow_meta.cudftbl");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  cudf::io::experimental::write_cudftable(cudf::io::experimental::cudftable_writer_options::builder(
                                            cudf::io::sink_info{filepath}, expected)
                                            .compression(cudf::io::compression_type::SNAPPY)
                                            .build());

  // metadata_length lives at offset 24. A value near UINT64_MAX overflows the
  // header_size + metadata_length sum used to compute the block index offset.
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(24);
  uint64_t bad_meta_length = std::numeric_limits<uint64_t>::max();
  file.write(reinterpret_cast<char*>(&bad_meta_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cudftable(
      cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{filepath})
        .build()),
    cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
