/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "writer.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/file_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using cudf::io::sink_info;
using ndsh::vortex_writer_options;
using ndsh::write_vortex;

class VortexWriterTest : public cudf::test::BaseFixture {
 protected:
  temp_directory const directory{"vortex_writer"};

  std::string path(std::string const& name = "table.vortex") const
  {
    return directory.path() + name;
  }

  void expect_rejected_without_touching_files(vortex_writer_options const& options,
                                              std::vector<std::string> const& paths)
  {
    for (auto const& file : paths) {
      ASSERT_FALSE(std::filesystem::exists(file));
    }
    EXPECT_THROW(write_vortex(options), cudf::logic_error);
    for (auto const& file : paths) {
      EXPECT_FALSE(std::filesystem::exists(file));
    }

    // Validation must precede opening the destination, including truncation of an existing file.
    std::string const sentinel{"do not truncate"};
    for (auto const& file : paths) {
      std::ofstream out{file, std::ios::binary};
      ASSERT_TRUE(out.is_open());
      out << sentinel;
      ASSERT_TRUE(out.good());
    }
    EXPECT_THROW(write_vortex(options), cudf::logic_error);
    for (auto const& file : paths) {
      std::ifstream in{file, std::ios::binary};
      ASSERT_TRUE(in.is_open());
      EXPECT_EQ((std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}}),
                sentinel);
      in.close();
      ASSERT_TRUE(std::filesystem::remove(file));
    }
  }
};

TEST_F(VortexWriterTest, OptionsDefaultsAndAggregate)
{
  vortex_writer_options defaults;
  EXPECT_TRUE(defaults.names.empty());
  EXPECT_EQ(defaults.rows_per_chunk, 16 << 20);
  EXPECT_EQ(defaults.table.num_columns(), 0);

  cudf::test::fixed_width_column_wrapper<int32_t> numbers{1, 2};
  cudf::table_view const table{{numbers}};
  vortex_writer_options const options{sink_info{path()}, table, {"number"}, 1};
  EXPECT_EQ(options.sink.filepaths(), std::vector<std::string>{path()});
  EXPECT_EQ(options.names, std::vector<std::string>{"number"});
  EXPECT_EQ(options.rows_per_chunk, 1);
  EXPECT_EQ(options.table.num_rows(), table.num_rows());
  EXPECT_EQ(options.table.column(0).head<int32_t>(), table.column(0).head<int32_t>());

  EXPECT_FALSE(std::filesystem::exists(path()));
}

class VortexWriterEnabledTest : public VortexWriterTest {
 protected:
  void SetUp() override
  {
    int device{};
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    if (device != 0) { GTEST_SKIP() << "The Vortex writer currently requires CUDA device 0"; }
  }

  // Writer-only smoke checks, not a decoder or a table-equality assertion. The pinned Vortex
  // revision d196f601 writes VTXF at both ends. No reader adapter or private FFI handles are used.
  void expect_finalized_file(std::string const& file)
  {
    ASSERT_TRUE(std::filesystem::is_regular_file(file));
    ASSERT_GT(std::filesystem::file_size(file), 8u);
    std::ifstream in{file, std::ios::binary};
    ASSERT_TRUE(in.is_open());
    std::array<char, 4> magic{};
    in.read(magic.data(), magic.size());
    ASSERT_TRUE(in.good());
    EXPECT_EQ(std::string(magic.data(), magic.size()), "VTXF");
    in.seekg(-static_cast<std::streamoff>(magic.size()), std::ios::end);
    in.read(magic.data(), magic.size());
    ASSERT_TRUE(in.good());
    EXPECT_EQ(std::string(magic.data(), magic.size()), "VTXF");
  }
};

TEST_F(VortexWriterEnabledTest, RejectsInvalidNamesBeforeOpeningFile)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{1, 2};
  vortex_writer_options options{sink_info{path()}, cudf::table_view{{numbers, numbers}}};
  std::vector<std::vector<std::string>> const invalid_names{
    {"only_one"}, {"a", "b", "extra"}, {"duplicate", "duplicate"}, {"a", std::string{"b\0c", 3}}};
  for (auto const& names : invalid_names) {
    SCOPED_TRACE(::testing::PrintToString(names));
    options.names = names;
    expect_rejected_without_touching_files(options, {path()});
  }
}

TEST_F(VortexWriterEnabledTest, RejectsNonpositiveChunkSizeBeforeOpeningFile)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{1, 2};
  for (auto const rows : {0, -1}) {
    SCOPED_TRACE(rows);
    vortex_writer_options const options{sink_info{path()}, cudf::table_view{{numbers}}, {}, rows};
    expect_rejected_without_touching_files(options, {path()});
  }
}

TEST_F(VortexWriterEnabledTest, RejectsZeroColumnsBeforeOpeningFile)
{
  vortex_writer_options const options{sink_info{path()}, cudf::table_view{}};
  expect_rejected_without_touching_files(options, {path()});
}

TEST_F(VortexWriterEnabledTest, RejectsUnsupportedSinks)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{1, 2};
  vortex_writer_options options{sink_info{}, cudf::table_view{{numbers}}};
  for (auto const& sink : {sink_info{},
                           sink_info{std::vector<std::string>{}},
                           sink_info{std::string{}},
                           sink_info{std::string{"s3://bucket/table.vortex"}},
                           sink_info{std::string{"https://example.com/table.vortex"}}}) {
    options.sink = sink;
    EXPECT_THROW(write_vortex(options), cudf::logic_error);
  }

  options.sink = sink_info{std::vector<std::string>{path("first"), path("second")}};
  expect_rejected_without_touching_files(options, {path("first"), path("second")});

  // A C-string conversion must not silently accept the prefix of a path containing NUL.
  options.sink = sink_info{path() + std::string{"\0suffix", 7}};
  expect_rejected_without_touching_files(options, {path()});

  std::vector<char> buffer{'k', 'e', 'e', 'p'};
  auto const original = buffer;
  options.sink        = sink_info{&buffer};
  EXPECT_THROW(write_vortex(options), cudf::logic_error);
  EXPECT_EQ(buffer, original);

  auto custom_sink = cudf::io::data_sink::create(&buffer);
  options.sink     = sink_info{custom_sink.get()};
  EXPECT_THROW(write_vortex(options), cudf::logic_error);
  EXPECT_EQ(buffer, original);
  EXPECT_EQ(custom_sink->bytes_written(), original.size());
}

TEST_F(VortexWriterEnabledTest, RejectsNestedAndDictionaryColumnsBeforeOpeningFile)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{1, 2};
  cudf::test::lists_column_wrapper<int32_t> lists{{1}, {2, 3}};
  cudf::test::structs_column_wrapper structs{numbers};
  cudf::test::dictionary_column_wrapper<int32_t> dictionary{1, 2};
  for (auto const& column : std::vector<cudf::column_view>{lists, structs, dictionary}) {
    SCOPED_TRACE(static_cast<int>(column.type().id()));
    vortex_writer_options const options{sink_info{path()}, cudf::table_view{{numbers, column}}};
    expect_rejected_without_touching_files(options, {path()});
  }
}

TEST_F(VortexWriterEnabledTest, RejectsUnsupportedChronoTypesEvenWithZeroRows)
{
  for (auto const type : {cudf::type_id::DURATION_DAYS,
                          cudf::type_id::DURATION_SECONDS,
                          cudf::type_id::DURATION_MILLISECONDS,
                          cudf::type_id::DURATION_MICROSECONDS,
                          cudf::type_id::DURATION_NANOSECONDS,
                          cudf::type_id::TIMESTAMP_SECONDS,
                          cudf::type_id::TIMESTAMP_MILLISECONDS,
                          cudf::type_id::TIMESTAMP_MICROSECONDS,
                          cudf::type_id::TIMESTAMP_NANOSECONDS}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto column = cudf::make_empty_column(cudf::data_type{type});
    vortex_writer_options const options{sink_info{path()}, cudf::table_view{{column->view()}}};
    expect_rejected_without_touching_files(options, {path()});
  }
}

TEST_F(VortexWriterEnabledTest, CreatesZeroRowFileWithGeneratedNames)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers;
  cudf::test::strings_column_wrapper strings;
  vortex_writer_options const options{
    sink_info{path()}, cudf::table_view{{numbers, strings}}, {}, 1};
  ASSERT_NO_THROW(write_vortex(options));
  expect_finalized_file(path());
}

template <typename T>
class VortexWriterNumericTest : public VortexWriterEnabledTest {};
TYPED_TEST_SUITE(VortexWriterNumericTest, cudf::test::NumericTypes);

TYPED_TEST(VortexWriterNumericTest, WritesNullableNumericAndBooleanColumns)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> column{{0, 1, 1, 0}, {true, false, true, true}};
  vortex_writer_options const options{sink_info{this->path()}, cudf::table_view{{column}}, {}, 2};
  ASSERT_NO_THROW(write_vortex(options));
  this->expect_finalized_file(this->path());
}

template <typename T>
class VortexWriterDecimalTest : public VortexWriterEnabledTest {};
TYPED_TEST_SUITE(VortexWriterDecimalTest, cudf::test::FixedPointTypes);

TYPED_TEST(VortexWriterDecimalTest, WritesNullableDecimals)
{
  for (auto const scale : {-2, 0, 2}) {
    SCOPED_TRACE(scale);
    cudf::test::fixed_point_column_wrapper<typename TypeParam::rep> column{
      {-1234, 0, 5678}, {true, false, true}, numeric::scale_type{scale}};
    vortex_writer_options const options{
      sink_info{this->path()}, cudf::table_view{{column}}, {"decimal"}, 1};
    ASSERT_NO_THROW(write_vortex(options));
    this->expect_finalized_file(this->path());
  }
}

TEST_F(VortexWriterEnabledTest, WritesDayTimestamps)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> days{
    {-1, 0, 1, 20000}, {true, false, true, true}};
  vortex_writer_options const options{sink_info{path()}, cudf::table_view{{days}}, {"day"}, 2};
  ASSERT_NO_THROW(write_vortex(options));
  expect_finalized_file(path());
}

TEST_F(VortexWriterEnabledTest, WritesSlicedNullableStringsAndNumbersOnNondefaultStream)
{
  cuda::stream stream{cuda::device_ref{0}};
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{
    {999, -1, 2, 3, 4, 5, 6, 999}, {true, true, false, true, true, false, true, true}, stream};
  cudf::test::strings_column_wrapper strings{
    {"excluded prefix", "alpha", "", "ignored", "\xc3\xa9", "omega", "last", "excluded suffix"},
    {true, true, true, false, true, false, true, true},
    stream};
  auto const sliced = cudf::slice(cudf::table_view{{numbers, strings}}, {1, 7}, stream).front();
  ASSERT_EQ(sliced.column(0).offset(), 1);
  ASSERT_EQ(sliced.column(1).offset(), 1);

  cudf::test::memory_resource_test_harness harness{mr()};
  // Cover single-row chunks, a short final chunk, and a chunk larger than the slice.
  for (auto const chunk_rows : {1, 2, 4, 16}) {
    SCOPED_TRACE(chunk_rows);
    vortex_writer_options const options{sink_info{path()}, sliced, {"number", "text"}, chunk_rows};
    // The supplied resource owns staging buffers; slice/compaction scratch uses the current one.
    ASSERT_NO_THROW(write_vortex(options, stream, harness.temporary_mr()));
    EXPECT_EQ(cudaStreamQuery(stream.get()), cudaSuccess);
    // Inspect immediately: write_vortex, not the caller, must finish file finalization.
    expect_finalized_file(path());
  }
  harness.expect_temporary_allocation_activity(stream);
  harness.expect_no_live_allocations(stream);
}

}  // namespace
