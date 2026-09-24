/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file vortex_io_test.cpp
 * @brief Verify Vortex/cuDF round trips, projections, metadata, and owning read results.
 *
 * Covers slices, chunking, bounded staging, stream completion, recovery, and concurrent reads.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/file_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <vortex/host_staging.hpp>
#include <vortex/vortex_io.hpp>
#include <vortex/writer.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Declare after caller-owned buffers to drain before their destruction, including on unwinding.
// This cannot protect buffers already freed inside a failing import/export helper.
class host_buffer_fence {
 public:
  explicit host_buffer_fence(cudaStream_t stream) : stream_{stream} {}
  ~host_buffer_fence()
  {
    if (pending_) {
      auto const status = cudaStreamSynchronize(stream_);
      EXPECT_EQ(status, cudaSuccess) << "Host buffer fence failed: " << cudaGetErrorString(status);
    }
  }

  host_buffer_fence(host_buffer_fence const&)            = delete;
  host_buffer_fence& operator=(host_buffer_fence const&) = delete;

  void wait()
  {
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream_));
    pending_ = false;
  }

 private:
  cudaStream_t stream_;
  bool pending_{true};
};

class VortexIOTest : public cudf::test::BaseFixture {
 protected:
  static rmm::cuda_stream make_stream()
  {
    CUDF_CUDA_TRY(cudaSetDevice(0));
    return rmm::cuda_stream{rmm::cuda_stream::flags::non_blocking};
  }

  void TearDown() override { stream_fence.wait(); }

  std::string file(std::string const& name) const { return directory.path() + name; }

  temp_directory const directory{"ndsh-vortex-io"};
  // Never replace the current resource: it and this stream must outlive all contexts/tables.
  rmm::cuda_stream stream_owner{make_stream()};
  cudaStream_t const stream{stream_owner.value()};
  // RMM destroys the stream without waiting. Also drain if TearDown fails or is bypassed.
  host_buffer_fence stream_fence{stream};
};

struct column_spec {
  char const* name;
  ArrowType arrow_type;
  cudf::type_id cudf_type;
  int32_t precision{0};
  int32_t arrow_scale{0};
};

std::array const columns{
  column_spec{"bool flag", NANOARROW_TYPE_BOOL, cudf::type_id::BOOL8},
  column_spec{"i8", NANOARROW_TYPE_INT8, cudf::type_id::INT8},
  column_spec{"i16", NANOARROW_TYPE_INT16, cudf::type_id::INT16},
  column_spec{"i32", NANOARROW_TYPE_INT32, cudf::type_id::INT32},
  column_spec{"i64", NANOARROW_TYPE_INT64, cudf::type_id::INT64},
  column_spec{"u8", NANOARROW_TYPE_UINT8, cudf::type_id::UINT8},
  column_spec{"u16", NANOARROW_TYPE_UINT16, cudf::type_id::UINT16},
  column_spec{"u32", NANOARROW_TYPE_UINT32, cudf::type_id::UINT32},
  column_spec{"u64", NANOARROW_TYPE_UINT64, cudf::type_id::UINT64},
  column_spec{"f32", NANOARROW_TYPE_FLOAT, cudf::type_id::FLOAT32},
  column_spec{"f64", NANOARROW_TYPE_DOUBLE, cudf::type_id::FLOAT64},
  column_spec{"utf8.text", NANOARROW_TYPE_STRING, cudf::type_id::STRING},
  column_spec{"date_days", NANOARROW_TYPE_DATE32, cudf::type_id::TIMESTAMP_DAYS},
  column_spec{"decimal32", NANOARROW_TYPE_DECIMAL32, cudf::type_id::DECIMAL32, 9, 2},
  column_spec{"decimal64", NANOARROW_TYPE_DECIMAL64, cudf::type_id::DECIMAL64, 18, 4},
  column_spec{"decimal128", NANOARROW_TYPE_DECIMAL128, cudf::type_id::DECIMAL128, 38, 6},
  column_spec{
    "decimal_negative_scale", NANOARROW_TYPE_DECIMAL128, cudf::type_id::DECIMAL128, 38, -2},
  column_spec{"all_valid", NANOARROW_TYPE_INT32, cudf::type_id::INT32},
  column_spec{"all_null", NANOARROW_TYPE_INT32, cudf::type_id::INT32}};

std::vector<std::string> column_names(std::vector<std::size_t> const& selected = {})
{
  std::vector<std::string> names;
  auto const count = selected.empty() ? columns.size() : selected.size();
  for (std::size_t c = 0; c < count; ++c) {
    names.emplace_back(columns[selected.empty() ? c : selected[c]].name);
  }
  return names;
}

struct host_table {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
};

int decimal_bitwidth(ArrowType type)
{
  switch (type) {
    case NANOARROW_TYPE_DECIMAL32: return 32;
    case NANOARROW_TYPE_DECIMAL64: return 64;
    case NANOARROW_TYPE_DECIMAL128: return 128;
    default: throw std::runtime_error("Not a supported decimal type");
  }
}

template <typename T>
int append_signed(ArrowArray* array, int64_t row)
{
  auto const value = row % 17 == 0   ? std::numeric_limits<T>::min()
                     : row % 17 == 1 ? std::numeric_limits<T>::max()
                                     : static_cast<T>(row % 201 - 100);
  return ArrowArrayAppendInt(array, value);
}

template <typename T>
int append_unsigned(ArrowArray* array, int64_t row)
{
  auto const value = row % 17 == 0 ? std::numeric_limits<T>::max() : static_cast<T>(row % 251);
  return ArrowArrayAppendUInt(array, value);
}

int append_fixture_value(ArrowArray* array, column_spec const& spec, int64_t row)
{
  switch (spec.arrow_type) {
    case NANOARROW_TYPE_BOOL: return ArrowArrayAppendInt(array, row % 2);
    case NANOARROW_TYPE_INT8: return append_signed<int8_t>(array, row);
    case NANOARROW_TYPE_INT16: return append_signed<int16_t>(array, row);
    case NANOARROW_TYPE_INT32: return append_signed<int32_t>(array, row);
    case NANOARROW_TYPE_INT64: return append_signed<int64_t>(array, row);
    case NANOARROW_TYPE_UINT8: return append_unsigned<uint8_t>(array, row);
    case NANOARROW_TYPE_UINT16: return append_unsigned<uint16_t>(array, row);
    case NANOARROW_TYPE_UINT32: return append_unsigned<uint32_t>(array, row);
    case NANOARROW_TYPE_UINT64: return append_unsigned<uint64_t>(array, row);
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE: return ArrowArrayAppendDouble(array, (row % 1001 - 500) * 0.25);
    case NANOARROW_TYPE_DATE32: return ArrowArrayAppendInt(array, row % 40001 - 20000);
    case NANOARROW_TYPE_STRING: {
      // Repetition encourages dictionary compression without depending on the writer's heuristics.
      constexpr std::array<std::string_view, 8> strings{
        "",
        "repeated string long enough to make dictionary compression worthwhile",
        "repeated string long enough to make dictionary compression worthwhile",
        "caf\xc3\xa9",
        std::string_view{"a\0b", 3},
        "another repeated string long enough to make dictionary compression worthwhile",
        "",
        "\xe6\x97\xa5\xe6\x9c\xac"};
      auto const value = strings[static_cast<std::size_t>(row) % strings.size()];
      return ArrowArrayAppendString(array, {value.data(), static_cast<int64_t>(value.size())});
    }
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128: {
      ArrowDecimal value;
      ArrowDecimalInit(&value, decimal_bitwidth(spec.arrow_type), spec.precision, spec.arrow_scale);
      if (spec.arrow_type == NANOARROW_TYPE_DECIMAL128 && row % 7 < 2) {
        NANOARROW_THROW_NOT_OK(ArrowDecimalSetDigits(
          &value,
          ArrowCharView(row % 7 == 0 ? "12345678901234567890123456789012345678"
                                     : "-12345678901234567890123456789012345678")));
      } else {
        ArrowDecimalSetInt(&value, (row % 10001 - 5000) * 101);
      }
      return ArrowArrayAppendDecimal(array, &value);
    }
    default: throw std::runtime_error("Unsupported fixture type");
  }
}

host_table make_fixture(int64_t first_row,
                        cudf::size_type rows,
                        std::vector<std::size_t> const& selected = {})
{
  auto const count = selected.empty() ? columns.size() : selected.size();
  host_table result;
  ArrowSchemaInit(result.schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(result.schema.get(), count));
  result.schema->flags &= ~ARROW_FLAG_NULLABLE;
  for (std::size_t c = 0; c < count; ++c) {
    auto const index = selected.empty() ? c : selected[c];
    auto const& spec = columns[index];
    auto child       = result.schema->children[c];
    if (spec.precision != 0) {
      NANOARROW_THROW_NOT_OK(
        ArrowSchemaSetTypeDecimal(child, spec.arrow_type, spec.precision, spec.arrow_scale));
    } else {
      NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(child, spec.arrow_type));
    }
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child, spec.name));
    if (index == columns.size() - 2) { child->flags &= ~ARROW_FLAG_NULLABLE; }
  }

  NANOARROW_THROW_NOT_OK(
    ArrowArrayInitFromSchema(result.array.get(), result.schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(result.array.get()));
  for (int64_t i = 0; i < rows; ++i) {
    auto const row = first_row + i;
    for (std::size_t c = 0; c < count; ++c) {
      auto const index = selected.empty() ? c : selected[c];
      bool const is_null =
        index == columns.size() - 1 ||
        (index != columns.size() - 2 && (row + static_cast<int64_t>(index) * 3) % 11 == 3);
      NANOARROW_THROW_NOT_OK(
        is_null ? ArrowArrayAppendNull(result.array->children[c], 1)
                : append_fixture_value(result.array->children[c], columns[index], row));
    }
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(result.array.get()));
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(result.array.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr));
  return result;
}

std::unique_ptr<cudf::table> make_source(cudaStream_t stream,
                                         int64_t first,
                                         cudf::size_type rows,
                                         std::vector<std::size_t> const& selected = {})
{
  auto fixture = make_fixture(first, rows, selected);
  host_buffer_fence fence{stream};
  auto source =
    cudf::from_arrow(fixture.schema.get(), fixture.array.get(), cuda::stream_ref{stream});
  fence.wait();
  return source;
}

void check_table(cudf::io::table_with_metadata const& actual,
                 host_table const& expected,
                 cudaStream_t stream,
                 std::vector<std::size_t> const& selected = {},
                 bool generated_names                     = false)
{
  ASSERT_NE(actual.tbl, nullptr) << "Reader returned no owning table";
  ASSERT_EQ(actual.tbl->num_rows(), expected.array->length);
  EXPECT_EQ(actual.metadata.num_rows_per_source,
            std::vector<std::size_t>{static_cast<std::size_t>(expected.array->length)});
  auto const count = selected.empty() ? columns.size() : selected.size();
  ASSERT_EQ(actual.tbl->num_columns(), static_cast<cudf::size_type>(count));
  ASSERT_EQ(actual.metadata.schema_info.size(), count);
  auto const view = actual.tbl->view();
  for (std::size_t c = 0; c < count; ++c) {
    auto const& spec   = columns[selected.empty() ? c : selected[c]];
    auto const& column = view.column(static_cast<cudf::size_type>(c));
    SCOPED_TRACE(spec.name);
    ASSERT_EQ(column.type().id(), spec.cudf_type);
    if (spec.precision != 0) { EXPECT_EQ(column.type().scale(), -spec.arrow_scale); }
    auto const name = generated_names ? "_col" + std::to_string(selected.empty() ? c : selected[c])
                                      : std::string{spec.name};
    EXPECT_EQ(actual.metadata.schema_info[c].name, name);
    EXPECT_EQ(column.null_count(), expected.array->children[c]->null_count);
  }

  host_buffer_fence fence{stream};
  auto expected_table =
    cudf::from_arrow(expected.schema.get(), expected.array.get(), cuda::stream_ref{stream});
  fence.wait();
  // Ignore redundant all-valid masks without relaxing EQUAL's exact floating-point comparison.
  auto const without_all_valid_mask = [](cudf::column_view column) {
    if (column.has_nulls()) { return column; }
    return cudf::column_view{column.type(),
                             column.size(),
                             column.head(),
                             nullptr,
                             0,
                             column.offset(),
                             {column.child_begin(), column.child_end()}};
  };
  for (cudf::size_type c = 0; c < view.num_columns(); ++c) {
    SCOPED_TRACE(columns[selected.empty() ? c : selected[c]].name);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(without_all_valid_mask(expected_table->view().column(c)),
                                   without_all_valid_mask(view.column(c)),
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   cuda::stream_ref{stream});
  }
}

cudf::io::table_with_metadata read_completed(ndsh::vortex_io const& io,
                                             std::string const& path,
                                             cudaStream_t stream,
                                             std::size_t batch_rows,
                                             std::vector<std::string> const& names = {})
{
  auto result = io.read_vortex(path, batch_rows, names);
  // Check before any test-side synchronization could hide an unfinished consumer stream.
  CUDF_CUDA_TRY(cudaStreamQuery(stream));
  return result;
}

void write_source(ndsh::vortex_io const& io,
                  std::string const& path,
                  cudaStream_t stream,
                  int64_t first,
                  cudf::size_type rows,
                  cudf::size_type slice_begin,
                  cudf::size_type chunk_rows)
{
  auto source = make_source(stream, first, rows + (slice_begin == 0 ? 0 : slice_begin + 4));
  auto input  = source->view();
  if (slice_begin != 0) {
    input = cudf::slice(input, {slice_begin, slice_begin + rows}, cuda::stream_ref{stream}).front();
    ASSERT_EQ(input.column(0).offset(), slice_begin) << "Test input is not actually sliced";
  }
  io.write_vortex(path, input, column_names(), chunk_rows);
  CUDF_CUDA_TRY(cudaStreamQuery(stream));
}

struct round_trip_case {
  char const* name;
  cudf::size_type rows;
  cudf::size_type slice_begin;
  cudf::size_type chunk_rows;
  std::size_t batch_rows;
};

class VortexIORoundTripTest : public VortexIOTest,
                              public ::testing::WithParamInterface<round_trip_case> {};

TEST_P(VortexIORoundTripTest, RoundTrip)
{
  auto const& [name, rows, slice_begin, chunk_rows, batch_rows] = GetParam();
  auto const path = file(std::string{name} + ".vortex");
  {
    ndsh::vortex_io writer{stream};
    ASSERT_NO_FATAL_FAILURE(write_source(writer, path, stream, 0, rows, slice_begin, chunk_rows));
  }
  ndsh::vortex_io reader{stream};
  check_table(
    read_completed(reader, path, stream, batch_rows), make_fixture(slice_begin, rows), stream);
}

// Nonzero scan sizes must not straddle physical blocks. Keep non-byte-aligned slices.
INSTANTIATE_TEST_SUITE_P(Layouts,
                         VortexIORoundTripTest,
                         ::testing::Values(round_trip_case{"SlicedLayout", 69, 5, 11, 0},
                                           round_trip_case{"SlicedFixed", 69, 5, 14, 7},
                                           round_trip_case{"WordBoundary", 520, 0, 520, 507},
                                           round_trip_case{"Empty", 0, 0, 7, 0}),
                         [](auto const& info) { return info.param.name; });

TEST_F(VortexIOTest, BoundedStringHostStaging)
{
  std::vector<std::size_t> const selected{11};
  constexpr cudf::size_type parent_rows = 4096;
  auto source                           = make_source(stream, 0, parent_rows, selected);
  auto const strings                    = source->view();
  std::vector<cudf::column_metadata> const metadata{{columns[selected.front()].name}};
  auto schema = cudf::to_arrow_schema(strings, metadata);
  // Prefix and offset slices exercise both string-compaction conditions; keep the empty path too.
  std::array<std::array<cudf::size_type, 2>, 3> const ranges{{{0, 7}, {5, 12}, {5, 5}}};
  for (auto const& range : ranges) {
    SCOPED_TRACE(::testing::PrintToString(range));
    auto input    = cudf::slice(strings, {range[0], range[1]}, cuda::stream_ref{stream}).front();
    auto expected = make_fixture(range[0], range[1] - range[0], selected);
    auto host     = ndsh::detail::stage_host_chunk(
      input, cuda::stream_ref{stream}, cudf::get_current_device_resource_ref());
    CUDF_CUDA_TRY(cudaStreamQuery(stream));
    ASSERT_EQ(host->device_type, ARROW_DEVICE_CPU) << "Staging returned non-host data";
    // Both arrays are nanoarrow-owned. Inspect actual buffer sizes: ArrayView's
    // inferred size would miss an unnecessarily retained trailing parent payload.
    auto const* chars          = ArrowArrayBuffer(host->array.children[0], 2);
    auto const* expected_chars = ArrowArrayBuffer(expected.array->children[0], 2);
    EXPECT_EQ(chars->size_bytes, expected_chars->size_bytes)
      << "String staging retained bytes outside the row chunk";
    host_buffer_fence fence{stream};
    auto actual_table = cudf::from_arrow(schema.get(), &host->array, cuda::stream_ref{stream});
    auto expected_table =
      cudf::from_arrow(expected.schema.get(), expected.array.get(), cuda::stream_ref{stream});
    fence.wait();
    CUDF_TEST_EXPECT_TABLES_EQUAL(
      expected_table->view(), actual_table->view(), cuda::stream_ref{stream});
  }
}

TEST_F(VortexIOTest, AsyncResourceOwnedReadsAndCompletion)
{
  rmm::mr::cuda_async_memory_resource resource;
  // This fence runs after all table destructors, before the explicit resource dies.
  host_buffer_fence resource_fence{stream};
  {
    auto const first_path  = file("owned-first.vortex");
    auto const second_path = file("owned-second.vortex");
    std::array<cudf::io::table_with_metadata, 2> results;
    {
      ndsh::vortex_io io{stream, rmm::device_async_resource_ref{resource}};
      write_source(io, first_path, stream, 0, 37, 0, 37);
      write_source(io, second_path, stream, 10000, 37, 0, 14);
      // Keep fixed-size scan batches within physical blocks while exercising import scratch.
      results[0] = read_completed(io, first_path, stream, 0);
      // A different file must not overwrite buffers held by the earlier returned table.
      results[1] = read_completed(io, second_path, stream, 7);
    }
    ASSERT_TRUE(std::filesystem::remove(first_path));
    ASSERT_TRUE(std::filesystem::remove(second_path));
    check_table(results[0], make_fixture(0, 37), stream);
    check_table(results[1], make_fixture(10000, 37), stream);
  }
  resource_fence.wait();
}

TEST_F(VortexIOTest, OrderedProjectionAndMetadata)
{
  auto source = make_source(stream, 0, 9);
  ndsh::vortex_io io{stream};
  auto const path = file("projection.vortex");
  io.write_vortex(path, source->view(), column_names(), 3);
  std::vector<std::size_t> const selected{11, 16, 0, 8, 18, 12};
  auto projected = read_completed(io, path, stream, 0, column_names(selected));
  ASSERT_NO_FATAL_FAILURE(check_table(projected, make_fixture(0, 9, selected), stream, selected));
  for (std::size_t c = 0; c < selected.size(); ++c) {
    SCOPED_TRACE(c);
    EXPECT_EQ(projected.metadata.schema_info[c].is_nullable,
              source->view().column(static_cast<cudf::size_type>(selected[c])).nullable());
  }
}

TEST_F(VortexIOTest, InvalidProjectionAndRecovery)
{
  ndsh::vortex_io io{stream};
  auto const path = file("invalid-projection.vortex");
  write_source(io, path, stream, 0, 9, 0, 3);
  for (auto const& names :
       std::array<std::vector<std::string>, 2>{{{"i32", "unknown-column"}, {"i32", "i32"}}}) {
    SCOPED_TRACE(::testing::PrintToString(names));
    EXPECT_THROW((void)io.read_vortex(path, 0, names), std::exception);
    check_table(read_completed(io, path, stream, 0), make_fixture(0, 9), stream);
  }
}

TEST_F(VortexIOTest, MissingAndTruncatedFilesAndRecovery)
{
  ndsh::vortex_io io{stream};
  auto const valid_path     = file("file-recovery.vortex");
  auto const missing_path   = file("missing.vortex");
  auto const truncated_path = file("truncated.vortex");
  write_source(io, valid_path, stream, 100, 9, 0, 3);
  ASSERT_TRUE(std::filesystem::copy_file(valid_path, truncated_path));
  ASSERT_GT(std::filesystem::file_size(truncated_path), 8u);
  std::filesystem::resize_file(truncated_path, 8);
  for (auto const& path : {missing_path, truncated_path}) {
    SCOPED_TRACE(path);
    EXPECT_THROW((void)io.read_vortex(path, 0), std::exception);
    check_table(read_completed(io, valid_path, stream, 0), make_fixture(100, 9), stream);
  }
}

TEST_F(VortexIOTest, WriterDefaultNamesAndProjection)
{
  auto source     = make_source(stream, 0, 9);
  auto const path = file("writer-defaults.vortex");
  ndsh::write_vortex({cudf::io::sink_info{path}, source->view(), {}, 3}, cuda::stream_ref{stream});
  CUDF_CUDA_TRY(cudaStreamQuery(stream));
  ndsh::vortex_io io{stream};
  check_table(read_completed(io, path, stream, 0), make_fixture(0, 9), stream, {}, true);
  std::vector<std::size_t> const selected{11, 0};
  check_table(read_completed(io, path, stream, 0, {"_col11", "_col0"}),
              make_fixture(0, 9, selected),
              stream,
              selected,
              true);
}

TEST_F(VortexIOTest, WriterFailureAndRecovery)
{
  ndsh::vortex_io io{stream};
  auto source             = make_source(stream, 200, 9);
  auto const invalid_path = file("missing-parent/output.vortex");
  EXPECT_THROW(io.write_vortex(invalid_path, source->view(), column_names(), 3), std::exception);
  auto const valid_path = file("writer-recovery.vortex");
  io.write_vortex(valid_path, source->view(), column_names(), 3);
  check_table(read_completed(io, valid_path, stream, 0), make_fixture(200, 9), stream);
}

TEST_F(VortexIOTest, ConcurrentSharedAdapterReads)
{
  ndsh::vortex_io io{stream};
  auto const first_path  = file("concurrent-first.vortex");
  auto const second_path = file("concurrent-second.vortex");
  write_source(io, first_path, stream, 0, 37, 0, 37);
  write_source(io, second_path, stream, 10000, 37, 0, 14);
  std::array<cudf::io::table_with_metadata, 2> results;

  // Futures join even if a launch or get throws, before the captured adapter and result owners die.
  std::array<std::future<cudf::io::table_with_metadata>, 2> futures;
  futures[0] = std::async(std::launch::async, [&io, first_path] {
    CUDF_CUDA_TRY(cudaSetDevice(0));
    return io.read_vortex(first_path, 0);
  });
  futures[1] = std::async(std::launch::async, [&io, second_path] {
    CUDF_CUDA_TRY(cudaSetDevice(0));
    return io.read_vortex(second_path, 7);
  });
  for (std::size_t i = 0; i < futures.size(); ++i) {
    results[i] = futures[i].get();
  }
  // A worker cannot query completion while the other may still enqueue work on the shared stream.
  CUDF_CUDA_TRY(cudaStreamQuery(stream));
  check_table(results[0], make_fixture(0, 37), stream);
  check_table(results[1], make_fixture(10000, 37), stream);
}

}  // namespace

CUDF_TEST_PROGRAM_MAIN()
