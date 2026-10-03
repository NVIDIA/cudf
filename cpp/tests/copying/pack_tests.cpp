/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/copying/slice_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/null_mask.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

// Size of the serialized table header that precedes the column entries in the
// packed metadata buffer: version + num_columns + num_rows + pad, four 4-byte fields.
// Must match `serialized_table_header` in cpp/src/copying/pack.cpp.
auto constexpr metadata_header_size = 4 * sizeof(cudf::size_type);

namespace cx = cudf::experimental;

namespace {

struct compressed_region_header {
  int32_t version;
  uint32_t num_regions;
  uint64_t num_chunks;
  uint64_t legacy_metadata_bytes;
  uint64_t uncompressed_payload_bytes;
};

struct compressed_region_entry {
  uint64_t uncompressed_offset;
  uint64_t uncompressed_bytes;
  uint64_t data_bytes;
  uint64_t chunk_bytes;
  uint64_t chunk_begin;
  uint64_t num_chunks;
  int32_t type;
  uint32_t is_validity;
  int32_t compression;
  uint32_t reserved;
};

struct region_directory {
  compressed_region_header header;
  std::vector<compressed_region_entry> entries;
};

region_directory read_region_directory(std::vector<uint8_t> const& metadata)
{
  region_directory directory{};
  std::memcpy(&directory.header, metadata.data(), sizeof(compressed_region_header));
  directory.entries.resize(directory.header.num_regions);
  std::memcpy(directory.entries.data(),
              metadata.data() + sizeof(compressed_region_header),
              directory.entries.size() * sizeof(compressed_region_entry));
  return directory;
}

constexpr std::array compressed_codecs{
  cx::pack_compression::cascaded, cx::pack_compression::zstd, cx::pack_compression::snappy};

bool is_codec_enabled(cx::pack_compression compression)
{
  auto const io_type = [&] {
    switch (compression) {
      case cx::pack_compression::zstd: return cudf::io::compression_type::ZSTD;
      case cx::pack_compression::snappy: return cudf::io::compression_type::SNAPPY;
      default: return cudf::io::compression_type::NONE;
    }
  }();
  return io_type == cudf::io::compression_type::NONE ||
         (cudf::io::detail::is_compression_supported(io_type) &&
          cudf::io::detail::is_decompression_supported(io_type));
}

cx::pack_options make_options(cx::pack_compression compression)
{
  cx::pack_options options;
  options.compression = compression;
  return options;
}

// A small staging buffer forces pack_into() through many staging windows.
cx::pack_options make_windowed_options(cx::pack_compression compression)
{
  auto options                 = make_options(compression);
  options.staging_buffer_bytes = 8 * 1024;
  return options;
}

enum class destination_kind { device, pinned, pageable };

std::span<uint8_t> as_span(rmm::device_buffer& buffer)
{
  return {static_cast<uint8_t*>(buffer.data()), buffer.size()};
}

struct packed_output {
  rmm::device_buffer buffer;
  cudf::detail::host_vector<uint8_t> pinned;
  std::vector<uint8_t> pageable;
  cx::pack_result result;
  uint8_t const* data;

  [[nodiscard]] cx::packed_data_view view() const
  {
    return {result.metadata, std::span<uint8_t const>{data, result.payload_bytes}};
  }
};

packed_output pack_to(cx::pack_plan const& plan, destination_kind kind)
{
  auto const stream = cudf::get_default_stream();
  auto const bytes  = plan.sizes().payload_bytes;
  auto const align  = plan.sizes().payload_alignment;
  packed_output output{rmm::device_buffer(kind == destination_kind::device ? bytes : 0, stream),
                       cudf::detail::make_pinned_vector_async<uint8_t>(
                         kind == destination_kind::pinned ? bytes : 0, stream),
                       std::vector<uint8_t>(kind == destination_kind::pageable ? bytes + align : 0),
                       {},
                       nullptr};
  std::span<uint8_t> destination;
  switch (kind) {
    case destination_kind::device: destination = as_span(output.buffer); break;
    case destination_kind::pinned:
      destination = {output.pinned.data(), output.pinned.size()};
      break;
    case destination_kind::pageable: {
      auto const base = reinterpret_cast<std::uintptr_t>(output.pageable.data());
      destination     = {reinterpret_cast<uint8_t*>((base + align - 1) / align * align), bytes};
      break;
    }
  }
  output.result = cx::pack_into(plan, destination);
  output.data   = destination.data();
  stream.sync();
  return output;
}

packed_output pack_to_device(cx::pack_plan const& plan)
{
  return pack_to(plan, destination_kind::device);
}

void expect_materializes_to(cudf::table_view const& expected, cx::packed_data_view const& packed)
{
  auto const materialized = cx::materialize(packed);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, materialized->view());
}

}  // namespace

struct PackUnpackTest : public cudf::test::BaseFixture {
  void verify_column_metadata(cudf::column_view const& col,
                              cudf::packed_metadata_view::column_view const& meta)
  {
    EXPECT_EQ(meta.type(), col.type());
    EXPECT_EQ(meta.num_rows(), col.size());
    EXPECT_EQ(meta.null_count(), col.null_count());
    EXPECT_EQ(meta.num_children(), col.num_children());
    for (cudf::size_type i = 0; i < col.num_children(); i++) {
      verify_column_metadata(col.child(i), meta.child(i));
    }
  }

  void verify_metadata(cudf::table_view const& t, cudf::packed_columns const& packed)
  {
    auto view = cudf::packed_metadata_view(*packed.metadata);
    EXPECT_EQ(view.num_columns(), t.num_columns());
    EXPECT_EQ(view.num_rows(), t.num_rows());
    for (cudf::size_type i = 0; i < t.num_columns(); i++) {
      verify_column_metadata(t.column(i), view.column(i));
    }
  }

  void verify_prepared_compressed_round_trip(cudf::table_view const& input)
  {
    // Host destinations are always windowed; ExperimentalPackIntoHost covers the other pairings.
    constexpr std::array cases{std::pair{false, destination_kind::device},
                               std::pair{true, destination_kind::pinned},
                               std::pair{true, destination_kind::pageable}};
    for (auto const compression : compressed_codecs) {
      if (!is_codec_enabled(compression)) { continue; }
      for (auto const [windowed, kind] : cases) {
        SCOPED_TRACE(static_cast<int>(compression));
        SCOPED_TRACE(windowed);
        SCOPED_TRACE(static_cast<int>(kind));
        auto const options =
          windowed ? make_windowed_options(compression) : make_options(compression);
        auto const plan   = cx::prepare_pack(input, options);
        auto const packed = pack_to(plan, kind);
        EXPECT_LE(packed.result.payload_bytes, plan.sizes().payload_bytes);
        expect_materializes_to(input, packed.view());
      }
    }
  }

  void run_test(cudf::table_view const& t)
  {
    // verify pack/unpack works
    auto packed   = cudf::pack(t);
    auto unpacked = cudf::unpack(packed);
    CUDF_TEST_EXPECT_TABLES_EQUAL(t, unpacked);

    // verify packed_metadata_view matches the unpacked table (which reflects
    // the compacted sizes stored in the packed metadata, not the original sliced sizes)
    verify_metadata(unpacked, packed);

    // verify packed_size returns the correct size
    EXPECT_EQ(cudf::packed_size(t), packed.gpu_data->size());

    // verify pack_metadata itself works
    auto metadata = cudf::pack_metadata(
      unpacked, reinterpret_cast<uint8_t const*>(packed.gpu_data->data()), packed.gpu_data->size());
    EXPECT_EQ(metadata.size(), packed.metadata->size());
    EXPECT_EQ(
      std::equal(metadata.data(), metadata.data() + metadata.size(), packed.metadata->data()),
      true);

    verify_prepared_compressed_round_trip(t);
  }
  void run_test(std::vector<cudf::column_view> const& t) { run_test(cudf::table_view{t}); }
};

TEST_F(PackUnpackTest, ExperimentalPreparedPackInto)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4, 5},
                                                          {true, false, true, true, true});
  cudf::test::strings_column_wrapper strings({"alpha", "", "gamma", "delta", "epsilon"});
  auto const input = cudf::table_view{{numbers, strings}};

  auto const plan = cx::prepare_pack(input, make_options(cx::pack_compression::none));
  EXPECT_EQ(plan.sizes().payload_bytes, cudf::packed_size(input));
  EXPECT_GT(plan.sizes().metadata_bytes, 0);

  // A plan may be reused with another destination while the input remains alive and unchanged.
  for (int execution = 0; execution < 2; ++execution) {
    auto const packed = pack_to_device(plan);
    EXPECT_EQ(packed.result.payload_bytes, packed.buffer.size());
    EXPECT_EQ(packed.result.metadata.size(), plan.sizes().metadata_bytes);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, cx::unpack_view(packed.view()));
    expect_materializes_to(input, packed.view());
  }
}

TEST_F(PackUnpackTest, ExperimentalPackIntoHost)
{
  cudf::test::fixed_width_column_wrapper<int64_t> numbers({10, 20, 30, 40, 50},
                                                          {true, true, false, true, true});
  cudf::test::strings_column_wrapper strings({"mapped", "pinned", "host", "destination", "buffer"});
  auto const input = cudf::table_view{{numbers, strings}};

  for (auto const compression : {cx::pack_compression::none,
                                 cx::pack_compression::cascaded,
                                 cx::pack_compression::zstd,
                                 cx::pack_compression::snappy}) {
    for (auto const windowed : {false, true}) {
      auto const options =
        windowed ? make_windowed_options(compression) : make_options(compression);
      auto const plan = cx::prepare_pack(input, options);
      for (auto const kind : {destination_kind::pinned, destination_kind::pageable}) {
        SCOPED_TRACE(static_cast<int>(compression));
        SCOPED_TRACE(windowed);
        SCOPED_TRACE(static_cast<int>(kind));
        auto const packed = pack_to(plan, kind);
        if (compression == cx::pack_compression::none && kind == destination_kind::pinned) {
          CUDF_TEST_EXPECT_TABLES_EQUAL(input, cx::unpack_view(packed.view()));
        }
        expect_materializes_to(input, packed.view());
      }
    }
  }
}

TEST_F(PackUnpackTest, ExperimentalPackIntoHostBufferLargerThanStaging)
{
  constexpr cudf::size_type num_rows = 256 * 1024;
  auto const values =
    cudf::detail::make_counting_transform_iterator(0, [](int64_t i) { return i; });
  cudf::test::fixed_width_column_wrapper<int64_t> numbers(
    values, values + num_rows, cudf::test::iterators::null_at(7));
  auto const input = cudf::table_view{{numbers}};

  // The 2 MiB data buffer spans several 1 MiB copy batches, each larger than the staging buffer.
  auto options                 = make_options(cx::pack_compression::none);
  options.staging_buffer_bytes = 64 * 1024 + 3 * 64;
  auto const plan              = cx::prepare_pack(input, options);
  for (auto const kind : {destination_kind::pinned, destination_kind::pageable}) {
    SCOPED_TRACE(static_cast<int>(kind));
    expect_materializes_to(input, pack_to(plan, kind).view());
  }
}

TEST_F(PackUnpackTest, ExperimentalExplicitMemoryResources)
{
  cudf::test::fixed_width_column_wrapper<int64_t> numbers({1, 2, 3, 4, 5, 6},
                                                          {true, false, true, true, true, true});
  cudf::test::strings_column_wrapper strings(
    {"explicit", "memory", "", "resources", "for", "pack"});
  auto const input  = cudf::table_view{{numbers, strings}};
  auto const stream = cudf::get_default_stream();

  {
    auto harness = cudf::test::memory_resource_test_harness{};
    // Plan state is owned by the returned plan, so it comes from the output resource; planning
    // scratch comes from the temporary resource.
    auto const plan = [&] {
      auto const scope = harness.fail_on_current_device_resource_use();
      auto result      = cx::prepare_pack(
        input, make_options(cx::pack_compression::none), stream, harness.resources());
      harness.synchronize(stream);
      return result;
    }();
    harness.expect_output_allocations_live(stream);
    auto const temporary_before = harness.expect_temporary_allocation_activity(stream).total;
    harness.expect_temporary_allocations_released(stream);

    auto pinned =
      cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
    auto const packed = [&] {
      auto const scope = harness.fail_on_current_device_resource_use();
      auto result =
        cx::pack_into(plan, std::span<uint8_t>{pinned.data(), pinned.size()}, harness.resources());
      harness.synchronize(stream);
      return result;
    }();
    EXPECT_GT(harness.temporary_mr().get_bytes_counter().total, temporary_before);
    harness.expect_temporary_allocations_released(stream);

    auto const view = cx::packed_data_view{
      packed.metadata, std::span<uint8_t const>{pinned.data(), packed.payload_bytes}};
    auto materialized = [&] {
      auto const scope = harness.fail_on_current_device_resource_use();
      auto result      = cx::materialize(view, stream, harness.resources());
      harness.synchronize(stream);
      return result;
    }();
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
    materialized.reset();
  }

  {
    auto harness    = cudf::test::memory_resource_test_harness{};
    auto const plan = cx::prepare_pack(
      input, make_options(cx::pack_compression::cascaded), stream, harness.resources());
    harness.expect_output_allocations_live(stream);
    auto output       = rmm::device_buffer(plan.sizes().payload_bytes, stream);
    auto const packed = cx::pack_into(plan, as_span(output), harness.resources());
    harness.expect_temporary_allocation_activity(stream);
    harness.expect_temporary_allocations_released(stream);

    auto const temporary_before = harness.temporary_mr().get_bytes_counter().total;
    auto const view             = cx::packed_data_view{
      packed.metadata,
      std::span<uint8_t const>{static_cast<uint8_t const*>(output.data()), packed.payload_bytes}};
    auto materialized = cx::materialize(view, stream, harness.resources());
    harness.expect_temporary_allocations_released(stream);
    EXPECT_GT(harness.temporary_mr().get_bytes_counter().total, temporary_before);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }
}

TEST_F(PackUnpackTest, ExperimentalUnpackViewPageablePayload)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3});
  auto const input  = cudf::table_view{{numbers}};
  auto const packed = pack_to(cx::prepare_pack(input, make_options(cx::pack_compression::none)),
                              destination_kind::pageable);
  int device        = 0;
  int access        = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&access, cudaDevAttrPageableMemoryAccess, device));
  if (access != 0) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, cx::unpack_view(packed.view()));
  } else {
    EXPECT_THROW(cx::unpack_view(packed.view()), cudf::logic_error);
  }
}

TEST_F(PackUnpackTest, ExperimentalMaterializeOversizedUncompressedPayload)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4, 5});
  auto const input  = cudf::table_view{{numbers}};
  auto const plan   = cx::prepare_pack(input, make_options(cx::pack_compression::none));
  auto const stream = cudf::get_default_stream();
  rmm::device_buffer buffer(plan.sizes().payload_bytes + 4096, stream);
  auto const result = cx::pack_into(plan, as_span(buffer));
  auto const view   = cx::packed_data_view{
    result.metadata,
    std::span<uint8_t const>{static_cast<uint8_t const*>(buffer.data()), buffer.size()}};
  auto materialized = cx::materialize(view);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  auto columns = materialized->release();
  EXPECT_EQ(columns.front()->release().data->size(), 5 * sizeof(int32_t));
}

TEST_F(PackUnpackTest, ExperimentalMaterializeColumnSubset)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4, 5, 6},
                                                          {true, false, true, true, true, true});
  cudf::test::strings_column_wrapper strings({"a", "", "ccc", "dddd", "e", "ff"},
                                             {true, true, false, true, true, true});
  cudf::test::lists_column_wrapper<int64_t> lists{{1, 2}, {}, {3}, {4, 5, 6}, {7}, {8}};
  cudf::test::fixed_width_column_wrapper<int16_t> member({1, 2, 3, 4, 5, 6});
  cudf::test::structs_column_wrapper structs({member});
  auto const input = cudf::table_view{{numbers, strings, lists, structs}};
  std::vector<std::vector<cudf::size_type>> const selections{{2, 0}, {1, 1}, {3}, {}};

  for (auto const compression : {cx::pack_compression::none,
                                 cx::pack_compression::automatic,
                                 cx::pack_compression::cascaded,
                                 cx::pack_compression::zstd,
                                 cx::pack_compression::snappy}) {
    auto const plan = cx::prepare_pack(input, make_options(compression));
    for (auto const kind :
         {destination_kind::device, destination_kind::pinned, destination_kind::pageable}) {
      SCOPED_TRACE(static_cast<int>(compression));
      SCOPED_TRACE(static_cast<int>(kind));
      auto const packed = pack_to(plan, kind);
      for (auto const& selection : selections) {
        auto const materialized = cx::materialize(packed.view(), selection);
        ASSERT_EQ(materialized->num_columns(), static_cast<cudf::size_type>(selection.size()));
        if (!selection.empty()) {
          CUDF_TEST_EXPECT_TABLES_EQUAL(input.select(selection), materialized->view());
        }
      }
      for (auto const index : {-1, input.num_columns()}) {
        EXPECT_THROW(cx::materialize(packed.view(), std::vector<cudf::size_type>{index}),
                     std::out_of_range);
      }
    }
  }
}

TEST_F(PackUnpackTest, ExperimentalMaterializeSparseColumnSubset)
{
  // An incompressible middle column makes a pageable subset upload disjoint byte ranges. The first
  // column is stored raw with an odd length, and Cascaded requires aligned input for the last one.
  constexpr cudf::size_type num_rows = 512 * 1024 + 3;
  auto const noise                   = cudf::detail::make_counting_transform_iterator(
    0, [](int64_t i) { return (i * 2654435761) ^ (i << 17) ^ (i >> 5); });
  auto const runs = cudf::detail::make_counting_transform_iterator(
    0, [](int64_t i) { return (i / 3) % 7 + (i / 1000) * 1000; });
  cudf::test::fixed_width_column_wrapper<int8_t> first(noise, noise + num_rows);
  cudf::test::fixed_width_column_wrapper<int64_t> second(noise, noise + num_rows);
  cudf::test::fixed_width_column_wrapper<int64_t> third(runs, runs + num_rows);
  auto const input = cudf::table_view{{first, second, third}};
  std::vector<cudf::size_type> const selection{2, 0};

  for (auto const compression :
       {cx::pack_compression::none, cx::pack_compression::cascaded, cx::pack_compression::snappy}) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto const packed =
      pack_to(cx::prepare_pack(input, make_options(compression)), destination_kind::pageable);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input.select(selection),
                                  cx::materialize(packed.view(), selection)->view());
  }
}

TEST_F(PackUnpackTest, ExperimentalPackIntoRejectsInvalidDestination)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  auto const plan  = cx::prepare_pack(cudf::table_view{{col}});
  auto const sizes = plan.sizes();
  ASSERT_GT(sizes.payload_bytes, 0);
  rmm::device_buffer destination(sizes.payload_bytes + sizes.payload_alignment,
                                 cudf::get_default_stream());
  auto const span = as_span(destination);
  EXPECT_THROW(cx::pack_into(plan, span.first(sizes.payload_bytes - 1)), cudf::logic_error);
  EXPECT_THROW(cx::pack_into(plan, span.subspan(1, sizes.payload_bytes)), cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalCompressedPackMaterialize)
{
  std::vector<int32_t> values(64 * 1024, 7);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  for (auto const compression : compressed_codecs) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto const plan = cx::prepare_pack(input, make_options(compression));
    EXPECT_EQ(plan.sizes().uncompressed_payload_bytes, cudf::packed_size(input));

    // Compressed plans are reusable as well.
    for (int execution = 0; execution < 2; ++execution) {
      auto const packed = pack_to_device(plan);
      EXPECT_GT(packed.result.payload_bytes, 0);
      EXPECT_LE(packed.result.payload_bytes, packed.buffer.size());
      EXPECT_LT(packed.result.payload_bytes, plan.sizes().uncompressed_payload_bytes);
      EXPECT_THROW(cx::unpack_view(packed.view()), cudf::logic_error);
      expect_materializes_to(input, packed.view());
    }
  }
}

TEST_F(PackUnpackTest, ExperimentalCompressExistingPackedColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({31, 31, 31, 31, 31},
                                                          {true, false, true, true, true});
  cudf::test::strings_column_wrapper strings({"late", "compression", "after", "ordinary", "pack"});
  auto const input  = cudf::table_view{{numbers, strings}};
  auto const packed = cudf::pack(input);

  for (auto const compression : compressed_codecs) {
    if (!is_codec_enabled(compression)) { continue; }
    SCOPED_TRACE(static_cast<int>(compression));
    auto const plan = cx::make_pack_plan_builder(packed, make_options(compression)).build();
    EXPECT_EQ(plan.sizes().uncompressed_payload_bytes, packed.gpu_data->size());
    expect_materializes_to(input, pack_to_device(plan).view());
  }
}

TEST_F(PackUnpackTest, ExperimentalExistingPackedColumnsPerRegionCompression)
{
  if (!is_codec_enabled(cx::pack_compression::zstd)) { GTEST_SKIP() << "Zstd is disabled"; }
  std::vector<int32_t> values(4096, 17);
  std::vector<std::string> words(4096, "existing-packed-region-selection");
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  cudf::test::strings_column_wrapper strings(words.begin(), words.end());
  auto const input  = cudf::table_view{{numbers, strings}};
  auto const packed = cudf::pack(input);

  auto builder = cx::make_pack_plan_builder(packed, make_options(cx::pack_compression::none));
  for (auto& region : builder.regions()) {
    if (region.info.kind == cx::pack_region_kind::string_characters) {
      region.codec = cx::pack_compression::zstd;
    }
  }
  auto const plan = std::move(builder).build();
  EXPECT_EQ(plan.sizes().uncompressed_payload_bytes, packed.gpu_data->size());

  auto const result  = pack_to_device(plan);
  auto const entries = read_region_directory(result.result.metadata).entries;
  auto const count   = [&](cx::pack_compression compression) {
    return std::count_if(entries.begin(), entries.end(), [&](auto const& entry) {
      return static_cast<cx::pack_compression>(entry.compression) == compression;
    });
  };
  EXPECT_EQ(count(cx::pack_compression::zstd), 1);
  EXPECT_EQ(count(cx::pack_compression::none), std::ssize(entries) - 1);
  expect_materializes_to(input, result.view());
}

TEST_F(PackUnpackTest, ExperimentalCompressContiguousSplitPartitions)
{
  constexpr cudf::size_type num_rows = 64 * 1024;
  auto const values =
    cudf::detail::make_counting_transform_iterator(0, [](int64_t i) { return (i / 16) % 1000; });
  cudf::test::fixed_width_column_wrapper<int64_t> numbers(
    values, values + num_rows, cudf::test::iterators::null_at(5));
  // Strings and lists are all empty in the partition [1000, 20000).
  auto const is_empty_row = [](cudf::size_type i) { return i >= 1000 && i < 20000; };
  auto const words        = cudf::detail::make_counting_transform_iterator(
    0, [&](int32_t i) { return is_empty_row(i) ? std::string{} : std::to_string(i % 100); });
  cudf::test::strings_column_wrapper strings(words, words + num_rows);
  std::vector<cudf::size_type> list_offsets(num_rows + 1, 0);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    list_offsets[i + 1] = list_offsets[i] + (is_empty_row(i) ? 0 : 2);
  }
  auto const elements =
    cudf::detail::make_counting_transform_iterator(0, [](int32_t i) { return i % 7; });
  auto lists = cudf::make_lists_column(
    num_rows,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(list_offsets.begin(),
                                                            list_offsets.end())
      .release(),
    cudf::test::fixed_width_column_wrapper<int32_t>(elements, elements + list_offsets.back())
      .release(),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  auto const members = cudf::detail::make_counting_transform_iterator(
    0, [](int32_t i) { return static_cast<int16_t>(i % 300); });
  cudf::test::fixed_width_column_wrapper<int16_t> member(members, members + num_rows);
  cudf::test::structs_column_wrapper structs({member}, cudf::test::iterators::null_at(30000));
  auto const input = cudf::table_view{{numbers, strings, lists->view(), structs}};

  std::vector<cudf::size_type> const splits{0, 1000, 20000, 20001, 50000};
  auto const partitions = cudf::contiguous_split(input, splits);
  auto const expected   = cudf::split(input, splits);
  ASSERT_EQ(partitions.size(), expected.size());
  for (std::size_t i = 0; i < partitions.size(); ++i) {
    SCOPED_TRACE(i);
    auto const plan =
      cx::make_pack_plan_builder(partitions[i].data, make_options(cx::pack_compression::cascaded))
        .build();
    expect_materializes_to(expected[i], pack_to_device(plan).view());
  }
}

TEST_F(PackUnpackTest, ExperimentalExistingPackedColumnsUncompressedCopy)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4});
  auto const input  = cudf::table_view{{numbers}};
  auto const packed = cudf::pack(input);
  auto const plan =
    cx::make_pack_plan_builder(packed, make_options(cx::pack_compression::none)).build();
  auto const result = pack_to_device(plan);
  EXPECT_EQ(result.result.payload_bytes, packed.gpu_data->size());
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, cx::unpack_view(result.view()));
}

TEST_F(PackUnpackTest, ExperimentalExistingPackedColumnsRequireMatchingLayout)
{
  // Two columns stored in reverse order: the total size matches the planned layout, but the
  // buffer offsets do not.
  constexpr cudf::size_type num_rows = 64;
  constexpr std::size_t column_bytes = num_rows * sizeof(int32_t);
  auto const stream                  = cudf::get_default_stream();
  auto data = std::make_unique<rmm::device_buffer>(2 * column_bytes, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(data->data(), 0, data->size(), stream.get()));
  auto const* const base = static_cast<uint8_t const*>(data->data());
  auto const type        = cudf::data_type{cudf::type_id::INT32};
  auto const input =
    cudf::table_view{{cudf::column_view{type, num_rows, base + column_bytes, nullptr, 0},
                      cudf::column_view{type, num_rows, base, nullptr, 0}}};
  auto metadata =
    std::make_unique<std::vector<uint8_t>>(cudf::pack_metadata(input, base, data->size()));
  cudf::packed_columns const packed{std::move(metadata), std::move(data)};
  EXPECT_THROW(cx::make_pack_plan_builder(packed, make_options(cx::pack_compression::cascaded)),
               cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalCascadedUsesNativeTypedRegions)
{
  cudf::test::fixed_width_column_wrapper<int16_t> small({1, 1, 2, 3, 5},
                                                        {true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<int64_t> large({10, 20, 30, 40, 50});
  cudf::test::fixed_width_column_wrapper<float> reals({1.5F, 2.5F, 3.5F, 4.5F, 5.5F});
  cudf::test::strings_column_wrapper strings({"typed", "regions", "are", "separate", "frames"});
  auto const input = cudf::table_view{{small, large, reals, strings}};

  auto const plan   = cx::prepare_pack(input, make_options(cx::pack_compression::cascaded));
  auto const packed = pack_to_device(plan);
  ASSERT_EQ(packed.result.metadata.size(), plan.sizes().metadata_bytes);
  auto const entries = read_region_directory(packed.result.metadata).entries;
  ASSERT_GE(entries.size(), 6);
  for (auto const type : {cudf::type_id::INT16,
                          cudf::type_id::INT64,
                          cudf::type_id::FLOAT32,
                          cudf::type_id::STRING}) {
    EXPECT_TRUE(std::any_of(entries.begin(),
                            entries.end(),
                            [type](auto const& entry) {
                              return entry.is_validity == 0 &&
                                     static_cast<cudf::type_id>(entry.type) == type;
                            }))
      << static_cast<int>(type);
  }
  EXPECT_TRUE(std::any_of(
    entries.begin(), entries.end(), [](auto const& entry) { return entry.is_validity != 0; }));

  expect_materializes_to(input, packed.view());
}

TEST_F(PackUnpackTest, ExperimentalAutomaticPerRegionCompression)
{
  constexpr cudf::size_type rows = 32 * 1024;
  std::vector<int32_t> values(rows, 7);
  std::vector<std::string> words(rows, "automatic-region-selection");
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  cudf::test::strings_column_wrapper strings(words.begin(), words.end());
  auto const input = cudf::table_view{{numbers, strings}};

  auto const plan   = cx::prepare_pack(input, make_options(cx::pack_compression::automatic));
  auto const packed = pack_to_device(plan);

  auto const entries = read_region_directory(packed.result.metadata).entries;
  auto const uses    = [&](cx::pack_compression compression) {
    return std::any_of(entries.begin(), entries.end(), [&](auto const& entry) {
      return static_cast<cx::pack_compression>(entry.compression) == compression;
    });
  };
  EXPECT_TRUE(uses(cx::pack_compression::cascaded));
  EXPECT_TRUE(uses(cx::pack_compression::snappy));
  expect_materializes_to(input, packed.view());
}

TEST_F(PackUnpackTest, ExperimentalAutomaticFallsBackToUncompressedRegions)
{
  // Full-width hashed values leave no codec anything to save.
  constexpr cudf::size_type num_rows = 32 * 1024;
  auto const noise = cudf::detail::make_counting_transform_iterator(0, [](int64_t i) {
    auto x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
    x      = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x      = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return static_cast<int64_t>(x ^ (x >> 31));
  });
  cudf::test::fixed_width_column_wrapper<int64_t> numbers(noise, noise + num_rows);
  auto const input = cudf::table_view{{numbers}};

  auto const plan   = cx::prepare_pack(input, make_options(cx::pack_compression::automatic));
  auto const packed = pack_to_device(plan);

  for (auto const& entry : read_region_directory(packed.result.metadata).entries) {
    EXPECT_EQ(static_cast<cx::pack_compression>(entry.compression), cx::pack_compression::none);
  }
  expect_materializes_to(input, packed.view());
}

TEST_F(PackUnpackTest, ExperimentalExpertPerRegionCompression)
{
  std::vector<int32_t> values(4096, 17);
  std::vector<bool> validity(4096, true);
  validity[3] = false;
  std::vector<std::string> words(4096, "expert-region-selection");
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(
    values.begin(), values.end(), validity.begin());
  cudf::test::strings_column_wrapper strings(words.begin(), words.end());
  auto const input = cudf::table_view{{numbers, strings}};

  auto builder = cx::make_pack_plan_builder(input, make_options(cx::pack_compression::automatic));
  std::vector<cx::pack_region_info> observed;
  for (auto& region : builder.regions()) {
    observed.push_back(region.info);
    switch (region.info.kind) {
      case cx::pack_region_kind::validity: region.codec = cx::pack_compression::none; break;
      case cx::pack_region_kind::offsets: region.codec = cx::pack_compression::cascaded; break;
      case cx::pack_region_kind::string_characters:
        region.codec = cx::pack_compression::zstd;
        break;
      case cx::pack_region_kind::data: region.codec = cx::pack_compression::snappy; break;
    }
  }

  auto const plan = std::move(builder).build();
  ASSERT_GE(observed.size(), 4);
  EXPECT_TRUE(std::any_of(observed.begin(), observed.end(), [](auto const& region) {
    return region.column_index == 0 && region.kind == cx::pack_region_kind::data &&
           region.type == cudf::type_id::INT32;
  }));
  EXPECT_TRUE(std::any_of(observed.begin(), observed.end(), [](auto const& region) {
    return region.column_index == 1 && region.kind == cx::pack_region_kind::string_characters;
  }));

  auto const packed = pack_to_device(plan);
  expect_materializes_to(input, packed.view());
}

TEST_F(PackUnpackTest, ExperimentalCompressedInputValidation)
{
  std::vector<int32_t> values(32 * 1024, 23);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  auto const plan   = cx::prepare_pack(input, make_options(cx::pack_compression::zstd));
  auto const packed = pack_to_device(plan);

  ASSERT_GT(packed.result.payload_bytes, 1);
  auto truncated    = packed.view();
  truncated.payload = truncated.payload.first(truncated.payload.size() - 1);
  EXPECT_THROW(cx::materialize(truncated), cudf::logic_error);
}

TEST_F(PackUnpackTest, SingleColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, SingleColumnFixedWidthNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col1({1, 2, 3, 4, 5, 6, 7});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, MultiColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> col2({7, 8, 6, 5, 4, 3, 2},
                                                     {true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<double> col3({8, 4, 2, 0, 7, 1, 3},
                                                      {false, true, true, true, true, true, true});

  this->run_test({col1, col2, col3});
}

TEST_F(PackUnpackTest, MultiColumnWithStrings)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::strings_column_wrapper col2({"Lorem", "ipsum", "dolor", "sit", "amet", "ort", "ral"},
                                          {true, false, true, true, true, false, true});
  cudf::test::strings_column_wrapper col3({"", "this", "is", "a", "column", "of", "strings"});

  this->run_test({col1, col2, col3});
}
// clang-format on

TEST_F(PackUnpackTest, EmptyColumns)
{
  {
    auto empty_string = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }

  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }

  {
    cudf::test::fixed_width_column_wrapper<int> col0;
    cudf::test::dictionary_column_wrapper<int> col1;
    cudf::test::strings_column_wrapper col2;
    cudf::test::lists_column_wrapper<int> col3;
    cudf::test::structs_column_wrapper col4({});

    cudf::table_view src_table({col0, col1, col2, col3, col4});
    this->run_test(src_table);
  }
}

std::vector<std::unique_ptr<cudf::column>> generate_lists(bool include_validity)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  if (include_validity) {
    auto valids = cudf::test::iterators::valids_at_multiples_of(2);
    cudf::test::lists_column_wrapper<int> list0{{1, 2, 3},
                                                {4, 5},
                                                {6},
                                                {{7, 8}, valids},
                                                {9, 10, 11},
                                                LCW{},
                                                LCW{},
                                                {{-1, -2, -3, -4, -5}, valids},
                                                {{100, -200}, valids}};

    cudf::test::lists_column_wrapper<int> list1{{{{1, 2, 3}, valids}, {4, 5}},
                                                {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                                {LCW{6}},
                                                {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                                {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                                {LCW{}},
                                                {LCW{-10}, {-100, -200}},
                                                {{-10, -200}, LCW{}, {8, 9}},
                                                {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

    std::vector<std::unique_ptr<cudf::column>> out;
    out.push_back(list0.release());
    out.push_back(list1.release());
    return out;
  }

  cudf::test::lists_column_wrapper<int> list0{
    {1, 2, 3}, {4, 5}, {6}, {7, 8}, {9, 10, 11}, LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-100, -200}};

  cudf::test::lists_column_wrapper<int> list1{{{1, 2, 3}, {4, 5}},
                                              {LCW{}, LCW{}, {7, 8}, LCW{}},
                                              {LCW{6}},
                                              {{7, 8}, {9, 10, 11}, LCW{}},
                                              {LCW{}, {-1, -2, -3, -4, -5}},
                                              {LCW{}},
                                              {{-10}, {-100, -200}},
                                              {{-10, -200}, LCW{}, {8, 9}},
                                              {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(list0.release());
  out.push_back(list1.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_structs(bool include_validity)
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin())
      : cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end());

  // 3. Boolean "is_human" column.
  std::vector<bool> is_human{true, true, false, false, false, false, true, true, true};
  std::vector<bool> is_human_validity{true, true, true, false, true, true, true, true, false};
  auto is_human_col =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<bool>(
          is_human.begin(), is_human.end(), is_human_validity.begin())
      : cudf::test::fixed_width_column_wrapper<bool>(is_human.begin(), is_human.end());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    include_validity
      ? cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col},
                                           struct_validity.begin())
      : cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col});

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_struct_of_list()
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  std::vector<bool> list_validity{true, true, true, true, true, false, true, false, true};
  cudf::test::lists_column_wrapper<cudf::string_view> list(
    {{{"abc", "d", "edf"}, {"jjj"}},
     {{"dgaer", "-7"}, LCW{}},
     {LCW{}},
     {{"qwerty"}, {"ral", "ort", "tal"}, {"five", "six"}},
     {LCW{}, LCW{}, {"eight", "nine"}},
     {LCW{}},
     {{"fun"}, {"a", "bc", "def", "ghij", "klmno", "pqrstu"}},
     {{"seven", "zz"}, LCW{}, {"xyzzy"}},
     {LCW{"negative 3", "  ", "cleveland"}}},
    list_validity.begin());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<cudf::column>> generate_list_of_struct()
{
  // 1. String "names" column.
  std::vector<std::string> names{"Vimes",
                                 "Carrot",
                                 "Angua",
                                 "Cheery",
                                 "Detritus",
                                 "Slant",
                                 "Fred",
                                 "Todd",
                                 "Kevin",
                                 "Abc",
                                 "Def",
                                 "Xyz",
                                 "Five",
                                 "Seventeen",
                                 "Dol",
                                 "Est"};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102, -1, -2, -3, -4, -5, -6, -7};
  std::vector<bool> ages_validity = {true,
                                     true,
                                     true,
                                     true,
                                     false,
                                     true,
                                     false,
                                     false,
                                     true,
                                     false,
                                     false,
                                     false,
                                     false,
                                     true,
                                     true,
                                     true};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 false,
                                                 false,
                                                 true,
                                                 false,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 true};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column}, struct_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{true, true, true, true, true, false, true, false, true};

  cudf::test::fixed_width_column_wrapper<int> offsets{0, 1, 4, 5, 7, 7, 10, 13, 14, 16};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_validity.begin(), list_validity.begin() + 9);
  auto list = [&] {
    auto tmp = cudf::make_lists_column(
      9, offsets.release(), struct_column.release(), null_count, std::move(null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();

  std::vector<std::unique_ptr<cudf::column>> out;
  out.push_back(std::move(list));
  return out;
}

TEST_F(PackUnpackTest, Lists)
{
  // lists
  {
    auto cols = generate_lists(false);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // lists with validity
  {
    auto cols = generate_lists(true);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, Structs)
{
  // structs
  {
    auto cols = generate_structs(false);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // structs with validity
  {
    auto cols = generate_structs(true);
    std::vector<cudf::column_view> col_views;
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(col_views),
                   [](std::unique_ptr<cudf::column> const& col) {
                     return static_cast<cudf::column_view>(*col);
                   });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, NestedTypes)
{
  // build one big table containing, lists, structs, structs<list>, list<struct>
  std::vector<cudf::column_view> col_views;

  auto lists = generate_lists(true);
  std::transform(
    lists.begin(),
    lists.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto structs = generate_structs(true);
  std::transform(
    structs.begin(),
    structs.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto struct_of_list = generate_struct_of_list();
  std::transform(
    struct_of_list.begin(),
    struct_of_list.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  auto list_of_struct = generate_list_of_struct();
  std::transform(
    list_of_struct.begin(),
    list_of_struct.end(),
    std::back_inserter(col_views),
    [](std::unique_ptr<cudf::column> const& col) { return static_cast<cudf::column_view>(*col); });

  cudf::table_view src_table(col_views);
  this->run_test(src_table);
}

TEST_F(PackUnpackTest, NestedEmpty)
{
  // this produces an empty strings column with no children,
  // nested inside a list
  {
    auto empty_string = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(1,
                                        offsets.release(),
                                        std::move(empty_string),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty strings column with children that have no data,
  // nested inside a list
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(1,
                                        offsets.release(),
                                        std::move(empty_string),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list       = cudf::make_lists_column(1,
                                        offsets.release(),
                                        std::move(empty_list),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list       = cudf::make_lists_column(1,
                                        offsets.release(),
                                        std::move(empty_list),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty struct column with children that have no data,
  // nested inside a list
  {
    cudf::test::fixed_width_column_wrapper<int> ints{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> floats{4, 3, 2, 1, 0};
    auto struct_column = cudf::test::structs_column_wrapper({ints, floats});
    auto empty_struct  = cudf::empty_like(struct_column);
    auto offsets       = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list          = cudf::make_lists_column(1,
                                        offsets.release(),
                                        std::move(empty_struct),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, NestedSliced)
{
  // list
  {
    auto valids = cudf::test::iterators::valids_at_multiples_of(2);

    using LCW = cudf::test::lists_column_wrapper<int>;

    cudf::test::lists_column_wrapper<int> col0{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{6, 12}},
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    cudf::test::strings_column_wrapper col1{
      "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred"};
    cudf::test::fixed_width_column_wrapper<float> col2{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::make_unique<cudf::column>(col2));
    children.push_back(std::make_unique<cudf::column>(col0));
    children.push_back(std::make_unique<cudf::column>(col1));
    auto col3 = cudf::make_structs_column(static_cast<cudf::column_view>(col0).size(),
                                          std::move(children),
                                          0,
                                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    cudf::table_view t({col0, col1, col2, *col3});
    this->run_test(t);
  }

  // struct
  {
    cudf::test::fixed_width_column_wrapper<int> a{0, 1, 2, 3, 4, 5, 6, 7};
    cudf::test::fixed_width_column_wrapper<float> b{
      {0, -1, -2, -3, -4, -5, -6, -7}, {true, true, true, false, false, false, false, true}};
    cudf::test::strings_column_wrapper c{{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"},
                                         {false, false, true, true, true, true, true, true}};
    std::vector<bool> list_validity{true, false, true, false, true, false, true, true};
    cudf::test::lists_column_wrapper<int16_t> d{
      {{0, 1}, {2, 3, 4}, {5, 6}, {7}, {8, 9, 10}, {11, 12}, {}, {15, 16, 17}},
      list_validity.begin()};
    cudf::test::fixed_width_column_wrapper<int> _a{10, 20, 30, 40, 50, 60, 70, 80};
    cudf::test::fixed_width_column_wrapper<float> _b{-10, -20, -30, -40, -50, -60, -70, -80};
    cudf::test::strings_column_wrapper _c{"aa", "", "ccc", "dddd", "eeeee", "f", "gg", "hhh"};
    cudf::test::structs_column_wrapper e({_a, _b, _c},
                                         {true, true, true, false, true, true, true, false});
    cudf::test::structs_column_wrapper s({a, b, c, d, e},
                                         {true, true, false, true, true, true, true, true});

    auto split = cudf::split(s, {2, 5});

    this->run_test(cudf::table_view({split[0]}));
    this->run_test(cudf::table_view({split[1]}));
    this->run_test(cudf::table_view({split[2]}));
  }
}

TEST_F(PackUnpackTest, EmptyTable)
{
  // no columns
  {
    cudf::table_view t;
    this->run_test(t);
  }

  // no rows
  {
    cudf::test::fixed_width_column_wrapper<int> a;
    cudf::test::strings_column_wrapper b;
    cudf::test::lists_column_wrapper<float> c;
    cudf::table_view t({a, b, c});
    this->run_test(t);
  }
}

TEST_F(PackUnpackTest, ZeroColumnsWithRows)
{
  // A zero-column table with rows survives a pack/unpack round-trip.
  cudf::table_view t{std::vector<cudf::column_view>{}, 7};
  auto unpacked = cudf::unpack(cudf::pack(t));
  EXPECT_EQ(unpacked.num_columns(), 0);
  EXPECT_EQ(unpacked.num_rows(), 7);

  // A genuinely empty (0, 0) table round-trips to (0, 0).
  cudf::table_view empty{std::vector<cudf::column_view>{}, 0};
  auto unpacked_empty = cudf::unpack(cudf::pack(empty));
  EXPECT_EQ(unpacked_empty.num_columns(), 0);
  EXPECT_EQ(unpacked_empty.num_rows(), 0);
}

TEST_F(PackUnpackTest, UnpackMetadataSpan)
{
  auto const unpack_and_test = [](cudf::table_view const& input,
                                  cudf::packed_columns const packed) {
    auto unpacked =
      cudf::unpack(*packed.metadata, reinterpret_cast<uint8_t const*>(packed.gpu_data->data()));
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, unpacked);
  };

  cudf::table_view only_rows{std::vector<cudf::column_view>{}, 7};
  unpack_and_test(only_rows, cudf::pack(only_rows));

  cudf::table_view empty{};
  auto empty_packed = cudf::pack(empty);
  ASSERT_TRUE(empty_packed.metadata->empty());
  unpack_and_test(empty, std::move(empty_packed));
}

TEST_F(PackUnpackTest, UnpackMetadataSpanRejectsTruncatedBuffer)
{
  std::vector<uint8_t> truncated_header(1);
  EXPECT_THROW(cudf::unpack(truncated_header, nullptr), cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int> column{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({column}));
  auto truncated_column =
    std::span<uint8_t const>{packed.metadata->data(), packed.metadata->size() - 1};
  EXPECT_THROW(
    cudf::unpack(truncated_column, reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, SlicedEmpty)
{
  // empty sliced column. this is specifically testing the corner case:
  // - a sliced column of size 0
  // - having children that are of size > 0
  //
  cudf::test::strings_column_wrapper a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<int> b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper _a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<float> _b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> _c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::structs_column_wrapper d({_a, _b, _c});

  cudf::table_view t({a, b, c, d});

  auto sliced   = cudf::split(t, {0});
  auto packed   = cudf::pack(t);
  auto unpacked = cudf::unpack(packed);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t, unpacked);
}

TEST_F(PackUnpackTest, LongOffsets)
{
  auto str = make_long_offsets_string_column();
  cudf::table_view tbl({*str});
  this->run_test(tbl);
}

TEST_F(PackUnpackTest, DISABLED_LongOffsetsAndChars)
{
  auto str = make_long_offsets_and_chars_string_column();
  cudf::table_view tbl({*str});
  this->run_test(tbl);
}

TEST_F(PackUnpackTest, MetadataViewEmptyBuffer)
{
  auto packed = cudf::pack(cudf::table_view{});
  ASSERT_TRUE(packed.metadata->empty());
  auto view = cudf::packed_metadata_view(*packed.metadata);
  EXPECT_EQ(view.num_columns(), 0);
  EXPECT_EQ(view.num_rows(), 0);
  EXPECT_THROW(std::ignore = view.column(0), std::out_of_range);
}

TEST_F(PackUnpackTest, MetadataViewRejectsNonMultipleSize)
{
  // Pack a valid table, then present its metadata with one byte chopped off
  // so the size is not a multiple of the serialized entry size.
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));
  ASSERT_GT(packed.metadata->size(), 1);
  auto truncated = std::span<uint8_t const>(packed.metadata->data(), packed.metadata->size() - 1);
  EXPECT_THROW(cudf::packed_metadata_view{truncated}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsTruncatedBuffer)
{
  // Pack a multi-column table, then lop off one serialized entry so
  // the header claims more columns than the buffer actually contains.
  cudf::test::fixed_width_column_wrapper<int> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col2{4.0f, 5.0f, 6.0f};
  cudf::test::fixed_width_column_wrapper<double> col3{7.0, 8.0, 9.0};
  auto packed = cudf::pack(cudf::table_view({col1, col2, col3}));

  // Metadata has a table header plus 3 column entries. Remove the last entry so
  // the header still says "3 columns" but only 2 column entries remain.
  auto const entry_size     = (packed.metadata->size() - metadata_header_size) / 3;
  auto const truncated_size = packed.metadata->size() - entry_size;

  auto truncated = std::span<uint8_t const>(packed.metadata->data(), truncated_size);
  EXPECT_THROW(cudf::packed_metadata_view{truncated}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsTooLongBuffer)
{
  // Pack a valid table, then extend the buffer by one entry so the tree
  // doesn't consume the entire buffer.
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto const entry_size = packed.metadata->size() - metadata_header_size;  // 1 column entry
  auto extended         = *packed.metadata;
  extended.resize(packed.metadata->size() + entry_size, 0);

  EXPECT_THROW(cudf::packed_metadata_view{extended}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewRejectsCorruptedChildCount)
{
  // Pack a table with a struct column, then corrupt the struct's num_children
  // field so traversal would read past the buffer.
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));

  // The metadata layout is: [header, struct, ints_child, floats_child].
  // The struct entry is the first column entry. We corrupt its num_children from 2 to
  // something larger so the tree claims more entries than exist.
  auto corrupted = *packed.metadata;

  // The num_children field is the
  // second-to-last 4-byte value in each entry (before the trailing pad).
  auto const entry_size = (corrupted.size() - metadata_header_size) / 3;  // 3 column entries
  auto const num_children_offset = metadata_header_size                   // skip table header
                                   + entry_size - 2 * sizeof(int32_t);    // num_children in struct
  cudf::size_type bad_children = 10;
  std::memcpy(corrupted.data() + num_children_offset, &bad_children, sizeof(bad_children));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsNegativeColumnCount)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted = *packed.metadata;
  // num_columns follows the leading version field in the header.
  auto constexpr num_columns_offset = sizeof(std::int32_t);
  cudf::size_type const negative    = -1;
  std::memcpy(corrupted.data() + num_columns_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

// num_rows follows the leading version and num_columns fields in the header.
auto constexpr num_rows_offset = 2 * sizeof(std::int32_t);

TEST_F(PackUnpackTest, MetadataRejectsNegativeRowCount)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted                 = *packed.metadata;
  cudf::size_type const negative = -1;
  std::memcpy(corrupted.data() + num_rows_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsRowCountInconsistentWithColumns)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  // The column has 3 rows; a header row count that disagrees must be rejected.
  auto corrupted              = *packed.metadata;
  cudf::size_type const wrong = 99;
  std::memcpy(corrupted.data() + num_rows_offset, &wrong, sizeof(wrong));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsRowCountInconsistentAcrossColumns)
{
  cudf::test::fixed_width_column_wrapper<int> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int> col2{4, 5, 6};
  auto packed = cudf::pack(cudf::table_view({col1, col2}));

  // Leave the header and column 0 at 3 rows but corrupt column 1's size to 4. Validating only
  // the first column would miss this; every top-level column must match the recorded row count.
  auto corrupted        = *packed.metadata;
  auto const entry_size = (corrupted.size() - metadata_header_size) / 2;  // 2 column entries
  // size is the first field after the 8-byte data_type (two int32s) in each entry.
  auto const col1_size_offset = metadata_header_size + entry_size + 2 * sizeof(std::int32_t);
  cudf::size_type const wrong = 4;
  std::memcpy(corrupted.data() + col1_size_offset, &wrong, sizeof(wrong));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsUnsupportedVersion)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));

  auto corrupted = *packed.metadata;
  // The version is the leading value of the header.
  std::int32_t const unknown_version = 999;
  std::memcpy(corrupted.data(), &unknown_version, sizeof(unknown_version));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataRejectsNegativeChildCount)
{
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));

  auto corrupted        = *packed.metadata;
  auto const entry_size = (corrupted.size() - metadata_header_size) / 3;  // 3 column entries
  auto const num_children_offset = metadata_header_size + entry_size - 2 * sizeof(int32_t);
  cudf::size_type const negative = -1;
  std::memcpy(corrupted.data() + num_children_offset, &negative, sizeof(negative));

  EXPECT_THROW(cudf::packed_metadata_view{corrupted}, cudf::logic_error);
  EXPECT_THROW(
    cudf::unpack(corrupted.data(), reinterpret_cast<uint8_t const*>(packed.gpu_data->data())),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, MetadataViewColumnIndexOutOfRange)
{
  cudf::test::fixed_width_column_wrapper<int> col{1, 2, 3};
  auto packed = cudf::pack(cudf::table_view({col}));
  auto view   = cudf::packed_metadata_view(*packed.metadata);

  EXPECT_THROW(std::ignore = view.column(-1), std::out_of_range);
  EXPECT_THROW(std::ignore = view.column(view.num_columns()), std::out_of_range);
}

TEST_F(PackUnpackTest, MetadataViewChildIndexOutOfRange)
{
  cudf::test::fixed_width_column_wrapper<int> ints{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> floats{4.0f, 5.0f, 6.0f};
  auto struct_col = cudf::test::structs_column_wrapper({ints, floats});
  auto packed     = cudf::pack(cudf::table_view({struct_col}));
  auto view       = cudf::packed_metadata_view(*packed.metadata);
  auto col_meta   = view.column(0);

  EXPECT_THROW(std::ignore = col_meta.child(-1), std::out_of_range);
  EXPECT_THROW(std::ignore = col_meta.child(col_meta.num_children()), std::out_of_range);
}
