/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/copying/slice_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

// Size of the serialized table header that precedes the column entries in the
// packed metadata buffer: version + num_columns + num_rows + pad, four 4-byte fields.
// Must match `serialized_table_header` in cpp/src/copying/pack.cpp.
auto constexpr metadata_header_size = 4 * sizeof(cudf::size_type);

namespace {

struct compressed_region_header {
  uint64_t magic;
  uint32_t version;
  uint32_t num_regions;
  uint64_t legacy_metadata_bytes;
  uint64_t uncompressed_payload_bytes;
};

struct compressed_region_entry {
  uint64_t uncompressed_offset;
  uint64_t uncompressed_bytes;
  uint64_t payload_offset;
  uint64_t payload_bytes;
  int32_t type;
  uint32_t is_validity;
  int32_t compression;
  uint32_t reserved;
};

template <typename T>
T read_compressed_region_metadata(std::vector<uint8_t> const& metadata, std::size_t offset)
{
  T value;
  std::memcpy(&value, metadata.data() + offset, sizeof(T));
  return value;
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

  void verify_prepared_compressed_round_trip(cudf::table_view const& input,
                                             cudf::experimental::pack_compression compression)
  {
    auto const stream   = cudf::get_default_stream();
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);
    rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan,
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
    auto const packed_view = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression};
    auto materialized = cudf::experimental::materialize(packed_view, stream);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }

  void run_test(cudf::table_view const& t)
  {
    // verify pack/unpack works
    auto packed   = cudf::pack(t);
    auto unpacked = cudf::unpack(packed);
    CUDF_TEST_EXPECT_TABLES_EQUAL(t, unpacked);

    // verify packed_metadata_view matches the unpacked table (which reflects
    // the compacted sizes stored in the packed metadata, not the original sliced sizes)
    if (!packed.metadata->empty()) { verify_metadata(unpacked, packed); }

    // verify packed_size returns the correct size
    EXPECT_EQ(cudf::packed_size(t), packed.gpu_data->size());

    // verify pack_metadata itself works
    auto metadata = cudf::pack_metadata(
      unpacked, reinterpret_cast<uint8_t const*>(packed.gpu_data->data()), packed.gpu_data->size());
    EXPECT_EQ(metadata.size(), packed.metadata->size());
    EXPECT_EQ(
      std::equal(metadata.data(), metadata.data() + metadata.size(), packed.metadata->data()),
      true);

    for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                   cudf::experimental::pack_compression::zstd,
                                   cudf::experimental::pack_compression::snappy}) {
      SCOPED_TRACE(static_cast<int>(compression));
      verify_prepared_compressed_round_trip(t, compression);
    }
  }
  void run_test(std::vector<cudf::column_view> const& t) { run_test(cudf::table_view{t}); }
};

TEST_F(PackUnpackTest, ExperimentalPreparedPackInto)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4, 5},
                                                          {true, false, true, true, true});
  cudf::test::strings_column_wrapper strings({"alpha", "", "gamma", "delta", "epsilon"});
  auto const input = cudf::table_view{{numbers, strings}};

  auto const plan_sizes = cudf::experimental::prepare_pack(input, stream).sizes();
  EXPECT_EQ(plan_sizes.payload_bytes, cudf::packed_size(input, stream));
  EXPECT_GT(plan_sizes.metadata_bytes, 0);

  // Build a second plan for execution. This also verifies that sizing is stable across plans.
  auto plan = cudf::experimental::prepare_pack(input, stream);
  EXPECT_EQ(plan.sizes().payload_bytes, plan_sizes.payload_bytes);

  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto const destination_span =
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()};
  auto result = cudf::experimental::pack_into(plan, destination_span);

  EXPECT_EQ(result.payload_bytes, destination.size());
  EXPECT_EQ(result.metadata.size(), plan.sizes().metadata_bytes);

  auto const packed_view = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                     destination.size()}};
  auto const unpacked = cudf::experimental::unpack_view(packed_view);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, unpacked);

  auto materialized = cudf::experimental::materialize(packed_view, stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());

  // A plan may be reused with another destination while the input remains alive and unchanged.
  rmm::device_buffer second_destination(plan.sizes().payload_bytes, stream);
  auto second_result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(second_destination.data()),
                               second_destination.size()});
  auto const second_view = cudf::experimental::packed_data_view{
    second_result.metadata,
    cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(second_destination.data()),
                                     second_destination.size()}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, cudf::experimental::unpack_view(second_view));
}

TEST_F(PackUnpackTest, ExperimentalPackIntoPinnedHost)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int64_t> numbers({10, 20, 30, 40, 50},
                                                          {true, true, false, true, true});
  cudf::test::strings_column_wrapper strings({"mapped", "pinned", "host", "destination", "buffer"});
  auto const input = cudf::table_view{{numbers, strings}};
  auto plan        = cudf::experimental::prepare_pack(input, stream);

  auto destination =
    cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
  ASSERT_TRUE(destination.get_allocator().is_device_accessible());

  auto result = cudf::experimental::pack_into(
    plan, cudf::device_span<uint8_t>{destination.data(), destination.size()});
  auto const packed_view = cudf::experimental::packed_data_view{
    result.metadata, cudf::device_span<uint8_t const>{destination.data(), destination.size()}};

  // The packed view directly references mapped host memory. Materialization performs the owning
  // host-to-device copy without an intermediate packed device allocation.
  auto const unpacked = cudf::experimental::unpack_view(packed_view);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, unpacked);
  auto materialized = cudf::experimental::materialize(packed_view, stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalPackIntoRejectsSmallDestination)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  auto plan = cudf::experimental::prepare_pack(cudf::table_view{{col}});
  ASSERT_GT(plan.sizes().payload_bytes, 0);
  rmm::device_buffer destination(plan.sizes().payload_bytes - 1, cudf::get_default_stream());
  EXPECT_THROW(
    cudf::experimental::pack_into(
      plan,
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()}),
    cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalPackIntoRejectsMisalignedDestination)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4});
  auto plan = cudf::experimental::prepare_pack(cudf::table_view{{col}}, stream);
  ASSERT_GT(plan.sizes().payload_bytes, 0);
  rmm::device_buffer destination(plan.sizes().payload_bytes + plan.sizes().payload_alignment,
                                 stream);
  auto* const misaligned = static_cast<uint8_t*>(destination.data()) + 1;
  EXPECT_THROW(cudf::experimental::pack_into(
                 plan, cudf::device_span<uint8_t>{misaligned, plan.sizes().payload_bytes}),
               cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalCascadedPackMaterialize)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(64 * 1024, 7);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  auto plan           = cudf::experimental::prepare_pack(input, options, stream);

  EXPECT_EQ(plan.sizes().uncompressed_payload_bytes, cudf::packed_size(input, stream));
  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

  EXPECT_EQ(result.compression, cudf::experimental::pack_compression::cascaded);
  EXPECT_GT(result.payload_bytes, 0);
  EXPECT_LE(result.payload_bytes, destination.size());
  EXPECT_LT(result.payload_bytes, plan.sizes().uncompressed_payload_bytes);

  auto const packed_view = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                     result.payload_bytes},
    result.compression};
  EXPECT_THROW(cudf::experimental::unpack_view(packed_view), cudf::logic_error);
  auto materialized = cudf::experimental::materialize(packed_view, stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalCompressExistingPackedColumns)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({31, 31, 31, 31, 31},
                                                          {true, false, true, true, true});
  cudf::test::strings_column_wrapper strings({"late", "compression", "after", "ordinary", "pack"});
  auto const input = cudf::table_view{{numbers, strings}};
  auto packed      = cudf::pack(input, stream);

  for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                 cudf::experimental::pack_compression::zstd,
                                 cudf::experimental::pack_compression::snappy}) {
    for (auto const output_mode : {cudf::experimental::compressed_output_mode::compact,
                                   cudf::experimental::compressed_output_mode::reserved}) {
      SCOPED_TRACE(static_cast<int>(compression));
      SCOPED_TRACE(static_cast<int>(output_mode));
      auto options        = cudf::experimental::pack_options{};
      options.compression = compression;
      options.output_mode = output_mode;
      auto plan           = cudf::experimental::prepare_pack(packed, options, stream);

      EXPECT_EQ(plan.sizes().uncompressed_payload_bytes, packed.gpu_data->size());
      rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
      auto result = cudf::experimental::pack_into(
        plan,
        cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
      auto const packed_view = cudf::experimental::packed_data_view{
        result.metadata,
        cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                         result.payload_bytes},
        result.compression};
      auto materialized = cudf::experimental::materialize(packed_view, stream);
      CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
    }
  }

  auto expert_options        = cudf::experimental::pack_options{};
  expert_options.compression = cudf::experimental::pack_compression::automatic;
  auto builder = cudf::experimental::make_pack_plan_builder(packed, expert_options, stream);
  for (auto& region : builder.regions()) {
    region.options.codec =
      region.info.kind == cudf::experimental::pack_region_kind::string_characters
        ? cudf::experimental::pack_compression::snappy
        : cudf::experimental::pack_compression::cascaded;
  }
  auto expert_plan = std::move(builder).build();
  rmm::device_buffer destination(expert_plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    expert_plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
  auto materialized = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalExistingPackedColumnsRequireCompression)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4});
  auto packed = cudf::pack(cudf::table_view{{numbers}});
  EXPECT_THROW(cudf::experimental::prepare_pack(
                 packed, cudf::experimental::pack_options{}, cudf::get_default_stream()),
               cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalCascadedUsesNativeTypedRegions)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int16_t> small({1, 1, 2, 3, 5},
                                                        {true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<int64_t> large({10, 20, 30, 40, 50});
  cudf::test::fixed_width_column_wrapper<float> reals({1.5F, 2.5F, 3.5F, 4.5F, 5.5F});
  cudf::test::strings_column_wrapper strings({"typed", "regions", "are", "separate", "frames"});
  auto const input = cudf::table_view{{small, large, reals, strings}};

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  auto plan           = cudf::experimental::prepare_pack(input, options, stream);
  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

  ASSERT_EQ(result.metadata.size(), plan.sizes().metadata_bytes);
  auto const header = read_compressed_region_metadata<compressed_region_header>(result.metadata, 0);
  EXPECT_EQ(header.magic, 0x4355444650524547ULL);
  EXPECT_EQ(header.version, 2);
  ASSERT_GE(header.num_regions, 6);

  bool saw_int16               = false;
  bool saw_int64               = false;
  bool saw_float32             = false;
  bool saw_string              = false;
  bool saw_validity            = false;
  std::size_t uncompressed_end = 0;
  auto const* payload          = static_cast<uint8_t const*>(destination.data());
  for (std::size_t i = 0; i < header.num_regions; ++i) {
    auto const entry = read_compressed_region_metadata<compressed_region_entry>(
      result.metadata, sizeof(compressed_region_header) + i * sizeof(compressed_region_entry));
    EXPECT_EQ(entry.uncompressed_offset, uncompressed_end);
    uncompressed_end += entry.uncompressed_bytes;
    EXPECT_LE(entry.payload_offset + entry.payload_bytes, result.payload_bytes);
    auto const type = static_cast<cudf::type_id>(entry.type);
    saw_int16 |= type == cudf::type_id::INT16 && entry.is_validity == 0;
    saw_int64 |= type == cudf::type_id::INT64 && entry.is_validity == 0;
    saw_float32 |= type == cudf::type_id::FLOAT32 && entry.is_validity == 0;
    saw_string |= type == cudf::type_id::STRING && entry.is_validity == 0;
    saw_validity |= entry.is_validity != 0;
  }
  EXPECT_EQ(uncompressed_end, header.uncompressed_payload_bytes);
  EXPECT_TRUE(saw_int16);
  EXPECT_TRUE(saw_int64);
  EXPECT_TRUE(saw_float32);
  EXPECT_TRUE(saw_string);
  EXPECT_TRUE(saw_validity);

  auto const packed_view = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{payload, result.payload_bytes},
    result.compression};
  auto materialized = cudf::experimental::materialize(packed_view, stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalAutomaticPerRegionCompression)
{
  auto const stream              = cudf::get_default_stream();
  constexpr cudf::size_type rows = 32 * 1024;
  std::vector<int32_t> values(rows, 7);
  std::vector<std::string> words(rows, "automatic-region-selection");
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  cudf::test::strings_column_wrapper strings(words.begin(), words.end());
  auto const input = cudf::table_view{{numbers, strings}};

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::automatic;
  auto plan           = cudf::experimental::prepare_pack(input, options, stream);
  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

  EXPECT_EQ(result.compression, cudf::experimental::pack_compression::automatic);
  auto const header = read_compressed_region_metadata<compressed_region_header>(result.metadata, 0);
  bool saw_cascaded = false;
  bool saw_snappy   = false;
  for (std::size_t i = 0; i < header.num_regions; ++i) {
    auto const entry = read_compressed_region_metadata<compressed_region_entry>(
      result.metadata, sizeof(compressed_region_header) + i * sizeof(compressed_region_entry));
    auto const compression = static_cast<cudf::experimental::pack_compression>(entry.compression);
    saw_cascaded |= compression == cudf::experimental::pack_compression::cascaded;
    saw_snappy |= compression == cudf::experimental::pack_compression::snappy;
  }
  EXPECT_TRUE(saw_cascaded);
  EXPECT_TRUE(saw_snappy);

  auto materialized = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalAutomaticFallsBackToUncompressedRegions)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(32 * 1024, 11);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  auto options                        = cudf::experimental::pack_options{};
  options.compression                 = cudf::experimental::pack_compression::automatic;
  options.automatic_min_savings_bytes = std::numeric_limits<std::size_t>::max();
  auto plan                           = cudf::experimental::prepare_pack(input, options, stream);
  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

  auto const header = read_compressed_region_metadata<compressed_region_header>(result.metadata, 0);
  for (std::size_t i = 0; i < header.num_regions; ++i) {
    auto const entry = read_compressed_region_metadata<compressed_region_entry>(
      result.metadata, sizeof(compressed_region_header) + i * sizeof(compressed_region_entry));
    EXPECT_EQ(static_cast<cudf::experimental::pack_compression>(entry.compression),
              cudf::experimental::pack_compression::none);
  }

  auto materialized = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalExpertPerRegionCompression)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(4096, 17);
  std::vector<bool> validity(4096, true);
  validity[3] = false;
  std::vector<std::string> words(4096, "expert-region-selection");
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(
    values.begin(), values.end(), validity.begin());
  cudf::test::strings_column_wrapper strings(words.begin(), words.end());
  auto const input = cudf::table_view{{numbers, strings}};

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::automatic;
  auto builder        = cudf::experimental::make_pack_plan_builder(input, options, stream);
  std::vector<cudf::experimental::pack_region_info> observed;
  for (auto& region : builder.regions()) {
    observed.push_back(region.info);
    region.options.compression_chunk_bytes = 32 * 1024;
    switch (region.info.kind) {
      case cudf::experimental::pack_region_kind::validity:
        region.options.codec = cudf::experimental::pack_compression::none;
        break;
      case cudf::experimental::pack_region_kind::offsets:
        region.options.codec                   = cudf::experimental::pack_compression::cascaded;
        region.options.cascaded_num_RLEs       = 1;
        region.options.cascaded_num_deltas     = 1;
        region.options.cascaded_use_bitpacking = true;
        break;
      case cudf::experimental::pack_region_kind::string_characters:
        region.options.codec = cudf::experimental::pack_compression::zstd;
        break;
      case cudf::experimental::pack_region_kind::data:
        region.options.codec = cudf::experimental::pack_compression::snappy;
        break;
    }
  }

  auto plan = std::move(builder).build();
  ASSERT_GE(observed.size(), 4);
  EXPECT_TRUE(std::any_of(observed.begin(), observed.end(), [](auto const& region) {
    return region.column_index == 0 && region.kind == cudf::experimental::pack_region_kind::data &&
           region.type == cudf::type_id::INT32;
  }));
  EXPECT_TRUE(std::any_of(observed.begin(), observed.end(), [](auto const& region) {
    return region.column_index == 1 &&
           region.kind == cudf::experimental::pack_region_kind::string_characters;
  }));

  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
  EXPECT_EQ(result.compression, cudf::experimental::pack_compression::automatic);

  auto materialized = cudf::experimental::materialize(
    cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression},
    stream);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
}

TEST_F(PackUnpackTest, ExperimentalReservedTypedRegions)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> numbers({1, 2, 3, 4}, {true, false, true, true});
  cudf::test::strings_column_wrapper strings({"one", "two", "three", "four"});
  auto const input = cudf::table_view{{numbers, strings}};

  for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                 cudf::experimental::pack_compression::zstd,
                                 cudf::experimental::pack_compression::snappy}) {
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    options.output_mode = cudf::experimental::compressed_output_mode::reserved;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);
    rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan,
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
    auto const packed_view = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression};
    auto materialized = cudf::experimental::materialize(packed_view, stream);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }
}

TEST_F(PackUnpackTest, ExperimentalCompressedPackIntoPinnedHost)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int64_t> values(32 * 1024, 42);
  cudf::test::fixed_width_column_wrapper<int64_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                 cudf::experimental::pack_compression::zstd,
                                 cudf::experimental::pack_compression::snappy}) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);
    auto destination =
      cudf::detail::make_pinned_vector_async<uint8_t>(plan.sizes().payload_bytes, stream);
    ASSERT_TRUE(destination.get_allocator().is_device_accessible());

    auto result = cudf::experimental::pack_into(
      plan, cudf::device_span<uint8_t>{destination.data(), destination.size()});
    auto const packed_view = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{destination.data(), result.payload_bytes},
      result.compression};
    auto materialized = cudf::experimental::materialize(packed_view, stream);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }
}

TEST_F(PackUnpackTest, ExperimentalCompressedPlanReuse)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(32 * 1024, 17);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                 cudf::experimental::pack_compression::zstd,
                                 cudf::experimental::pack_compression::snappy}) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);

    for (int execution = 0; execution < 2; ++execution) {
      rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
      auto result = cudf::experimental::pack_into(
        plan,
        cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
      auto const packed_view = cudf::experimental::packed_data_view{
        result.metadata,
        cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                         result.payload_bytes},
        result.compression};
      auto materialized = cudf::experimental::materialize(packed_view, stream);
      CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
    }
  }
}

TEST_F(PackUnpackTest, ExperimentalReservedCompressedOutput)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(32 * 1024, 19);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  for (auto const compression : {cudf::experimental::pack_compression::cascaded,
                                 cudf::experimental::pack_compression::zstd,
                                 cudf::experimental::pack_compression::snappy}) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    options.output_mode = cudf::experimental::compressed_output_mode::reserved;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);
    rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan,
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

    EXPECT_EQ(result.output_mode, cudf::experimental::compressed_output_mode::reserved);
    EXPECT_EQ(result.payload_bytes, plan.sizes().payload_bytes);
    auto const packed_view = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression};
    auto materialized = cudf::experimental::materialize(packed_view, stream);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }
}

TEST_F(PackUnpackTest, ExperimentalCompressedInputValidation)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(32 * 1024, 23);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  auto options        = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::zstd;
  auto plan           = cudf::experimental::prepare_pack(input, options, stream);
  rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
  auto result = cudf::experimental::pack_into(
    plan,
    cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});
  auto const payload = static_cast<uint8_t const*>(destination.data());

  auto const wrong_codec = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{payload, result.payload_bytes},
    cudf::experimental::pack_compression::snappy};
  EXPECT_THROW(cudf::experimental::materialize(wrong_codec, stream), cudf::logic_error);

  ASSERT_GT(result.payload_bytes, 1);
  auto const truncated = cudf::experimental::packed_data_view{
    result.metadata,
    cudf::device_span<uint8_t const>{payload, result.payload_bytes - 1},
    result.compression};
  EXPECT_THROW(cudf::experimental::materialize(truncated, stream), cudf::logic_error);
}

TEST_F(PackUnpackTest, ExperimentalZstdAndSnappyPackMaterialize)
{
  auto const stream = cudf::get_default_stream();
  std::vector<int32_t> values(64 * 1024, 123);
  cudf::test::fixed_width_column_wrapper<int32_t> numbers(values.begin(), values.end());
  auto const input = cudf::table_view{{numbers}};

  for (auto const compression :
       {cudf::experimental::pack_compression::zstd, cudf::experimental::pack_compression::snappy}) {
    SCOPED_TRACE(static_cast<int>(compression));
    auto options        = cudf::experimental::pack_options{};
    options.compression = compression;
    auto plan           = cudf::experimental::prepare_pack(input, options, stream);
    rmm::device_buffer destination(plan.sizes().payload_bytes, stream);
    auto result = cudf::experimental::pack_into(
      plan,
      cudf::device_span<uint8_t>{static_cast<uint8_t*>(destination.data()), destination.size()});

    EXPECT_EQ(result.compression, compression);
    EXPECT_GT(result.payload_bytes, 0);
    EXPECT_LE(result.payload_bytes, destination.size());
    EXPECT_LT(result.payload_bytes, plan.sizes().uncompressed_payload_bytes);

    auto const packed_view = cudf::experimental::packed_data_view{
      result.metadata,
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(destination.data()),
                                       result.payload_bytes},
      result.compression};
    auto materialized = cudf::experimental::materialize(packed_view, stream);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, materialized->view());
  }
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
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty strings column with children that have no data,
  // nested inside a list
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

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
    auto list          = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_struct), 0, rmm::device_buffer{});

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
    auto col3 = cudf::make_structs_column(
      static_cast<cudf::column_view>(col0).size(), std::move(children), 0, rmm::device_buffer{});

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
