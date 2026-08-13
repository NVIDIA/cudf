/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Copy fixed-width column data to a host vector
 */
template <typename T>
auto host_row_mask_data(cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  return cudf::detail::make_host_vector<T>(
    cudf::device_span<T const>(column.data<T>(), static_cast<size_t>(column.size())), stream);
}

/**
 * @brief Creates a parquet buffer with zero-rows and same schema as table from
 * `create_parquet_with_stats`
 */
template <typename T>
std::vector<char> create_empty_parquet_with_stats()
{
  auto const non_empty = std::get<0>(create_parquet_with_stats<T, 1>());
  auto const empty     = cudf::empty_like(non_empty->view());

  cudf::io::table_input_metadata output_metadata(empty->view());
  output_metadata.column_metadata[0].set_name("col0");
  output_metadata.column_metadata[1].set_name("col1");
  output_metadata.column_metadata[2].set_name("col2");

  std::vector<char> buffer;
  auto out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, empty->view())
      .metadata(std::move(output_metadata))
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::write_parquet(out_opts);
  return buffer;
}

/**
 * @brief Build a scalar literal matching a filter column type
 */
template <typename T>
auto make_scalar(cudf::size_type value, rmm::cuda_stream_view stream)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return cudf::timestamp_scalar<T>(T(typename T::duration(value)), true, stream);
  } else if constexpr (cudf::is_duration<T>()) {
    return cudf::duration_scalar<T>(T(value), true, stream);
  } else {
    return cudf::numeric_scalar<T>(static_cast<T>(value), true, stream);
  }
}

}  // namespace

struct HybridScanMultifileFiltersTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanMultifileFiltersTest, Metadata)
{
  using T = cudf::timestamp_ms;

  // Create two parquet sources, each with 4 row groups and 5000 rows per row
  // group
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto constexpr num_sources        = 2;

  // Build sources with different seeds
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr num_concat = 1;
  auto constexpr seed       = 0xbad;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, num_concat>());
                 });

  // Filtering AST - col0 < 100
  auto literal_value     = make_scalar<T>(100, cudf::get_default_stream());
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Construct reader from footer bytes
  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    cudf::host_span<cudf::host_span<uint8_t const> const>{inputs.footer_byte_spans}, options);

  // Get parquet metadata and check
  auto parquet_metadata = reader->parquet_metadatas();
  ASSERT_EQ(parquet_metadata.size(), num_sources);
  for (auto const& meta : parquet_metadata) {
    ASSERT_FALSE(meta.row_groups.empty());
    EXPECT_FALSE(meta.row_groups[0].columns[0].offset_index.has_value());
    EXPECT_FALSE(meta.row_groups[0].columns[0].column_index.has_value());
  }

  // Setup page index
  auto const page_index_byte_ranges = reader->page_index_byte_ranges();
  ASSERT_EQ(page_index_byte_ranges.size(), num_sources);
  EXPECT_TRUE(std::all_of(page_index_byte_ranges.begin(),
                          page_index_byte_ranges.end(),
                          [](auto const& range) { return not range.is_empty(); }));

  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> page_index_buffers;
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  page_index_buffers.reserve(num_sources);
  page_index_byte_spans.reserve(num_sources);

  auto iter = cuda::zip_iterator(page_index_byte_ranges.begin(), inputs.datasources.begin());
  std::for_each(iter, iter + num_sources, [&](auto const& pair) {
    auto const& [pgidx_byte_range, datasource] = pair;
    page_index_buffers.emplace_back(
      cudf::io::parquet::fetch_page_index_to_host(*datasource, pgidx_byte_range));
    page_index_byte_spans.emplace_back(*page_index_buffers.back());
  });

  reader->setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const>{page_index_byte_spans});

  // Check if page index is now present in each parquet metadata
  parquet_metadata = reader->parquet_metadatas();
  for (auto const& meta : parquet_metadata) {
    EXPECT_TRUE(meta.row_groups[0].columns[0].offset_index.has_value());
    EXPECT_TRUE(meta.row_groups[0].columns[0].column_index.has_value());
  }

  // Check all row groups
  auto input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_TRUE(std::all_of(
    input_row_group_indices.begin(), input_row_group_indices.end(), [](auto const& rgs) {
      return rgs == (std::vector<cudf::size_type>{0, 1, 2, 3});
    }));

  // Set explicit row groups (per-source) via options
  options.set_row_groups({{0, 1}, {2, 3}});
  input_row_group_indices = reader->all_row_groups(options);

  // Check if the row groups are set correctly
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_EQ(input_row_group_indices[0], (std::vector<cudf::size_type>{0, 1}));
  EXPECT_EQ(input_row_group_indices[1], (std::vector<cudf::size_type>{2, 3}));
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices),
            2 * rows_per_row_group * num_sources);

  // Construct a new reader from a span of existing FileMetaData
  auto const reader_with_existing_metadata =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      cudf::host_span<cudf::io::parquet::FileMetaData const>{parquet_metadata}, options);

  // Check if the new metadata is the same as the existing one
  auto const new_metadata = reader_with_existing_metadata->parquet_metadatas();
  ASSERT_EQ(new_metadata.size(), num_sources);
  EXPECT_TRUE(std::all_of(new_metadata.begin(), new_metadata.end(), [&](auto const& meta) {
    return meta.row_groups.size() == parquet_metadata.front().row_groups.size();
  }));
}

TEST_F(HybridScanMultifileFiltersTest, EmptySource)
{
  using T = uint32_t;

  // Create two parquet source. First one with non-zero rows and the second one with zero rows.
  auto constexpr num_sources = 2;
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  file_buffers.emplace_back(create_empty_parquet_with_stats<T>());

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Check parquet metadata
  auto const parquet_metadata = reader->parquet_metadatas();
  ASSERT_EQ(parquet_metadata.size(), num_sources);
  EXPECT_FALSE(parquet_metadata.front().row_groups.empty());
  EXPECT_TRUE(parquet_metadata.back().row_groups.empty());

  // Check row group indices
  auto const all_rgs = reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), num_sources);
  EXPECT_EQ(all_rgs.front(), (std::vector<cudf::size_type>{0, 1, 2, 3}));
  EXPECT_TRUE(all_rgs.back().empty());

  // Check page index byte ranges
  auto const page_index_byte_ranges = reader->page_index_byte_ranges();
  ASSERT_EQ(page_index_byte_ranges.size(), num_sources);
  EXPECT_FALSE(page_index_byte_ranges.front().is_empty());
  EXPECT_TRUE(page_index_byte_ranges.back().is_empty());

  auto const passes = reader->construct_row_group_passes(all_rgs, 1);
  ASSERT_EQ(passes.size(), all_rgs.front().size());
  for (auto const& pass : passes) {
    ASSERT_EQ(pass.size(), num_sources);
    ASSERT_EQ(pass.front().size(), 1);
    EXPECT_TRUE(pass.back().empty());
  }
}

TEST_F(HybridScanMultifileFiltersTest, RowGroupPasses)
{
  using T = uint32_t;

  auto constexpr num_sources = 2;
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  std::transform(cuda::counting_iterator(0),
                 cuda::counting_iterator(num_sources),
                 std::back_inserter(file_buffers),
                 [&](auto i) {
                   srand(0xced + i);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });
  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto const all_rgs = reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), num_sources);
  EXPECT_TRUE(std::all_of(all_rgs.begin(), all_rgs.end(), [](auto const& rgs) {
    return rgs == (std::vector<cudf::size_type>{0, 1, 2, 3});
  }));

  // Invalid row group indices => throw error
  {
    auto invalid_rgs = all_rgs;
    invalid_rgs.pop_back();
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(invalid_rgs, 0)),
                 std::invalid_argument);
  }

  // Empty row group indices => throw error
  {
    auto const empty_rgs = std::vector<std::vector<cudf::size_type>>(num_sources);
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(empty_rgs, 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(empty_rgs, 1)),
                 std::invalid_argument);
  }

  // Zero pass read limit => single pass with all row groups
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 0);
    ASSERT_EQ(passes.size(), 1);
    ASSERT_EQ(passes.front().size(), num_sources);
    EXPECT_EQ(passes.front(), all_rgs);
  }

  // Small pass read limit => each row group in its own pass
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 1);
    ASSERT_EQ(passes.size(), num_sources * all_rgs.front().size());
    for (auto const& pass : passes) {
      ASSERT_EQ(pass.size(), num_sources);
      auto const pass_num_row_groups =
        std::accumulate(pass.begin(), pass.end(), std::size_t{0}, [](auto sum, auto const& rgs) {
          return sum + rgs.size();
        });
      EXPECT_EQ(pass_num_row_groups, 1);
    }
  }

  // Large pass read limit => multiple passes
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 10'000);
    ASSERT_GT(passes.size(), 1);
    auto const pass_num_row_groups = [](auto const& pass) {
      return std::accumulate(
        pass.begin(), pass.end(), std::size_t{0}, [](auto sum, auto const& rgs) {
          return sum + rgs.size();
        });
    };
    EXPECT_TRUE(std::any_of(passes.begin(), passes.end(), [&](auto const& pass) {
      return pass_num_row_groups(pass) > 1;
    }));

    auto flattened = std::vector<std::pair<std::size_t, cudf::size_type>>{};
    for (auto const& pass : passes) {
      ASSERT_EQ(pass.size(), num_sources);
      for (auto source_index = std::size_t{0}; source_index < pass.size(); ++source_index) {
        std::transform(
          pass[source_index].begin(),
          pass[source_index].end(),
          std::back_inserter(flattened),
          [source_index](auto const rg_index) { return std::pair{source_index, rg_index}; });
      }
    }

    auto expected = std::vector<std::pair<std::size_t, cudf::size_type>>{};
    for (auto source_index = std::size_t{0}; source_index < all_rgs.size(); ++source_index) {
      std::transform(
        all_rgs[source_index].begin(),
        all_rgs[source_index].end(),
        std::back_inserter(expected),
        [source_index](auto const rg_index) { return std::pair{source_index, rg_index}; });
    }
    EXPECT_EQ(flattened, expected);
  }
}

TEST_F(HybridScanMultifileFiltersTest, RowGroupPassesSingleSourceParity)
{
  using T = cudf::duration_ms;

  srand(0xced);

  std::vector<std::vector<char>> file_buffers;
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const multifile_reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
  auto const single_file_reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      inputs.footer_byte_spans.front(), options);

  auto const all_rgs = multifile_reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), 1);
  auto constexpr pass_read_limit = std::size_t{10'000};
  auto const multifile_passes =
    multifile_reader->construct_row_group_passes(all_rgs, pass_read_limit);
  auto const single_file_passes =
    single_file_reader->construct_row_group_passes(all_rgs.front(), pass_read_limit);

  auto projected_passes = std::vector<std::vector<cudf::size_type>>{};
  projected_passes.reserve(multifile_passes.size());
  for (auto const& pass : multifile_passes) {
    ASSERT_EQ(pass.size(), 1);
    projected_passes.push_back(pass.front());
  }
  EXPECT_EQ(projected_passes, single_file_passes);
}

TEST_F(HybridScanMultifileFiltersTest, ErrorFilterRowGroupsWithByteRanges)
{
  using T                    = uint32_t;
  auto constexpr num_sources = 2;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xb47e;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // Setting `skip_bytes` or `num_bytes` is ambiguous when reading multiple sources. The reader is
  // expected to throw an exception if row groups are filtered using byte range in this case.
  {
    auto const options = cudf::io::parquet_reader_options::builder().skip_bytes(1000).build();
    auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    auto const row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(row_group_indices.size(), num_sources);
    EXPECT_THROW(
      std::ignore = reader->filter_row_groups_with_byte_range(row_group_indices, options),
      std::invalid_argument);
  }
  {
    auto const options = cudf::io::parquet_reader_options::builder().num_bytes(1000).build();
    auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    auto const row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(row_group_indices.size(), num_sources);
    EXPECT_THROW(
      std::ignore = reader->filter_row_groups_with_byte_range(row_group_indices, options),
      std::invalid_argument);
  }
}

TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithStats)
{
  using T                           = cudf::duration_ms;
  auto constexpr num_sources        = 2;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;

  // Two sources, each with 4 row groups and ascending strings in col2
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xc001;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1, false>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // Filter - col0 < 50 and col2 > "000010000"
  auto literal_value0 = make_scalar<T>(50, cudf::get_default_stream());
  auto literal0       = cudf::ast::literal(literal_value0);
  auto col_ref0       = cudf::ast::column_reference(0);
  auto filter1        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal0);

  auto literal_value2 = cudf::string_scalar("000010000", true, cudf::get_default_stream());
  auto literal2       = cudf::ast::literal(literal_value2);
  auto col_ref2       = cudf::ast::column_reference(2);
  auto filter2        = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal2, col_ref2);

  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, filter1, filter2);

  auto options      = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Each source has 4 row groups (20000 rows / 5000 rows per row group)
  auto input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices),
            num_sources * 4 * rows_per_row_group);

  // Each source prunes down to a single surviving row group
  auto stats_filtered = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  ASSERT_EQ(stats_filtered.size(), num_sources);
  for (std::size_t i = 0; i < stats_filtered.size(); ++i) {
    EXPECT_EQ(stats_filtered[i].size(), 1) << "Source index: " << i;
  }
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), num_sources * rows_per_row_group);

  // Custom per-source indices that prune all row groups via stats, including an empty source
  input_row_group_indices = {{1, 2}, {}};
  stats_filtered          = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  ASSERT_EQ(stats_filtered.size(), num_sources);
  EXPECT_TRUE(stats_filtered.front().empty());
  EXPECT_TRUE(stats_filtered.back().empty());
}

TEST_F(HybridScanMultifileFiltersTest, BuildAllTrueRowMask)
{
  using T                    = uint64_t;
  auto constexpr num_sources = 2;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xa11;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  auto const options = cudf::io::parquet_reader_options::builder().build();
  auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto test_all_true_row_mask =
    [&](cudf::host_span<std::vector<cudf::size_type> const> row_group_indices) {
      auto const row_mask = reader->build_all_true_row_mask(row_group_indices, stream, mr);

      auto const expected_num_rows = reader->total_rows_in_row_groups(row_group_indices);

      EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
      EXPECT_EQ(row_mask->size(), expected_num_rows);
      EXPECT_EQ(row_mask->null_count(), 0);
      auto const host_row_mask = host_row_mask_data<bool>(row_mask->view(), stream);
      EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true), expected_num_rows);
    };

  auto row_group_indices = std::vector<std::vector<cudf::size_type>>{{0, 2}, {1, 3}};
  test_all_true_row_mask(row_group_indices);

  row_group_indices = reader->all_row_groups(options);
  test_all_true_row_mask(row_group_indices);
}

template <typename T>
struct HybridScanMultifilePageIndexRowMaskTest : public HybridScanMultifileFiltersTest {};

// Unsigned numeric types except booleans for page index stats tests
using SignedIntegralTypesNotBool =
  cudf::test::ContainedIn<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>>;
using PageIndexRowMaskTestTypes =
  cudf::test::RemoveIf<SignedIntegralTypesNotBool,
                       cudf::test::Concat<cudf::test::IntegralTypesNotBool>>;

TYPED_TEST_SUITE(HybridScanMultifilePageIndexRowMaskTest, PageIndexRowMaskTestTypes);

TYPED_TEST(HybridScanMultifilePageIndexRowMaskTest, BuildRowMaskWithPageIndexStats)
{
  using T                    = TypeParam;
  auto constexpr num_sources = 4;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xa11b;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1, false>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  auto options      = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const input_row_group_indices = reader->all_row_groups(options);

  auto const test_filter_data_pages_with_stats = [&](
                                                   cudf::ast::operation const& filter_expression,
                                                   cudf::size_type const expected_surviving_rows) {
    options.set_filter(filter_expression);
    reader->reset_column_selection();

    auto const row_mask =
      reader->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);

    auto const expected_num_rows = reader->total_rows_in_row_groups(input_row_group_indices);
    EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
    EXPECT_EQ(row_mask->size(), expected_num_rows);
    EXPECT_EQ(row_mask->null_count(), 0);

    auto const host_row_mask = host_row_mask_data<bool>(row_mask->view(), stream);
    EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true),
              expected_surviving_rows);
  };

  // Calling the page-index row mask builder before setting up the page index should raise an error.
  {
    auto literal_value     = make_scalar<T>(100, stream);
    auto const literal     = cudf::ast::literal(literal_value);
    auto const col_ref     = cudf::ast::column_name_reference("col0");
    auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);
    options.set_filter(filter_expression);
    EXPECT_THROW(std::ignore = reader->build_row_mask_with_page_index_stats(
                   input_row_group_indices, options, stream, mr),
                 std::runtime_error);
  }

  setup_page_indexes(*reader, inputs);

  // Filtering AST - table[0] < 100
  {
    auto literal_value = make_scalar<T>(100, stream);
    auto const literal = cudf::ast::literal(literal_value);
    auto const col_ref = cudf::ast::column_name_reference("col0");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref);
    auto constexpr expected_surviving_rows =
      num_sources * num_ordered_rows / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[2] >= 10000
  {
    auto literal_value = cudf::string_scalar("000010000", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);
    auto constexpr expected_surviving_rows =
      num_sources * num_ordered_rows / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] < 50 AND table[2] < "000010000"
  {
    auto literal_value1 = make_scalar<T>(50, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000010000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    auto constexpr expected_surviving_rows = num_sources * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] > 150 OR table[2] < "000005000"
  {
    auto literal_value1 = make_scalar<T>(150, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000005000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    auto constexpr expected_surviving_rows = 2 * num_sources * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }
}

TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithDictionaryPages)
{
  using T                    = uint32_t;
  auto constexpr num_sources = 2;
  auto stream                = cudf::get_default_stream();
  auto mr                    = cudf::get_current_device_resource_ref();

  // 2 sources, each `dictionary_policy::ALWAYS` with a per-source constant `col2`
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  srand(0xd1c7);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(100)));  // col2 == "0100"
  srand(0xfeed);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(200)));  // col2 == "0200"

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // Filter: `col2 == "0100"` (present only in source A's dictionary)
  auto literal_value = cudf::string_scalar("0100", true, stream);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_name_reference("col2");
  auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto options      = cudf::io::parquet_reader_options::builder().filter(filter).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Page index is needed to detect dictionary-only encoded pages
  setup_page_indexes(*reader, inputs);

  auto const dict_filtered =
    filter_row_groups_with_dictionaries(inputs, *reader, options, stream, mr);

  // Source A keeps all 4 row groups (col2 == "0100"); source B is fully pruned (only "0200")
  ASSERT_EQ(dict_filtered.size(), num_sources);
  EXPECT_EQ(dict_filtered.front(), (std::vector<cudf::size_type>{0, 1, 2, 3}));
  EXPECT_TRUE(dict_filtered.back().empty());
}

TEST_F(HybridScanMultifileFiltersTest, MismatchedSchemaDictionaryPruningCollision)
{
  using T                    = cudf::duration_ms;
  auto constexpr num_sources = 2;
  auto stream                = cudf::get_default_stream();
  auto mr                    = cudf::get_current_device_resource_ref();

  // Source A: default column order/names, col2 == "0200" (pruned by the filter).
  // Source B: same columns emitted as {col2, col0, col1}, col2 == "0100" (survives).
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  srand(0xd1c7);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(200)));
  srand(0xfeed);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(
    100, cudf::io::compression_type::AUTO, {"col2", "col0", "col1"}, {2, 0, 1})));

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // Filter: `col2 == "0100"`
  auto literal_value = cudf::string_scalar("0100", true, stream);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_name_reference("col2");
  auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto options = cudf::io::parquet_reader_options::builder()
                   .allow_mismatched_pq_schemas(true)
                   .column_names({"col2"})
                   .filter(filter)
                   .build();

  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    cudf::host_span<cudf::host_span<uint8_t const> const>{inputs.footer_byte_spans}, options);

  // Ensure the reorder genuinely differs the per-source schemas
  auto const metadatas = reader->parquet_metadatas();
  ASSERT_EQ(metadatas.size(), num_sources);
  EXPECT_EQ(metadatas.front().schema.at(1).name, "col0");
  EXPECT_EQ(metadatas.back().schema.at(1).name, "col2");

  // Page index is needed to detect dictionary-only encoded pages
  setup_page_indexes(*reader, inputs);

  auto const dict_filtered =
    filter_row_groups_with_dictionaries(inputs, *reader, options, stream, mr);

  // Source A is pruned (col2 == "0200"), source B survives (col2 == "0100").
  ASSERT_EQ(dict_filtered.size(), num_sources);
  EXPECT_TRUE(dict_filtered.front().empty()) << "Source A should be pruned (col2 == \"0200\")";
  EXPECT_EQ(dict_filtered.back(), (std::vector<cudf::size_type>{0, 1, 2, 3}))
    << "Source B should survive (col2 == \"0100\")";
}
