/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"
#include "hybrid_scan_composer.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <cuda/iterator>

namespace {

/**
 * @brief Helper to test the hybrid scan reader
 *
 * Concatenates the input table and writes it to parquet. Then reads it back using the regular and
 * the chunked hybrid scan readers. The filter expression used is as follows: table[0] >= 100. The
 * output filter and payload tables from both readers are compared with the expected table (read via
 * the mainline parquet reader) for equivalence (as nullability may be different for the read
 * tables).
 *
 *
 * @note The first column in the input table must be constructed with
 * `cudf::test::ascending<uint32_t>()`
 *
 * @tparam num_concat Number of times to concatenate the table before writing to parquet
 * @tparam num_rows Number of rows in the input table
 *
 * @param columns List of column views in the input table
 */
template <int num_concat = 2, int num_rows = num_ordered_rows>
void test_hybrid_scan(std::vector<cudf::column_view> const& columns,
                      bool case_sensitive_names = true)
{
  // Input table
  auto table    = cudf::table_view{columns};
  auto expected = cudf::concatenate(std::vector<table_view>(num_concat, table));
  table         = expected->view();
  cudf::io::table_input_metadata expected_metadata(table);
  expected_metadata.column_metadata[0].set_name("col0");

  // Parquet buffer
  std::vector<char> parquet_buffer;
  {
    cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&parquet_buffer}, table)
        .metadata(std::move(expected_metadata))
        .row_group_size_rows(num_rows)
        .max_page_size_rows(page_size_for_ordered_tests)
        .compression(cudf::io::compression_type::AUTO)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    cudf::io::write_parquet(out_opts);
  }

  // Filtering AST - table[0] >= 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0 = cudf::ast::column_name_reference(case_sensitive_names ? "col0" : "Col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                      bloom_filter_alignment);

  auto datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));
  auto datasource_ref = std::ref(*datasource);

  // Read parquet using the hybrid scan reader
  auto const [read_filter_table, read_payload_table] = hybrid_scan(
    datasource_ref, filter_expression, {}, case_sensitive_names, stream, mr, aligned_mr);

  // Read parquet using the chunked hybrid scan reader
  auto const [read_filter_table_chunked, read_payload_table_chunked] = chunked_hybrid_scan(
    datasource_ref, filter_expression, {}, case_sensitive_names, stream, mr, aligned_mr);

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  auto const options =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
      .filter(filter_expression)
      .case_sensitive_names(case_sensitive_names)
      .build();
  auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);

  CUDF_EXPECTS(
    expected_tbl->num_rows() == read_filter_table->num_rows(),
    "Tables read by the mainline and hybrid scan readers must have the same number of rows");

  // Check equivalence for the filter column
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table_chunked->view());

  // Check equivalence for the payload columns: [num_filter_columns, num_columns)
  auto payload_column_indices = std::vector<cudf::size_type>(columns.size() - num_filter_columns);
  std::iota(payload_column_indices.begin(), payload_column_indices.end(), num_filter_columns);
  auto const expected_payload_table = expected_tbl->select(payload_column_indices);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_payload_table, read_payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_payload_table, read_payload_table_chunked->view());

  // Read parquet using the hybrid scan reader in a single step
  auto const read_single_step_table = hybrid_scan_single_step(
    datasource_ref, filter_expression, {}, case_sensitive_names, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->view(), read_single_step_table->view());

  // Read parquet using the chunked hybrid scan reader in a single step
  auto const read_chunked_single_step_table = chunked_hybrid_scan_single_step(
    datasource_ref, filter_expression, {}, case_sensitive_names, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->view(), read_chunked_single_step_table->view());
}

/**
 * @brief Helper to test the hybrid scan reader with filter pushdown and column selection
 *
 * Reads the input parquet buffer using hybrid scan single and two-step compositions with specified
 * column selection.
 *
 * @param parquet_buffer Input parquet buffer
 * @param filter_expression Filter expression
 * @param filter_column_name Name of filter column
 * @param payload_column_indices Indices of payload columns to read
 * @param payload_column_names Names of payload columns to read, if any
 * @param case_sensitive_names Whether column names are case sensitive
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @param aligned_mr Aligned memory resource
 *
 * @return Read table
 */
std::unique_ptr<cudf::table> test_hybrid_scan_column_selection(
  cudf::host_span<char const> parquet_buffer,
  cudf::ast::operation const& filter_expression,
  std::string_view filter_column_name,
  std::vector<cudf::size_type> const& payload_column_indices,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor& aligned_mr)
{
  auto datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));
  auto datasource_ref = std::ref(*datasource);

  {
    auto const [read_filter_table, read_payload_table] = hybrid_scan(datasource_ref,
                                                                     filter_expression,
                                                                     payload_column_names,
                                                                     case_sensitive_names,
                                                                     stream,
                                                                     mr,
                                                                     aligned_mr);
    auto const [read_filter_table_chunked, read_payload_table_chunked] =
      chunked_hybrid_scan(datasource_ref,
                          filter_expression,
                          payload_column_names,
                          case_sensitive_names,
                          stream,
                          mr,
                          aligned_mr);

    // Read parquet using the main reader
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(parquet_buffer))
        .filter(filter_expression)
        .case_sensitive_names(case_sensitive_names);
    auto const expected_tbl = cudf::io::read_parquet(options, stream, mr).tbl;

    // Validate
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}),
                                       read_filter_table_chunked->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select(payload_column_indices),
                                       read_payload_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select(payload_column_indices),
                                       read_payload_table_chunked->view());
  }

  // All column names for single step readers
  auto all_column_names = std::optional<std::vector<std::string>>{};
  if (payload_column_names.has_value()) {
    all_column_names = std::vector<std::string>{};
    if (not filter_column_name.empty()) { all_column_names->emplace_back(filter_column_name); }
    all_column_names->insert(all_column_names->end(),
                             payload_column_names.value().begin(),
                             payload_column_names.value().end());
  }

  auto read_single_step = hybrid_scan_single_step(
    datasource_ref, filter_expression, all_column_names, case_sensitive_names, stream, mr);

  auto const read_chunked_single_step = chunked_hybrid_scan_single_step(
    datasource_ref, filter_expression, all_column_names, case_sensitive_names, stream, mr);

  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(parquet_buffer))
      .filter(filter_expression)
      .case_sensitive_names(case_sensitive_names);
  if (all_column_names.has_value()) { options.set_column_names(all_column_names.value()); }
  auto const expected_tbl = cudf::io::read_parquet(options, stream, mr).tbl;

  // Validate
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->view(), read_single_step->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->view(), read_chunked_single_step->view());

  return read_single_step;
}

}  // namespace

// Base test fixture for tests
struct HybridScanTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanTest, FilterRowGroupsOnlyAndScanSelectColumns)
{
  srand(0xc0ffee);
  using T = uint32_t;

  // A table with several row groups each containing a single page per column. The data page and row
  // group stats are identical so only row groups can be pruned using stats
  auto constexpr num_concat            = 1;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("Col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                      bloom_filter_alignment);
  auto constexpr case_sensitive_names = false;

  // No column selection (all columns)
  {
    auto const payload_column_indices = std::vector<cudf::size_type>{1, 2};
    std::ignore                       = test_hybrid_scan_column_selection(parquet_buffer,
                                                    filter_expression,
                                                    "col0",
                                                    payload_column_indices,
                                                                          {},
                                                    case_sensitive_names,
                                                    stream,
                                                    mr,
                                                    aligned_mr);
  }

  // Columns: col0, col2
  {
    auto const payload_column_names   = std::vector<std::string>{"Col0", "Col2"};
    auto const payload_column_indices = std::vector<cudf::size_type>{2};
    std::ignore                       = test_hybrid_scan_column_selection(parquet_buffer,
                                                    filter_expression,
                                                    "col0",
                                                    payload_column_indices,
                                                    payload_column_names,
                                                    case_sensitive_names,
                                                    stream,
                                                    mr,
                                                    aligned_mr);
  }

  // Columns: col2, col1
  {
    auto const payload_column_names   = std::vector<std::string>{"cOl2", "coL1"};
    auto const payload_column_indices = std::vector<cudf::size_type>{2, 1};
    std::ignore                       = test_hybrid_scan_column_selection(parquet_buffer,
                                                    filter_expression,
                                                    "col0",
                                                    payload_column_indices,
                                                    payload_column_names,
                                                    case_sensitive_names,
                                                    stream,
                                                    mr,
                                                    aligned_mr);
  }
}

TEST_F(HybridScanTest, FilterDataPagesOnlyAndScanAllColumns)
{
  srand(0xf00d);
  using T = cudf::duration_ms;

  // A table concatenated with itself results in a parquet file with a row group per concatenated
  // table, each containing multiple pages per column. All row groups will be identical so only data
  // pages can be pruned using page index stats
  auto constexpr num_concat            = 2;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::duration_scalar<T>(T{100});
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("Col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                      bloom_filter_alignment);
  auto constexpr case_sensitive_names = false;

  auto const payload_column_indices = std::vector<cudf::size_type>{1, 2};
  auto read_table                   = test_hybrid_scan_column_selection(parquet_buffer,
                                                      filter_expression,
                                                      "col0",
                                                                        {1, 2},
                                                                        {},
                                                      case_sensitive_names,
                                                      stream,
                                                      mr,
                                                      aligned_mr);

  // Check equivalence (equal without checking nullability) with the original table with the
  // applied boolean mask
  {
    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto predicate = cudf::compute_column(written_table->view(), filter_expression);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto expected = cudf::apply_boolean_mask(written_table->view(), *predicate);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), read_table->view());
  }
}

TEST_F(HybridScanTest, MaterializeStrings)
{
  std::mt19937 gen(0xbee);

  auto const num_concat   = 2;
  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // str(non-nullable)
  auto col1 = testdata::ascending<cudf::string_view>();

  // str(nullable)
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  std::bernoulli_distribution bn(0.7f);
  auto string_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto col2 =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, string_valids};

  test_hybrid_scan<num_concat, num_rows>({col0, col1, col2});
}

TEST_F(HybridScanTest, MaterializeLists)
{
  std::mt19937 gen(0xadd);

  auto const num_concat            = 2;
  auto constexpr num_rows          = num_ordered_rows;
  auto constexpr lists_per_row     = 2;
  auto constexpr max_vals_per_list = 3;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // list<int32_t(nullable)>(nullable)
  auto col1 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, true);

  // list<list<int32_t(nullable)>(nullable)>(nullable)
  auto col2 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, true);

  // list<int32_t(non-nullable)>(non-nullable)
  auto col3 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, false);

  // list<list<int32_t(non-nullable)>(non-nullable)>(non-nullable)
  auto col4 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, false);

  // list<list<bool(nullable)>(nullable)>(nullable)
  auto col5 = make_parquet_list_list_col<bool>(0, num_rows, lists_per_row, max_vals_per_list, true);

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2, *col3, *col4, *col5});
}

TEST_F(HybridScanTest, MaterializeListsOfStrings)
{
  std::mt19937 gen(0xc0c0a);

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // list<str(non-nullable)>(non-nullable)
  auto col1 = make_list_str_column(gen, false, false);

  // list<str(non-nullable)>(nullable)
  auto col2 = make_list_str_column(gen, false, true);

  // list<str(nullable)>(non-nullable)
  auto col3 = make_list_str_column(gen, true, false);

  // list<list<str(nullable)>>(nullable)
  auto col4 = make_list_str_column(gen, true, true);

  test_hybrid_scan({col0, *col1, *col2, *col3, *col4}, false);
}

TEST_F(HybridScanTest, ConsecutivePrunedPageOffsets)
{
  std::mt19937 gen(0x5ca1e);
  auto constexpr num_rows = num_ordered_rows;

  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::ascending<cudf::string_view>();
  auto col2 = make_parquet_list_col<int32_t>(gen, num_rows, 3, true);
  col2      = cudf::purge_nonempty_nulls(col2->view());
  auto col3 = make_parquet_list_list_col<int32_t>(0, num_rows, 2, 3, true);
  col3      = cudf::purge_nonempty_nulls(col3->view());
  auto col4 = make_list_str_column(gen, true, true);
  col4      = cudf::purge_nonempty_nulls(col4->view());

  auto const input = cudf::table_view{{col0, col1, *col2, *col3, *col4}};

  std::string filepath = "ConsecutivePrunedPageOffsets.parquet";
  {
    auto metadata = cudf::io::table_input_metadata(input);
    metadata.column_metadata[0].set_name("col0");

    auto const write_options =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(std::move(metadata))
        .row_group_size_rows(num_rows)
        .max_page_size_rows(page_size_for_ordered_tests)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .build();
    cudf::io::write_parquet(write_options);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto aligned_mr   = rmm::mr::aligned_resource_adaptor(mr, bloom_filter_alignment);

  // Helper to validate monotonic string and list offsets
  auto const expect_monotonic_offsets = [](auto& self, cudf::column_view const& column) -> void {
    EXPECT_FALSE(cudf::has_nonempty_nulls(column));

    if (column.type().id() == cudf::type_id::STRING) {
      auto const offsets      = cudf::strings_column_view{column}.offsets();
      auto const host_offsets = cudf::test::to_host<cudf::size_type>(offsets).first;
      EXPECT_TRUE(std::is_sorted(host_offsets.begin(), host_offsets.end()));
      return;
    }

    if (column.type().id() == cudf::type_id::LIST) {
      auto const lists        = cudf::lists_column_view{column};
      auto const host_offsets = cudf::test::to_host<cudf::size_type>(lists.offsets()).first;
      EXPECT_TRUE(std::is_sorted(host_offsets.begin(), host_offsets.end()));
      self(self, lists.child());
      return;
    }

    for (auto index = cudf::size_type{0}; index < column.num_children(); ++index) {
      self(self, column.child(index));
    }
  };

  // Helper to validate the `offsets` children of string and list column
  auto const validate = [&](cudf::ast::operation const& filter,
                            std::vector<cudf::size_type> const& expected_slices) {
    auto datasource = cudf::io::datasource::create({filepath});

    auto const expected = cudf::concatenate(cudf::slice(input, expected_slices));
    std::unique_ptr<cudf::table> filter_table;
    std::unique_ptr<cudf::table> payload_table;
    ASSERT_NO_THROW(std::tie(filter_table, payload_table) =
                      hybrid_scan(*datasource, filter, {}, true, stream, mr, aligned_mr));

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view().select({0}), filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view().select({1, 2, 3, 4}),
                                       payload_table->view());
    for (auto const& column : payload_table->view()) {
      expect_monotonic_offsets(expect_monotonic_offsets, column);
    }
  };

  auto const col_ref = cudf::ast::column_name_reference{"col0"};

  // Prune two leading pages.
  auto middle_value = cudf::numeric_scalar<uint32_t>{2 * page_size_for_ordered_tests / 100};
  auto middle       = cudf::ast::literal{middle_value};
  auto const keep_trailing =
    cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, col_ref, middle};
  validate(keep_trailing, {2 * page_size_for_ordered_tests, num_rows});

  // Prune two trailing pages.
  auto const keep_leading = cudf::ast::operation{cudf::ast::ast_operator::LESS, col_ref, middle};
  validate(keep_leading, {0, 2 * page_size_for_ordered_tests});

  // Prune two middle pages.
  auto lower_value      = cudf::numeric_scalar<uint32_t>{page_size_for_ordered_tests / 100};
  auto upper_value      = cudf::numeric_scalar<uint32_t>{3 * page_size_for_ordered_tests / 100};
  auto lower            = cudf::ast::literal{lower_value};
  auto upper            = cudf::ast::literal{upper_value};
  auto const keep_first = cudf::ast::operation{cudf::ast::ast_operator::LESS, col_ref, lower};
  auto const keep_last =
    cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, col_ref, upper};
  auto const keep_outer =
    cudf::ast::operation{cudf::ast::ast_operator::LOGICAL_OR, keep_first, keep_last};
  validate(keep_outer, {0, page_size_for_ordered_tests, 3 * page_size_for_ordered_tests, num_rows});
}

TEST_F(HybridScanTest, MaterializeStructs)
{
  std::mt19937 gen(0xbaLL);

  auto const num_concat   = 2;
  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // Validity helpers
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 121; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  // strings helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  // struct<list<str(nullable)>(nullable), int(nullable), float(non-nullable)>(nullable)
  auto values    = cuda::counting_iterator<int>{0};
  auto col1_list = make_list_str_column(gen, true, true);
  cudf::test::fixed_width_column_wrapper<int> col1_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> col1_floats(values, values + num_rows);
  std::vector<std::unique_ptr<cudf::column>> col1_children;
  col1_children.push_back(std::move(col1_list));
  col1_children.push_back(col1_ints.release());
  col1_children.push_back(col1_floats.release());
  cudf::test::structs_column_wrapper _col1(std::move(col1_children), struct_valids);
  auto col1 = cudf::purge_nonempty_nulls(_col1);

  // struct<str(nullable), str(non-nullable), bool(nullable)>(nullable)
  auto col2_str = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  auto col2_str_non_nullable =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows};
  auto col2_bool = cudf::test::fixed_width_column_wrapper<bool>(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> col2_children;
  col2_children.push_back(col2_str.release());
  col2_children.push_back(col2_str_non_nullable.release());
  col2_children.push_back(col2_bool.release());
  cudf::test::structs_column_wrapper _col2(std::move(col2_children));
  auto col2 = cudf::purge_nonempty_nulls(_col2);

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2});
}

TEST_F(HybridScanTest, MaterializeListsOfStructs)
{
  std::mt19937 gen(0xcaLL);

  auto constexpr num_concat = 2;
  auto constexpr num_rows   = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // Validity helpers
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 150; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  // list<struct<list<str(nullable)>(nullable), int(nullable),
  // float(non-nullable)>(nullable)>(nullable)
  auto struct1_list = make_list_str_column(gen, true, true);
  auto values       = cuda::counting_iterator<int>{0};
  cudf::test::fixed_width_column_wrapper<float> struct1_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> struct1_children;
  struct1_children.push_back(std::move(struct1_list));
  struct1_children.push_back(struct1_floats.release());
  cudf::test::structs_column_wrapper _struct1(std::move(struct1_children), struct_valids);
  auto struct1 = cudf::purge_nonempty_nulls(_struct1);

  auto col1_offsets_iter = cuda::counting_iterator<int32_t>{0};
  auto col1_offsets_col  = cudf::test::fixed_width_column_wrapper<int32_t>(
    col1_offsets_iter, col1_offsets_iter + num_rows + 1);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col1 = cudf::make_lists_column(
    num_rows, col1_offsets_col.release(), std::move(struct1), null_count, std::move(null_mask));

  // strings helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  // list<struct<str(nullable), str(non-nullable), bool(nullable)>(non-nullable)>(nullable)
  auto struct2_str =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  auto struct2_str_non_nullable =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows};
  auto struct2_bool =
    cudf::test::fixed_width_column_wrapper<bool>(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> struct2_children;
  struct2_children.push_back(struct2_str.release());
  struct2_children.push_back(struct2_str_non_nullable.release());
  struct2_children.push_back(struct2_bool.release());
  cudf::test::structs_column_wrapper _struct2(std::move(struct2_children));
  auto struct2 = cudf::purge_nonempty_nulls(_struct2);

  auto col2_offsets_iter = cuda::counting_iterator<int32_t>{0};
  auto col2_offsets_col  = cudf::test::fixed_width_column_wrapper<int32_t>(
    col2_offsets_iter, col2_offsets_iter + num_rows + 1);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col2 = cudf::make_lists_column(
    num_rows, col2_offsets_col.release(), std::move(struct2), null_count, std::move(null_mask));

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2});
}

TEST_F(HybridScanTest, MaterializeMixedPayloadColumns)
{
  std::mt19937 gen(0xcaffe);

  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();
  // str(non-nullable)
  auto col1 = testdata::ascending<cudf::string_view>();

  // Validity helpers
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 121; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  // str and list<str> helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  // list<bool(nullable)>(nullable)
  auto bools_iter = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto bools_col =
    cudf::test::fixed_width_column_wrapper<bool>(bools_iter, bools_iter + num_rows, valids);
  auto offsets_iter = cuda::counting_iterator<int32_t>{0};
  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets_iter, offsets_iter + num_rows + 1);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col2 = cudf::make_lists_column(
    num_rows, offsets_col.release(), bools_col.release(), null_count, std::move(null_mask));
  // list<list<bool(nullable)>(nullable)>(nullable)
  auto col3 = make_parquet_list_list_col<bool>(0, num_rows, 5, 8, true);

  // list<str(nullable)>(must be non-nullable)
  auto const make_list_str_column = [&](bool is_nullable) {
    constexpr int string_per_row  = 3;
    constexpr int num_string_rows = num_rows * string_per_row;
    cudf::test::strings_column_wrapper string_col{
      string_iter, string_iter + num_string_rows, valids};
    auto offset_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type idx) { return idx * string_per_row; });
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                    offset_iter + num_rows + 1);
    auto [null_mask, null_count] = [&]() {
      if (is_nullable) {
        return cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
      } else {
        return std::make_pair(rmm::device_buffer{}, 0);
      }
    }();
    return cudf::make_lists_column(
      num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
  };

  // str(nullable)
  auto col4 = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};

  // list<str(nullable)>(non-nullable)
  auto col5 = make_list_str_column(false);

  // list<str(nullable)>(nullable)
  auto col6 = make_list_str_column(true);

  // struct<list<str(nullable)>(nullable), int(nullable), float(nullable)>(nullable)
  auto values    = cuda::counting_iterator<int>{0};
  auto col7_list = make_list_str_column(true);
  cudf::test::fixed_width_column_wrapper<int> col7_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> col7_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> col7_children;
  col7_children.push_back(std::move(col7_list));
  col7_children.push_back(col7_ints.release());
  col7_children.push_back(col7_floats.release());
  cudf::test::structs_column_wrapper _col7(std::move(col7_children), struct_valids);
  auto col7 = cudf::purge_nonempty_nulls(_col7);

  // struct<str(nullable), bool(nullable)>(nullable)
  auto col8_str = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  cudf::test::fixed_width_column_wrapper<bool> col8_bools(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> col8_children;
  col8_children.push_back(col8_str.release());
  col8_children.push_back(col8_bools.release());
  cudf::test::structs_column_wrapper _col8(std::move(col8_children), struct_valids);
  auto col8 = cudf::purge_nonempty_nulls(_col8);

  // list<list<str(nullable)>(nullable)>(nullable)
  constexpr int string_per_row  = 3;
  constexpr int lists_per_list  = 2;
  constexpr int num_string_rows = num_rows * string_per_row * lists_per_list;
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows, valids};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> list_offsets(
    offset_iter, offset_iter + (num_rows * lists_per_list) + 1);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_valids, list_valids + (num_rows * lists_per_list));

  auto list_col = cudf::make_lists_column(num_rows * lists_per_list,
                                          list_offsets.release(),
                                          string_col.release(),
                                          null_count,
                                          std::move(null_mask));

  auto list_list_offsets_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * lists_per_list; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> list_list_offsets(
    list_list_offsets_iter, list_list_offsets_iter + num_rows + 1);
  auto list_list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 80; });
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_list_valids, list_list_valids + num_rows);

  auto col9 = cudf::make_lists_column(
    num_rows, list_list_offsets.release(), std::move(list_col), null_count, std::move(null_mask));

  auto constexpr num_concat = 3;
  test_hybrid_scan<num_concat, num_rows>(
    {col0, col1, *col2, *col3, col4, *col5, *col6, *col7, *col8, *col9}, false);
}

TEST_F(HybridScanTest, ExtendedFilterExpressions)
{
  srand(0xbeef);
  using T = uint64_t;

  auto constexpr num_concat            = 1;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                      bloom_filter_alignment);

  // Create datasource from buffer
  auto const datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));
  auto const datasource_ref = std::ref(*datasource);

  auto col_ref0 = cudf::ast::column_reference(0);
  auto col_ref1 = cudf::ast::column_reference(1);

  auto constexpr case_sensitive_names = true;

  // Filter: (col0 < 100) and (col0 < col1)
  {
    auto literal_value = cudf::numeric_scalar<T>(100);
    auto literal       = cudf::ast::literal(literal_value);
    auto col0_lt_100   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal);
    auto col0_lt_col1  = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, col_ref1);
    auto filter =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col0_lt_100, col0_lt_col1);

    auto [filter_table, payload_table] = hybrid_scan(
      datasource_ref, filter, std::nullopt, case_sensitive_names, stream, mr, aligned_mr);

    auto read_single_step = hybrid_scan_single_step(
      datasource_ref, filter, std::nullopt, case_sensitive_names, stream, mr);

    auto predicate = cudf::compute_column(written_table->view(), filter);
    auto expected  = cudf::apply_boolean_mask(written_table->view(), *predicate);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), read_single_step->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({1, 0}), filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({2}), payload_table->view());
  }

  // Filter: (col0 < 10) or ((col0 + col1 > 0) and (col1 < 1))
  {
    auto literal_10_value = cudf::numeric_scalar<T>(10);
    auto literal_10       = cudf::ast::literal(literal_10_value);
    auto literal_0_value  = cudf::numeric_scalar<T>(0);
    auto literal_0        = cudf::ast::literal(literal_0_value);
    auto literal_1_value  = cudf::numeric_scalar<T>(1);
    auto literal_1        = cudf::ast::literal(literal_1_value);

    auto col0_lt_10     = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal_10);
    auto col0_plus_col1 = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref0, col_ref1);
    auto col0_plus_col1_gt_0 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col0_plus_col1, literal_0);
    auto col1_lt_1 = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal_1);
    auto col0_plus_col1_gt_0_expr =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col0_plus_col1_gt_0, col1_lt_1);

    auto filter = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, col0_lt_10, col0_plus_col1_gt_0_expr);

    auto [filter_table, payload_table] = hybrid_scan(
      datasource_ref, filter, std::nullopt, case_sensitive_names, stream, mr, aligned_mr);

    auto read_single_step = hybrid_scan_single_step(
      datasource_ref, filter, std::nullopt, case_sensitive_names, stream, mr);

    auto predicate = cudf::compute_column(written_table->view(), filter);
    auto expected  = cudf::apply_boolean_mask(written_table->view(), *predicate);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), read_single_step->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({1, 0}), filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({2}), payload_table->view());
  }
}

TEST_F(HybridScanTest, DecimalTypeOption)
{
  auto const data = std::vector<int32_t>{1000, 2000, 3000, 4000, 5000};
  auto col        = cudf::test::fixed_point_column_wrapper<int32_t>(
    data.begin(), data.end(), numeric::scale_type{-2});

  std::vector<char> parquet_buffer;
  {
    auto const table = cudf::table_view{{col}};
    auto opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&parquet_buffer}, table)
        .build();
    cudf::io::write_parquet(opts);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto datasource   = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));

  auto const read_with_decimal_type = [&](cudf::type_id decimal_type_id) {
    auto options =
      cudf::io::parquet_reader_options::builder().decimal_width(decimal_type_id).build();

    auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      *footer_buffer, options);

    auto const row_groups   = reader->all_row_groups(options);
    auto const chunk_ranges = reader->all_column_chunks_byte_ranges(row_groups, options);
    auto [buffers, col_data, tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(*datasource, chunk_ranges, stream, mr);
    tasks.get();

    return reader->materialize_all_columns(row_groups, col_data, options, stream, mr);
  };

  {
    auto result = read_with_decimal_type(cudf::type_id::DECIMAL128);
    EXPECT_EQ(result.tbl->view().column(0).type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(result.tbl->view().column(0).type().scale(), -2);
  }
  {
    auto result = read_with_decimal_type(cudf::type_id::DECIMAL64);
    EXPECT_EQ(result.tbl->view().column(0).type().id(), cudf::type_id::DECIMAL64);
    EXPECT_EQ(result.tbl->view().column(0).type().scale(), -2);
  }
}

TEST_F(HybridScanTest, StructChildFilterColumn)
{
  // struct<a:int32_t, b:int32_t> column
  auto child_a    = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4};
  auto child_b    = cudf::test::fixed_width_column_wrapper<int32_t>{10, 11, 12, 13, 14};
  auto struct_col = cudf::test::structs_column_wrapper{{child_a, child_b}}.release();

  auto input = cudf::table_view({*struct_col});

  cudf::io::table_input_metadata input_metadata(input);
  input_metadata.column_metadata[0].set_name("struct");
  input_metadata.column_metadata[0].child(0).set_name("a");
  input_metadata.column_metadata[0].child(1).set_name("b");

  auto const filepath = temp_env->get_temp_filepath("struct_col.parquet");
  {
    auto write_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(std::move(input_metadata))
        .build();
    cudf::io::write_parquet(write_opts);
  }

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const col_ref = cudf::ast::column_name_reference("struct.a");
  auto scalar_val    = cudf::numeric_scalar<int32_t>(3);
  auto literal       = cudf::ast::literal(scalar_val);
  auto filter_expr   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);

  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expr).build();

  auto datasource          = cudf::io::datasource::create(filepath);
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto row_groups = reader->all_row_groups(options);
  auto row_mask   = reader->build_all_true_row_mask(row_groups, stream, mr);

  // Error case: Materialize filter column (struct_col.a)
  auto const filter_byte_ranges = reader->filter_column_chunks_byte_ranges(row_groups, options);
  auto [filter_bufs, filter_data, filter_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource, filter_byte_ranges, stream, mr);
  filter_tasks.get();

  using cudf::io::parquet::experimental::use_data_page_mask;
  auto row_mask_mutable = row_mask->mutable_view();
  EXPECT_THROW(
    std::ignore = reader->materialize_filter_columns(
      row_groups, filter_data, row_mask_mutable, use_data_page_mask::NO, options, stream, mr),
    std::invalid_argument);
}

TEST_F(HybridScanTest, AllRowsPrunedReportsInputRowGroups)
{
  using cudf::io::parquet::experimental::use_data_page_mask;

  auto constexpr num_rows = 10;

  // Single row group, single filter column (col0) and single payload column (col1)
  auto values = cuda::counting_iterator<uint32_t>{0};
  cudf::test::fixed_width_column_wrapper<uint32_t> col0(values, values + num_rows);
  cudf::test::fixed_width_column_wrapper<uint32_t> col1(values, values + num_rows);
  auto table = cudf::table_view{{col0, col1}};

  std::string const filepath = temp_env->get_temp_filepath("AllRowsPruned.parquet");
  {
    cudf::io::table_input_metadata input_metadata(table);
    input_metadata.column_metadata[0].set_name("col0");
    auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table)
                      .metadata(std::move(input_metadata))
                      .build();
    cudf::io::write_parquet(out_opts);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto scalar    = cudf::numeric_scalar<uint32_t>(0, true, stream);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  auto datasource          = cudf::io::datasource::create(filepath);
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto const row_groups = reader->all_row_groups(options);
  auto false_iter       = cuda::make_constant_iterator(false);
  auto row_mask =
    cudf::test::fixed_width_column_wrapper<bool>(false_iter, false_iter + num_rows).release();
  auto row_mask_view = row_mask->mutable_view();

  auto const filter_byte_ranges = reader->filter_column_chunks_byte_ranges(row_groups, options);
  auto [filter_buffers, filter_data, filter_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource, filter_byte_ranges, stream, mr);
  filter_tasks.get();

  auto const filter_result = reader->materialize_filter_columns(
    row_groups, filter_data, row_mask_view, use_data_page_mask::YES, options, stream, mr);
  EXPECT_EQ(filter_result.tbl->num_rows(), 0);
  EXPECT_EQ(filter_result.metadata.num_input_row_groups,
            static_cast<cudf::size_type>(row_groups.size()));

  // Payload columns: same expectation
  auto const payload_byte_ranges = reader->payload_column_chunks_byte_ranges(row_groups, options);
  auto [payload_buffers, payload_data, payload_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource, payload_byte_ranges, stream, mr);
  payload_tasks.get();

  auto const payload_result = reader->materialize_payload_columns(
    row_groups, payload_data, row_mask_view, use_data_page_mask::YES, options, stream, mr);
  EXPECT_EQ(payload_result.tbl->num_rows(), 0);
  EXPECT_EQ(payload_result.metadata.num_input_row_groups,
            static_cast<cudf::size_type>(row_groups.size()));
}

TEST_F(HybridScanTest, ChunkedReadRowMaskPerPass)
{
  using T = uint32_t;

  // A multi-row-group table so that the row groups can be split into more than one pass
  auto constexpr num_concat            = 4;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - col0 < 100
  auto literal_value     = cudf::numeric_scalar<T>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  auto datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));

  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto const page_index_byte_range = reader->page_index_byte_range();
  if (not page_index_byte_range.is_empty()) {
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
    reader->setup_page_index(*page_index_buffer);
  }

  auto const all_row_groups = reader->all_row_groups(options);

  std::vector<std::unique_ptr<cudf::table>> filter_tables;
  std::vector<std::unique_ptr<cudf::table>> payload_tables;

  auto const materialize_pass = [&](cudf::host_span<cudf::size_type const> row_group_indices) {
    // Build a per-pass row mask for just this pass's row groups
    auto row_mask =
      reader->build_row_mask_with_page_index_stats(row_group_indices, options, stream, mr);
    auto row_mask_view = row_mask->mutable_view();

    // Materialize filter columns for this pass
    auto const filter_byte_ranges =
      reader->filter_column_chunks_byte_ranges(row_group_indices, options);
    auto [filter_buffers, filter_data, filter_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        *datasource, filter_byte_ranges, stream, mr);
    filter_tasks.get();

    filter_tables.push_back(
      reader
        ->materialize_filter_columns(row_group_indices,
                                     filter_data,
                                     row_mask_view,
                                     cudf::io::parquet::experimental::use_data_page_mask::YES,
                                     options,
                                     stream,
                                     mr)
        .tbl);

    // Materialize payload columns for this pass using the same per-pass row mask
    auto const payload_byte_ranges =
      reader->payload_column_chunks_byte_ranges(row_group_indices, options);
    auto [payload_buffers, payload_data, payload_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        *datasource, payload_byte_ranges, stream, mr);
    payload_tasks.get();

    payload_tables.push_back(
      reader
        ->materialize_payload_columns(row_group_indices,
                                      payload_data,
                                      row_mask_view,
                                      cudf::io::parquet::experimental::use_data_page_mask::YES,
                                      options,
                                      stream,
                                      mr)
        .tbl);
  };

  // Split the row groups into two passes and materialize each pass independently
  auto const row_group_span = cudf::host_span<cudf::size_type const>(all_row_groups);
  ASSERT_GT(row_group_span.size(), 1);
  auto const split = row_group_span.size() / 2;
  materialize_pass(row_group_span.subspan(0, split));
  materialize_pass(row_group_span.subspan(split, row_group_span.size() - split));

  auto const filter_table = cudf::concatenate(
    std::vector<cudf::table_view>{filter_tables[0]->view(), filter_tables[1]->view()});
  auto const payload_table = cudf::concatenate(
    std::vector<cudf::table_view>{payload_tables[0]->view(), payload_tables[1]->view()});

  // Read the same file with the mainline parquet reader for comparison
  auto const expected_options =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
      .filter(filter_expression)
      .build();
  auto const expected = cudf::io::read_parquet(expected_options, stream).tbl;

  // The filter column (col0) and payload columns (col1, col2) must match the mainline reader
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({0}), filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({1, 2}), payload_table->view());
}

TEST_F(HybridScanTest, RowGroupPassesMatchesChunkedReader)
{
  auto constexpr num_rg      = 10;
  auto constexpr rows_per_rg = 1'000;

  // Create a per-row-group table (each write() call produces one row group)
  auto values = cuda::counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int32_t> col0(values, values + rows_per_rg);
  cudf::test::fixed_width_column_wrapper<float> col1(values, values + rows_per_rg);
  auto chunk_table = cudf::table_view{{col0, col1}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  std::string parquet_filepath =
    temp_env->get_temp_filepath("RowGroupPassesMatchesChunkedReader.parquet");
  {
    auto opts =
      cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{parquet_filepath})
        .build();
    auto writer = cudf::io::chunked_parquet_writer(opts, stream);
    for (int i = 0; i < num_rg; ++i) {
      writer.write(chunk_table);
    }
    writer.close();
    stream.synchronize();
  }

  // Pick a pass_read_limit that forces multiple passes but groups some row groups together
  std::size_t const pass_read_limit = 2'048;

  // Table chunks from hybrid scan passes
  std::vector<std::unique_ptr<cudf::table>> hybrid_scan_tables;
  {
    auto options    = cudf::io::parquet_reader_options::builder().build();
    auto datasource = cudf::io::datasource::create(parquet_filepath);

    auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      *footer_buffer, options);

    auto const all_row_groups = reader->all_row_groups(options);
    auto const passes         = reader->construct_row_group_passes(all_row_groups, pass_read_limit);

    for (auto const& pass_row_groups : passes) {
      auto const chunk_byte_ranges =
        reader->all_column_chunks_byte_ranges(pass_row_groups, options);
      auto [buffers, col_data, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
        *datasource, chunk_byte_ranges, stream, mr);
      tasks.get();

      reader->setup_chunking_for_all_columns(
        0, pass_read_limit, pass_row_groups, col_data, options, stream, mr);

      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_all_columns_chunk();
        hybrid_scan_tables.push_back(std::move(chunk.tbl));
      }
    }
  }

  // Read with the chunked parquet reader using the same pass_read_limit and chunk_read_limit == 0
  std::vector<std::unique_ptr<cudf::table>> chunked_reader_tables;
  {
    auto opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(parquet_filepath)).build();
    auto chunked_reader = cudf::io::chunked_parquet_reader(0, pass_read_limit, opts, stream, mr);
    while (chunked_reader.has_next()) {
      auto chunk = chunked_reader.read_chunk();
      chunked_reader_tables.push_back(std::move(chunk.tbl));
    }
  }

  // Check
  EXPECT_EQ(hybrid_scan_tables.size(), chunked_reader_tables.size());
  auto iter = cuda::make_zip_iterator(hybrid_scan_tables.begin(), chunked_reader_tables.begin());
  std::for_each(iter, iter + hybrid_scan_tables.size(), [&](auto const& iter) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cuda::std::get<0>(iter)->view(),
                                       cuda::std::get<1>(iter)->view());
  });
}
