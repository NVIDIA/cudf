/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <url_log_fragments.hpp>

#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr auto output_count = std::size_t{6};

constexpr std::string_view usage =
  "usage: url_log_transforms INPUT.csv OUTPUT.csv <regex|precompiled> ROWS\n"
  "       url_log_transforms INPUT.csv OUTPUT.csv lto ROWS "
  "<--warm|--cold>\n"
  "       url_log_transforms <usage|--help>\n";

// Extracts RFC 3986-style hierarchical URI components from unstructured log lines.
[[nodiscard]] std::unique_ptr<cudf::table> run_regex(cudf::column_view input,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  // Derived from RFC 3986 Appendix B (https://www.rfc-editor.org/info/rfc3986/#page-50). The
  // authority capture is expanded into optional userinfo plus host and port, and Appendix C
  // delimiters bound the URI within a log line.
  static auto program = cudf::strings::regex_program::create(
    R"((?:^|[^A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*):\/\/(?:[^@\/?# \t\n\r"<>]*@)?(\[[^\]\/?# \t\n\r"<>]*\]|[^\/:?# \t\n\r"<>]*)(?::([0-9]*))?([^?# \t\n\r"<>]*)(?:\?([^# \t\n\r"<>]*))?(?:#([^ \t\n\r"<>]*))?(?:$|[ \t\n\r"<>]))");
  auto extracted = cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
  auto columns   = extracted->release();
  auto empty     = cudf::string_scalar{"", true, stream, mr};
  for (auto& column : columns) {
    column = cudf::replace_nulls(column->view(), empty, stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// Decomposes key-value URL tokens using only precompiled libcudf string primitives.
[[nodiscard]] std::unique_ptr<cudf::table> run_precompiled(cudf::column_view input,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  // Materialize the delimiters used by each partitioning stage.
  auto empty            = cudf::string_scalar{"", true, stream, mr};
  auto scheme_separator = cudf::string_scalar{"://", true, stream, mr};
  auto marker_separator = cudf::string_scalar{"=", true, stream, mr};
  auto token_separator  = cudf::string_scalar{" ", true, stream, mr};
  auto hash             = cudf::string_scalar{"#", true, stream, mr};
  auto question         = cudf::string_scalar{"?", true, stream, mr};
  auto slash            = cudf::string_scalar{"/", true, stream, mr};
  auto at               = cudf::string_scalar{"@", true, stream, mr};
  auto right_bracket    = cudf::string_scalar{"]", true, stream, mr};
  auto left_bracket     = cudf::string_scalar{"[", true, stream, mr};
  auto colon            = cudf::string_scalar{":", true, stream, mr};

  // Mark rows containing an authority-style URI and split at the first "://".
  auto has_url =
    cudf::strings::contains(cudf::strings_column_view{input}, scheme_separator, stream, mr);
  auto scheme_table =
    cudf::strings::partition(cudf::strings_column_view{input}, scheme_separator, stream, mr);
  auto scheme_columns = scheme_table->release();

  // Extract the scheme from the key-value token immediately preceding "://".
  auto marker_table = cudf::strings::rpartition(
    cudf::strings_column_view{scheme_columns[0]->view()}, marker_separator, stream, mr);
  auto marker_columns = marker_table->release();

  // Stop at the first space so later log fields are excluded from the URI.
  auto token_table = cudf::strings::partition(
    cudf::strings_column_view{scheme_columns[2]->view()}, token_separator, stream, mr);
  auto token_columns = token_table->release();

  // Split off the fragment; everything after the first '#' belongs to it.
  auto fragment_table =
    cudf::strings::partition(cudf::strings_column_view{token_columns[0]->view()}, hash, stream, mr);
  auto fragment_columns = fragment_table->release();

  // Split the pre-fragment portion at the first '?' to isolate the query.
  auto query_table = cudf::strings::partition(
    cudf::strings_column_view{fragment_columns[0]->view()}, question, stream, mr);
  auto query_columns = query_table->release();

  // Split the remaining hierarchical part at its first slash into authority and path.
  auto authority_path_table = cudf::strings::partition(
    cudf::strings_column_view{query_columns[0]->view()}, slash, stream, mr);
  auto authority_path_columns = authority_path_table->release();

  // Reattach the slash delimiter to produce the RFC path value.
  auto path = cudf::strings::concatenate(
    cudf::table_view{{authority_path_columns[1]->view(), authority_path_columns[2]->view()}},
    empty,
    cudf::string_scalar{"", false, stream, mr},
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Remove optional userinfo by retaining everything after the authority's last '@'.
  auto has_userinfo = cudf::strings::contains(
    cudf::strings_column_view{authority_path_columns[0]->view()}, at, stream, mr);
  auto userinfo_table = cudf::strings::rpartition(
    cudf::strings_column_view{authority_path_columns[0]->view()}, at, stream, mr);
  auto userinfo_columns = userinfo_table->release();
  auto host_port        = cudf::copy_if_else(userinfo_columns[2]->view(),
                                      authority_path_columns[0]->view(),
                                      has_userinfo->view(),
                                      stream,
                                      mr);

  // Bracketed IP literals and regular hosts require different port splitting rules.
  auto is_ip_literal = cudf::strings::starts_with(
    cudf::strings_column_view{host_port->view()}, left_bracket, stream, mr);
  auto bracket_table = cudf::strings::partition(
    cudf::strings_column_view{host_port->view()}, right_bracket, stream, mr);
  auto bracket_columns = bracket_table->release();

  // Preserve both brackets as part of an IP-literal host.
  auto bracket_host = cudf::strings::concatenate(
    cudf::table_view{{bracket_columns[0]->view(), bracket_columns[1]->view()}},
    empty,
    cudf::string_scalar{"", false, stream, mr},
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // For an IP literal, parse an optional port only after the closing bracket.
  auto bracket_port_table = cudf::strings::partition(
    cudf::strings_column_view{bracket_columns[2]->view()}, colon, stream, mr);
  auto bracket_port_columns = bracket_port_table->release();

  // For a regular authority, treat the final colon as the port separator.
  auto has_regular_port =
    cudf::strings::contains(cudf::strings_column_view{host_port->view()}, colon, stream, mr);
  auto regular_table =
    cudf::strings::rpartition(cudf::strings_column_view{host_port->view()}, colon, stream, mr);
  auto regular_columns = regular_table->release();
  auto regular_host    = cudf::copy_if_else(
    regular_columns[0]->view(), host_port->view(), has_regular_port->view(), stream, mr);
  auto regular_port =
    cudf::copy_if_else(regular_columns[2]->view(), empty, has_regular_port->view(), stream, mr);

  // Select the bracketed or regular host/port result for each row.
  auto host = cudf::copy_if_else(
    bracket_host->view(), regular_host->view(), is_ip_literal->view(), stream, mr);
  auto port = cudf::copy_if_else(
    bracket_port_columns[2]->view(), regular_port->view(), is_ip_literal->view(), stream, mr);

  // Convert missing components to empty strings and blank rows without a URL.
  auto normalize = [&](cudf::column_view column) {
    auto no_nulls = cudf::replace_nulls(column, empty, stream, mr);
    return cudf::copy_if_else(no_nulls->view(), empty, has_url->view(), stream, mr);
  };

  // Return the six columns in the same order used by the regex and LTO implementations.
  std::vector<std::unique_ptr<cudf::column>> result;
  result.reserve(output_count);
  result.push_back(normalize(marker_columns[2]->view()));
  result.push_back(normalize(host->view()));
  result.push_back(normalize(port->view()));
  result.push_back(normalize(path->view()));
  result.push_back(normalize(query_columns[2]->view()));
  result.push_back(normalize(fragment_columns[2]->view()));
  return std::make_unique<cudf::table>(std::move(result));
}

// Runs the two exports from the precompiled URL-parser module.
[[nodiscard]] std::unique_ptr<cudf::table> run_jit(cudf::column_view input,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  cudf::transform_output const size_spec{cudf::data_type{cudf::type_id::INT32},
                                         cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const size_outputs(output_count, size_spec);
  cudf::transform_input inputs[] = {input};
  cudf::lto_fragment module{cudf::lto_binary_type::FATBIN,
                            url_log_fragments::get(url_log_fragments::url_components)};
  auto sizes = cudf::transform({"compute_url_component_sizes", {&module, 1}},
                               inputs,
                               size_outputs,
                               {},
                               std::nullopt,
                               stream,
                               mr);

  std::vector<std::unique_ptr<cudf::column>> offsets;
  offsets.reserve(output_count);
  for (auto& string_sizes : sizes->view()) {
    auto run_ends = cudf::scan(string_sizes,
                               *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                               cudf::scan_type::INCLUSIVE,
                               cudf::null_policy::EXCLUDE,
                               stream,
                               mr);
    auto zero     = cudf::numeric_scalar<int32_t>{0, true, stream, mr};
    auto first    = cudf::make_column_from_scalar(zero, 1, stream, mr);
    offsets.push_back(cudf::concatenate(
      std::vector<cudf::column_view>{first->view(), run_ends->view()}, stream, mr));
  }

  cudf::transform_output const output_spec{cudf::data_type{cudf::type_id::STRING},
                                           cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const outputs(output_count, output_spec);
  return cudf::transform({"write_url_components", {&module, 1}},
                         inputs,
                         outputs,
                         std::move(offsets),
                         std::nullopt,
                         stream,
                         mr);
}

}  // namespace

int main(int argc, char const** argv)
try {
  if (argc == 2 &&
      (std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "usage")) {
    std::cout << usage;
    return EXIT_SUCCESS;
  }
  if (argc != 5 && argc != 6) {
    throw std::invalid_argument("invalid arguments; run url_log_transforms --help for usage");
  }

  auto input_path  = std::string{argv[1]};
  auto output_path = std::string{argv[2]};
  auto impl        = std::string_view{argv[3]};
  if (impl != "regex" && impl != "precompiled" && impl != "lto") {
    throw std::invalid_argument("executor must be regex, precompiled, or lto");
  }
  auto requested_rows = std::stoll(argv[4]);
  auto is_jit         = impl == "lto";
  if (is_jit && argc != 6) { throw std::invalid_argument("lto requires a warm-up control"); }
  if (!is_jit && argc != 5) {
    throw std::invalid_argument("regex and precompiled do not accept a warm-up control");
  }
  auto warmup_control = argc == 6 ? std::string_view{argv[5]} : std::string_view{"none"};
  if (is_jit && warmup_control != "--warm" && warmup_control != "--cold") {
    throw std::invalid_argument("warm-up control must be --warm or --cold");
  }
  if (requested_rows < 0 || requested_rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("ROWS is outside the cudf::size_type range");
  }
  nvtxRangePush("url_log_process");
  auto process_start = std::chrono::steady_clock::now();
  auto rows          = static_cast<cudf::size_type>(requested_rows);
  auto stream        = cudf::get_default_stream();
  auto upstream_mr   = cudf::get_current_device_resource_ref();
  // Tracks setup, measured work, and output
  rmm::mr::statistics_resource_adaptor whole_stats{upstream_mr};
  auto whole_mr = rmm::device_async_resource_ref{whole_stats};
  cudf::set_current_device_resource(whole_mr);

  nvtxRangePush("url_log_setup");
  auto read_options = cudf::io::csv_reader_options::builder(cudf::io::source_info{input_path})
                        .header(0)
                        .use_cols_names({"LogLine"})
                        .build();
  auto input = cudf::io::read_csv(read_options).tbl;
  if (rows != input->num_rows()) {
    input =
      cudf::sample(input->view(), rows, cudf::sample_with_replacement::TRUE, 0, stream, whole_mr);
  }
  stream.sync();
  auto input_view          = input->get_column(0).view();
  auto logical_input_bytes = cudf::strings_column_view{input_view}.chars_size(stream);
  nvtxRangePop();

  // Tracks measured work; nested allocations also update whole_stats.
  rmm::mr::statistics_resource_adaptor measured_stats{whole_mr};
  auto measured_mr   = rmm::device_async_resource_ref{measured_stats};
  auto run_transform = [&](rmm::device_async_resource_ref mr) {
    if (impl == "regex") {
      return run_regex(input_view, stream, mr);
    } else if (impl == "precompiled") {
      return run_precompiled(input_view, stream, mr);
    } else {
      return run_jit(input_view, stream, mr);
    }
  };

  std::unique_ptr<cudf::table> result;
  auto warmup_duration = std::chrono::steady_clock::duration::zero();

  if (warmup_control == "--warm") {
    // Do not track warm-up allocations.
    cudf::set_current_device_resource(upstream_mr);
    stream.sync();
    auto warmup_start = std::chrono::steady_clock::now();
    nvtxRangePush("url_log_warmup");
    result = run_transform(upstream_mr);
    stream.sync();
    nvtxRangePop();
    warmup_duration = std::chrono::steady_clock::now() - warmup_start;
    result.reset();
    cudf::set_current_device_resource(whole_mr);
  }

  // Measured allocations update both statistics scopes.
  cudf::set_current_device_resource(measured_mr);
  stream.sync();
  auto measured_start = std::chrono::steady_clock::now();
  nvtxRangePush("url_log_measured");
  result = run_transform(measured_mr);
  stream.sync();
  nvtxRangePop();
  auto measured_duration = std::chrono::steady_clock::now() - measured_start;

  if (output_path != "-") {
    // Exclude output serialization from measured statistics.
    cudf::set_current_device_resource(whole_mr);
    auto write_options =
      cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, result->view())
        .include_header(true)
        .names({"protocol", "host", "port", "path", "query", "fragment"})
        .build();
    cudf::io::write_csv(write_options);
  }

  // Read measured and broader workload scopes separately.
  auto measured_bytes         = measured_stats.get_bytes_counter();
  auto whole_bytes            = whole_stats.get_bytes_counter();
  auto output_allocated_bytes = result->alloc_size();
  auto input_gib      = static_cast<double>(logical_input_bytes) / static_cast<double>(1ULL << 30);
  auto whole_duration = std::chrono::steady_clock::now() - process_start;
  auto warmup_seconds = std::chrono::duration<double>{warmup_duration}.count();
  auto measured_seconds = std::chrono::duration<double>{measured_duration}.count();
  auto whole_seconds    = std::chrono::duration<double>{whole_duration}.count();
  std::cout << std::format(
    "executor={}\nwarmup_control={}\nrows={}\nwarmup_seconds={}\n"
    "measured_cpu_wall_seconds={}\nrows_per_second={}\n"
    "input_gib_per_second={}\nlogical_input_bytes={}\noutput_allocated_bytes={}\n"
    "peak_memory_bytes={}\n"
    "total_allocated_bytes={}\nallocated_bytes_per_call={}\nmeasured_gpu_peak_bytes={}\n"
    "measured_gpu_allocation_volume_bytes={}\nwhole_workload_seconds={}\n"
    "whole_gpu_peak_bytes={}\nwhole_gpu_allocation_volume_bytes={}\n",
    impl,
    warmup_control,
    rows,
    warmup_seconds,
    measured_seconds,
    static_cast<double>(rows) / measured_seconds,
    input_gib / measured_seconds,
    logical_input_bytes,
    output_allocated_bytes,
    measured_bytes.peak,
    measured_bytes.total,
    measured_bytes.total,
    measured_bytes.peak,
    measured_bytes.total,
    whole_seconds,
    whole_bytes.peak,
    whole_bytes.total);
  result.reset();
  input.reset();
  cudf::set_current_device_resource(upstream_mr);
  nvtxRangePop();
  return EXIT_SUCCESS;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return EXIT_FAILURE;
}
