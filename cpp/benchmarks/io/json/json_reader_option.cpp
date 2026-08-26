/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

template <json_lines JsonLines>
void BM_json_read_options(nvbench::state& state, nvbench::type_list<nvbench::enum_type<JsonLines>>)
{
  constexpr auto json_lines_bool = JsonLines == json_lines::YES;

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const data_types = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::DECIMAL),
                                             static_cast<int32_t>(data_type::STRING),
                                             static_cast<int32_t>(data_type::LIST),
                                             static_cast<int32_t>(data_type::STRUCT)});

  auto const tbl = create_random_table(
    cycle_dtypes(data_types, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();
  cudf::io::json_writer_options const write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .lines(json_lines_bool)
      .na_rep("null")
      .rows_per_chunk(100'000);
  cudf::io::write_json(write_opts);

  cudf::io::json_reader_options read_options =
    cudf::io::json_reader_options::builder(source_sink.make_source_info()).lines(json_lines_bool);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());
      timer.start();
      auto const result        = cudf::io::read_json(read_options);
      auto const num_rows_read = result.tbl->num_rows();
      auto const num_cols_read = result.tbl->num_columns();
      timer.stop();
      CUDF_EXPECTS(num_rows_read == view.num_rows(), "Benchmark did not read the entire table");
      CUDF_EXPECTS(num_cols_read == num_cols, "Unexpected number of columns");
    });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * num_cols / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <row_selection RowSelection,
          normalize_single_quotes NormalizeSingleQuotes,
          normalize_whitespace NormalizeWhitespace,
          mixed_types_as_string MixedTypesAsString,
          recovery_mode RecoveryMode>
void BM_jsonlines_read_options(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<RowSelection>,
                                                  nvbench::enum_type<NormalizeSingleQuotes>,
                                                  nvbench::enum_type<NormalizeWhitespace>,
                                                  nvbench::enum_type<MixedTypesAsString>,
                                                  nvbench::enum_type<RecoveryMode>>)
{
  constexpr auto normalize_single_quotes_bool =
    NormalizeSingleQuotes == normalize_single_quotes::YES;
  constexpr auto normalize_whitespace_bool  = NormalizeWhitespace == normalize_whitespace::YES;
  constexpr auto mixed_types_as_string_bool = MixedTypesAsString == mixed_types_as_string::YES;
  constexpr auto recovery_mode_enum         = RecoveryMode == recovery_mode::RECOVER_WITH_NULL
                                                ? cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL
                                                : cudf::io::json_recovery_mode_t::FAIL;
  size_t const num_chunks                   = state.get_int64("num_chunks");
  if (num_chunks > 1 && RowSelection == row_selection::ALL) {
    state.skip(
      "No point running the same benchmark multiple times for different num_chunks when all rows "
      "are being selected anyway");
    return;
  }

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const data_types = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::DECIMAL),
                                             static_cast<int32_t>(data_type::STRING),
                                             static_cast<int32_t>(data_type::LIST),
                                             static_cast<int32_t>(data_type::STRUCT)});

  auto const tbl = create_random_table(
    cycle_dtypes(data_types, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();
  cudf::io::json_writer_options const write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .lines(true)
      .na_rep("null")
      .rows_per_chunk(100'000);
  cudf::io::write_json(write_opts);

  cudf::io::json_reader_options read_options =
    cudf::io::json_reader_options::builder(source_sink.make_source_info())
      .lines(true)
      .normalize_single_quotes(normalize_single_quotes_bool)
      .normalize_whitespace(normalize_whitespace_bool)
      .mixed_types_as_string(mixed_types_as_string_bool)
      .recovery_mode(recovery_mode_enum);

  size_t const chunk_size = cudf::util::div_rounding_up_safe(source_sink.size(), num_chunks);
  auto mem_stats_logger   = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());
      cudf::size_type num_rows_read = 0;
      cudf::size_type num_cols_read = 0;
      timer.start();
      switch (RowSelection) {
        case row_selection::ALL: {
          auto const result = cudf::io::read_json(read_options);
          num_rows_read     = result.tbl->num_rows();
          num_cols_read     = result.tbl->num_columns();
          break;
        }
        case row_selection::BYTE_RANGE: {
          for (uint64_t chunk = 0; chunk < num_chunks; chunk++) {
            read_options.set_byte_range_offset(chunk * chunk_size);
            read_options.set_byte_range_size(chunk_size);
            auto const result = cudf::io::read_json(read_options);
            num_rows_read += result.tbl->num_rows();
            num_cols_read = result.tbl->num_columns();
            if (num_cols_read)
              CUDF_EXPECTS(num_cols_read == num_cols, "Unexpected number of columns");
          }
          break;
        }
        default: CUDF_FAIL("Unsupported row selection method");
      }
      timer.stop();
      CUDF_EXPECTS(num_rows_read == view.num_rows(), "Benchmark did not read the entire table");
    });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * num_cols / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH_TYPES(BM_jsonlines_read_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<row_selection::ALL>,
                                      nvbench::enum_type_list<normalize_single_quotes::NO,
                                                              normalize_single_quotes::YES>,
                                      nvbench::enum_type_list<normalize_whitespace::NO>,
                                      nvbench::enum_type_list<mixed_types_as_string::NO>,
                                      nvbench::enum_type_list<recovery_mode::FAIL>))
  .set_name("jsonlines_reader_normalize_single_quotes")
  .set_type_axes_names({"row_selection",
                        "normalize_single_quotes",
                        "normalize_whitespace",
                        "mixed_types_as_string",
                        "recovery_mode"})
  .set_min_samples(6)
  .add_int64_axis("num_chunks", nvbench::range(1, 1, 1));

NVBENCH_BENCH_TYPES(
  BM_jsonlines_read_options,
  NVBENCH_TYPE_AXES(nvbench::enum_type_list<row_selection::ALL>,
                    nvbench::enum_type_list<normalize_single_quotes::NO>,
                    nvbench::enum_type_list<normalize_whitespace::NO, normalize_whitespace::YES>,
                    nvbench::enum_type_list<mixed_types_as_string::NO>,
                    nvbench::enum_type_list<recovery_mode::FAIL>))
  .set_name("jsonlines_reader_normalize_whitespace")
  .set_type_axes_names({"row_selection",
                        "normalize_single_quotes",
                        "normalize_whitespace",
                        "mixed_types_as_string",
                        "recovery_mode"})
  .set_min_samples(6)
  .add_int64_axis("num_chunks", nvbench::range(1, 1, 1));

NVBENCH_BENCH_TYPES(
  BM_jsonlines_read_options,
  NVBENCH_TYPE_AXES(nvbench::enum_type_list<row_selection::ALL>,
                    nvbench::enum_type_list<normalize_single_quotes::NO>,
                    nvbench::enum_type_list<normalize_whitespace::NO>,
                    nvbench::enum_type_list<mixed_types_as_string::NO, mixed_types_as_string::YES>,
                    nvbench::enum_type_list<recovery_mode::RECOVER_WITH_NULL, recovery_mode::FAIL>))
  .set_name("jsonlines_reader_mixed_types_as_string")
  .set_type_axes_names({"row_selection",
                        "normalize_single_quotes",
                        "normalize_whitespace",
                        "mixed_types_as_string",
                        "recovery_mode"})
  .set_min_samples(6)
  .add_int64_axis("num_chunks", nvbench::range(1, 1, 1));

NVBENCH_BENCH_TYPES(
  BM_jsonlines_read_options,
  NVBENCH_TYPE_AXES(nvbench::enum_type_list<row_selection::ALL, row_selection::BYTE_RANGE>,
                    nvbench::enum_type_list<normalize_single_quotes::NO>,
                    nvbench::enum_type_list<normalize_whitespace::NO>,
                    nvbench::enum_type_list<mixed_types_as_string::NO>,
                    nvbench::enum_type_list<recovery_mode::FAIL>))
  .set_name("jsonlines_reader_row_selection")
  .set_type_axes_names({"row_selection",
                        "normalize_single_quotes",
                        "normalize_whitespace",
                        "mixed_types_as_string",
                        "recovery_mode"})
  .set_min_samples(6)
  .add_int64_axis("num_chunks", nvbench::range(1, 5, 1));

NVBENCH_BENCH_TYPES(BM_json_read_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<json_lines::YES, json_lines::NO>))
  .set_name("json_reader")
  .set_type_axes_names({"json_lines"})
  .set_min_samples(6);
