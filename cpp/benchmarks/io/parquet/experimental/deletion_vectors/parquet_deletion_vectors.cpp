/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>
#include <roaring/roaring64.h>

#include <random>

namespace {
/**
 * @brief Serializes a roaring64 bitmap to a vector of cuda::std::byte
 *
 * @param deletion_vector Pointer to the roaring64 bitmap to serialize
 *
 * @return Host vector of bytes containing the serialized roaring64 bitmap
 */
auto serialize_roaring_bitmap(roaring64_bitmap_t const* roaring_bitmap)
{
  auto const num_bytes = roaring64_bitmap_portable_size_in_bytes(roaring_bitmap);
  CUDF_EXPECTS(num_bytes > 0, "Roaring64 bitmap is empty");
  auto serialized_bitmap = std::vector<cuda::std::byte>(num_bytes);
  std::ignore            = roaring64_bitmap_portable_serialize(
    roaring_bitmap, reinterpret_cast<char*>(serialized_bitmap.data()));
  return serialized_bitmap;
}

/**
 * @brief Builds a host vector of expected row indices from the specified row group offsets and
 * row counts
 *
 * @param row_group_offsets Row group offsets
 * @param row_group_num_rows Number of rows in each row group
 * @param num_rows Total number of table rows
 *
 * @return Host vector of expected row indices
 */
auto build_row_indices(cudf::host_span<std::size_t const> row_group_offsets,
                       cudf::host_span<cudf::size_type const> row_group_num_rows,
                       cudf::size_type num_rows)
{
  auto const num_row_groups = static_cast<cudf::size_type>(row_group_num_rows.size());

  // Row group span offsets
  auto row_group_span_offsets = thrust::host_vector<cudf::size_type>(num_row_groups + 1);
  row_group_span_offsets[0]   = 0;
  thrust::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  // Expected row indices data
  auto expected_row_indices = thrust::host_vector<std::size_t>(num_rows);
  std::fill(expected_row_indices.begin(), expected_row_indices.end(), 1);

  // Scatter row group row offsets to expected row indices
  thrust::scatter(row_group_offsets.begin(),
                  row_group_offsets.end(),
                  row_group_span_offsets.begin(),
                  expected_row_indices.begin());

  // Inclusive scan to compute the rest of the expected row indices
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_row_groups},
                [&](cudf::size_type i) {
                  auto start_row_index = row_group_span_offsets[i];
                  auto end_row_index   = row_group_span_offsets[i + 1];
                  thrust::inclusive_scan(expected_row_indices.begin() + start_row_index,
                                         expected_row_indices.begin() + end_row_index,
                                         expected_row_indices.begin() + start_row_index);
                });

  return expected_row_indices;
}

/**
 * @brief Builds a roaring64 deletion vector and a (host) row mask vector based on the specified
 * probability of a row being deleted
 *
 * @param row_group_offsets Row group row offsets
 * @param row_group_num_rows Number of rows in each row group
 * @param num_rows Number of rows in the table
 * @param deletion_probability The probability of a row being deleted
 *
 * @return Serialized roaring64 bitmap buffer
 */
auto build_deletion_vector(cudf::host_span<std::size_t const> row_group_offsets,
                           cudf::host_span<cudf::size_type const> row_group_num_rows,
                           cudf::size_type num_rows,
                           float deletion_probability)
{
  std::mt19937 engine{0xbaLL};
  std::bernoulli_distribution dist(deletion_probability);

  auto row_indices = build_row_indices(row_group_offsets, row_group_num_rows, num_rows);

  CUDF_EXPECTS(std::cmp_equal(row_indices.size(), num_rows),
               "Row indices vector must have the same number of rows as the table");

  auto input_row_mask = thrust::host_vector<bool>(num_rows);
  std::generate(input_row_mask.begin(), input_row_mask.end(), [&]() { return dist(engine); });

  auto deletion_vector = roaring64_bitmap_create();

  // Context for the roaring64 bitmap for faster (bulk) add operations
  auto roaring64_context =
    roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_rows},
                [&](auto row_idx) {
                  // Insert provided host row index if the row is deleted in the row mask
                  if (not input_row_mask[row_idx]) {
                    roaring64_bitmap_add_bulk(
                      deletion_vector, &roaring64_context, row_indices[row_idx]);
                  }
                });

  return serialize_roaring_bitmap(deletion_vector);
}

auto setup_table_and_deletion_vector(nvbench::state& state)
{
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const rows_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("rows_per_row_group"));
  auto const num_row_groups       = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const deletion_probability = static_cast<float>(state.get_float64("deletion_probability"));
  auto const source_type          = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows             = rows_per_row_group * num_row_groups;

  cuio_source_sink_pair source_sink(source_type);

  // Create a table and write it to parquet sink
  {
    auto const d_types = std::vector<cudf::type_id>{
      cudf::type_id::FLOAT64,
      cudf::type_id::DURATION_MICROSECONDS,
      cudf::type_id::TIMESTAMP_MILLISECONDS,
      cudf::type_id::STRING,
    };

    auto const table = create_random_table(cycle_dtypes(d_types, num_columns),
                                           row_count{num_rows},
                                           data_profile_builder().null_probability(0.10),
                                           0xbad);
    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), table->view())
        .row_group_size_rows(rows_per_row_group)
        .compression(cudf::io::compression_type::NONE);
    cudf::io::write_parquet(write_opts);
  }

  // Row offsets for each row group - arbitrary, only used to build the index column
  auto row_group_offsets = std::vector<std::size_t>(num_row_groups);
  row_group_offsets[0]   = static_cast<std::size_t>(std::llround(2e9));
  std::for_each(
    cuda::counting_iterator<int>{1}, cuda::counting_iterator{num_row_groups}, [&](auto i) {
      row_group_offsets[i] = std::llround(row_group_offsets[i - 1] + 0.5e9);
    });

  // Row group splits
  auto row_group_splits = std::vector<cudf::size_type>(num_row_groups - 1);
  {
    std::mt19937 engine{0xf00d};
    std::uniform_int_distribution<cudf::size_type> dist{1, num_rows};
    std::generate(row_group_splits.begin(), row_group_splits.end(), [&]() { return dist(engine); });
    std::sort(row_group_splits.begin(), row_group_splits.end());
  }

  // Number of rows in each row group
  auto row_group_num_rows = std::vector<cudf::size_type>{};
  {
    row_group_num_rows.reserve(num_row_groups);
    auto previous_split = cudf::size_type{0};
    std::transform(row_group_splits.begin(),
                   row_group_splits.end(),
                   std::back_inserter(row_group_num_rows),
                   [&](auto current_split) {
                     auto current_split_size = current_split - previous_split;
                     previous_split          = current_split;
                     return current_split_size;
                   });
    row_group_num_rows.push_back(num_rows - row_group_splits.back());
  }

  auto deletion_vector =
    build_deletion_vector(row_group_offsets, row_group_num_rows, num_rows, deletion_probability);

  return std::tuple{std::move(source_sink),
                    std::move(row_group_offsets),
                    std::move(row_group_num_rows),
                    std::move(deletion_vector)};
}

}  // namespace

void BM_parquet_deletion_vectors(nvbench::state& state)
{
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const rows_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("rows_per_row_group"));
  auto const num_rows = rows_per_row_group * num_row_groups;

  auto [source_sink, row_group_offsets, row_group_num_rows, deletion_vector] =
    setup_table_and_deletion_vector(state);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {std::numeric_limits<cudf::size_type>::max()},
    .row_group_offsets          = std::move(row_group_offsets),
    .row_group_num_rows         = std::move(row_group_num_rows),
  };

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();
      std::ignore = cudf::io::parquet::experimental::read_parquet(read_opts, deletion_vector_info);
      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / time, "rows_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

void BM_parquet_chunked_deletion_vectors(nvbench::state& state)
{
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const rows_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("rows_per_row_group"));
  auto const num_rows         = rows_per_row_group * num_row_groups;
  auto const chunk_read_limit = static_cast<cudf::size_type>(state.get_int64("chunk_read_limit"));
  auto const pass_read_limit  = static_cast<cudf::size_type>(state.get_int64("pass_read_limit"));

  auto [source_sink, row_group_offsets, row_group_num_rows, deletion_vector] =
    setup_table_and_deletion_vector(state);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {std::numeric_limits<cudf::size_type>::max()},
    .row_group_offsets          = std::move(row_group_offsets),
    .row_group_num_rows         = std::move(row_group_num_rows),
  };

  auto num_chunks       = 0;
  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_opts.get_source().filepaths());

               timer.start();
               auto reader = cudf::io::parquet::experimental::chunked_parquet_reader(
                 chunk_read_limit, pass_read_limit, read_opts, deletion_vector_info);
               do {
                 auto const result = reader.read_chunk();
                 num_chunks++;
               } while (reader.has_next());
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(num_chunks, "num_table_chunks");
  state.add_element_count(static_cast<double>(num_rows) / time, "bytes_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_mem_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_parquet_deletion_vectors)
  .set_name("parquet_deletion_vectors")
  .set_min_samples(4)
  .add_int64_power_of_two_axis("num_row_groups", nvbench::range(4, 14, 2))
  .add_int64_axis("rows_per_row_group", {5'000, 10'000})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_float64_axis("deletion_probability", {0.25, 0.65})
  .add_int64_axis("num_cols", {4});

NVBENCH_BENCH(BM_parquet_chunked_deletion_vectors)
  .set_name("parquet_chunked_deletion_vectors")
  .set_min_samples(4)
  .add_int64_power_of_two_axis("num_row_groups", nvbench::range(4, 14, 2))
  .add_int64_axis("rows_per_row_group", {5'000, 10'000})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_int64_axis("chunk_read_limit", {4'096'000})
  .add_int64_axis("pass_read_limit", {10'240'000, 102'400'000})
  .add_float64_axis("deletion_probability", {0.50})
  .add_int64_axis("num_cols", {4});
