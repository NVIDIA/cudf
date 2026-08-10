/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"
#include "io/parquet/reader_impl_preprocess_utils.cuh"
#include "io/utilities/time_utils.hpp"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/sequence.h>

#include <numeric>

namespace cudf::io::parquet::experimental::detail {

namespace {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::PageInfo;

/**
 * @brief Decode the dictionary page information from each column chunk
 *
 * @param chunks Host device span of column chunk descriptors, one per input column chunk
 * @param pages Host device span of empty page headers to fill in, one per input column chunk
 * @param stream CUDA stream
 */
void decode_dictionary_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                                    cudf::detail::hostdevice_span<PageInfo> pages,
                                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  rmm::device_uvector<chunk_page_info> chunk_page_info(chunks.size(), stream);
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<cuda::std::size_t>{0},
                   cuda::counting_iterator{chunks.size()},
                   [cpi = chunk_page_info.begin(), pages = pages.device_begin()] __device__(
                     auto page_idx) { cpi[page_idx].pages = &pages[page_idx]; });

  parquet::kernel_error error_code(stream);

  parquet::detail::decode_page_headers(chunks, chunk_page_info, error_code.data(), stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL("Parquet header parsing failed with code(s) " +
              parquet::kernel_error::to_string(error));
  }

  // Setup dictionary page for each chunk
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   pages.device_begin(),
                   pages.device_end(),
                   [chunks = chunks.device_begin()] __device__(PageInfo const& p) {
                     if (p.flags & parquet::detail::PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pages.device_to_host_async(stream);
  chunks.device_to_host_async(stream);
  stream.synchronize();
}

}  // namespace

void hybrid_scan_reader_impl::prepare_row_groups(
  read_mode mode, std::span<std::vector<size_type> const> row_group_indices)
{
  std::tie(_file_itm_data.global_skip_rows,
           _file_itm_data.global_num_rows,
           _file_itm_data.row_groups,
           _file_itm_data.num_rows_per_source,
           _file_itm_data.num_input_row_groups,
           _file_itm_data.surviving_row_groups) =
    _extended_metadata->select_row_groups({},
                                          cudf::host_span<std::vector<size_type> const>{
                                            row_group_indices.data(), row_group_indices.size()},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          _stream);

  CUDF_EXPECTS(
    std::cmp_less_equal(_file_itm_data.global_num_rows, std::numeric_limits<size_type>::max()),
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit. "
    "For reading larger number of rows, please use chunked_parquet_reader.",
    std::overflow_error);

  // Inclusive scan the number of rows per source
  _file_itm_data.exclusive_sum_num_rows_per_source.resize(
    _file_itm_data.num_rows_per_source.size());
  std::inclusive_scan(_file_itm_data.num_rows_per_source.cbegin(),
                      _file_itm_data.num_rows_per_source.cend(),
                      _file_itm_data.exclusive_sum_num_rows_per_source.begin());

  // Check for offset indexes.
  _has_offset_index =
    _extended_metadata->has_offset_index(_file_itm_data.row_groups, _input_columns);

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty()) {
    // fills in chunk information without physically loading or decompressing
    // the associated data
    create_global_chunk_info();

    // compute schedule of input reads.
    compute_input_passes(mode);
  }

  _file_preprocessed = true;
}

bool hybrid_scan_reader_impl::setup_column_chunks(
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data)
{
  auto const& row_groups_info = _pass_itm_data->row_groups;
  auto& chunks                = _pass_itm_data->chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  size_type chunk_count          = 0;
  for (auto const& rg : row_groups_info) {
    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto const& col = _input_columns[i];
      // look up metadata
      auto& col_meta =
        _extended_metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }

      CUDF_EXPECTS(column_chunk_data[chunk_count].data() != nullptr and
                     column_chunk_data[chunk_count].size() > 0,
                   "Encountered an invalid column chunk data span");
      // Set pointer to compressed data from the device span
      chunks[chunk_count].compressed_data = column_chunk_data[chunk_count].data();

      chunk_count++;
    }
  }
  return total_decompressed_size > 0;
}

void hybrid_scan_reader_impl::setup_compressed_data(
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data)
{
  auto& pass = *_pass_itm_data;

  // This function should never be called if `num_rows == 0`.
  CUDF_EXPECTS(_pass_itm_data->num_rows > 0, "Number of reading rows must not be zero.");

  auto& chunks = pass.chunks;

  pass.has_compressed_data = setup_column_chunks(column_chunk_data);

  // Process dataset chunk pages into output columns
  auto const total_pages = _has_offset_index ? count_page_headers_with_pgidx(chunks, _stream)
                                             : count_page_headers(chunks, _stream);
  if (total_pages <= 0) { return; }
  rmm::device_uvector<PageInfo> unsorted_pages(total_pages, _stream);

  // decoding of column/page information
  parquet::detail::decode_page_headers(pass, unsorted_pages, _has_offset_index, _stream);
  CUDF_EXPECTS(pass.page_offsets.size() - 1 == static_cast<size_t>(_input_columns.size()),
               "Encountered page_offsets / num_columns mismatch");
}

std::tuple<bool,
           cudf::detail::hostdevice_vector<ColumnChunkDesc>,
           cudf::detail::hostdevice_vector<PageInfo>>
hybrid_scan_reader_impl::prepare_dictionaries(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<int const> dictionary_col_schemas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  // Create row group information for the input row group indices
  auto const row_groups_info = std::get<2>(
    _extended_metadata->select_row_groups({},
                                          cudf::host_span<std::vector<size_type> const>{
                                            row_group_indices.data(), row_group_indices.size()},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          _stream));

  CUDF_EXPECTS(
    row_groups_info.size() * dictionary_col_schemas.size() == dictionary_page_data.size(),
    "Dictionary page data size must match the number of row groups times the number of columns "
    "with dictionaries and an (in)equality predicate");

  // Number of column chunks
  auto const total_column_chunks = dictionary_page_data.size();

  // Boolean to check if any of the column chunnks have compressed data
  auto has_compressed_data = false;

  // Initialize column chunk descriptors
  auto chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
    total_column_chunks, stream);
  auto chunk_idx = 0;

  // For all row groups
  for (auto const& rg : row_groups_info) {
    auto const& row_group = _extended_metadata->get_row_group(rg.index, rg.source_index);

    // For all columns with dictionary page and (in)equality predicate
    for (auto col_schema_idx : dictionary_col_schemas) {
      // look up metadata
      auto& col_meta =
        _extended_metadata->get_column_metadata(rg.index, rg.source_index, col_schema_idx);
      auto& schema = _extended_metadata->get_schema(
        _extended_metadata->map_schema_index(col_schema_idx, rg.source_index), rg.source_index);

      // dictionary data buffer for this column chunk
      auto& dict_page_data = dictionary_page_data[chunk_idx];

      // Check if the column chunk has compressed data
      has_compressed_data |=
        col_meta.codec != Compression::UNCOMPRESSED and col_meta.total_compressed_size > 0;

      auto const [clock_rate, logical_type] = parquet::detail::conversion_info(
        parquet::detail::to_type_id(schema,
                                    options.is_enabled_convert_strings_to_categories(),
                                    options.get_timestamp_type().id(),
                                    options.get_decimal_width()),
        options.get_timestamp_type().id(),
        schema.type,
        schema.logical_type);

      // Create a column chunk descriptor - zero/null values for all fields that are not needed
      chunks[chunk_idx] = ColumnChunkDesc(static_cast<int64_t>(dict_page_data.size()),
                                          const_cast<uint8_t*>(dict_page_data.data()),
                                          col_meta.num_values,
                                          schema.type,
                                          schema.type_length,
                                          0,  // start_row
                                          0,  // num_rows
                                          0,  // max_definition_level
                                          0,  // max_repetition_level
                                          0,  // max_nesting_depth
                                          0,  // def_level_bits
                                          0,  // rep_level_bits
                                          col_meta.codec,
                                          logical_type,
                                          clock_rate,
                                          0,  // src_col_index
                                          col_schema_idx,
                                          nullptr,  // chunk_info
                                          0.0f,     // list_bytes_per_row_est
                                          false,    // strings_to_categorical
                                          rg.source_index);
      // Set the number of dictionary and data pages
      chunks[chunk_idx].num_dict_pages = static_cast<int32_t>(dict_page_data.size() > 0);
      chunks[chunk_idx].num_data_pages = 0;  // Always zero at this stage
      chunk_idx++;
    }
  }

  // Copy the column chunk descriptors to the device
  chunks.host_to_device_async(stream);

  // Create page infos for each column chunk's dictionary page
  cudf::detail::hostdevice_vector<PageInfo> pages(total_column_chunks, stream);

  // Decode dictionary page headers
  decode_dictionary_page_headers(chunks, pages, stream);

  return {has_compressed_data, std::move(chunks), std::move(pages)};
}

namespace {

/**
 * @brief Computes the updated row mask value such that out_row_mask[i] = true, iff in_row_mask[i]
 * is valid and true. This is inline with the masking behavior of cudf::apply_boolean_mask.
 */
struct row_mask_update_fn {
  bool is_nullable;
  bool const* in_row_mask;
  bitmask_type const* in_bitmask;

  __device__ bool operator()(cudf::size_type row_idx) const
  {
    if (is_nullable and not bit_is_set(in_bitmask, row_idx)) { return false; }
    return in_row_mask[row_idx];
  }
};

/**
 * @brief Checks if a row is pruned (valid and false)
 */
struct is_row_pruned_fn {
  bool is_nullable;
  bool const* row_mask;
  bitmask_type const* bitmask;
  __device__ bool operator()(cudf::size_type row_idx) const
  {
    if (is_nullable and not bit_is_set(bitmask, row_idx)) { return false; }
    return not row_mask[row_idx];
  }
};

}  // namespace

bool hybrid_scan_reader_impl::are_all_rows_pruned(cudf::column_view const& row_mask,
                                                  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_mask.type().id() == type_id::BOOL8,
               "Input row mask column must be a boolean column");
  return cudf::detail::all_of(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{row_mask.size()},
    is_row_pruned_fn{row_mask.nullable(), row_mask.begin<bool>(), row_mask.null_mask()},
    stream);
}

void hybrid_scan_reader_impl::update_row_mask(cudf::column_view const& in_row_mask,
                                              cudf::mutable_column_view& out_row_mask,
                                              cudf::size_type out_row_mask_offset,
                                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // Total number of output row mask rows to be updated from the input
  auto const total_rows = static_cast<cudf::size_type>(in_row_mask.size());

  CUDF_EXPECTS(out_row_mask_offset + total_rows <= out_row_mask.size(),
               "Input and output row mask columns must have the same number of rows");
  CUDF_EXPECTS(out_row_mask.type().id() == type_id::BOOL8,
               "Output row mask column must be a boolean column");
  CUDF_EXPECTS(in_row_mask.type().id() == type_id::BOOL8,
               "Input row mask column must be a boolean column");

  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
    cuda::counting_iterator<cudf::size_type>{0},
    out_row_mask.begin<bool>() + out_row_mask_offset,
    total_rows,
    row_mask_update_fn{in_row_mask.nullable(), in_row_mask.begin<bool>(), in_row_mask.null_mask()},
    stream.value()));

  // Make sure the null mask of the output row mask column is all valid after the update. This is
  // to correctly assess if a payload column data page can be pruned. An invalid row in the row mask
  // column means the corresponding data page cannot be pruned.
  if (out_row_mask.nullable()) {
    cudf::set_null_mask(out_row_mask.null_mask(),
                        out_row_mask_offset,
                        out_row_mask_offset + total_rows,
                        true,
                        stream);
    out_row_mask.set_null_count(out_row_mask.null_count(0, out_row_mask.size(), stream));
  }
}

}  // namespace cudf::io::parquet::experimental::detail
