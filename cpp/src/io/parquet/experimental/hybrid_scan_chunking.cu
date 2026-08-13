/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/reader_impl_chunking.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>

#include <numeric>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::ColumnChunkDesc;
using parquet::detail::pass_intermediate_data;

void hybrid_scan_reader_impl::handle_chunking(
  read_mode mode,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  host_span<bool const> data_page_mask)
{
  // if this is our first time in here, setup the first pass.
  if (!_pass_itm_data) {
    // setup the next pass
    setup_next_pass(column_chunk_data);

    // Must be called as soon as we create the pass
    set_pass_page_mask(data_page_mask);
  }

  auto& pass = *_pass_itm_data;

  // if we already have a subpass in flight.
  if (pass.subpass != nullptr) {
    // if it still has more chunks in flight, there's nothing more to do
    if (pass.subpass->current_output_chunk < pass.subpass->output_chunk_read_info.size()) {
      return;
    }

    // increment rows processed
    pass.processed_rows += pass.subpass->num_rows;

    // release the old subpass (will free memory)
    pass.subpass.reset();

    // otherwise we are done with the pass entirely
    if (pass.processed_rows == pass.num_rows) {
      // release the old pass
      _pass_itm_data.reset();

      _file_itm_data._current_input_pass++;

      // no more passes. we are absolutely done with this file.
      CUDF_EXPECTS(_file_itm_data._current_input_pass == _file_itm_data.num_passes(),
                   "Hybrid scan reader must only create one pass per chunking setup");
      return;
    }
  }

  // setup the next sub pass
  setup_next_subpass(mode);
}

void hybrid_scan_reader_impl::setup_next_pass(
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data)
{
  auto const num_passes = _file_itm_data.num_passes();
  CUDF_EXPECTS(num_passes == 1,
               "The hybrid scan reader currently only supports single-pass read mode");

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<pass_intermediate_data>(_stream);

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty() && _file_itm_data._current_input_pass < num_passes) {
    auto& pass = *_pass_itm_data;

    // setup row groups to be loaded for this pass
    auto const row_group_start =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass];
    auto const row_group_end =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass + 1];
    auto const num_row_groups = row_group_end - row_group_start;
    pass.row_groups.resize(num_row_groups);
    std::copy(_file_itm_data.row_groups.begin() + row_group_start,
              _file_itm_data.row_groups.begin() + row_group_end,
              pass.row_groups.begin());

    CUDF_EXPECTS(_file_itm_data._current_input_pass < num_passes,
                 "Encountered an invalid read pass index");

    auto const chunks_per_rowgroup = _input_columns.size();
    auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

    auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
    auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

    pass.chunks = cudf::detail::hostdevice_vector<ColumnChunkDesc>(num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    pass.skip_rows = _file_itm_data.global_skip_rows;
    pass.num_rows  = _file_itm_data.global_num_rows;

    // Setup page information for the chunk (which we can access without decompressing)
    setup_compressed_data(column_chunk_data);

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(pass.pages,
                           pass.chunks,
                           uses_custom_row_bounds(read_mode::READ_ALL)
                             ? std::nullopt
                             : std::make_optional(pass.num_rows),
                           _stream);

    // Get the decompressed size of dictionary pages to help estimate memory usage
    auto const decomp_dict_data_size = std::accumulate(
      pass.pages.begin(), pass.pages.end(), size_t{0}, [&pass](size_t acc, auto const& page) {
        if (pass.chunks[page.chunk_idx].codec != Compression::UNCOMPRESSED &&
            (page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY)) {
          return acc + page.uncompressed_page_size;
        }
        return acc;
      });

    // store off how much memory we've used so far. This includes the compressed page data and the
    // decompressed dictionary data. we will subtract this from the available total memory for the
    // subpasses
    auto chunk_iter = thrust::make_transform_iterator(pass.chunks.d_begin(),
                                                      parquet::detail::get_chunk_compressed_size{});
    pass.base_mem_size =
      decomp_dict_data_size +
      cudf::detail::reduce(
        chunk_iter, chunk_iter + pass.chunks.size(), size_t{0}, cuda::std::plus<size_t>{}, _stream);

    // if we are doing subpass reading, generate more accurate num_row estimates for list columns.
    // this helps us to generate more accurate subpass splits.
    if (pass.has_compressed_data && _input_pass_read_limit != 0) {
      if (_has_offset_index) {
        generate_list_column_row_counts(is_estimate_row_counts::NO);
      } else {
        generate_list_column_row_counts(is_estimate_row_counts::YES);
      }
    }

    _stream.synchronize();
  }
}

}  // namespace cudf::io::parquet::experimental::detail
